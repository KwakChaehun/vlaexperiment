"""Measured GPU projection microbenchmarks versus existing PIM shape simulations.

Single-vector timing comparisons only; no full VLA/PIM speedup claim. The GPU
also evaluates 16/50-vector GEMM to expose token reuse, without fabricating PIM
multi-vector timings. All allocations/transfers are outside timed regions.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch


def stats(values):
    return dict(p50=float(np.median(values)), p95=float(np.percentile(values, 95)),
                minimum=float(min(values)), mean=float(np.mean(values)))


@torch.inference_mode()
def benchmark(case, layout, dtype_name, vectors, samples):
    out_dim, in_dim = case["shape" if layout == "logical" else "padded_shape"]
    dtype = getattr(torch, dtype_name)
    w = torch.randn((out_dim, in_dim), device="cuda", dtype=dtype) * .02
    x = torch.randn((vectors, in_dim), device="cuda", dtype=dtype)
    y = torch.empty((vectors, out_dim), device="cuda", dtype=dtype)
    wt = w.t()
    flush = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    for _ in range(30):
        torch.mm(x, wt, out=y)
    torch.cuda.synchronize()
    assert torch.isfinite(y).all().item()
    # CUDA graph provides a launch-amortized repeated-weight, cache-hot baseline.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            torch.mm(x, wt, out=y)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(32):
            torch.mm(x, wt, out=y)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    graph_us, hot_us, wall_us = [], [], []
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(samples):
        start.record(); graph.replay(); end.record(); end.synchronize()
        graph_us.append(start.elapsed_time(end) * 1000 / 32)
    for _ in range(samples):
        before = time.perf_counter_ns()
        start.record(); torch.mm(x, wt, out=y); end.record(); end.synchronize()
        wall_us.append((time.perf_counter_ns() - before) / 1000)
        hot_us.append(start.elapsed_time(end) * 1000)
    # Flush well beyond 3090 L2 before each timed matmul. Flush time excluded.
    # This approximates a weight-streaming layer, not a measured full-model cache.
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(samples)]
    for start, end in pairs:
        flush.zero_()
        start.record(); torch.mm(x, wt, out=y); end.record()
    torch.cuda.synchronize()
    cold_us = [a.elapsed_time(b) * 1000 for a, b in pairs]
    return dict(name=case["name"], layout=layout, dtype=dtype_name, vectors=vectors,
                shape=[out_dim, in_dim], samples=samples,
                hot_graph_per_matmul_us=stats(graph_us), hot_eager_device_us=stats(hot_us),
                hot_eager_host_sync_us=stats(wall_us), cold_device_us=stats(cold_us),
                raw_us=dict(hot_graph=graph_us, hot_eager_device=hot_us,
                            hot_host_sync=wall_us, cold_device=cold_us))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "gpu_measurements.jsonl").exists():
        raise ValueError("Use a new output directory")
    source = Path("reports/pim_simulator_20261117/shape_summary.json")
    pim = json.loads(source.read_text())
    torch.manual_seed(917)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    def telemetry():
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,"
            "clocks.sm,clocks.mem,power.draw,power.limit,memory.used,utilization.gpu",
            "--format=csv"], text=True)
    manifest = dict(torch=torch.__version__, cuda=torch.version.cuda, device=props.name,
                    sm_count=props.multi_processor_count, memory_bytes=props.total_memory,
                    samples_per_run=args.samples, repeats=args.repeats,
                    seed=917, cache_flush_bytes=64 * 1024 * 1024, graph_matmuls=32,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    pim_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    telemetry_before=telemetry(), scope="Projection kernels only; resident random GPU tensors; PIM dummy inputs")
    variants = list(itertools.product(pim["cases"], ["logical", "padded"],
                                     ["float16", "bfloat16"], [1, 16, 50]))
    all_rows = []
    for repeat in range(args.repeats):
        random.Random(917 + repeat).shuffle(variants)
        for case, layout, dtype, vectors in variants:
            row = benchmark(case, layout, dtype, vectors, args.samples)
            row["repeat"] = repeat
            all_rows.append(row)
            with (out / "gpu_measurements.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"Completed repeat {repeat + 1}/{args.repeats}, {len(all_rows)} cases", flush=True)
    summary = []
    fields = {"hot_graph_per_matmul_us": "hot_graph", "hot_eager_device_us": "hot_eager_device",
              "hot_eager_host_sync_us": "hot_host_sync", "cold_device_us": "cold_device"}
    for case, layout, dtype, vectors in variants:
        rows = [r for r in all_rows if (r["name"], r["layout"], r["dtype"], r["vectors"]) ==
                (case["name"], layout, dtype, vectors)]
        entry = {k: rows[0][k] for k in ("name", "layout", "dtype", "vectors", "shape")}
        entry["samples"] = sum(r["samples"] for r in rows)
        for field, raw in fields.items():
            entry[field] = stats([v for r in rows for v in r["raw_us"][raw]])
        if vectors == 1 and dtype == "float16":
            entry["pim_us"] = case["pim_us"]
            entry["cold_gpu_over_pim_kernel_ratio"] = entry["cold_device_us"]["p50"] / case["pim_us"]
        summary.append(entry)
    manifest["telemetry_after"] = telemetry()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "gpu_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps([r for r in summary if r["layout"] == "logical" and
                      r["dtype"] == "float16" and r["vectors"] == 1], indent=2))


if __name__ == "__main__":
    main()
