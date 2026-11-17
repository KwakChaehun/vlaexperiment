"""Hardware measurements; arithmetic intensity is only an analytical lower-traffic estimate."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def profile_gemm(args):
    import torch
    if args.iterations < 2 or args.warmup < 1 or min(args.hidden, args.width) < 1:
        raise ValueError("Positive dimensions, warmup>=1 and iterations>=2 are required")
    if any(h < 1 for h in args.horizons) or args.state_tokens < 0:
        raise ValueError("H must be positive; state_tokens must be nonnegative")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("GEMM latency profiling requires CUDA; CPU fixtures are separate")
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    weight = torch.randn(args.hidden, args.width, device=device, dtype=dtype)
    results = []
    for horizon in args.horizons:
        rows = horizon + args.state_tokens
        x = torch.randn(rows, args.hidden, device=device, dtype=dtype)
        y = torch.empty(rows, args.width, device=device, dtype=dtype)
        for _ in range(args.warmup):
            torch.mm(x, weight, out=y)
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(device):
                start.record()
                torch.mm(x, weight, out=y)
                end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        flops = 2 * rows * args.hidden * args.width
        minimum_bytes = (rows * args.hidden + args.hidden * args.width + rows * args.width) * x.element_size()
        median = float(np.median(samples))
        results.append({"horizon": horizon, "state_tokens": args.state_tokens, "batch": 1,
                        "gemm_m": rows, "gemm_k": args.hidden, "gemm_n": args.width,
                        "dtype": args.dtype, "gpu": torch.cuda.get_device_name(device),
                        "latency_p50_ms": median, "latency_p95_ms": float(np.quantile(samples, .95)),
                        "tflops": flops / (median / 1000) / 1e12,
                        "minimum_tensor_traffic_bytes": minimum_bytes,
                        "ideal_arithmetic_intensity_flop_per_byte": flops / minimum_bytes,
                        "caveat": "Repeated GEMM may cache weights. Does not measure DRAM bandwidth or PIM speedup."})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    return results


def profile_policy(config, output, sample_index=0):
    import torch
    from .dataset import PairedDataset
    from .offline import policies
    from .runner import timed
    heavy, _ = policies({**config, "methods": ["A"]})
    dataset = PairedDataset(config["dataset"], config.get("language_key", "task"))
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    try:
        obs = dataset.observation(sample_index)
        horizon, steps = config["horizons"][0], config["flow_steps"][0]
        for seed in range(3):
            timed(heavy, heavy.predict, obs, horizon, steps, seed)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True, with_flops=True) as profiler:
            heavy.predict(obs, horizon, steps, 0)
            heavy.synchronize()
        profiler.export_chrome_trace(str(path / "trace.json"))
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        table = profiler.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=80)
        (path / "operators.txt").write_text(table, encoding="utf-8")
    finally:
        dataset.close()
