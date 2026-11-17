"""Cycle-simulate single-vector GR00T projection shapes, NOT full inference.

Requires the locally built Samsung PIMSimulator described in docs/pim_simulation.md.
Uses its 64-channel timing-only GEMV fixture with dummy data and resident weights.
No checkpoint weights or activations are executed; no GPU speedup is calculated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KEYS = {
    "query": "attn1.to_q.weight",
    "key": "attn1.to_k.weight",
    "ff_up": "ff.net.0.proj.weight",
    "ff_down": "ff.net.2.weight",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "shape_summary.json").exists():
        raise ValueError("Use a new output directory")
    simulator = ROOT / ".runtime/PIMSimulator"
    prefix = "action_head.model.transformer_blocks.0."
    metadata = {}
    for file in (ROOT / "checkpoints/gr00t_n16_libero").glob("*.safetensors"):
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        metadata.update({k: v for k, v in header.items() if k.startswith(prefix)})
    cases = []
    for name, suffix in KEYS.items():
        entry = metadata[prefix + suffix]
        output_dim, input_dim = entry["shape"]
        # 64 channels * 8 PIM blocks/channel * 8 GRF_B = 4096 outputs/tile.
        # 8 GRF_A * 16 FP16 elements/burst * 2 bank parities = 256 inputs.
        padded_out = ((output_dim + 4095) // 4096) * 4096
        padded_in = ((input_dim + 255) // 256) * 256
        cases.append(dict(name=name, tensor=prefix + suffix,
                          checkpoint_dtype=entry["dtype"], shape=[output_dim, input_dim],
                          padded_shape=[padded_out, padded_in], vectors=1))
    source = '#include "tests/PIMBenchTestCases.h"\n'
    for case in cases:
        output_dim, input_dim = case["padded_shape"]
        source += (
            f'TEST_F(PIMBenchFixture, vla_{case["name"]}) {{\n'
            f'  setPIMBenchTestCase(KernelType::GEMV, {output_dim}, {input_dim});\n'
            '  executeKernel();\n  executePIMKernel();\n}\n'
        )
    (simulator / "src/tests/VLAShapeProbe.cpp").write_text(source)
    (out / "VLAShapeProbe.cpp").write_text(source)
    with (out / "shape_build.log").open("w") as log:
        subprocess.run([str(ROOT / ".runtime/pim-tools/bin/scons"), "NO_LIBRARY=1"],
                       cwd=simulator, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out / "shape_run.log").open("w") as log:
        subprocess.run(["./sim", "--gtest_filter=PIMBenchFixture.vla_*"],
                       cwd=simulator, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    log = (out / "shape_run.log").read_text()
    hardware = (simulator / "ini/HBM2_samsung_2M_16B_x64.ini").read_text()
    tck_ns = float(re.search(r"^tCK\s*=\s*([\d.]+)", hardware, re.M)[1])
    for case in cases:
        name = "PIMBenchFixture.vla_" + case["name"]
        block = log.split("[ RUN      ] " + name + "\n")[1].split("[       OK ]")[0]
        cycles = [int(x) for x in re.findall(r"> Cycle : (\d+)", block)]
        if len(cycles) != 2 or min(cycles) <= 0:
            raise ValueError(f"Invalid simulator output: {name}")
        case.update(memory_traffic_only_baseline_cycles=cycles[0], pim_cycles=cycles[1],
                    pim_us=cycles[1] * tck_ns / 1000)
    result = dict(
        upstream="https://github.com/SAITPublic/PIMSimulator",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=simulator,
                                         text=True).strip(),
        channels=64, pim_blocks_per_channel=8, tck_ns=tck_ns,
        scope="Timing-only FP16 single-vector GEMV shape probes; dummy data; no model inference",
        exclusions=["weight preload", "host/device link", "explicit host output readback",
                    "attention softmax", "normalization and activation", "full H50 or H16 sequence",
                    "flow steps", "functional checks on model tensors", "BF16/FP16 accuracy"],
        baseline="Upstream non-PIM fixture counts memory traffic only; NOT RTX3090 compute",
        cases=cases,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "shape_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
