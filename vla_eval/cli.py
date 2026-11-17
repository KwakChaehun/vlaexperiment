from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from .artifacts import Artifacts
from .runner import check_config, execute_trial, load_factory, trial_grid


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    check_config(config)
    return config


def main():
    parser = argparse.ArgumentParser(description="VLA precision evaluation: A/B/C/D and H/K/N sweeps")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local Python and optional GPU dependencies")
    for name in ("run", "offline", "plan"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        if name != "plan":
            sub.add_argument("--output")
        if name == "offline":
            sub.add_argument("--max-episodes", type=int)
            sub.add_argument("--max-samples-per-episode", type=int)
            sub.add_argument("--sample-stride", type=int)
            sub.add_argument("--episode-ids", nargs="+")
    training = commands.add_parser("train-small")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--regularization", type=float, default=1.0)
    profile = commands.add_parser("profile-gemm")
    profile.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    profile.add_argument("--hidden", type=int, default=2048)
    profile.add_argument("--width", type=int, default=8192)
    profile.add_argument("--state-tokens", type=int, default=1)
    profile.add_argument("--device", default="cuda:0")
    profile.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    profile.add_argument("--warmup", type=int, default=20)
    profile.add_argument("--iterations", type=int, default=100)
    profile.add_argument("--seed", type=int, default=0)
    profile.add_argument("--output", required=True)
    profile_model = commands.add_parser("profile-policy")
    profile_model.add_argument("--config", required=True)
    profile_model.add_argument("--output", required=True)
    profile_model.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"python": sys.executable, "numpy": importlib.util.find_spec("numpy") is not None,
                  "gr00t": importlib.util.find_spec("gr00t") is not None}
        if importlib.util.find_spec("torch") is not None:
            import torch
            result.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
            if torch.cuda.is_available():
                result.update(gpu=torch.cuda.get_device_name(0),
                              vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
        print(json.dumps(result, indent=2))
        return
    if args.command == "profile-gemm":
        from .profiling import profile_gemm
        print(json.dumps(profile_gemm(args), indent=2))
        return
    config = read_config(args.config)
    if args.command == "offline":
        selection = dict(config.get("selection", {}))
        for key in ("max_episodes", "max_samples_per_episode", "sample_stride", "episode_ids"):
            value = getattr(args, key)
            if value is not None:
                selection[key] = value
        if selection:
            config["selection"] = selection
    if args.command == "plan":
        trials = list(trial_grid(config))
        pairs = sorted({(t.horizon, t.execute, t.flow_steps) for t, _ in trials})
        print(json.dumps({"trials": len(trials), "valid_H_K_N": pairs,
                          "invalid_K_gt_H_skipped": True,
                          "backend": config.get("heavy_factory", config.get("backend"))}, indent=2))
        return
    if args.command == "train-small":
        from .offline import train_small
        path = Path(args.output)
        if path.exists() or path.suffix != ".npz":
            raise ValueError("Small output must be a new .npz path")
        path.parent.mkdir(parents=True, exist_ok=True)
        train_small(config, path, args.regularization)
        print(f"Saved Small baseline: {path}")
        return
    if args.command == "profile-policy":
        from .profiling import profile_policy
        profile_policy(config, args.output, args.sample_index)
        print(f"Saved profiler trace: {args.output}")
        return
    output = args.output or f"results/{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    artifacts = Artifacts(output, config)
    try:
        if args.command == "offline":
            from .offline import evaluate
            evaluate(config, artifacts.emit, artifacts.record_backend)
        else:
            heavy, small, env_factory = load_factory(config["backend"])(config)
            artifacts.record_backend(heavy, small)
            # Warm up in a separate environment with a dedicated seed.
            warm_env = env_factory(2147483, {**config.get("environment", {}), **config["scenarios"][0]})
            try:
                warm_obs = warm_env.observe()
                h, n = config["horizons"][0], config["flow_steps"][0]
                for i in range(config.get("warmup_calls", 2)):
                    heavy.predict(warm_obs, h, n, i)
                    small.predict(warm_obs, h, n, i)
                heavy.synchronize()
                small.synchronize()
            finally:
                if hasattr(warm_env, "close"):
                    warm_env.close()
            for index, (trial, scenario) in enumerate(trial_grid(config), start=1):
                artifacts.emit(execute_trial(trial, scenario, config, heavy, small, env_factory))
                if index % 50 == 0:
                    print(f"Completed {index} trials", flush=True)
        summary = artifacts.finish()
        print(f"Saved {len(artifacts.rows)} trials, {len(summary)} groups: {output}")
        if config.get("backend", "").startswith("vla_eval.toy:"):
            print("CPU FIXTURE ONLY: these results do not measure VLA accuracy, GPU speed, or PIM benefit.")
    except BaseException as error:
        artifacts.fail(error)
        raise


if __name__ == "__main__":
    main()
