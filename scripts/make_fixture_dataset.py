"""Create a small paired CPU fixture for checking the offline I/O pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vla_eval.toy import ToyPrecisionEnv, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/fixture.npz")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()
    if args.episodes < 4 or args.horizon < 1:
        raise ValueError("Need >=4 episodes and a positive horizon")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecasts, arrivals, targets = [], [], []
    for seed in range(args.episodes):
        env = ToyPrecisionEnv(seed, {})
        forecast = env.forecast().observation
        arrival = env.observe()
        forecasts.append(forecast.state)
        arrivals.append(arrival.state)
        targets.append(plan(arrival, args.horizon))
    split = ["train" if i < int(args.episodes * .6) else "val"
             if i < int(args.episodes * .8) else "test" for i in range(args.episodes)]
    np.savez_compressed(output, episode_id=np.array([f"fixture_{i}" for i in range(args.episodes)]),
                        split=np.array(split), state=np.stack(arrivals), forecast_state=np.stack(forecasts),
                        actions=np.stack(targets), action_mask=np.ones((args.episodes, args.horizon), bool),
                        forecast_source=np.array(["predicted_observation"] * args.episodes),
                        forecast_generation_ms=np.zeros(args.episodes))
    print(f"Saved CPU FIXTURE ONLY: {output}")


if __name__ == "__main__":
    main()
