from __future__ import annotations

from dataclasses import asdict
import numpy as np
from .contracts import validate_chunk
from .dataset import PairedDataset
from .runner import load_factory, timed, trial_grid
from .small import RidgeSmall


def policies(config):
    if "heavy_factory" in config:
        heavy = load_factory(config["heavy_factory"])(config.get("heavy", {}))
        small = RidgeSmall(config["small_checkpoint"]) if config.get("small_checkpoint") else None
        if set(config["methods"]) - {"A", "prior_only", "resume_only"} and small is None:
            raise ValueError("B/C/D require small_checkpoint or a custom backend factory")
        return heavy, small
    heavy, small, _ = load_factory(config["backend"])(config)
    return heavy, small


def evaluate(config, emit, report_backend=None):
    heavy, small = policies(config)
    if report_backend is not None:
        report_backend(heavy, small)
    dataset = PairedDataset(config["dataset"], config.get("language_key", "task"))
    try:
        split = config.get("split", "test")
        diagnostic = split == "train"
        if diagnostic and not (config.get("allow_train_diagnostics", False)
                               and set(config["methods"]) == {"A"}):
            raise ValueError("Train-split diagnostics require explicit allow_train_diagnostics and methods=[A]")
        indices = dataset.select_indices(split, **config.get("selection", {}))
        if not indices:
            raise ValueError(f"No samples in {split}")
        if hasattr(small, "train_episode_ids"):
            overlap = small.train_episode_ids & set(dataset.data["episode_id"][indices].astype(str))
            if overlap:
                raise ValueError("Small training episodes overlap with evaluation episodes")
        warmed = set()
        for trial, scenario in trial_grid(config):
            h = trial.horizon
            if h > dataset.data["actions"].shape[1]:
                raise ValueError("Requested H exceeds stored target length")
            if (h, trial.flow_steps) not in warmed:
                warm_indices = indices
                warm_obs = dataset.observation(warm_indices[0])
                for warm_seed in range(config.get("warmup_calls", 2)):
                    heavy.predict(warm_obs, h, trial.flow_steps, warm_seed)
                    if small is not None:
                        small.predict(warm_obs, h, trial.flow_steps, warm_seed)
                heavy.synchronize()
                if small is not None:
                    small.synchronize()
                warmed.add((h, trial.flow_steps))
            for index in indices:
                target = dataset.data["actions"][index, :h]
                mask = dataset.data["action_mask"][index, :h]
                if not mask.any():
                    continue
                method = trial.method
                pre_ms, elapsed = 0.0, 0.0
                forecast, prior, cache, source = None, None, None, None
                if method in {"C", "D", "prior_only", "shuffled_prior", "resume_only"}:
                    forecast_index = index
                    if method == "shuffled_prior":
                        candidates = [j for j in indices if dataset.data["episode_id"][j]
                                      != dataset.data["episode_id"][index]]
                        if not candidates:
                            raise ValueError("shuffled_prior needs at least two distinct episodes")
                        forecast_index = candidates[(index + trial.seed) % len(candidates)]
                    source = dataset.forecast_source(forecast_index)
                    forecast = dataset.observation(forecast_index, forecast=True)
                    if method in {"D", "resume_only"}:
                        cache, pre_ms = timed(heavy, heavy.prefill, forecast, h, trial.flow_steps,
                                              trial.partial_steps, trial.seed)
                    else:
                        prior, pre_ms = timed(heavy, heavy.predict, forecast, h, trial.flow_steps,
                                              trial.seed)
                        prior = validate_chunk(prior, h)
                    pre_ms += dataset.forecast_latency_ms(forecast_index)
                # Causal ordering: actual arrival inputs first accessed AFTER precompute.
                obs = dataset.observation(index)
                if method == "A":
                    actions, elapsed = timed(heavy, heavy.predict, obs, h, trial.flow_steps, trial.seed)
                elif method == "B":
                    actions, elapsed = timed(small, small.predict, obs, h, trial.flow_steps, trial.seed)
                elif method in {"D", "resume_only"}:
                    actions, resume_ms = timed(heavy, heavy.resume, obs, cache)
                    elapsed = resume_ms
                    if method == "D":
                        actions, correction_ms = timed(small, small.correct, obs, actions, obs)
                        elapsed += correction_ms
                elif method == "prior_only":
                    actions = prior
                else:
                    actions, elapsed = timed(small, small.correct, obs, prior, forecast)
                actions = validate_chunk(actions, h)
                used_mask = mask[:trial.execute]
                if not used_mask.any():
                    continue
                mse = float(np.mean((actions[:trial.execute][used_mask]
                                     - target[:trial.execute][used_mask]) ** 2))
                reach_ms = 1000 * scenario.get("reach_duration_s", config.get(
                    "environment", {}).get("reach_duration_s", 0.5))
                row = {**asdict(trial), "suite": "offline_" + trial.suite,
                       "scenario": scenario["name"], "seed": trial.seed,
                       "pair_id": f"sample:{index}/seed:{trial.seed}",
                       "inference_seed": trial.seed, "sample_index": index,
                       "episode_id": str(dataset.data["episode_id"][index]),
                       "diagnostic_only": diagnostic,
                       "latency_mode": "measured_wall_with_analytical_overlap",
                       "success": None, "action_mse": mse,
                       "first_action_mse": float(np.mean((actions[0] - target[0]) ** 2)),
                       "full_chunk_action_mse": float(np.mean((actions[mask] - target[mask]) ** 2)),
                       "precompute_ms": pre_ms, "post_inference_ms": elapsed,
                       "first_action_delay_ms": max(0, pre_ms - reach_ms) + elapsed,
                       "generated_actions": h, "used_actions": int(used_mask.sum()),
                       "wasted_actions": h - int(used_mask.sum()), "forecast_source": source}
                emit(row)
    finally:
        dataset.close()


def train_small(config, output, regularization):
    dataset = PairedDataset(config["dataset"], config.get("language_key", "task"))
    heavy, _ = policies({**config, "methods": ["A"], "small_checkpoint": None})
    from .small import fit_ridge, observation_features
    try:
        h = config["horizons"][0]
        features, priors, targets, ids = [], [], [], []
        for index in dataset.indices("train"):
            if not dataset.data["action_mask"][index, :h].all():
                continue
            dataset.forecast_source(index)
            forecast = dataset.observation(index, forecast=True)
            prior = validate_chunk(heavy.predict(forecast, h, config["flow_steps"][0], index), h)
            obs = dataset.observation(index)
            features.append(observation_features(obs))
            priors.append(prior)
            targets.append(dataset.data["actions"][index, :h])
            ids.append(str(dataset.data["episode_id"][index]))
        if not features:
            raise ValueError("No training records have a full valid target chunk")
        fit_ridge(features, priors, targets, ids, output, regularization)
    finally:
        dataset.close()
