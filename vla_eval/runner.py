from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import math
import time
import numpy as np
from .contracts import validate_chunk


METHODS = {"A", "B", "C", "D", "prior_only", "shuffled_prior", "resume_only"}


def load_factory(spec):
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def check_config(config):
    for key in ("seeds", "horizons", "execute", "flow_steps"):
        values = config.get(key)
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a non-empty list without duplicates")
        minimum = 0 if key == "seeds" else 1
        if any(isinstance(x, bool) or not isinstance(x, int) or x < minimum for x in values):
            raise ValueError(f"{key} must contain integers >= {minimum}")
    if not set(config.get("methods", [])) <= METHODS or not config.get("methods"):
        raise ValueError("Unknown or missing methods")
    if len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Duplicate methods")
    if config.get("latency_mode", "wall") not in {"wall", "zero", "model"}:
        raise ValueError("latency_mode must be wall, zero, or model")
    fraction = config.get("partial_fraction", 0.5)
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("partial_fraction must be in [0, 1)")
    if not config.get("scenarios"):
        raise ValueError("At least one scenario is required")
    names = [s["name"] for s in config["scenarios"]]
    if len(names) != len(set(names)):
        raise ValueError("Scenario names must be unique")
    for scenario in config["scenarios"]:
        env = {**config.get("environment", {}), **scenario}
        for key, default in (("control_hz", 20), ("max_actions", 80)):
            if not math.isfinite(env.get(key, default)) or env.get(key, default) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        reach = env.get("reach_duration_s", 0.5)
        if not math.isfinite(reach) or reach < 0:
            raise ValueError("reach_duration_s must be finite and nonnegative")
    for key, value in config.get("modeled_latency_ms", {}).items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid modeled latency: {key}")
    if not any(k <= h for k in config["execute"] for h in config["horizons"]):
        raise ValueError("No valid K <= H combinations")


def timed(owner, function, *args):
    owner.synchronize()
    start = time.perf_counter()
    value = function(*args)
    owner.synchronize()
    return value, (time.perf_counter() - start) * 1000


@dataclass(frozen=True)
class Trial:
    suite: str
    scenario: str
    method: str
    seed: int
    horizon: int
    execute: int
    flow_steps: int
    partial_steps: int


def trial_grid(config):
    check_config(config)
    for scenario in config["scenarios"]:
        for horizon in config["horizons"]:
            for execute in config["execute"]:
                if execute > horizon:
                    continue
                for steps in config["flow_steps"]:
                    partial = min(steps - 1, int(steps * config.get("partial_fraction", 0.5)))
                    for seed in config["seeds"]:
                        for method in config["methods"]:
                            yield Trial(config.get("suite", "comparison"), scenario["name"], method,
                                        seed, horizon, execute, steps, partial), scenario


def execute_trial(trial, scenario, config, heavy, small, env_factory):
    env_config = {**config.get("environment", {}), **scenario}
    env = env_factory(trial.seed, env_config)
    h, k, n = trial.horizon, trial.execute, trial.flow_steps
    mode = config.get("latency_mode", "wall")
    model_latency = config.get("modeled_latency_ms", {})
    def delay(stage, measured):
        if mode == "zero":
            return 0.0
        if mode == "model":
            if stage not in model_latency:
                raise ValueError(f"Missing modeled_latency_ms.{stage}")
            return model_latency[stage]
        return measured

    precompute_ms = 0.0
    forecast = None
    prior = None
    cache = None
    generated = used = calls = deadline_misses = 0
    post_times, trace = [], []
    pre_delay = 0.0
    forecast_ms = 0.0
    try:
        if trial.method in {"C", "D", "prior_only", "shuffled_prior", "resume_only"}:
            forecast_start = time.perf_counter()
            forecast = env.forecast()  # Always BEFORE the first arrival observe().
            forecast_ms = (time.perf_counter() - forecast_start) * 1000
            if trial.method == "shuffled_prior":
                other = env_factory(trial.seed + 1000003, env_config)
                try:
                    forecast_start = time.perf_counter()
                    forecast = other.forecast()
                    forecast_ms = (time.perf_counter() - forecast_start) * 1000
                finally:
                    if hasattr(other, "close"):
                        other.close()
            if trial.method in {"D", "resume_only"}:
                cache, precompute_ms = timed(heavy, heavy.prefill, deepcopy(forecast.observation),
                                             h, n, trial.partial_steps, trial.seed)
                pre_delay = delay("partial_pre", precompute_ms)
            else:
                prior, precompute_ms = timed(heavy, heavy.predict, deepcopy(forecast.observation),
                                             h, n, trial.seed)
                prior = validate_chunk(prior, h)
                generated += h
                pre_delay = delay("heavy_pre", precompute_ms)
            precompute_ms += forecast_ms
            if mode == "wall":
                pre_delay += forecast_ms
        reach_ms = 1000 * env_config.get("reach_duration_s", 0.5)
        residual_wait_ms = max(0.0, pre_delay - reach_ms)
        # Sequential measurement + overlap accounting, not actual concurrent GPU execution.
        if residual_wait_ms:
            env.advance(residual_wait_ms / 1000)
        first_delay_ms = 0.0
        while not env.done():
            observation = env.observe()
            first = calls == 0
            sample_seed = trial.seed if first else trial.seed * 100003 + calls
            if trial.method == "A" or (trial.method == "resume_only" and not first):
                actions, elapsed = timed(heavy, heavy.predict, observation, h, n, sample_seed)
                stage = "heavy_post"
                generated += h
            elif first and trial.method == "prior_only":
                actions, elapsed, stage = prior, 0.0, "prior_post"
            elif first and trial.method in {"C", "shuffled_prior"}:
                actions, elapsed = timed(small, small.correct, observation, prior,
                                         forecast.observation)
                stage = "correction_post"
            elif first and trial.method in {"D", "resume_only"}:
                resumed, resume_ms = timed(heavy, heavy.resume, observation, cache)
                resumed = validate_chunk(resumed, h)
                if trial.method == "resume_only":
                    # Isolate reuse quality without the Small corrector as a confound.
                    actions, elapsed, stage = resumed, resume_ms, "resume_post"
                else:
                    # Legacy D: fresh-conditioned Heavy continuation + Small residual.
                    actions, correction_ms = timed(small, small.correct, observation, resumed, observation)
                    elapsed = resume_ms + correction_ms
                    stage = "resume_and_correction_post"
                generated += h
            else:
                actions, elapsed = timed(small, small.predict, observation, h, n, sample_seed)
                stage = "small_post"
                generated += h
            actions = validate_chunk(actions, h)
            modeled_ms = delay(stage, elapsed)
            if first:
                first_delay_ms = residual_wait_ms + modeled_ms
            # Observation is held fixed during synchronous inference; world time keeps advancing.
            env.advance(modeled_ms / 1000)
            executed_this_call = 0
            for action in actions[:k]:
                if env.done():
                    break
                env.step(action)
                used += 1
                executed_this_call += 1
            if not executed_this_call and not env.done():
                raise RuntimeError("Environment made no progress")
            deadline_ms = 1000 * k / env_config.get("control_hz", 20)
            deadline_misses += int(modeled_ms > deadline_ms)
            calls += 1
            post_times.append(elapsed)
            trace.append({"call": calls, "stage": stage, "measured_ms": elapsed,
                          "applied_delay_ms": modeled_ms, "generated_horizon": h,
                          "executed": executed_this_call, "deadline_budget_ms": deadline_ms})
            if calls > env_config.get("max_policy_calls", 10000):
                raise RuntimeError("Environment exceeded max_policy_calls")
        metrics = env.metrics()
        return {**vars(trial), "latency_mode": mode, **metrics,
                "precompute_ms": precompute_ms, "residual_wait_ms": residual_wait_ms,
                "forecast_generation_ms": forecast_ms, "intervention_scope": "arrival_only",
                "first_action_delay_ms": first_delay_ms,
                "post_inference_ms": float(np.mean(post_times)) if post_times else 0.0,
                "policy_calls": calls, "generated_actions": generated, "used_actions": used,
                "wasted_actions": generated - used,
                "generation_utilization": used / generated if generated else None,
                "generated_actions_per_executed_action": generated / used if used else None,
                "post_inference_per_executed_action_ms": sum(post_times) / used if used else None,
                "deadline_miss_rate": deadline_misses / max(calls, 1),
                "total_episode_time_s": env_config.get("reach_duration_s", 0.5)
                                        + metrics["environment_time_s"],
                "forecast_source": forecast.source if forecast else None,
                "trace": trace}
    finally:
        if hasattr(env, "close"):
            env.close()
