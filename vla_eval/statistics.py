from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


GROUP_KEYS = ("suite", "scenario", "method", "horizon", "execute", "flow_steps",
              "partial_steps", "latency_mode")


def wilson(successes, count):
    if not count:
        return [None, None]
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    result = []
    for keys, samples in groups.items():
        item = dict(zip(GROUP_KEYS, keys))
        known = [s for s in samples if s.get("success") is not None]
        successes = sum(int(s["success"]) for s in known)
        item.update(trials=len(samples),
                    success_rate=successes / len(known) if known else None,
                    success_ci95=wilson(successes, len(known)))
        for key in ("position_error_mm", "rotation_error_deg", "first_action_delay_ms",
                    "post_inference_ms", "precompute_ms", "environment_time_s",
                    "total_episode_time_s", "generated_actions", "used_actions",
                    "wasted_actions", "deadline_miss_rate", "action_mse", "first_action_mse",
                    "full_chunk_action_mse", "generation_utilization",
                    "generated_actions_per_executed_action", "post_inference_per_executed_action_ms"):
            values = [s[key] for s in samples if s.get(key) is not None]
            if values:
                a = np.asarray(values, dtype=float)
                item[key + "_mean"] = float(a.mean())
                item[key + "_p50"] = float(np.quantile(a, 0.5))
                item[key + "_p95"] = float(np.quantile(a, 0.95))
        result.append(item)
    return result


def paired_comparisons(rows, bootstrap_samples=2000):
    """Compare same seed/scenario/H/K/N. Never pool unpaired trials."""
    groups = defaultdict(dict)
    for r in rows:
        key = tuple(r[k] for k in GROUP_KEYS if k != "method")
        pair = r.get("pair_id", str(r["seed"]))
        if pair in groups[key].get(r["method"], {}):
            raise ValueError("Duplicate paired trial identifier")
        groups[key].setdefault(r["method"], {})[pair] = r
    comparisons = []
    for key, methods in groups.items():
        contrasts = [(method, "B") for method in
                     ("A", "C", "D", "prior_only", "shuffled_prior")]
        contrasts += [("resume_only", "A"), ("D", "A")]
        for method, reference in contrasts:
            if method not in methods or reference not in methods:
                continue
            seeds = sorted(methods[method].keys() & methods[reference].keys())
            if not seeds:
                continue
            for metric in ("success", "position_error_mm", "first_action_delay_ms", "action_mse"):
                clusters = defaultdict(list)
                for seed in seeds:
                    a, b = methods[method][seed], methods[reference][seed]
                    if a.get(metric) is not None and b.get(metric) is not None:
                        # Frames and repeated flow seeds from one demonstration are correlated.
                        cluster = a.get("episode_id", seed)
                        clusters[cluster].append(float(a[metric]) - float(b[metric]))
                values = [np.mean(v) for v in clusters.values()]
                if not values:
                    continue
                delta = np.asarray(values)
                rng = np.random.default_rng(442)
                # Bound memory for larger real evaluation sets.
                boot = [delta[rng.integers(0, len(delta), len(delta))].mean()
                        for _ in range(bootstrap_samples)]
                comparisons.append({**dict(zip((k for k in GROUP_KEYS if k != "method"), key)),
                                    "comparison": method + "-" + reference, "metric": metric,
                                    "pairs": sum(len(v) for v in clusters.values()),
                                    "independent_units": len(delta),
                                    "mean_delta": float(delta.mean()),
                                    "paired_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist()})
    return comparisons
