"""Paired H/K contrasts with N and all other grouping variables held fixed."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from .statistics import GROUP_KEYS


CONTEXT_KEYS = tuple(k for k in GROUP_KEYS if k not in {"horizon", "execute"})
METRICS = ("success", "position_error_mm", "rotation_error_deg", "first_action_delay_ms",
           "post_inference_ms", "post_inference_per_executed_action_ms",
           "total_episode_time_s", "deadline_miss_rate", "generation_utilization",
           "generated_actions_per_executed_action", "first_action_mse")


def _paired_delta(candidate, reference, metric, bootstrap_samples):
    units = defaultdict(list)
    for pair in sorted(candidate.keys() & reference.keys()):
        a, b = candidate[pair], reference[pair]
        if a.get(metric) is None or b.get(metric) is None:
            continue
        delta = float(a[metric]) - float(b[metric])
        if not np.isfinite(delta):
            raise ValueError(f"Non-finite {metric} in horizon comparison")
        # Repeated snapshots/noise samples from an episode form one sampling unit.
        units[a.get("episode_id", pair)].append(delta)
    if not units:
        return None
    values = np.asarray([np.mean(v) for v in units.values()])
    rng = np.random.default_rng(442)
    bootstrap = np.empty(bootstrap_samples)
    # Bound temporary index allocation for large episode sets.
    batch = max(1, min(128, 1000000 // len(values)))
    for start in range(0, bootstrap_samples, batch):
        stop = min(bootstrap_samples, start + batch)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        bootstrap[start:stop] = values[indices].mean(axis=1)
    return {"pairs": sum(len(v) for v in units.values()), "independent_units": len(values),
            "mean_delta": float(values.mean()),
            "paired_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist()}


def horizon_comparisons(rows, bootstrap_samples=2000):
    """Compare smaller settings with the largest observed setting on the same axis.

    The reference is the largest *evaluated* H or K, not necessarily a model's
    native horizon. Joint diagonal contrasts cannot identify which axis helped.
    Offline snapshots support generation comparisons only: K does not cause
    feedback there, and MSE over different prefixes is not a K quality test.
    """
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        context = tuple(row[k] for k in CONTEXT_KEYS)
        setting = (row["horizon"], row["execute"])
        if not 1 <= setting[1] <= setting[0]:
            raise ValueError("Horizon analysis requires 1 <= K <= H")
        pair = row.get("pair_id", str(row["seed"]))
        if pair in groups[context][setting]:
            raise ValueError("Duplicate H/K paired trial identifier")
        groups[context][setting][pair] = row
    result = []
    for context, settings in groups.items():
        contrasts = []
        for k in sorted({k for h, k in settings}):
            horizons = sorted(h for h, execute in settings if execute == k)
            contrasts.extend(("generation", (h, k), (horizons[-1], k)) for h in horizons[:-1])
        if not str(context[0]).startswith("offline_"):
            for h in sorted({h for h, k in settings}):
                executions = sorted(k for horizon, k in settings if horizon == h)
                contrasts.extend(("execution", (h, k), (h, executions[-1]))
                                 for k in executions[:-1])
            diagonal = sorted(h for h, k in settings if h == k)
            if diagonal:
                contrasts.extend(("joint", (h, h), (diagonal[-1], diagonal[-1]))
                                 for h in diagonal[:-1])
        for axis, candidate, reference in contrasts:
            metrics = METRICS + (("action_mse",) if axis == "generation" else ())
            for metric in metrics:
                estimate = _paired_delta(settings[candidate], settings[reference], metric,
                                         bootstrap_samples)
                if estimate is None:
                    continue
                result.append({**dict(zip(CONTEXT_KEYS, context)), "axis": axis,
                               "candidate": {"H": candidate[0], "K": candidate[1]},
                               "reference": {"H": reference[0], "K": reference[1]},
                               "reference_rule": "largest evaluated value on this axis",
                               "metric": metric,
                               "positive_favors_candidate": metric in {"success", "generation_utilization"},
                               "effect_attribution": "combined H and K; cannot separate effects"
                                   if axis == "joint" else "one axis varied, other axis fixed",
                               **estimate})
    return result
