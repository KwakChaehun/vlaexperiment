"""CPU ridge baseline for Small-only and prior correction using identical weights.

This is a cheap baseline, not a substitute for a trained small VLA. Training
features include available observations and the Heavy prior, never test labels.
"""
from __future__ import annotations

import numpy as np


def observation_features(obs):
    features = [np.asarray(obs.state, dtype=np.float64).ravel()]
    for key in sorted(obs.images):
        image = np.asarray(obs.images[key], dtype=np.float64)
        if image.ndim == 4:
            image = image[-1]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Small image features expect [height, width, 3] or [time, height, width, 3]")
        # Spatial color grid preserves rough localization. A learned encoder can replace this.
        height, width = image.shape[:2]
        if min(height, width) < 4:
            raise ValueError("Images must be at least 4x4")
        grid = [cell.mean(axis=(0, 1)) / 255 for row in np.array_split(image, 4, axis=0)
                for cell in np.array_split(row, 4, axis=1)]
        features.append(np.asarray(grid).ravel())
    return np.concatenate(features)


class RidgeSmall:
    name = "shared_ridge_small_and_residual_baseline"

    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as data:
            self.weight = data["weight"]
            self.mean = data["mean"]
            self.scale = data["scale"]
            self.horizon = int(data["horizon"])
            self.action_dim = int(data["action_dim"])
            self.train_episode_ids = set(data["train_episode_ids"].tolist())

    def synchronize(self):
        pass

    def _infer(self, obs, prior):
        x = np.r_[observation_features(obs), prior.ravel(), float(np.any(prior))]
        if len(x) != len(self.mean):
            raise ValueError("Small checkpoint feature schema differs from observation")
        y = np.r_[(x - self.mean) / self.scale, 1.0] @ self.weight
        return (prior + y.reshape(self.horizon, self.action_dim)).astype(np.float32)

    def predict(self, obs, horizon, steps, seed):
        if horizon != self.horizon:
            raise ValueError("Ridge baseline has a fixed trained horizon")
        return self._infer(obs, np.zeros((horizon, self.action_dim)))

    def correct(self, obs, prior, forecast):
        if prior.shape != (self.horizon, self.action_dim):
            raise ValueError("Prior shape differs from trained Small checkpoint")
        return self._infer(obs, prior)


def fit_ridge(features, priors, targets, episode_ids, output, regularization=1.0):
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    features, priors, targets = map(lambda a: np.asarray(a, dtype=np.float64),
                                   (features, priors, targets))
    if targets.ndim != 3 or priors.shape != targets.shape or features.ndim != 2:
        raise ValueError("Expected features [samples,F], priors and targets [samples,H,D]")
    if not (len(features) == len(priors) == len(episode_ids)) or not len(features):
        raise ValueError("Sample counts differ or dataset is empty")
    if not all(np.isfinite(a).all() for a in (features, priors, targets)):
        raise ValueError("Training data contains non-finite values")
    count, horizon, action_dim = targets.shape
    flat_prior = priors.reshape(count, -1)
    # Joint training with/without a prior gives B and C exactly the same parameter budget.
    x_b = np.c_[features, np.zeros_like(flat_prior), np.zeros(count)]
    x_c = np.c_[features, flat_prior, np.any(flat_prior != 0, axis=1).astype(float)]
    x = np.r_[x_b, x_c]
    y = np.r_[targets.reshape(count, -1), (targets - priors).reshape(count, -1)]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1
    design = np.c_[(x - mean) / scale, np.ones(len(x))]
    penalty = np.eye(design.shape[1]) * regularization
    penalty[-1, -1] = 0
    weight = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    np.savez_compressed(output, weight=weight, mean=mean, scale=scale, horizon=horizon,
                        action_dim=action_dim, train_episode_ids=np.unique(episode_ids).astype(str))
