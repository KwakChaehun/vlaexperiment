"""Pickle-free paired observations for open-loop screening and Small training."""
from __future__ import annotations

import numpy as np
from .contracts import Observation


class PairedDataset:
    def __init__(self, path, language_key="task"):
        self.data = np.load(path, allow_pickle=False)
        self.language_key = language_key
        try:
            required = {"episode_id", "split", "state", "actions", "action_mask"}
            if not required <= set(self.data.files):
                raise ValueError(f"Missing dataset arrays: {sorted(required - set(self.data.files))}")
            self.count = len(self.data["episode_id"])
            if self.count == 0:
                raise ValueError("Dataset is empty")
            if any(len(self.data[k]) != self.count for k in required):
                raise ValueError("Dataset arrays have different sample counts")
            actions, mask = self.data["actions"], self.data["action_mask"]
            if actions.ndim != 3 or mask.shape != actions.shape[:2]:
                raise ValueError("actions must be [samples,H,D], action_mask [samples,H]")
            if mask.dtype != np.bool_:
                raise ValueError("action_mask must have boolean dtype")
            # Prefix masks only: padded trajectory tails must never become expert targets.
            if np.any(np.diff(mask.astype(int), axis=1) > 0):
                raise ValueError("Action masks must be contiguous valid prefixes")
            if not np.isfinite(actions[mask]).all():
                raise ValueError("Valid target actions contain non-finite values")
            splits = self.data["split"].astype(str)
            if not set(splits) <= {"train", "val", "test"}:
                raise ValueError("split must be train, val, or test")
            episodes = self.data["episode_id"].astype(str)
            seen = {}
            for episode, split in zip(episodes, splits):
                if episode in seen and seen[episode] != split:
                    raise ValueError(f"Episode {episode} appears in multiple splits")
                seen[episode] = split
        except BaseException:
            self.data.close()
            raise

    def indices(self, split):
        return np.flatnonzero(self.data["split"].astype(str) == split).tolist()

    def select_indices(self, split, *, episode_ids=None, max_episodes=None,
                       max_samples_per_episode=None, sample_stride=1):
        """Select once before the trial grid so every method/H/K uses the same samples."""
        for name, value in (("max_episodes", max_episodes),
                            ("max_samples_per_episode", max_samples_per_episode),
                            ("sample_stride", sample_stride)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if sample_stride is None:
            raise ValueError("sample_stride cannot be None")
        available = self.indices(split)
        names = self.data["episode_id"].astype(str)
        ordered = list(dict.fromkeys(names[available].tolist()))
        if episode_ids is not None:
            if not isinstance(episode_ids, list) or not episode_ids or len(set(episode_ids)) != len(episode_ids):
                raise ValueError("episode_ids must be a nonempty unique list")
            missing = set(episode_ids) - set(ordered)
            if missing:
                raise ValueError(f"Episodes are absent from split {split}: {sorted(missing)}")
            ordered = list(episode_ids)
        if max_episodes is not None:
            ordered = ordered[:max_episodes]
        selected = []
        for episode in ordered:
            indices = [i for i in available if names[i] == episode][::sample_stride]
            if max_samples_per_episode is not None and len(indices) > max_samples_per_episode:
                positions = (np.array([len(indices) // 2]) if max_samples_per_episode == 1
                             else np.linspace(0, len(indices) - 1, max_samples_per_episode, dtype=int))
                indices = [indices[int(p)] for p in positions]
            selected.extend(indices)
        return selected

    def observation(self, index, forecast=False):
        prefix = "forecast_" if forecast else ""
        key = prefix + "state"
        if key not in self.data:
            raise ValueError(f"Missing {key}; never substitute arrival observations for forecasts")
        state = np.asarray(self.data[key][index], dtype=np.float32)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("state must be a finite vector")
        image_keys = [k[len("image__"):] for k in self.data.files if k.startswith("image__")]
        images = {}
        for camera in image_keys:
            k = prefix + "image__" + camera
            if k not in self.data:
                raise ValueError(f"Missing {k}; forecast images must be supplied separately")
            images[camera] = self.data[k][index]
        native_keys = [k[len("native_state__"):] for k in self.data.files
                       if k.startswith("native_state__")]
        native = None
        instruction = str(self.data["instruction"][index]) if "instruction" in self.data else ""
        if native_keys:
            states = {}
            for name in native_keys:
                k = prefix + "native_state__" + name
                if k not in self.data:
                    raise ValueError(f"Missing {k}")
                value = np.asarray(self.data[k][index], dtype=np.float32)
                if value.ndim != 2:
                    raise ValueError("native state must have sample shape [time,dimension]")
                states[name] = value[None]
            videos = {}
            for camera, value in images.items():
                if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
                    raise ValueError("Native images must be uint8 [time,height,width,3]")
                videos[camera] = value[None]
            native = {"state": states, "video": videos,
                      "language": {self.language_key: [[instruction]]}}
        return Observation(state, images, instruction, native)

    def forecast_source(self, index):
        if "forecast_source" not in self.data:
            raise ValueError("Precompute experiments require forecast_source provenance")
        source = str(self.data["forecast_source"][index])
        if source not in {"stale_observation", "predicted_observation", "rendered_prediction"}:
            raise ValueError("Forecast source must describe information available before arrival")
        return source

    def forecast_latency_ms(self, index):
        source = self.forecast_source(index)
        if "forecast_generation_ms" not in self.data:
            if source == "stale_observation":
                return 0.0
            raise ValueError("Predicted/rendered forecasts require measured forecast_generation_ms")
        value = float(self.data["forecast_generation_ms"][index])
        if not np.isfinite(value) or value < 0:
            raise ValueError("forecast_generation_ms must be finite and nonnegative")
        return value

    def close(self):
        self.data.close()
