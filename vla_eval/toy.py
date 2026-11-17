"""CPU plumbing fixture, NOT a robot simulator or a learned VLA benchmark."""
from __future__ import annotations

import math
import numpy as np
from .contracts import FlowState, Forecast, Observation


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def plan(obs: Observation, horizon: int, coarse: bool = False) -> np.ndarray:
    # Observed state: tool x/y/theta, visually estimated target x/y/theta.
    tool, target = obs.state[:3].copy(), obs.state[3:6].copy()
    if coarse:
        target[:2] = np.round(target[:2] / 0.002) * 0.002
        target[2] = np.round(target[2] / 0.035) * 0.035
    result = []
    for _ in range(horizon):
        error = target - tool
        error[2] = wrap(error[2])
        action = np.clip(0.45 * error, [-0.002, -0.002, -0.04], [0.002, 0.002, 0.04])
        result.append(action)
        tool += action
    return np.asarray(result, dtype=np.float32)


class ToyPrecisionEnv:
    def __init__(self, seed: int, config: dict):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.target = self.rng.normal(size=3) * np.array([0.006, 0.006, 0.1])
        self.tool = self.target + np.array([0.009, -0.006, 0.12])
        self.predicted_tool = self.tool.copy()
        # Arrival perturbations are not exposed to forecast().
        self.arrival_delta = self.rng.normal(size=3) * np.array([
            config.get("arrival_position_mm", 2.0) / 1000,
            config.get("arrival_position_mm", 2.0) / 1000,
            math.radians(config.get("arrival_rotation_deg", 3.0)),
        ])
        self.arrived = False
        self.time = 0.0
        self.actions = 0
        self.collisions = 0
        self.success = False

    def forecast(self):
        if self.arrived:
            raise RuntimeError("Forecast must be requested before observing arrival")
        return Forecast(Observation(np.r_[self.predicted_tool, self.target].astype(np.float32),
                                    instruction="align the tool with the target"),
                        "toy_predicted_proprioception_and_static_target")

    def observe(self):
        if not self.arrived:
            self.tool += self.arrival_delta
            self.arrived = True
        target = self.current_target().copy()
        target += self.rng.normal(size=3) * np.array([0.00015, 0.00015, 0.003])
        return Observation(np.r_[self.tool, target].astype(np.float32),
                           instruction="align the tool with the target")

    def current_target(self):
        return self.target + np.array([
            self.config.get("target_speed_mm_s", 0.0) / 1000 * self.time, 0, 0])

    def advance(self, seconds):
        if seconds < 0:
            raise ValueError("Cannot advance by negative time")
        self.time += seconds

    def step(self, action):
        self.tool += np.asarray(action)
        self.tool[2] = wrap(self.tool[2])
        self.actions += 1
        self.advance(1 / self.config.get("control_hz", 20))
        e = self.tool - self.current_target()
        self.success = bool(np.linalg.norm(e[:2]) <= self.config.get("tolerance_mm", 0.5) / 1000
                            and abs(wrap(e[2])) <= math.radians(
                                self.config.get("rotation_tolerance_deg", 1.0)))

    def done(self):
        return self.success or self.actions >= self.config.get("max_actions", 80)

    def metrics(self):
        e = self.tool - self.current_target()
        return {"success": self.success, "position_error_mm": float(np.linalg.norm(e[:2]) * 1000),
                "rotation_error_deg": float(abs(wrap(e[2])) * 180 / np.pi),
                "collisions": self.collisions, "environment_time_s": self.time,
                "executed_actions": self.actions}


class ToyHeavy:
    name = "toy_analytic_flow_fixture"

    def synchronize(self):
        pass

    def prefill(self, obs, horizon, steps, completed, seed):
        if not 0 <= completed <= steps or steps < 1:
            raise ValueError("Invalid flow split")
        latent = np.random.default_rng(seed).normal(0, 0.001, (horizon, 3))
        target = plan(obs, horizon)
        # Exact constant conditional velocity fixture; real flow is learned.
        velocity = target - latent
        latent += completed / steps * velocity
        return FlowState(latent, completed, steps, horizon)

    def resume(self, obs, cache):
        target = plan(obs, cache.horizon)
        if cache.completed == cache.total:
            return np.asarray(cache.latent, dtype=np.float32)
        # Toy bridge changes conditioning and maps the remaining interval to target.
        return target

    def predict(self, obs, horizon, steps, seed):
        return self.resume(obs, self.prefill(obs, horizon, steps, 0, seed))


class ToySmall:
    name = "toy_coarse_controller_fixture"

    def synchronize(self):
        pass

    def predict(self, obs, horizon, steps, seed):
        return plan(obs, horizon, coarse=True)

    def correct(self, obs, prior, forecast):
        # A deliberately simple baseline, no privileged simulator state.
        correction = plan(obs, len(prior), coarse=True) - plan(forecast, len(prior), coarse=True)
        return prior + correction


def make_bundle(config):
    if config.get("small_checkpoint"):
        from .small import RidgeSmall
        small = RidgeSmall(config["small_checkpoint"])
    else:
        small = ToySmall()
    return ToyHeavy(), small, lambda seed, env_config: ToyPrecisionEnv(seed, env_config)
