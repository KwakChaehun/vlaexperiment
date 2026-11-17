from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np


@dataclass
class Observation:
    """Only sensor-visible inputs. Simulator truth must stay in the environment."""

    state: np.ndarray
    images: dict[str, np.ndarray] = field(default_factory=dict)
    instruction: str = ""
    native: dict[str, Any] | None = None


@dataclass
class Forecast:
    """Construct before reading the actual arrival observation."""

    observation: Observation
    source: str


@dataclass
class FlowState:
    latent: Any
    completed: int
    total: int
    horizon: int


class Policy(Protocol):
    name: str

    def predict(self, obs: Observation, horizon: int, steps: int, seed: int) -> np.ndarray: ...
    def prefill(self, obs: Observation, horizon: int, steps: int, completed: int,
                seed: int) -> FlowState: ...
    def resume(self, obs: Observation, cache: FlowState) -> np.ndarray: ...
    def synchronize(self) -> None: ...


class SmallPolicy(Protocol):
    def predict(self, obs: Observation, horizon: int, steps: int, seed: int) -> np.ndarray: ...
    def correct(self, obs: Observation, prior: np.ndarray,
                forecast: Observation) -> np.ndarray: ...
    def synchronize(self) -> None: ...


class Environment(Protocol):
    """Factory must create independent instances with identical seeded arrival states."""

    def forecast(self) -> Forecast: ...
    def observe(self) -> Observation: ...
    def advance(self, seconds: float) -> None: ...
    def step(self, action: np.ndarray) -> None: ...
    def done(self) -> bool: ...
    def metrics(self) -> dict[str, Any]: ...


def validate_chunk(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] != horizon or actions.shape[1] < 1:
        raise ValueError(f"Expected [{horizon}, action_dim], received {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy returned non-finite actions")
    return actions
