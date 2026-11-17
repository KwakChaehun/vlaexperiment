"""Model-independent Euler integration with an explicit, resumable flow time."""
from __future__ import annotations

from .contracts import FlowState, validate_chunk


class EulerFlowPolicy:
    """An engine supplies condition/noise/velocity/decode/synchronize methods.

    D discards old image-dependent features, retains only x_t and t, and
    computes fresh conditioning. This is an experimental conditioning switch,
    not a mathematically equivalent continuation of fresh Heavy inference.
    """

    def __init__(self, engine):
        self.engine = engine
        self.name = engine.name

    def synchronize(self):
        self.engine.synchronize()

    def _advance(self, latent, condition, start, stop, total):
        for index in range(start, stop):
            velocity = self.engine.velocity(latent, index / total, condition)
            if velocity.shape != latent.shape:
                raise ValueError("Flow velocity shape differs from latent shape")
            latent = latent + velocity / total
        return latent

    def prefill(self, obs, horizon, steps, completed, seed):
        if steps < 1 or not 0 <= completed <= steps:
            raise ValueError("Invalid flow time range")
        condition = self.engine.condition(obs, horizon)
        latent = self.engine.noise(horizon, seed, condition)
        if len(latent.shape) != 3 or latent.shape[1] != horizon:
            raise ValueError("Engine must allocate [batch, requested H, action_dim] noise")
        latent = self._advance(latent, condition, 0, completed, steps)
        return FlowState(latent, completed, steps, horizon)

    def resume(self, obs, cache):
        if cache.total < 1 or not 0 <= cache.completed <= cache.total:
            raise ValueError("Invalid saved flow state")
        condition = self.engine.condition(obs, cache.horizon)
        latent = self._advance(cache.latent, condition, cache.completed, cache.total, cache.total)
        return validate_chunk(self.engine.decode(latent, condition), cache.horizon)

    def predict(self, obs, horizon, steps, seed):
        # Encode once for normal inference. D intentionally re-encodes on resume.
        if steps < 1:
            raise ValueError("steps must be positive")
        condition = self.engine.condition(obs, horizon)
        latent = self.engine.noise(horizon, seed, condition)
        if len(latent.shape) != 3 or latent.shape[1] != horizon:
            raise ValueError("Requested H differs from actual allocated H")
        latent = self._advance(latent, condition, 0, steps, steps)
        return validate_chunk(self.engine.decode(latent, condition), horizon)
