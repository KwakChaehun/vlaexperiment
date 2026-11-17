"""Optional official openpi policy binding for native-horizon A-only diagnostics.

This binding never downloads model weights and does not implement H shortening
or partial flow reuse. Install openpi in its own supported GPU environment.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from .contracts import validate_chunk


class OpenPiNativePolicy:
    name = "openpi_native_horizon_diagnostic"

    def __init__(self, checkpoint, config_name="pi05_libero", flow_steps=10, device="cuda:0"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Prepare a local openpi checkpoint first: {checkpoint}")
        from openpi.policies import policy_config
        from openpi.training import config as training_config
        config = training_config.get_config(config_name)
        self.horizon = int(config.model.action_horizon)
        self.action_dim = int(config.model.action_dim)
        self.flow_steps = flow_steps
        self.policy = policy_config.create_trained_policy(
            config, checkpoint, sample_kwargs={"num_steps": flow_steps}, pytorch_device=device)
        self.metadata = {"checkpoint": str(checkpoint.resolve()), "config_name": config_name,
                         "native_horizon": self.horizon, "flow_steps": flow_steps,
                         "scope": "A-only, native H; no H shortening or flow resume"}

    def synchronize(self):
        # The official infer() converts actions to host NumPy before returning.
        pass

    def predict(self, obs, horizon, steps, seed):
        if horizon != self.horizon:
            raise ValueError(f"This binding generates native H={self.horizon}; it cannot test shortened H")
        if steps != self.flow_steps:
            raise ValueError("N must match the flow_steps used to construct this policy")
        if np.asarray(obs.state).shape != (8,):
            raise ValueError("pi05_libero expects an 8-dimensional observation state")
        images = {}
        for key in ("image", "wrist_image"):
            value = np.asarray(obs.images[key])
            if value.ndim == 4:
                value = value[-1]
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError("LIBERO images must be uint8 RGB")
            images[key] = value
        # Do not rotate saved LeRobot images; simulator raw images need their own preprocessing.
        inputs = {"observation/image": images["image"],
                  "observation/wrist_image": images["wrist_image"],
                  "observation/state": np.asarray(obs.state, dtype=np.float32),
                  "prompt": obs.instruction}
        # Official infer accepts explicit noise; create it independently of policy call history.
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((self.horizon, self.action_dim)).astype(np.float32)
        output = self.policy.infer(inputs, noise=noise)
        return validate_chunk(output["actions"], self.horizon)

    def prefill(self, *args):
        raise NotImplementedError("Native openpi binding does not implement partial flow")

    def resume(self, *args):
        raise NotImplementedError("Native openpi binding does not implement partial flow")


def make_policy(config):
    return OpenPiNativePolicy(**config)
