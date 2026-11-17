"""Optional experimental binding to NVIDIA Isaac-GR00T's N1.7 Python API.

No checkpoint download or dependency installation occurs on import.
Verified against upstream API source, not validated on a checkpoint here.
Other GR00T generations need separate adapters; never silently fall back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from .flow import EulerFlowPolicy


@dataclass
class Condition:
    backbone: Any
    features: Any
    inputs: Any
    states: dict
    horizon: int
    mask: Any


class Gr00tN17Engine:
    def __init__(self, model_path, embodiment_tag, device="cuda:0", allow_short_horizon=False):
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.torch = torch
        self.policy = Gr00tPolicy(model_path=model_path, embodiment_tag=embodiment_tag,
                                  device=device, strict=True)
        self.head = self.policy.model.action_head
        if self.head.__class__.__name__ != "Gr00tN1d7ActionHead":
            raise ValueError("This adapter requires Gr00tN1d7ActionHead; use a version-specific adapter")
        self.native_horizon = int(self.head.config.action_horizon)
        self.allow_short_horizon = allow_short_horizon
        self.name = "gr00t_n1d7_experimental_euler"
        self.metadata = {"checkpoint": str(model_path), "embodiment": str(embodiment_tag),
                         "native_horizon": self.native_horizon,
                         "allow_short_horizon": allow_short_horizon,
                         "short_horizon_status": "OOD unless checkpoint was trained for these lengths"}

    def synchronize(self):
        if self.head.device.type == "cuda":
            self.torch.cuda.synchronize(self.head.device)

    def condition(self, obs, horizon):
        from gr00t.data.types import MessageType
        if not 1 <= horizon <= self.native_horizon:
            raise ValueError(f"H must be in [1, {self.native_horizon}]")
        if horizon != self.native_horizon and not self.allow_short_horizon:
            raise ValueError("Changing generation H is experimental/OOD; explicitly enable allow_short_horizon")
        if obs.native is None:
            raise ValueError("GR00T requires Observation.native with video/state/language modalities")
        p, torch = self.policy, self.torch
        p.check_observation(obs.native)
        samples = p._unbatch_observation(obs.native)
        if len(samples) != 1:
            raise ValueError("Paired latency evaluation requires batch=1")
        processed = [p.processor([{"type": MessageType.EPISODE_STEP.value,
                                   "content": p._to_vla_step_data(sample)}]) for sample in samples]
        batch = p.collate_fn(processed)
        if set(batch) != {"inputs"}:
            raise ValueError("Upstream collator API changed; inspect before using this adapter")
        with torch.inference_mode():
            vision_inputs, action_inputs = p.model.prepare_input(batch["inputs"])
            backbone = p.model.backbone(vision_inputs)
            features = self.head._encode_features(backbone, action_inputs)
        states = {key: np.stack([s["state"][key] for s in samples])
                  for key in p.modality_configs["state"].modality_keys}
        mask = None
        if self.head.config.strict_action_padding_mask:
            mask = action_inputs.get("action_mask")
            if mask is None:
                raise ValueError("Checkpoint requires an action padding mask")
            mask = mask[:, :horizon].to(self.head.device, dtype=self.head.dtype)
            if mask.shape[1] != horizon:
                raise ValueError("Checkpoint padding mask cannot cover requested horizon")
        return Condition(backbone, features, action_inputs, states, horizon, mask)

    def noise(self, horizon, seed, condition):
        torch = self.torch
        generator = torch.Generator(device=self.head.device).manual_seed(seed)
        x = torch.randn((1, horizon, self.head.action_dim), generator=generator,
                        device=self.head.device, dtype=self.head.dtype)
        return x if condition.mask is None else x * condition.mask

    def velocity(self, latent, t, condition):
        torch, head = self.torch, self.head
        with torch.inference_mode():
            ticks = torch.full((latent.shape[0],), int(t * head.num_timestep_buckets),
                               device=latent.device, dtype=torch.long)
            action_features = head.action_encoder(latent, ticks, condition.inputs.embodiment_id)
            if head.config.add_pos_embed:
                positions = torch.arange(condition.horizon, device=latent.device)
                action_features = action_features + head.position_embedding(positions)[None]
            hidden_mask = None
            if condition.mask is not None:
                valid = condition.mask.bool().any(dim=-1)
                action_features = action_features * valid[..., None]
                state_mask = torch.ones(condition.features.state_features.shape[:2],
                                        device=latent.device, dtype=torch.bool)
                hidden_mask = torch.cat([state_mask, valid], dim=1)
            arguments = dict(hidden_states=torch.cat([condition.features.state_features,
                                                      action_features], dim=1),
                             encoder_hidden_states=condition.features.backbone_features,
                             timestep=ticks, hidden_attention_mask=hidden_mask)
            if head.config.use_alternate_vl_dit:
                arguments.update(image_mask=condition.backbone.image_mask,
                                 backbone_attention_mask=condition.backbone.backbone_attention_mask)
            else:
                arguments["encoder_attention_mask"] = (
                    condition.backbone.backbone_attention_mask
                    if head.config.strict_vl_padding_mask else None)
            hidden = head.model(**arguments)
            velocity = head.action_decoder(hidden, condition.inputs.embodiment_id)[:, -condition.horizon:]
            return velocity if condition.mask is None else velocity * condition.mask

    def decode(self, latent, condition):
        decoded = self.policy.processor.decode_action(
            latent.float().cpu().numpy(), self.policy.embodiment_tag, condition.states)
        keys = self.policy.modality_configs["action"].modality_keys
        pieces = [decoded[k] for k in keys]
        if any(v.shape[:2] != (1, condition.horizon) for v in pieces):
            raise ValueError("Processor does not support this generation H; do not truncate and claim acceleration")
        return np.concatenate(pieces, axis=-1)[0].astype(np.float32)


def make_policy(config):
    return EulerFlowPolicy(Gr00tN17Engine(**config))
