"""Small action-conditioned feature forecast and VLM-free arrival conditioning."""
import numpy as np
import torch
from torch import nn

from .gr00t16_arrival import Gr00t16Arrival


class DirectFeaturePilot(Gr00t16Arrival):
    def condition_from_features(self, obs, feature, masks):
        """Actual state encoding, with supplied POST-vlln backbone features.

        Native processor (including image preprocessing) is paid, but neither
        vision encoder nor VLM is called. Masks depend on the unchanged prompt
        and fixed camera/token layout and come from the last full observation.
        """
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype
        from transformers.feature_extraction_utils import BatchFeature
        native = self.native(obs)
        self.policy.check_observation(native)
        sample = self.policy._unbatch_observation(native)[0]
        processed = self.policy.processor([{'type': MessageType.EPISODE_STEP.value,
                                            'content': self.policy._to_vla_step_data(sample)}])
        batch = _rec_to_dtype(self.policy.collate_fn([processed]), torch.bfloat16)
        _, ai = self.policy.model.prepare_input(batch['inputs'])
        features = BatchFeature(data={
            'backbone_features': feature.to(device=self.head.device, dtype=torch.bfloat16),
            'state_features': self.head.state_encoder(ai.state, ai.embodiment_id)})
        backbone = BatchFeature(data={k: v.to(self.head.device) for k, v in masks.items()})
        if feature.shape[1] != backbone.backbone_attention_mask.shape[1]:
            raise ValueError('Token layout changed')
        return backbone, ai, features, native['state']


def context(state, previous_state, actions, horizon):
    """Only origin state/history and already committed action prefix are visible."""
    planned = np.zeros((16, 7), dtype=np.float32)
    planned[:horizon] = np.asarray(actions, np.float32)[:horizon]
    valid = (np.arange(16) < horizon).astype(np.float32)
    return np.concatenate([state, np.asarray(state)-np.asarray(previous_state),
                           planned.ravel(), valid, [horizon/16]]).astype(np.float32)


class FeaturePredictor(nn.Module):
    def __init__(self, basis, mean, scale):
        super().__init__()
        self.register_buffer('basis', basis)
        self.register_buffer('context_mean', mean)
        self.register_buffer('context_scale', scale)
        rank = basis.shape[1]
        self.context_net = nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU(), nn.Linear(64, 32))
        self.token_net = nn.Sequential(nn.Linear(rank*3+32+5, 128), nn.SiLU(),
                                       nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, rank))
        nn.init.zeros_(self.token_net[-1].weight)
        nn.init.zeros_(self.token_net[-1].bias)

    def coefficients(self, current, previous, ctx, image_mask):
        # current/previous are projected features, B x tokens x rank.
        pos = torch.linspace(0, 1, current.shape[1], device=current.device)
        positional = torch.stack([pos, pos.square(), torch.sin(pos*6.2831853),
                                  torch.cos(pos*6.2831853)], -1)[None].expand(current.shape[0], -1, -1)
        c = self.context_net((ctx-self.context_mean)/self.context_scale)
        parts = [current, current-previous, current.mean(1, keepdim=True).expand_as(current),
                 c[:, None].expand(-1, current.shape[1], -1), positional, image_mask[..., None].float()]
        return self.token_net(torch.cat(parts, -1))

    def forward(self, current, previous, ctx, image_mask):
        z = current.float() @ self.basis
        p = previous.float() @ self.basis
        delta = self.coefficients(z, p, ctx, image_mask) @ self.basis.T
        return (current.float()+delta).to(current.dtype)
