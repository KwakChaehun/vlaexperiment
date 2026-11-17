"""N1.6 LIBERO pilot: real shortened latents and per-camera vision reuse.

Requires NVIDIA's n1d6 branch, separately from the N1.7 adapter. All VLM
layers are rerun even when image encoder features are cached. No KV splicing.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np


class Gr00t16Pilot:
    def __init__(self, checkpoint, device="cuda:0"):
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.torch = torch
        torch.set_num_threads(4)
        self.policy = Gr00tPolicy(EmbodimentTag.LIBERO_PANDA, checkpoint, device=device)
        self.head = self.policy.model.action_head
        if type(self.head).__name__ != "Gr00tN1d6ActionHead":
            raise ValueError("N1.6 checkpoint and code required")
        self.native_horizon = self.head.config.action_horizon
        self.native_steps = self.head.num_inference_timesteps
        self.output_horizon = len(self.policy.modality_configs['action'].delta_indices)
        self.eagle = self.policy.model.backbone.model
        self.original_extract = self.eagle.extract_feature
        self.camera_keys = self.policy.modality_configs['video'].modality_keys
        if self.camera_keys != ['image', 'wrist_image']:
            raise ValueError(f"Unexpected camera order: {self.camera_keys}")
        self.cache = {}
        self.cache_enabled = False
        self.image_keys = []
        self.stage = {}
        self.eagle.extract_feature = self._extract
        self.last = {}

    def synchronize(self):
        self.torch.cuda.synchronize(self.policy.model.device)

    def clear_cache(self):
        self.cache.clear()

    def _extract(self, pixel_values, image_flags=None):
        self.synchronize()
        start = time.perf_counter()
        if not self.cache_enabled:
            result = self.original_extract(pixel_values, image_flags)
            self.synchronize()
            self.stage['vision_ms'] = (time.perf_counter()-start)*1000
            self.stage['vision_cache_hits'] = 0
            return result
        if len(pixel_values) != 2 or image_flags is not None:
            raise ValueError("Cache supports one sample, two cameras, no image flags")
        result, hits = [], 0
        for index, (pixels, key) in enumerate(zip(pixel_values, self.image_keys)):
            self.synchronize()
            before = time.perf_counter()
            entry = self.cache.get(index)
            if entry is not None and entry[0] == key:
                feature = entry[1]
                hits += 1
            else:
                feature = self.original_extract([pixels])
                self.cache[index] = (key, feature.detach())
            self.synchronize()
            self.stage[self.camera_keys[index]+'_vision_ms'] = (time.perf_counter()-before)*1000
            result.append(feature)
        result = self.torch.cat(result, dim=0)
        self.synchronize()
        self.stage['vision_ms'] = (time.perf_counter()-start)*1000
        self.stage['vision_cache_hits'] = hits
        return result

    def native(self, obs):
        keys = ['x','y','z','roll','pitch','yaw','gripper']
        state = np.asarray(obs.state, np.float32)
        return {'video': {k: np.ascontiguousarray(obs.images[k])[None,None]
                          for k in self.camera_keys},
                'state': {k: state[i:i+1][None,None] if i < 6 else state[6:8][None,None]
                          for i,k in enumerate(keys)},
                'language': {self.policy.language_key: [[obs.instruction]]}}

    def condition(self, obs, cache=False):
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype
        self.stage = {}
        self.cache_enabled = cache
        self.synchronize()
        start = time.perf_counter()
        # Hash cost is included in preprocessing; cache never reuses changed pixels.
        self.image_keys = [hashlib.sha256(np.ascontiguousarray(obs.images[k]).tobytes()).hexdigest()
                           for k in self.camera_keys] if cache else []
        native = self.native(obs)
        self.policy.check_observation(native)
        sample = self.policy._unbatch_observation(native)[0]
        processed = self.policy.processor([{'type':MessageType.EPISODE_STEP.value,
                                            'content':self.policy._to_vla_step_data(sample)}])
        batch = _rec_to_dtype(self.policy.collate_fn([processed]), self.torch.bfloat16)
        vi, ai = self.policy.model.prepare_input(batch['inputs'])
        self.synchronize()
        self.stage['preprocess_ms'] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        backbone = self.policy.model.backbone(vi)
        self.synchronize()
        self.stage['backbone_ms'] = (time.perf_counter()-start)*1000
        self.stage['vlm_ms'] = self.stage['backbone_ms'] - self.stage['vision_ms']
        start = time.perf_counter()
        features = self.head._encode_features(backbone, ai)
        self.synchronize()
        self.stage['state_and_fusion_ms'] = (time.perf_counter()-start)*1000
        return backbone, ai, features, native['state']

    def sample(self, condition, horizon, steps, seed):
        torch, head = self.torch, self.head
        if not 1 <= horizon <= self.native_horizon or steps < 1:
            raise ValueError("Invalid H/N")
        backbone, ai, features, states = condition
        generator = torch.Generator(device=head.device).manual_seed(seed)
        # Generate native noise first: identical prefixes for every H.
        x = torch.randn((1,self.native_horizon,head.action_dim), generator=generator,
                        device=head.device,dtype=features.backbone_features.dtype)[:,:horizon].clone()
        self.last['latent_shape'] = list(x.shape)
        for step in range(steps):
            ticks = torch.full((1,), int(step / float(steps) * head.num_timestep_buckets),
                               device=head.device)
            act = head.action_encoder(x, ticks, ai.embodiment_id)
            if head.config.add_pos_embed:
                act = act + head.position_embedding(torch.arange(horizon,device=head.device))[None]
            args = dict(hidden_states=torch.cat([features.state_features,act],dim=1),
                        encoder_hidden_states=features.backbone_features,timestep=ticks)
            if head.config.use_alternate_vl_dit:
                args.update(image_mask=backbone.image_mask,
                            backbone_attention_mask=backbone.backbone_attention_mask)
            velocity = head.action_decoder(head.model(**args),ai.embodiment_id)[:,-horizon:]
            x = x + (1.0/steps)*velocity
        return x

    def predict(self, obs, horizon=16, steps=4, seed=0, cache=False):
        torch = self.torch
        self.last = {}
        self.synchronize()
        total = time.perf_counter()
        with torch.inference_mode():
            condition = self.condition(obs,cache)
            start = time.perf_counter()
            latent = self.sample(condition,horizon,steps,seed)
            self.synchronize()
            self.stage['ae_ms'] = (time.perf_counter()-start)*1000
            start = time.perf_counter()
            decoded = self.policy.processor.decode_action(latent.float().cpu().numpy(),
                       self.policy.embodiment_tag,condition[3])
            actions = np.concatenate([decoded[k] for k in self.policy.modality_configs['action'].modality_keys],-1)[0]
            self.synchronize()
            self.stage['decode_ms'] = (time.perf_counter()-start)*1000
        self.stage['total_ms'] = (time.perf_counter()-total)*1000
        if actions.shape != (min(horizon,self.output_horizon),7) or not np.isfinite(actions).all():
            raise ValueError(f"Unexpected decoded actions: {actions.shape}")
        self.last.update(self.stage)
        self.last.update(H=horizon, N=steps, decoded_horizon=len(actions))
        return actions

    def parity(self, obs, seed=37):
        self.clear_cache()
        self.cache_enabled = False
        self.torch.manual_seed(seed)
        with self.torch.inference_mode():
            action,_ = self.policy.get_action(self.native(obs))
        native = np.concatenate([action[k] for k in self.policy.modality_configs['action'].modality_keys],-1)[0]
        custom = self.predict(obs,self.native_horizon,self.native_steps,seed)
        error = float(np.max(np.abs(native-custom)))
        if error > 1e-4:
            raise ValueError(f"Official sampler mismatch: {error}")
        cached = self.predict(obs,self.native_horizon,self.native_steps,seed,cache=True)
        split_error = float(np.max(np.abs(custom-cached)))
        cached_again = self.predict(obs,self.native_horizon,self.native_steps,seed,cache=True)
        hit_error = float(np.max(np.abs(cached-cached_again)))
        # Separate-image GPU kernels can differ in BF16; record rather than hide it.
        if hit_error != 0 or split_error > 0.03:
            raise ValueError(f"Vision cache mismatch: split={split_error}, hit={hit_error}")
        return {'official_sampler_max_abs':error,'split_vision_max_abs':split_error,
                'cache_hit_max_abs':hit_error,'native_generated_H':self.native_horizon,
                'native_decoded_H':self.output_horizon,'native_N':self.native_steps}
