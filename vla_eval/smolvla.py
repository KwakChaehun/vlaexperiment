"""Pinned SmolVLA research adapter; short H and condition switching are experiments.

Official LeRobot v0.4.4 model math is used directly. Preprocessing follows the
downloaded checkpoint's normalizer/tokenizer configuration. No task success is
inferred from offline action error.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np
from .contracts import Observation


@dataclass
class SmolCondition:
    pad_mask: object
    kv: object
    batch: dict
    preparation_ms: float
    backbone_ms: float


class SmolPilot:
    def __init__(self, checkpoint, vlm_config, threads=4, device='cpu'):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        from lerobot.configs.types import PolicyFeature, FeatureType
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
        self.torch, self.make_att_masks = torch, make_att_2d_masks
        self.device=torch.device(device)
        if self.device.type not in {'cpu','cuda'}:
            raise ValueError('The pilot supports cpu and cuda devices')
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install CUDA PyTorch and check nvidia-smi')
        torch.set_num_threads(threads)
        checkpoint = Path(checkpoint)
        raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        raw.pop("type")
        for name in ("input_features", "output_features"):
            raw[name] = {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                         for k, v in raw[name].items()}
        cfg = SmolVLAConfig(**raw)
        cfg.device, cfg.load_vlm_weights, cfg.vlm_model_name = str(self.device), False, str(vlm_config)
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg).eval().to(self.device)
        self.model, self.config = self.policy.model, cfg
        self.native_horizon = cfg.chunk_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(vlm_config))
        self.stats = load_file(str(checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        self.action_mean = self.stats["action.mean"].numpy()
        self.action_std = self.stats["action.std"].numpy()
        self.metadata = {"model": "HuggingFaceVLA/smolvla_libero",
                         "revision": "6721902bc4d61e50a3bfdb11dfb4cb626f05d102",
                         "lerobot": "0.4.4", "native_horizon": self.native_horizon,
                         "device": str(self.device), "dtype":"float32", "threads": threads,
                         "torch_version":torch.__version__,"cuda_runtime":torch.version.cuda,
                         "gpu_name":torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
                         "matmul_allow_tf32":bool(torch.backends.cuda.matmul.allow_tf32),
                         "short_horizon_status": "inference-only OOD; not retrained"}

    def batch(self, obs: Observation):
        torch = self.torch
        state = torch.from_numpy(np.asarray(obs.state).copy()).float()
        state = (state - self.stats["observation.state.mean"]) / (self.stats["observation.state.std"] + 1e-8)
        result = {"observation.state": state[None]}
        if len(self.config.image_features) != 2:
            raise ValueError("Pilot expects the pinned two-camera checkpoint")
        for key, source in zip(self.config.image_features, ("image", "wrist_image")):
            arr = np.asarray(obs.images[source])
            if arr.ndim == 4:
                arr = arr[-1]
            result[key] = torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].float() / 255
        language = self.tokenizer(obs.instruction.rstrip("\n") + "\n", return_tensors="pt",
                                  padding=self.config.pad_language_to, max_length=self.config.tokenizer_max_length,
                                  truncation=True)
        result["observation.language.tokens"] = language["input_ids"]
        result["observation.language.attention_mask"] = language["attention_mask"].bool()
        return {key:value.to(self.device) for key,value in result.items()}

    def synchronize(self):
        if self.device.type=='cuda':self.torch.cuda.synchronize(self.device)

    def condition(self, obs):
        torch = self.torch
        with torch.inference_mode():
            self.synchronize()
            start = time.perf_counter()
            batch = self.batch(obs)
            images, masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            self.synchronize();prepared = time.perf_counter()
            emb, pad, att = self.model.embed_prefix(
                images, masks, batch["observation.language.tokens"],
                batch["observation.language.attention_mask"], state=state)
            _, kv = self.model.vlm_with_expert.forward(
                attention_mask=self.make_att_masks(pad, att),
                position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                inputs_embeds=[emb, None], use_cache=self.config.use_cache, fill_kv_cache=True)
            self.synchronize();end = time.perf_counter()
        return SmolCondition(pad, kv, batch, (prepared-start)*1000, (end-prepared)*1000)

    def noise(self, h, seed):
        if not 1 <= h <= self.native_horizon:
            raise ValueError("H must be positive and no larger than the native horizon")
        # Common random prefix across H, independently of the call order.
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        return self.torch.randn((1, self.native_horizon, self.config.max_action_dim),
                                generator=g)[:, :h].clone().to(self.device)

    def integrate(self, condition, h, n, seed=0, latent=None, start_step=0, stop_step=None):
        torch = self.torch
        stop_step = n if stop_step is None else stop_step
        if not (n >= 1 and 0 <= start_step <= stop_step <= n):
            raise ValueError("Invalid flow step range")
        x = self.noise(h, seed) if latent is None else latent.clone()
        if tuple(x.shape) != (1, h, self.config.max_action_dim):
            raise ValueError("Latent H differs from the actual generated H")
        old = self.config.chunk_size
        self.config.chunk_size = h
        self.synchronize();start = time.perf_counter()
        dt = -1.0 / n
        try:
            with torch.inference_mode():
                for step in range(start_step, stop_step):
                    ticks = torch.full((1,), 1.0 + step * dt, dtype=torch.float32,device=self.device)
                    velocity = self.model.denoise_step(condition.pad_mask, condition.kv, x, ticks)
                    if velocity.shape != x.shape:
                        raise ValueError("Short H did not propagate into the action expert")
                    x = x + dt * velocity
        finally:
            self.config.chunk_size = old
        self.synchronize()
        return x, (time.perf_counter()-start)*1000

    def decode(self, x):
        normalized = x[0, :, :7].float().cpu().numpy()
        return normalized * self.action_std + self.action_mean

    def predict_condition(self, condition, h, n, seed):
        latent, elapsed = self.integrate(condition, h, n, seed)
        return self.decode(latent), latent, elapsed

    def sanity(self, obs, seed=0):
        """Compare custom split integration with the official native-H sampler."""
        torch = self.torch
        cond = self.condition(obs)
        h, n = self.native_horizon, self.config.num_steps
        with torch.inference_mode():
            official = self.policy.predict_action_chunk(cond.batch, noise=self.noise(h, seed))
            custom, _ = self.integrate(cond, h, n, seed)
        error = float((official - custom[:, :, :7]).abs().max())
        if error > 1e-4:
            raise ValueError(f"Native sampler parity failed: max abs difference {error}")
        return {"native_sampler_max_abs_difference": error,
                "preparation_ms": cond.preparation_ms, "backbone_ms": cond.backbone_ms}
