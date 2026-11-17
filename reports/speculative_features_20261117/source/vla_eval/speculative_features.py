"""Prediction-relative sparse MLP updates; attention is always recomputed.

This is a training-free diagnostic, not an implementation of Eventful Transformers.
Caches belong to one predicted future observation, never the last real frame.
Selection uses actual layer inputs, so attention/global change detection is paid for.
"""
from __future__ import annotations

import math
import time


class PredictedMLPCache:
    def __init__(self, pilot):
        self.pilot = pilot
        self.torch = pilot.torch
        self.mode = "off"
        self.bank = None
        self.ratio = 1.0
        self.stats = []
        self.originals = {}
        # Patch only the backbone. The action expert stays unchanged.
        for name, module in pilot.eagle.named_modules():
            if name.endswith(".mlp") and (
                name.startswith("vision_model.") or name.startswith("language_model.")
            ):
                self.originals[name] = (module, module.forward)
                module.forward = self._wrapper(name, module.forward)
        if not self.originals:
            raise ValueError("No backbone MLPs found")

    def _wrapper(self, name, original):
        def forward(hidden_states, *args, **kwargs):
            if self.mode == "off":
                return original(hidden_states, *args, **kwargs)
            if args or kwargs:
                raise ValueError("Unexpected MLP arguments")
            if self.mode == "capture":
                output = original(hidden_states)
                self.bank[name] = (hidden_states.detach().clone(), output.detach().clone())
                return output
            old_input, old_output = self.bank[name]
            if hidden_states.shape != old_input.shape:
                raise ValueError(f"Prediction/observation token shape differs: {name}")
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            prior = old_input.reshape_as(flat)
            n = flat.shape[0]
            k = min(n, max(0, math.ceil(n * self.ratio)))
            if k == n:
                output = original(hidden_states)
            elif k == 0:
                output = old_output.clone()
            else:
                # Rank the discrepancy from THIS predicted future, not temporal motion.
                score = (flat.float() - prior.float()).square().mean(-1)
                indices = score.topk(k, sorted=False).indices
                updated = original(flat.index_select(0, indices))
                output = old_output.reshape_as(flat).clone()
                output.index_copy_(0, indices, updated)
                output = output.reshape_as(hidden_states)
            self.stats.append((name, n, k))
            return output
        return forward

    def capture(self, obs, seed):
        p = self.pilot
        self.bank = {}
        self.mode = "capture"
        try:
            with self.torch.inference_mode():
                p.synchronize()
                begin = time.perf_counter()
                condition = p.condition(obs)
                part = p.integrate(condition, 50, 4, seed, stop=3)
                full = p.integrate(condition, 50, 4, seed, start=3, latent=part)
                actions = p.decode(condition, full)
                p.synchronize()
                ms = (time.perf_counter() - begin) * 1000
            return dict(mlps=self.bank, latent=part, actions=actions, pre_ms=ms)
        finally:
            self.mode = "off"
            self.bank = None

    def correct(self, obs, prediction, seed, ratio, steps=4, resume=True):
        if not 0 <= ratio <= 1:
            raise ValueError("ratio outside [0,1]")
        p = self.pilot
        self.bank = prediction["mlps"]
        self.ratio = ratio
        self.stats = []
        self.mode = "correct"
        try:
            with self.torch.inference_mode():
                p.synchronize()
                begin = time.perf_counter()
                condition = p.condition(obs)
                latent = p.integrate(condition, 50, steps, seed,
                    start=3 if resume else 0,
                    latent=prediction["latent"] if resume else None)
                actions = p.decode(condition, latent)
                p.synchronize()
                ms = (time.perf_counter() - begin) * 1000
            groups = {}
            for name, n, k in self.stats:
                group = "vision" if name.startswith("vision_model.") else "vlm"
                totals = groups.setdefault(group, [0, 0])
                totals[0] += n
                totals[1] += k
            return actions, ms, groups
        finally:
            self.mode = "off"
            self.bank = None

    @staticmethod
    def bytes(prediction):
        return sum(x.numel() * x.element_size() for pair in prediction["mlps"].values()
                   for x in pair) + prediction["latent"].numel() * prediction["latent"].element_size()

    def close(self):
        for module, original in self.originals.values():
            module.forward = original
