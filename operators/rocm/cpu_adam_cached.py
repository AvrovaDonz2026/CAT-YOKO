"""Opt-in CPU Adam with a BF16 parameter cache and direct device writeback.

CPU moment arithmetic retains CPUOffloadAdamW's exact primitive sequence. Each
update starts from the latest BF16 value, with no persistent FP32 master. Only
the parameter readback and the extra device temporary are removed.

Use in a standalone, single-threaded optimizer process. Versioned parameter
writes automatically rebuild the cache. Writes through ``p.data`` or another
storage alias can bypass version tracking and require ``invalidate(p)``. The
cache is private to the context, absent from checkpoints, and cleared on load
or exit. Native ``step_params`` is retained for parameter-offload callers.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from unittest.mock import patch
import weakref

import torch

from cat_yoko.optim import CPUOffloadAdamW

_NATIVE_UPDATE_ONE = CPUOffloadAdamW._update_one


def _signature(parameter):
    return (parameter._version, parameter.data_ptr(), parameter.storage_offset(),
            tuple(parameter.shape), tuple(parameter.stride()), parameter.dtype, parameter.device)


class CachedCPUAdamInstallation:
    def __init__(self, *, allow_cpu=False, optimizers=None):
        self.allow_cpu = allow_cpu
        self.targets = None if optimizers is None else weakref.WeakSet(optimizers)
        # Key by identity, since Tensor equality is not a scalar comparison.
        self.cache = {}
        self.fallback_depth = weakref.WeakKeyDictionary()
        self.parameter_updates = self.gpu_parameter_updates = 0
        self.cache_rebuilds = self.cache_reuses = self.invalidations = 0
        self.state_loads = self.fallback_parameter_updates = 0
        self.saved_parameter_readback_bytes = 0

    def applies(self, optimizer):
        return self.targets is None or optimizer in self.targets

    def invalidate(self, parameter=None):
        """Rebuild before the next update; call after unversioned alias writes."""
        if parameter is None:
            self.invalidations += len(self.cache)
            self.cache.clear()
        elif self.cache.pop(id(parameter), None) is not None:
            self.invalidations += 1

    def _entry(self, parameter):
        identity, signature = id(parameter), _signature(parameter)
        entry = self.cache.get(identity)
        if entry is None or entry[0]() is not parameter or entry[2] != signature:
            self.invalidate(parameter)
            shadow = parameter.detach().to(device="cpu", dtype=torch.bfloat16, copy=True)

            def expired(reference):
                current = self.cache.get(identity)
                if current is not None and current[0] is reference:
                    self.cache.pop(identity, None)

            entry = [weakref.ref(parameter, expired), shadow, signature]
            self.cache[identity] = entry
            self.cache_rebuilds += 1
        else:
            self.cache_reuses += 1
            if parameter.device.type == "cuda":
                self.saved_parameter_readback_bytes += parameter.numel() * parameter.element_size()
        return entry

    def _validate(self, optimizer, parameter):
        if optimizer.state_dtype != torch.float32 or not optimizer.retain_state:
            raise ValueError("cached CPU Adam requires retained FP32 moments")
        if parameter.dtype != torch.bfloat16 or not parameter.requires_grad:
            raise ValueError("cached CPU Adam requires trainable BF16 parameters")
        if parameter.device.type != "cuda" and not (self.allow_cpu and parameter.device.type == "cpu"):
            raise ValueError("cached CPU Adam requires CUDA/HIP parameters; allow_cpu is only for tests")
        if parameter.grad.is_sparse:
            raise ValueError("cached CPU Adam requires dense gradients")
        state = optimizer.state.get(parameter)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if (not torch.is_tensor(value) or value.dtype != torch.float32
                        or value.device.type != "cpu" or value.shape != parameter.shape):
                    raise ValueError("cached CPU Adam requires native CPU FP32 moments with matching shapes")

    @torch.no_grad()
    def update(self, optimizer, parameter, group):
        if parameter.grad is None:
            return
        self._validate(optimizer, parameter)
        entry = self._entry(parameter)
        lr, (beta1, beta2) = float(group["lr"]), group["betas"]
        eps, decay = float(group.get("eps", 1e-8)), float(group["weight_decay"])
        grad = parameter.grad.detach()
        parameter.grad = None
        if grad.device.type != "cpu" or grad.dtype != torch.float32:
            grad = grad.to(device="cpu", dtype=torch.float32)
        p32 = entry[1].to(dtype=torch.float32)
        state = optimizer.state[parameter]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros(p32.shape, dtype=torch.float32, device="cpu")
            state["exp_avg_sq"] = torch.zeros(p32.shape, dtype=torch.float32, device="cpu")
        elif torch.is_tensor(state.get("step")):
            state["step"] = int(state["step"].item())
        state["step"] = int(state.get("step", 0)) + 1
        step = int(state["step"])
        exp_avg, exp_avg_sq = state["exp_avg"].float(), state["exp_avg_sq"].float()
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        state["exp_avg"] = exp_avg.to(dtype=optimizer.state_dtype)
        state["exp_avg_sq"] = exp_avg_sq.to(dtype=optimizer.state_dtype)
        if decay != 0.0:
            p32.mul_(1.0 - lr * decay)
        denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**step)).add_(eps)
        p32.addcdiv_(exp_avg, denom, value=-lr / (1.0 - beta1**step))
        rounded = p32.to(dtype=torch.bfloat16)
        parameter.copy_(rounded)  # Same dtype: direct, blocking H2D, no GPU temporary.
        entry[1], entry[2] = rounded, _signature(parameter)
        self.parameter_updates += 1
        self.gpu_parameter_updates += int(parameter.device.type == "cuda")

    def report(self):
        return {"parameter_updates": self.parameter_updates, "gpu_parameter_updates": self.gpu_parameter_updates,
                "cache_rebuilds": self.cache_rebuilds, "cache_reuses": self.cache_reuses,
                "invalidations": self.invalidations, "state_loads": self.state_loads,
                "fallback_parameter_updates": self.fallback_parameter_updates,
                "cached_parameters": len(self.cache), "cache_bytes": sum(entry[1].numel() * entry[1].element_size() for entry in self.cache.values()),
                "saved_parameter_readback_bytes": self.saved_parameter_readback_bytes,
                "cache_dtype": "bfloat16", "moment_device": "cpu", "moment_dtype": "float32",
                "persistent_fp32_master": False, "checkpoint_format_changed": False,
                "writeback": "blocking same-dtype direct copy", "whole_training_speedup_measured": False}


@contextmanager
def use_cached_cpu_adam(*, allow_cpu=False, optimizers=None):
    """Temporarily cache BF16 parameters; retain native state/load/step APIs."""
    installation = CachedCPUAdamInstallation(allow_cpu=allow_cpu, optimizers=optimizers)
    original_update, original_load = CPUOffloadAdamW._update_one, CPUOffloadAdamW.load_state_dict
    original_parts = CPUOffloadAdamW.step_params

    def update(optimizer, parameter, group):
        if not installation.applies(optimizer):
            return original_update(optimizer, parameter, group)
        if installation.fallback_depth.get(optimizer, 0):
            changed = parameter.grad is not None
            result = _NATIVE_UPDATE_ONE(optimizer, parameter, group)
            installation.fallback_parameter_updates += int(changed)
            return result
        try:
            return installation.update(optimizer, parameter, group)
        except BaseException:
            installation.invalidate(parameter)
            raise

    def load(optimizer, state):
        if not installation.applies(optimizer):
            return original_load(optimizer, state)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                installation.invalidate(parameter)
        result = original_load(optimizer, state)
        installation.state_loads += 1
        return result

    def parts(optimizer, parameters):
        if not installation.applies(optimizer):
            return original_parts(optimizer, parameters)
        parameters = list(parameters)
        installation.fallback_depth[optimizer] = installation.fallback_depth.get(optimizer, 0) + 1
        try:
            return original_parts(optimizer, parameters)
        finally:
            installation.fallback_depth[optimizer] -= 1
            for parameter in parameters:
                installation.invalidate(parameter)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(CPUOffloadAdamW, "_update_one", update))
            stack.enter_context(patch.object(CPUOffloadAdamW, "load_state_dict", load))
            stack.enter_context(patch.object(CPUOffloadAdamW, "step_params", parts))
            yield installation
    finally:
        installation.invalidate()
