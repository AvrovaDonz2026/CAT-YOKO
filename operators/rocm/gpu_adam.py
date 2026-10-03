"""Opt-in FP32 Adam moments on the parameter device, without master weights.

This copies CPUOffloadAdamW's update sequence. CPU and GPU arithmetic can differ;
matching BF16 parameter rounding and FP32 moments must be measured before use.
Exports remain independent CPU FP32 snapshots. Context exit restores methods and
returns surviving optimizers' moments to CPU, including exceptional exits.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import math
from unittest.mock import patch
import weakref

import torch

from cat_yoko.optim import CPUOffloadAdamW


def _validate_optimizer(optimizer):
    if optimizer.state_dtype != torch.float32 or not optimizer.retain_state:
        raise ValueError("GPU Adam requires retained FP32 moments")


def _validate_parameter(parameter, *, allow_cpu):
    if parameter.dtype != torch.bfloat16 or not parameter.requires_grad:
        raise ValueError("GPU Adam requires trainable BF16 parameters")
    if parameter.device.type != "cuda" and not (allow_cpu and parameter.device.type == "cpu"):
        raise ValueError("GPU Adam requires CUDA/HIP parameters; allow_cpu is only for tests")
    if parameter.grad is not None and parameter.grad.is_sparse:
        raise ValueError("GPU Adam requires dense gradients")


def _device_moments(optimizer, parameter):
    state = optimizer.state.get(parameter)
    if not state:
        return
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("GPU Adam expects native step/exp_avg/exp_avg_sq state")
    step = state["step"]
    step = step.item() if torch.is_tensor(step) else step
    if not isinstance(step, (int, float)) or not math.isfinite(step) or int(step) != step or step < 0:
        raise ValueError("invalid Adam step counter")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state[key]
        if not torch.is_tensor(value) or value.dtype != torch.float32 or value.shape != parameter.shape:
            raise ValueError("GPU Adam expects FP32 moments with the parameter shape")
    state["step"] = int(step)
    for key in ("exp_avg", "exp_avg_sq"):
        state[key] = state[key].detach().to(device=parameter.device, dtype=torch.float32)


@torch.no_grad()
def update_one_on_parameter_device(optimizer, parameter, group, *, allow_cpu=False):
    """Perform one native-order update; create a fresh FP32 p from BF16 each time."""
    _validate_optimizer(optimizer)
    _validate_parameter(parameter, allow_cpu=allow_cpu)
    if parameter.grad is None:
        return False
    _device_moments(optimizer, parameter)
    lr, (beta1, beta2) = float(group["lr"]), group["betas"]
    eps, decay = float(group.get("eps", 1e-8)), float(group["weight_decay"])
    grad = parameter.grad.detach().to(device=parameter.device, dtype=torch.float32)
    p32 = parameter.detach().to(dtype=torch.float32)
    parameter.grad = None
    state = optimizer.state[parameter]
    if not state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p32, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p32, dtype=torch.float32)
    state["step"] = int(state["step"]) + 1
    step = state["step"]
    exp_avg, exp_avg_sq = state["exp_avg"].float(), state["exp_avg_sq"].float()
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    state["exp_avg"], state["exp_avg_sq"] = exp_avg, exp_avg_sq
    if decay != 0.0:
        p32.mul_(1.0 - lr * decay)
    denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**step)).add_(eps)
    p32.addcdiv_(exp_avg, denom, value=-lr / (1.0 - beta1**step))
    parameter.copy_(p32.to(dtype=parameter.dtype))
    return True


def _host_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _host_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_host_snapshot(item) for item in value)
    return copy.deepcopy(value)


class GPUAdamInstallation:
    """Counters and cleanup for one temporary optimizer implementation."""

    def __init__(self, *, allow_cpu=False, optimizers=None):
        self.allow_cpu = allow_cpu
        self.targets = None if optimizers is None else weakref.WeakSet(optimizers)
        self.optimizers = weakref.WeakSet()
        self.parameter_updates = self.optimized_calls = self.state_loads = self.state_exports = 0

    def applies(self, optimizer):
        return self.targets is None or optimizer in self.targets

    def remember(self, optimizer):
        _validate_optimizer(optimizer)
        if optimizer not in self.optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    _validate_parameter(parameter, allow_cpu=self.allow_cpu)
            self.optimizers.add(optimizer)

    def report(self):
        return {"parameter_updates": self.parameter_updates, "optimized_calls": self.optimized_calls,
                "state_loads": self.state_loads, "state_exports": self.state_exports,
                "moment_dtype": "float32", "parameter_dtype": "bfloat16",
                "persistent_fp32_master": False, "checkpoint_moments": "independent CPU FP32 snapshot"}

    def restore_cpu(self):
        for optimizer in list(self.optimizers):
            for state in optimizer.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key] = state[key].detach().to(device="cpu", dtype=torch.float32)


@contextmanager
def use_gpu_fp32_adam(*, allow_cpu=False, optimizers=None):
    """Patch native optimizer methods; enter before Trainer constructs/restores it.

    ``optimizers`` optionally restricts installation to existing instances for a
    paired benchmark. ``allow_cpu`` exercises identical code on CPU in tests.
    Global patches require a standalone process without unrelated training.
    """
    installation = GPUAdamInstallation(allow_cpu=allow_cpu, optimizers=optimizers)
    original_update = CPUOffloadAdamW._update_one
    original_load = CPUOffloadAdamW.load_state_dict
    original_export = CPUOffloadAdamW.state_dict

    def update(optimizer, parameter, group):
        if not installation.applies(optimizer):
            return original_update(optimizer, parameter, group)
        installation.remember(optimizer)
        changed = update_one_on_parameter_device(optimizer, parameter, group, allow_cpu=allow_cpu)
        if changed:
            installation.parameter_updates += 1
            installation.optimized_calls += int(parameter.device.type == "cuda")

    def load(optimizer, state):
        if not installation.applies(optimizer):
            return original_load(optimizer, state)
        installation.remember(optimizer)
        saved_groups = state.get("param_groups", [])
        if len(saved_groups) != len(optimizer.param_groups) or any(
            len(old.get("params", [])) != len(new["params"])
            for old, new in zip(saved_groups, optimizer.param_groups)
        ):
            raise ValueError("loaded GPU Adam parameter groups do not match")
        mapping = {identifier: parameter for old,new in zip(saved_groups,optimizer.param_groups)
                   for identifier,parameter in zip(old["params"],new["params"])}
        for identifier, values in state.get("state", {}).items():
            if identifier not in mapping or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("loaded GPU Adam state does not map to native parameters")
            for key in ("exp_avg", "exp_avg_sq"):
                value = values.get(key)
                if (not torch.is_tensor(value) or value.dtype != torch.float32
                        or value.shape != mapping[identifier].shape):
                    raise ValueError("loaded GPU Adam moments must already be FP32 with the parameter shape")
        # Native loader preserves group mapping/hooks and bypasses BF16 casts.
        result = original_load(optimizer, state)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                _device_moments(optimizer, parameter)
        installation.state_loads += 1
        return result

    def export(optimizer):
        if not installation.applies(optimizer):
            return original_export(optimizer)
        installation.remember(optimizer)
        installation.state_exports += 1
        return _host_snapshot(original_export(optimizer))

    with ExitStack() as patches:
        patches.enter_context(patch.object(CPUOffloadAdamW, "_update_one", update))
        patches.enter_context(patch.object(CPUOffloadAdamW, "load_state_dict", load))
        patches.enter_context(patch.object(CPUOffloadAdamW, "state_dict", export))
        try:
            yield installation
        finally:
            installation.restore_cpu()
