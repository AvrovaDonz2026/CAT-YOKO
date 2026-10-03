"""Experimental CPU-expression-order FP32 Adam on the parameter device.

Matching operation order does not prove GPU/CPU numerical parity. The existing
BF16 byte-equality and FP32 moment gates must pass before this is installed.
There is no persistent FP32 parameter master.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from operators.rocm import gpu_adam


@torch.no_grad()
def ordered_update(optimizer, parameter, group, *, allow_cpu=False):
    """Expose the CPU expression's multiplications and divisions separately."""
    gpu_adam._validate_optimizer(optimizer)
    gpu_adam._validate_parameter(parameter, allow_cpu=allow_cpu)
    if parameter.grad is None:
        return False
    gpu_adam._device_moments(optimizer, parameter)
    lr, (beta1, beta2) = float(group["lr"]), group["betas"]
    eps, decay = float(group.get("eps", 1e-8)), float(group["weight_decay"])
    grad = parameter.grad.detach().to(device=parameter.device, dtype=torch.float32)
    p32 = parameter.detach().to(dtype=torch.float32)
    parameter.grad = None
    state = optimizer.state[parameter]
    if not state:
        state.update(step=0, exp_avg=torch.zeros_like(p32), exp_avg_sq=torch.zeros_like(p32))
    state["step"] = int(state["step"]) + 1
    step = state["step"]
    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    # CPU addcmul evaluates (value * tensor1) * tensor2, with FP32
    # intermediate rounding. GPU addcmul may associate value differently.
    squared_increment = grad.mul(1.0 - beta2).mul_(grad)
    exp_avg_sq.mul_(beta2).add_(squared_increment)
    if decay != 0.0:
        p32.mul_(1.0 - lr * decay)
    # A device scalar tensor selects tensor division instead of host-scalar
    # reciprocal multiplication in the GPU implementation.
    correction = torch.tensor(math.sqrt(1.0 - beta2**step), dtype=torch.float32, device=parameter.device)
    denom = exp_avg_sq.sqrt().div_(correction).add_(eps)
    updated = exp_avg.mul(-lr / (1.0 - beta1**step)).div_(denom).add_(p32)
    parameter.copy_(updated.to(dtype=parameter.dtype))
    return True


@contextmanager
def use_ordered_gpu_fp32_adam(*, allow_cpu=False, optimizers=None):
    """Reuse the original guarded context, snapshots, counters and cleanup."""
    with patch.object(gpu_adam, "update_one_on_parameter_device", ordered_update):
        with gpu_adam.use_gpu_fp32_adam(allow_cpu=allow_cpu, optimizers=optimizers) as installation:
            yield installation


# The same context API allows an isolated caller to select this implementation.
use_gpu_fp32_adam = use_ordered_gpu_fp32_adam
