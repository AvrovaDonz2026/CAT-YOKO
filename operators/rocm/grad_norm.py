"""Experimental mixed-device clipping with one norm readback per device.

Each gradient still uses the native ``detach().norm(2)`` in its original dtype.
Only scalar readback is batched. Python floats are squared and summed in the
original parameter order, preserving the native rounding and clipping formula.
This reduces GPU synchronization opportunities; it does not fuse norm kernels
or establish an end-to-end training speedup. CPU-only workloads may pay extra
stack allocation overhead.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import math
from typing import Iterable, Iterator
from unittest.mock import patch

import torch
from torch import nn


def clip_grad_norm_mixed(params: Iterable[nn.Parameter], max_norm: float) -> float:
    """Match CAT-YOKO native clipping, batching scalar copies by device."""
    grads = [parameter.grad for parameter in params if parameter.grad is not None]
    if not grads:
        return 0.0

    groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
    for index, grad in enumerate(grads):
        norm = grad.detach().norm(2)
        groups.setdefault(norm.device, []).append((index, norm))

    values = [0.0] * len(grads)
    for entries in groups.values():
        # Scalar dtype promotion in stack is exact for BF16/FP16/FP32/FP64
        # norms. Do not cast the gradient or recompute its norm in FP32.
        host_norms = torch.stack([norm for _, norm in entries]).cpu().tolist()
        for (index, _norm), value in zip(entries, host_norms):
            values[index] = value

    sq = 0.0
    for value in values:
        sq += float(value ** 2)
    total_norm = math.sqrt(sq)
    coef = float(max_norm) / (total_norm + 1e-6)
    if coef < 1.0:
        for grad in grads:
            grad.mul_(coef)
    return total_norm


@contextmanager
def use_batched_grad_norm() -> Iterator[None]:
    """Temporarily patch offload and Trainer's imported reference; always restore.

    For standalone experiments only. Patches are process-wide, so unrelated
    training threads must not run inside this context. No model/optimizer state
    is installed or changed by entering or leaving the context.
    """
    import cat_yoko.offload as offload
    import cat_yoko.trainer as trainer

    with ExitStack() as stack:
        stack.enter_context(patch.object(offload, "clip_grad_norm_mixed", clip_grad_norm_mixed))
        stack.enter_context(patch.object(trainer, "clip_grad_norm_mixed", clip_grad_norm_mixed))
        yield
