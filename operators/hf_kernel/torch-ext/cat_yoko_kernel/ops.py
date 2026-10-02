# SPDX-License-Identifier: Apache-2.0
"""Frozen padded SwiGLU with native batched GEMM and SiLU backward.

This module depends only on PyTorch. It performs no expert routing, token
dispatch, reduction, model installation, or checkpoint conversion. The caller
must establish that all represented routed experts are frozen and identical
before passing one expert's master weights to ``pack_frozen_swiglu``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _check_tensor(name: str, tensor: torch.Tensor, *, ndim: int,
                  frozen: bool = False) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or not tensor.is_floating_point():
        raise ValueError(f"{name} must be a strided floating-point tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if frozen and tensor.requires_grad:
        raise ValueError(f"{name} must be frozen (requires_grad=False)")


def pack_frozen_swiglu(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one frozen expert as zero-stride views for ``experts`` experts.

    Gate/up weights have shape ``[I,H]`` and down has shape ``[H,I]``. Return
    fused gate/up ``[E,2I,H]`` and down ``[E,H,I]``. Gate/up concatenate once;
    the returned down view aliases the supplied master. Neither return value
    allocates E copies. Repack after changing a master weight: the fused gate/up
    tensor is a snapshot, while down retains its original storage alias.

    Only one master expert is accepted. This API cannot prove that distinct
    experts in a source model are identical; callers must verify that property
    before replacing their expert weights with these views. All weights must
    be frozen, floating point, and on the same device with the same dtype.
    """
    if isinstance(experts, bool) or not isinstance(experts, int) or experts < 1:
        raise ValueError("experts must be a positive integer")
    for name, weight in (("gate_weight", gate_weight), ("up_weight", up_weight),
                         ("down_weight", down_weight)):
        _check_tensor(name, weight, ndim=2, frozen=True)
    intermediate, hidden = gate_weight.shape
    if min(intermediate, hidden) < 1:
        raise ValueError("hidden and intermediate dimensions must be positive")
    if up_weight.shape != gate_weight.shape or down_weight.shape != (hidden, intermediate):
        raise ValueError("weights must have shapes gate/up=[I,H], down=[H,I]")
    if any(weight.device != gate_weight.device or weight.dtype != gate_weight.dtype
           for weight in (up_weight, down_weight)):
        raise ValueError("all weights must have the same device and dtype")
    fused = torch.cat((gate_weight, up_weight), dim=0)
    return (fused.unsqueeze(0).expand(experts, *fused.shape),
            down_weight.unsqueeze(0).expand(experts, *down_weight.shape))


class _SiluMulFn(torch.autograd.Function):
    """Native ``SiLU(gate) * up`` arithmetic with one forward output buffer."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        return F.silu(gate).mul_(up)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        gate, up = ctx.saved_tensors
        need_gate, need_up = ctx.needs_input_grad
        if dy is None or not (need_gate or need_up):
            return None, None
        # Preserve the native rounded dy*up and the ATen SiLU kernel's FP32
        # opmath for BF16/FP16. A BF16 sigmoid expansion differs near dSiLU=0.
        dgate = torch.ops.aten.silu_backward(dy * up, gate) if need_gate else None
        dup = dy * F.silu(gate) if need_up else None
        return dgate, dup


def frozen_swiglu(
    x_padded: torch.Tensor,
    fused_gate_up: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply frozen padded SwiGLU without changing expert/reduction shapes.

    Accept input ``[E,N,H]``, fused gate/up ``[E,2I,H]``, and down ``[E,H,I]``;
    return ``[E,N,H]``. This keeps the native bmm -> chunk -> SiLU*up -> bmm
    sequence and supports gradients with respect to ``x_padded`` only.

    Input and weights must share dtype/device. When autocast is enabled, its
    target dtype must also match: converting an expanded E-expert view would
    materialize E weight copies. Move/cast master weights before packing.
    No input, packed weight, or autograd-saved tensor is detached or cached.
    """
    _check_tensor("x_padded", x_padded, ndim=3)
    _check_tensor("fused_gate_up", fused_gate_up, ndim=3, frozen=True)
    _check_tensor("down_weight", down_weight, ndim=3, frozen=True)
    experts, _, hidden = x_padded.shape
    intermediate_twice = fused_gate_up.shape[1]
    if experts < 1 or hidden < 1 or intermediate_twice < 2 or intermediate_twice % 2:
        raise ValueError("positive E/H/I and an even fused gate/up dimension are required")
    intermediate = intermediate_twice // 2
    if (fused_gate_up.shape != (experts, 2 * intermediate, hidden)
            or down_weight.shape != (experts, hidden, intermediate)):
        raise ValueError("shapes must be x=[E,N,H], fused_gate_up=[E,2I,H], down=[E,H,I]")
    if any(weight.device != x_padded.device or weight.dtype != x_padded.dtype
           for weight in (fused_gate_up, down_weight)):
        raise ValueError("input and packed weights must have the same device and dtype")
    device_type = x_padded.device.type
    if torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) != x_padded.dtype:
        raise ValueError("autocast dtype must match input and packed weights; cast masters before packing")
    gate_up = torch.bmm(x_padded, fused_gate_up.transpose(1, 2))
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.bmm(_SiluMulFn.apply(gate, up), down_weight.transpose(1, 2))
