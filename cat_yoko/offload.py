"""Encoder / block CPU offload so C1 B1/B2 can take a GPU step on 32GB.

B0/B1: Encoder is frozen and the cache is detached, so the 16 encoder layers
(~8.2GiB bf16) can leave the GPU after the encoder forward.

B2: every EncoderBlock / DecoderBlock is checkpointed and moved to CPU between
layers. Adam moments stay on host RAM (see ``CPUOffloadAdamW``).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def module_device(mod: nn.Module) -> torch.device | None:
    p = next(mod.parameters(), None)
    return None if p is None else p.device


def same_device(a: torch.device | str, b: torch.device | str) -> bool:
    da, db = torch.device(a), torch.device(b)
    if da.type != db.type:
        return False
    if da.type == "cpu":
        return True
    return (da.index or 0) == (db.index or 0)


def move_module(mod: nn.Module, device: torch.device | str) -> None:
    cur = module_device(mod)
    if cur is not None and same_device(cur, device):
        return
    mod.to(device)


_after_block_backward = None


def set_after_block_backward(fn) -> None:
    """Optional hook: ``fn(blk)`` after a checkpointed block's backward (B2)."""
    global _after_block_backward
    _after_block_backward = fn


def _aux_of(blk: nn.Module, y: torch.Tensor) -> torch.Tensor:
    aux = getattr(getattr(blk, "mlp", None), "last_aux", None)
    if aux is None:
        return y.new_zeros(())
    return aux


def _autocast_ctx(device: torch.device, enabled: bool, dtype: torch.dtype):
    kind = "cuda" if device.type == "cuda" else device.type
    return torch.autocast(device_type=kind, dtype=dtype, enabled=enabled)


def _read_autocast(device: torch.device) -> tuple[bool, torch.dtype]:
    kind = "cuda" if device.type == "cuda" else device.type
    try:
        enabled = bool(torch.is_autocast_enabled(kind))
        dtype = torch.get_autocast_dtype(kind)
    except TypeError:
        enabled = bool(torch.is_autocast_enabled())
        dtype = torch.get_autocast_gpu_dtype() if kind == "cuda" else torch.bfloat16
    return enabled, dtype


def offload_checkpoint_block(blk: nn.Module, *tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``blk`` on ``tensors[0].device``, then move ``blk`` back to CPU.

    Backward reloads the block, recomputes, then offloads again (activation
    checkpoint + parameter offload).
    """
    need = any(t is not None and t.requires_grad for t in tensors) or any(
        p.requires_grad for p in blk.parameters()
    )
    if (not need) or (not torch.is_grad_enabled()):
        move_module(blk, tensors[0].device)
        y = blk(*tensors)
        aux = _aux_of(blk, y)
        move_module(blk, "cpu")
        return y, aux

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *ts):
            ctx.none_mask = tuple(t is None for t in ts)
            saved = tuple(t for t in ts if t is not None)
            ctx.save_for_backward(*saved)
            first = next(t for t in ts if t is not None)
            ctx.ac_enabled, ctx.ac_dtype = _read_autocast(first.device)
            move_module(blk, first.device)
            with torch.no_grad():
                y = blk(*ts)
                aux = _aux_of(blk, y)
            move_module(blk, "cpu")
            return y, aux

        @staticmethod
        def backward(ctx, gy, gaux):
            saved = list(ctx.saved_tensors)
            device = saved[0].device
            move_module(blk, device)
            inputs = []
            si = 0
            for is_none in ctx.none_mask:
                if is_none:
                    inputs.append(None)
                    continue
                t = saved[si]
                si += 1
                x = t.detach()
                if t.requires_grad:
                    x.requires_grad_(True)
                inputs.append(x)
            with torch.enable_grad(), _autocast_ctx(device, ctx.ac_enabled, ctx.ac_dtype):
                y = blk(*inputs)
                aux = _aux_of(blk, y)
            outs: list[torch.Tensor] = [y]
            grads: list[torch.Tensor | None] = [gy]
            if aux.requires_grad:
                outs.append(aux)
                grads.append(gaux if gaux is not None else torch.zeros_like(aux))
            torch.autograd.backward(tuple(outs), tuple(grads))
            move_module(blk, "cpu")
            cb = _after_block_backward
            if cb is not None:
                with torch.no_grad():
                    cb(blk)
            grads_out = []
            for is_none, x in zip(ctx.none_mask, inputs, strict=True):
                if is_none:
                    grads_out.append(None)
                else:
                    grads_out.append(x.grad)
            return tuple(grads_out)

    return _Fn.apply(*tensors)


def clip_grad_norm_mixed(params: Iterable[nn.Parameter], max_norm: float) -> float:
    """``clip_grad_norm_`` that allows mixed CPU / CUDA grads (block offload).

    Do not upcast whole grad tensors to fp32 — on B2 that would clone ~22GiB
    and blow a 62GiB cgroup.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    sq = 0.0
    for g in grads:
        sq += float(g.detach().norm(2).item() ** 2)
    total_norm = math.sqrt(sq)
    coef = float(max_norm) / (total_norm + 1e-6)
    if coef < 1.0:
        for g in grads:
            g.mul_(coef)
    return total_norm


def auto_offload_flags(
    *,
    phase: str,
    cfg_name: str,
    device: str,
    fsdp: bool,
    ddp: bool,
    offload_encoder: bool | None,
    offload_blocks: bool | None,
    optim_cpu: bool | None,
    deepspeed: bool = False,
) -> tuple[bool, bool, bool]:
    cuda = str(device).startswith("cuda")
    big = cfg_name == "CAT-YOKO-12B"
    dist = bool(fsdp or ddp)
    if deepspeed:
        if offload_encoder is True or offload_blocks is True:
            raise ValueError(
                "DeepSpeed ZeRO cannot mix native --offload-encoder/--offload-blocks; "
                "use --zero-offload-param"
            )
        if optim_cpu is True:
            raise ValueError(
                "DeepSpeed ZeRO cannot mix native --optim-cpu; use --zero-offload"
            )
        return False, False, False
    # Explicit --offload-encoder/--offload-blocks with DDP/FSDP would put
    # some params on CPU while the reducer expects a single device.
    if dist and (offload_encoder is True or offload_blocks is True):
        raise ValueError(
            "DDP/FSDP cannot mix CPU offload on 12B; use ZeRO or single GPU"
        )
    from cat_yoko.phases import PHASES

    ph = PHASES.get(phase)
    if offload_blocks is None:
        default_blocks = bool(ph.offload_blocks) if ph is not None else phase == "B2"
        offload_blocks = bool(big and cuda and default_blocks and not dist)
    if offload_encoder is None:
        default_enc = bool(ph.offload_encoder) if ph is not None else phase in {"B0", "B1"}
        offload_encoder = bool(
            big and cuda and default_enc and not offload_blocks and not dist
        )
    if optim_cpu is None:
        default_cpu = bool(ph.optim_cpu) if ph is not None else phase in {"B1", "B2"}
        optim_cpu = bool(big and cuda and default_cpu)
    return bool(offload_encoder), bool(offload_blocks), bool(optim_cpu)
