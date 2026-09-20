"""RoPE and RMSNorm."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Must-high-prec: variance in fp32 (KEEP_HIGH_PREC rms_norm / qk_norm).
        orig = x.dtype
        # CUDA fused rms_norm accumulates rstd in fp32 on fp16/bf16 input.
        # Skip the full activation ``x.float()`` copy (one HBM round per ln).
        # CPU keeps the explicit fp32 path so autocast probes match Llama math.
        if (
            x.is_cuda
            and orig in (torch.float16, torch.bfloat16)
            and hasattr(F, "rms_norm")
        ):
            w = self.weight if self.weight.dtype == orig else self.weight.to(dtype=orig)
            return F.rms_norm(x, (x.shape[-1],), w, self.eps)
        x32 = x.float()
        w = self.weight.float()
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x32, (x32.shape[-1],), w, self.eps).to(orig)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        y = x32 * torch.rsqrt(var + self.eps)
        return (w * y).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[-x2; x1]`` on the last dim. ``empty_like`` keeps ``x``'s strides.

    ``torch.cat`` of the two halves always packs a new contiguous tensor, so
    RoPE on a ``[B, H, S, D]`` view of ``[B, S, H, D]`` memory used to destroy
    the Flash-native BSHD layout.
    """
    d = int(x.size(-1))
    if d % 2:
        raise ValueError(f"RoPE head dim must be even, got {d}")
    h = d // 2
    out = torch.empty_like(x)
    out[..., :h] = -x[..., h:]
    out[..., h:] = x[..., :h]
    return out


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seq_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE. ``cos``/``sin`` are ``[S, D]``.

    Default ``seq_dim=-2`` is BHSD ``[B, H, S, D]``. Pass ``seq_dim=1`` for
    contiguous ``[B, S, H, D]`` (qk_norm + RoPE before the SDPA transpose).
    """
    if q.size(-1) != cos.size(-1) or k.size(-1) != sin.size(-1):
        raise ValueError(
            f"RoPE head dim mismatch q={tuple(q.shape)} k={tuple(k.shape)} "
            f"cos={tuple(cos.shape)}"
        )
    sd = seq_dim if seq_dim >= 0 else q.dim() + seq_dim
    seq = int(cos.size(0))
    if q.size(sd) != seq or k.size(sd) != seq:
        raise ValueError(
            f"RoPE seq {seq} != q.size({sd})={q.size(sd)} or k.size({sd})={k.size(sd)}"
        )
    shape = [1] * q.dim()
    shape[sd] = seq
    shape[-1] = int(cos.size(-1))
    cos_b = cos.reshape(*shape)
    sin_b = sin.reshape(*shape)
    q = q * cos_b + rotate_half(q) * sin_b
    k = k * cos_b + rotate_half(k) * sin_b
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cache_key: tuple[int, str, torch.dtype] | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, str(device), dtype)
        if (
            self._cache_key == key
            and self._cos is not None
            and self._sin is not None
            and self._cos.device == device
            and self._sin.dtype == dtype
        ):
            return self._cos, self._sin
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cache_key = key
        return self._cos, self._sin
