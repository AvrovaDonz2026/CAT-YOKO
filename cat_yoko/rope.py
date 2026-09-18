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
        # Must-high-prec: RMSNorm in fp32, then cast back (Llama-style).
        orig = x.dtype
        x32 = x.float()
        w = self.weight.float()
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x32, (x32.shape[-1],), w, self.eps).to(orig)
        var = x32.pow(2).mean(dim=-1, keepdim=True)
        y = x32 * torch.rsqrt(var + self.eps)
        return (w * y).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, D]; cos/sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
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
