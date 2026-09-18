"""Sliding-window self-attn and gated cross-attn (Phase B backend)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.config import CATYokoConfig
from cat_yoko.rope import RMSNorm, RotaryEmbedding, apply_rope


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Attention softmax in fp32 (C1+FP8 whitelist); output matches ``q.dtype``."""
    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), attn_mask=bias.float()
    )
    return out.to(q.dtype)


def _window_causal_bias(
    q_len: int,
    k_len: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive mask: 0 keep, -inf drop. Position t may see [t-window+1, t] ∩ [0, t].

    If ``doc_ids`` is ``[B, S]``, tokens from different packed documents cannot attend
    (document mask). Returned shape is ``[S, S]`` or ``[B, 1, S, S]``.
    """
    q = torch.arange(q_len, device=device)
    k = torch.arange(k_len, device=device)
    causal = k[None, :] <= q[:, None]
    win = (q[:, None] - k[None, :]) < window
    keep = causal & win
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    if doc_ids is None:
        return bias
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    bias = bias.view(1, q_len, k_len).expand(doc_ids.size(0), -1, -1)
    bias = bias.masked_fill(~same, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


class WindowAttention(nn.Module):
    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.n_win = cfg.n_win
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, s, d = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(s, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        bias = _window_causal_bias(s, s, self.n_win, x.device, q.dtype, doc_ids)
        out = _sdpa(q, k, v, bias)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, d))


class CrossAttention(nn.Module):
    """Decoder queries attend causally to the YOCO global cache."""

    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, d = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, h, hd).transpose(1, 2)
        v = v.view(b, s, h, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(s, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        bias = _window_causal_bias(s, s, s, x.device, q.dtype, doc_ids)  # window=s → causal only
        out = _sdpa(q, k, v, bias)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, d))
