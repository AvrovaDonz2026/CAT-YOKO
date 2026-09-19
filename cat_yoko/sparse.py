"""PyTorch sparse-attn helpers for Phase C. Not a CSA CUDA kernel.

C-topk: indexer top-k additive mask on SDPA.
C-hca: mean-pool KV every ``compress_m_hca`` tokens (HCA-labeled layers).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cat_yoko.attention import (
    _fused_qkv,
    _repeat_kv,
    _sdpa,
    _window_causal_bias,
)
from cat_yoko.rope import apply_rope


def compress_kv(k: torch.Tensor, v: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool sequence groups. ``k,v`` are ``[B, H, S, D]``."""
    if group <= 1:
        return k, v
    b, h, s, d = k.shape
    if s <= group:
        return k, v
    pad = (group - (s % group)) % group
    if pad:
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    n = k.size(2) // group
    k = k.view(b, h, n, group, d).mean(dim=3)
    v = v.view(b, h, n, group, d).mean(dim=3)
    return k, v


def hca_causal_bias(
    q_len: int,
    group: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query ``t`` may read compressed slots ``0 .. t // group``."""
    q = torch.arange(q_len, device=device)
    k = torch.arange(k_len, device=device)
    keep = k[None, :] <= (q[:, None] // max(int(group), 1))
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    if doc_ids is None:
        return bias
    # Compressed slot s_comp came from tokens [s*g, (s+1)*g). Drop cross-doc.
    g = max(int(group), 1)
    # Approximate: compare query doc with the last token of the compressed group.
    slot_pos = ((k + 1) * g - 1).clamp(max=q_len - 1)
    slot_doc = doc_ids[:, slot_pos]
    same = doc_ids[:, :, None] == slot_doc[:, None, :]
    bias = bias.view(1, q_len, k_len).expand(doc_ids.size(0), -1, -1)
    bias = bias.masked_fill(~same, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def hca_attend(attn, x: torch.Tensor, doc_ids: torch.Tensor | None, group: int) -> torch.Tensor:
    """WindowAttention forward with mean-pooled KV. Falls back when group ≥ S."""
    b, s, d = x.shape
    if group <= 1 or group >= s:
        return attn(x, doc_ids)
    h, hd, n_kv = attn.n_heads, attn.head_dim, attn.n_kv
    q, k, v = _fused_qkv(attn.q_proj, attn.k_proj, attn.v_proj, x)
    q = q.view(b, s, h, hd).transpose(1, 2)
    k = k.view(b, s, n_kv, hd).transpose(1, 2)
    v = v.view(b, s, n_kv, hd).transpose(1, 2)
    if attn.q_norm is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    cos, sin = attn.rope(s, x.device, x.dtype)
    q, k = apply_rope(q, k, cos, sin)
    k = _repeat_kv(k, h // n_kv)
    v = _repeat_kv(v, h // n_kv)
    k, v = compress_kv(k, v, group)
    bias = hca_causal_bias(s, group, k.size(2), x.device, q.dtype, doc_ids)
    out = _sdpa(q, k, v, bias)
    return attn.o_proj(out.transpose(1, 2).contiguous().view(b, s, d))


def mean_head_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int,
    doc_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-over-heads dense attn probs ``[B, S, S]`` for indexer KL."""
    h = q.size(-3)
    n_kv = k.size(-3)
    if n_kv != h:
        k = _repeat_kv(k, h // n_kv)
    scale = q.size(-1) ** -0.5
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    b, _, s, _ = q.shape
    bias = _window_causal_bias(s, s, window, q.device, torch.float32, doc_ids)
    if bias.dim() == 2:
        logits = logits + bias
    else:
        logits = logits + bias
    return torch.softmax(logits, dim=-1).mean(dim=1)
