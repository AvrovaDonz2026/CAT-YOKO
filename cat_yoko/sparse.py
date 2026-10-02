"""PyTorch sparse-attn helpers for Phase C. Not a CSA CUDA kernel.

Theorem B (``docs/ARCHITECTURE_THEORY.md``): a query at ``t`` sees

    V(t) = W(t) ∪ tokens in compressed blocks s < floor(t / m)

The sliding window covers the own-block hole. Indexer top-k may only
**delete** from the compressed set, never add. HCA is the same union with
mean-pooled slots (``m'``) instead of CSA ``m``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cat_yoko.attention import (
    _fused_qkv,
    _repeat_kv,
    _sdpa,
    _window_causal_bias,
    merge_heads,
    rope_after_qk_norm,
    split_heads,
    to_sdpa_layout,
)


_KEEP_CACHE: dict[tuple, torch.Tensor] = {}
_HCA_BIAS_CACHE: dict[tuple, torch.Tensor] = {}
_SPARSE_CACHE_MAX = 24


def compressed_keep_matrix(
    seq: int,
    group: int,
    device: torch.device,
) -> torch.Tensor:
    """``[S, S]`` bool: key ``p`` sits in a block strictly before query ``t``."""
    g = max(int(group), 1)
    dev = torch.device(device)
    key = ("comp", int(seq), g, dev.type, dev.index)
    hit = _KEEP_CACHE.get(key)
    if hit is not None:
        return hit
    t = torch.arange(seq, device=device)
    p = torch.arange(seq, device=device)
    out = (p.unsqueeze(0) // g) < (t.unsqueeze(1) // g)
    if len(_KEEP_CACHE) >= _SPARSE_CACHE_MAX:
        _KEEP_CACHE.clear()
    _KEEP_CACHE[key] = out
    return out


def window_keep_matrix(
    seq: int,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    """``[S, S]`` bool matching ``_window_causal_bias`` keep-set (no doc mask)."""
    t = torch.arange(seq, device=device)
    p = torch.arange(seq, device=device)
    causal = p.unsqueeze(0) <= t.unsqueeze(1)
    win = (t.unsqueeze(1) - p.unsqueeze(0)) < max(int(window), 1)
    return causal & win


def own_block_token_keep(seq: int, group: int, device: torch.device) -> torch.Tensor:
    """``[S, S]``: keys in the query's own compression block with ``p <= t``."""
    g = max(int(group), 1)
    t = torch.arange(seq, device=device)
    p = torch.arange(seq, device=device)
    same = (p.unsqueeze(0) // g) == (t.unsqueeze(1) // g)
    return same & (p.unsqueeze(0) <= t.unsqueeze(1))


def _same_doc(doc_ids: torch.Tensor | None, seq: int, device: torch.device) -> torch.Tensor | None:
    if doc_ids is None:
        return None
    return doc_ids[:, :, None] == doc_ids[:, None, :]


def lift_compressed(
    window_bias: torch.Tensor,
    compress_keep: torch.Tensor,
) -> torch.Tensor:
    """Unmask compressed keys that the sliding window dropped. Never adds future."""
    keep = compress_keep.bool()
    bias = window_bias
    if keep.dim() == 3:
        if bias.dim() == 2:
            bias = bias.view(1, 1, bias.size(0), bias.size(1)).expand(
                keep.size(0), 1, -1, -1
            )
        elif bias.dim() == 4 and bias.size(0) == 1 and keep.size(0) > 1:
            bias = bias.expand(keep.size(0), -1, -1, -1)
        keep = keep.unsqueeze(1)
    elif keep.dim() == 2 and bias.dim() == 4:
        keep = keep.view(1, 1, keep.size(0), keep.size(1)).expand_as(bias)
    zeros = torch.zeros_like(bias)
    return torch.where(keep.expand_as(bias), zeros, bias)


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


def hca_slot_keep(q_len: int, group: int, n_slots: int, device: torch.device) -> torch.Tensor:
    """``[S, n_slots]``: slot ``s`` is visible iff ``s < floor(t / group)`` (own block out)."""
    g = max(int(group), 1)
    t = torch.arange(q_len, device=device)
    slot = torch.arange(n_slots, device=device)
    return slot.unsqueeze(0) < (t.unsqueeze(1) // g)


def hca_causal_bias(
    q_len: int,
    group: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compressed-slot bias. Own block is **excluded** (theorem B)."""
    if doc_ids is None:
        dev = torch.device(device)
        key = (int(q_len), int(group), int(k_len), dev.type, dev.index, str(dtype))
        hit = _HCA_BIAS_CACHE.get(key)
        if hit is not None:
            return hit
    keep = hca_slot_keep(q_len, group, k_len, device)
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    if doc_ids is None:
        if len(_HCA_BIAS_CACHE) >= _SPARSE_CACHE_MAX:
            _HCA_BIAS_CACHE.clear()
        _HCA_BIAS_CACHE[key] = bias
        return bias
    g = max(int(group), 1)
    slot = torch.arange(k_len, device=device)
    slot_pos = ((slot + 1) * g - 1).clamp(max=q_len - 1)
    slot_doc = doc_ids[:, slot_pos]
    same = doc_ids[:, :, None] == slot_doc[:, None, :]
    bias = bias.view(1, q_len, k_len).expand(doc_ids.size(0), -1, -1)
    bias = bias.masked_fill(~same, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def _qkv_rope(attn, x: torch.Tensor):
    h, hd, n_kv = attn.n_heads, attn.head_dim, attn.n_kv
    q, k, v = _fused_qkv(attn.q_proj, attn.k_proj, attn.v_proj, x)
    q = split_heads(q, h, hd)
    k = split_heads(k, n_kv, hd)
    v = split_heads(v, n_kv, hd)
    q, k = rope_after_qk_norm(
        q, k, rope=attn.rope, q_norm=attn.q_norm, k_norm=attn.k_norm
    )
    return to_sdpa_layout(q), to_sdpa_layout(k), to_sdpa_layout(v), h, n_kv


def csa_attend(
    attn,
    x: torch.Tensor,
    doc_ids: torch.Tensor | None,
    compress_keep: torch.Tensor,
) -> torch.Tensor:
    """Window GQA with compressed-block keys lifted (union mask, not a CSA kernel)."""
    b, s, d = x.shape
    q, k, v, _, _ = _qkv_rope(attn, x)
    bias = _window_causal_bias(s, s, attn.n_win, x.device, q.dtype, doc_ids)
    same = _same_doc(doc_ids, s, x.device)
    keep = compress_keep
    if same is not None:
        while keep.dim() < same.dim():
            keep = keep.unsqueeze(0)
        keep = keep & same
    bias = lift_compressed(bias, keep)
    out = _sdpa(q, k, v, bias)
    return attn.o_proj(merge_heads(out))


def _cat_bias(win: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
    """Concat window keys and compressed slots on the last dim."""
    if win.dim() == 2 and comp.dim() == 2:
        return torch.cat([win, comp], dim=-1)
    if win.dim() == 2:
        win = win.view(1, 1, win.size(0), win.size(1))
    if comp.dim() == 2:
        comp = comp.view(1, 1, comp.size(0), comp.size(1))
    if win.size(0) == 1 and comp.size(0) > 1:
        win = win.expand(comp.size(0), -1, -1, -1)
    if comp.size(0) == 1 and win.size(0) > 1:
        comp = comp.expand(win.size(0), -1, -1, -1)
    if win.size(1) == 1 and comp.size(1) > 1:
        win = win.expand(-1, comp.size(1), -1, -1)
    if comp.size(1) == 1 and win.size(1) > 1:
        comp = comp.expand(-1, win.size(1), -1, -1)
    return torch.cat([win, comp], dim=-1)


def hca_attend(attn, x: torch.Tensor, doc_ids: torch.Tensor | None, group: int) -> torch.Tensor:
    """Concat sliding-window KV with mean-pooled long-range slots (own block out)."""
    b, s, d = x.shape
    if group <= 1 or group >= s:
        return attn(x, doc_ids)
    q, k, v, _, _ = _qkv_rope(attn, x)
    k_c, v_c = compress_kv(k, v, group)
    if k_c.size(2) == k.size(2):
        return attn(x, doc_ids)
    k_cat = torch.cat([k, k_c], dim=2)
    v_cat = torch.cat([v, v_c], dim=2)
    win = _window_causal_bias(s, s, attn.n_win, x.device, q.dtype, doc_ids)
    comp = hca_causal_bias(s, group, k_c.size(2), x.device, q.dtype, doc_ids)
    bias = _cat_bias(win, comp)
    out = _sdpa(q, k_cat, v_cat, bias)
    return attn.o_proj(merge_heads(out))


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
    logits = logits + bias
    return torch.softmax(logits, dim=-1).mean(dim=1)
