"""Opt-in FP32 MATH window tiles and document fragments for packed GQA.

Only the window SDPA call is replaced, preserving native projection/RoPE and
autograd boundaries. Extra-bias and unsupported shapes use the original
implementation. Importing this module never installs the candidate.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable
import weakref

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import cat_yoko.attention as native

_ORIGINAL_WINDOW_SDPA = native._window_sdpa
_DOC_PLAN_CACHE: dict[int, tuple] = {}


def new_stats() -> dict:
    return {"optimized_calls": 0, "tiled_calls": 0, "dense_packed_calls": 0,
            "fallback_calls": 0, "query_tokens": 0, "tile_batches": 0,
            "max_mask_elements": 0, "host_plan_builds": 0, "doc_fragment_calls": 0,
            "max_fragment_score_elements": 0, "fragment_score_elements": 0,
            "cross_attention_calls": 0}


def clear_doc_plan_cache() -> None:
    _DOC_PLAN_CACHE.clear()


def doc_plan_cache_size() -> int:
    return len(_DOC_PLAN_CACHE)


def _document_plan(docs: torch.Tensor, stats: dict | None):
    """Weak identity/version cache; a new/mutated batch never reuses old values."""
    identity = id(docs)
    try:
        version = docs._version
    except RuntimeError:  # Inference tensors have no mutation counter: do not cache.
        version = None
    entry = _DOC_PLAN_CACHE.get(identity)
    if version is not None and entry is not None and entry[0]() is docs and entry[1] == version:
        return entry[2]
    if stats is not None:
        stats["host_plan_builds"] += 1
    plan = []
    for row in docs.detach().cpu().tolist():
        starts, seen, valid = [0], set(), True
        if not row or any(type(value) not in (int, bool) for value in row):
            plan = None
            break
        seen.add(row[0])
        for position in range(1, len(row)):
            if row[position] != row[position - 1]:
                if row[position] in seen:
                    valid = False  # Native permits causal links to earlier same-ID runs.
                    break
                starts.append(position)
                seen.add(row[position])
        if not valid:
            plan = None
            break
        starts.append(len(row))
        plan.append(tuple(zip(starts[:-1], starts[1:])))
    if plan is not None:
        plan = tuple(plan)
    if version is not None:
        def remove(reference, key=identity):
            current = _DOC_PLAN_CACHE.get(key)
            if current is not None and current[0] is reference:
                _DOC_PLAN_CACHE.pop(key, None)

        _DOC_PLAN_CACHE[identity] = (weakref.ref(docs, remove), version, plan)
    return plan


def _dense_document_sdpa(q, k, v, docs, fallback, window, extra_bias, stats):
    plan = _document_plan(docs, stats)
    if plan is None:
        if stats is not None:
            stats["fallback_calls"] += 1
        return fallback(q, k, v, window, docs, extra_bias)
    if stats is not None:
        stats["optimized_calls"] += 1
        stats["dense_packed_calls"] += 1
        stats["query_tokens"] += q.size(0) * q.size(-2)
    # RoPE/norm has already run. Slices retain those positions and are never
    # re-rotated or offset. Only the attention keep-set is factored by document.
    with torch.autocast(device_type=q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        qf, kf, vf = (value.float() for value in (q, k, v))
        repeats = q.size(1) // k.size(1)
        batches = []
        for batch, fragments in enumerate(plan):
            pieces = []
            for start, end in fragments:
                qq, kk, vv = (value[batch:batch + 1, :, start:end, :] for value in (qf, kf, vf))
                if repeats != 1:
                    kk, vv = native._repeat_kv(kk, repeats), native._repeat_kv(vv, repeats)
                pieces.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True))
                if stats is not None:
                    elements = q.size(1) * (end - start) ** 2
                    stats["doc_fragment_calls"] += 1
                    stats["fragment_score_elements"] += elements
                    stats["max_fragment_score_elements"] = max(stats["max_fragment_score_elements"], elements)
            batches.append(torch.cat(pieces, dim=-2))
        native._note_sdpa("math_fp32", torch.float32)
        return torch.cat(batches, dim=0).to(q.dtype)


def packed_window_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
    doc_ids: torch.Tensor | None = None, extra_bias: torch.Tensor | None = None,
    *, tile_size: int = 256, min_seq_len: int = 0, fallback: Callable | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """Exact attention keep-set as native, with FP32 tilewise accumulation.

    All Q/K/V conversions happen once before tiling/fragmenting. K/V autograd therefore
    combines overlapping tile gradients in FP32 before casting back to BF16.
    Each tile sees its preceding window-1 keys and its own keys, then intersects
    causal, window, valid-position, and same-document predicates. Padded queries
    get one harmless valid key so softmax cannot produce NaN; they are cropped.
    A covering window uses separate causal matrices per contiguous document.
    No S×S document mask is created, though a single long document can still
    require a full matrix. Repeated IDs in separated runs retain native fallback.
    """
    if tile_size <= 0 or min_seq_len < 0:
        raise ValueError("tile_size must be positive and min_seq_len nonnegative")
    fallback = _ORIGINAL_WINDOW_SDPA if fallback is None else fallback
    seq, key_seq, width = int(q.size(-2)), int(k.size(-2)), int(window)
    tile = min(seq, max(int(tile_size), width))
    if (doc_ids is None or extra_bias is not None or seq != key_seq or width < 1
            or (tile >= seq and width < seq) or seq < min_seq_len):
        if stats is not None:
            stats["fallback_calls"] += 1
        return fallback(q, k, v, window, doc_ids, extra_bias)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("packed window expects [B,H,S,D] Q/K/V")
    batch, q_heads, _, dim = q.shape
    if (k.shape != v.shape or k.size(0) != batch or k.size(-1) != dim
            or q_heads % k.size(1) != 0):
        raise ValueError("Q/K/V must have matching batch, head dimension and valid GQA heads")
    if doc_ids.shape != (batch, seq):
        raise ValueError("doc_ids must have shape [B,S]")
    if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q/K/V must share device and dtype")
    if width >= seq:
        return _dense_document_sdpa(q, k, v, doc_ids, fallback, window, extra_bias, stats)
    kv_heads = k.size(1)
    lookback = width - 1
    padding = (-seq) % tile
    padded_seq = seq + padding
    tiles = padded_seq // tile
    key_width = lookback + tile
    if stats is not None:
        stats["optimized_calls"] += 1
        stats["tiled_calls"] += 1
        stats["query_tokens"] += batch * seq
        stats["tile_batches"] += batch * tiles
        stats["max_mask_elements"] = max(stats["max_mask_elements"], batch * tiles * tile * key_width)

    with torch.autocast(device_type=q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        qf, kf, vf = (value.float().contiguous() for value in (q, k, v))
        q_padded = F.pad(qf, (0, 0, 0, padding))
        k_padded = F.pad(kf, (0, 0, lookback, padding))
        v_padded = F.pad(vf, (0, 0, lookback, padding))
        qt = q_padded.reshape(batch, q_heads, tiles, tile, dim)
        kt = k_padded.unfold(2, key_width, tile).permute(0, 1, 2, 4, 3)
        vt = v_padded.unfold(2, key_width, tile).permute(0, 1, 2, 4, 3)
        qt = qt.permute(0, 2, 1, 3, 4).reshape(batch * tiles, q_heads, tile, dim)
        kt = kt.permute(0, 2, 1, 3, 4).reshape(batch * tiles, kv_heads, key_width, dim)
        vt = vt.permute(0, 2, 1, 3, 4).reshape(batch * tiles, kv_heads, key_width, dim)

        docs = doc_ids.to(device=q.device)
        q_docs = F.pad(docs, (0, padding)).reshape(batch, tiles, tile)
        k_docs = F.pad(docs, (lookback, padding)).unfold(1, key_width, tile)
        tile_start = torch.arange(tiles, device=q.device)[:, None, None] * tile
        query_pos = tile_start + torch.arange(tile, device=q.device)[None, :, None]
        key_pos = tile_start - lookback + torch.arange(key_width, device=q.device)[None, None, :]
        valid = ((query_pos < seq) & (key_pos >= 0) & (key_pos < seq)
                 & (key_pos <= query_pos) & (query_pos - key_pos < width))
        keep = valid.unsqueeze(0) & (q_docs[..., :, None] == k_docs[..., None, :])
        # An all-masked padded query must not poison overlapping K/V backward.
        # The first real key of the final tile is a harmless normalization anchor.
        padded_anchor = ((query_pos >= seq) & (key_pos == tile_start))
        keep = keep | padded_anchor.unsqueeze(0)
        bias = torch.zeros(keep.shape, device=q.device, dtype=torch.float32)
        bias = bias.masked_fill(~keep, torch.finfo(torch.float32).min)
        bias = bias.reshape(batch * tiles, 1, tile, key_width)
        repeats = q_heads // kv_heads
        # Same masked GQA head expansion as native _sdpa's FP32 MATH path.
        if repeats != 1:
            kt = native._repeat_kv(kt, repeats)
            vt = native._repeat_kv(vt, repeats)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=bias)
        native._note_sdpa("math_fp32", torch.float32)
        out = out.reshape(batch, tiles, q_heads, tile, dim).permute(0, 2, 1, 3, 4)
        return out.reshape(batch, q_heads, padded_seq, dim)[..., :seq, :].to(q.dtype)


@dataclass
class Installation:
    previous: Callable
    replacement: Callable
    removed: bool = False
    stats: dict = field(default_factory=new_stats)
    previous_cross: Callable | None = None
    replacement_cross: Callable | None = None

    def report(self) -> dict:
        """Mask count is candidate-created explicit masks; dense causal masks
        remain internal to MATH. Fragment score counts report dense work size.
        """
        return dict(self.stats)

    def remove(self) -> None:
        if self.removed:
            return
        if native._window_sdpa is not self.replacement:
            raise RuntimeError("packed attention patches must be removed in reverse installation order")
        if self.replacement_cross is not None and native.CrossAttention.forward is not self.replacement_cross:
            raise RuntimeError("packed cross attention patches must be removed in reverse installation order")
        native._window_sdpa = self.previous
        if self.previous_cross is not None:
            native.CrossAttention.forward = self.previous_cross
        self.removed = True


def install_packed_attention(*, tile_size: int = 256, min_seq_len: int = 2048) -> Installation:
    """Process-local patch. Never changes weights, module identities, or SDPA."""
    if tile_size <= 0 or min_seq_len < 0:
        raise ValueError("invalid packed attention tile/sequence settings")
    previous = native._window_sdpa
    previous_cross = native.CrossAttention.forward
    stats = new_stats()

    def replacement(q, k, v, window, doc_ids=None, extra_bias=None):
        return packed_window_sdpa(q, k, v, window, doc_ids, extra_bias,
                                  tile_size=tile_size, min_seq_len=min_seq_len, fallback=previous, stats=stats)

    def cross_forward(module, x, k, v, doc_ids=None, *, extra_bias=None, return_probs=False):
        if doc_ids is None or extra_bias is not None or return_probs or x.size(1) < min_seq_len:
            stats["fallback_calls"] += 1
            return previous_cross(module, x, k, v, doc_ids, extra_bias=extra_bias, return_probs=return_probs)
        seq = int(x.size(1))
        q = native.split_heads(module.q_proj(x), module.n_heads, module.head_dim)
        kk = native.split_heads(k, module.n_kv, module.head_dim)
        vv = native.split_heads(v, module.n_kv, module.head_dim)
        q, kk = native.rope_after_qk_norm(q, kk, rope=module.rope,
                                         q_norm=module.q_norm, k_norm=module.k_norm)
        qh, kh, vh = (native.to_sdpa_layout(value) for value in (q, kk, vv))
        stats["cross_attention_calls"] += 1
        out = packed_window_sdpa(qh, kh, vh, seq, doc_ids, tile_size=tile_size,
                                min_seq_len=min_seq_len, fallback=previous, stats=stats)
        return module.o_proj(native.merge_heads(out))

    native._window_sdpa = replacement
    native.CrossAttention.forward = cross_forward
    return Installation(previous, replacement, stats=stats,
                        previous_cross=previous_cross, replacement_cross=cross_forward)


def remove_packed_attention(installation: Installation) -> None:
    installation.remove()


@contextmanager
def packed_attention_context(*, tile_size: int = 256, min_seq_len: int = 2048):
    installation = install_packed_attention(tile_size=tile_size, min_seq_len=min_seq_len)
    try:
        yield installation
    finally:
        installation.remove()
