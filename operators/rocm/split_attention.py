"""Opt-in packed FP32 attention with one document split per Q/K/V tensor.

The attention calls, their views, and the once-only FP32 conversions retain
the packed reference's arithmetic. Only the autograd partition changes: a
single split joins disjoint document gradients instead of independently
expanding every sliced gradient to the complete sequence. Import is inert.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import cat_yoko.attention as native
from operators.rocm import packed_attention as packed


_BASE_DENSE = packed._dense_document_sdpa


def new_stats() -> dict:
    return {**packed.new_stats(), "split_optimized_calls": 0,
            "split_documents": 0, "split_fallback_calls": 0,
            "split_fallback_reasons": {}}


def _targets(stats, observer):
    return [target for index, target in enumerate((stats, observer))
            if target is not None and (index == 0 or target is not stats)]


def _fallback_note(stats, observer, reason):
    for target in _targets(stats, observer):
        target["split_fallback_calls"] = target.get("split_fallback_calls", 0) + 1
        reasons = target.setdefault("split_fallback_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1


def _execute(q, k, v, docs, fallback, window, extra_bias, stats, *,
             previous=_BASE_DENSE, observer=None):
    if q.size(0) != 1:
        _fallback_note(stats, observer, "batch_not_one")
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    plan = packed._document_plan(docs, stats)
    if plan is None:
        _fallback_note(stats, observer, "invalid_or_noncontiguous_document_ids")
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    fragments = plan[0]
    if len(fragments) < 2:
        _fallback_note(stats, observer, "fewer_than_two_documents")
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    lengths = tuple(end - start for start, end in fragments)
    # _document_plan produces a positive, contiguous partition of the row.
    # Retain a fail-closed guard in case a future planner changes that contract.
    if (len(plan) != 1 or any(length <= 0 for length in lengths)
            or fragments[0][0] != 0 or fragments[-1][1] != q.size(-2)
            or any(left[1] != right[0] for left, right in zip(fragments, fragments[1:]))):
        _fallback_note(stats, observer, "invalid_partition")
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    targets = _targets(stats, observer)
    for target in targets:
        for key, increment in (("split_optimized_calls", 1),
                ("split_documents", len(fragments)), ("optimized_calls", 1),
                ("dense_packed_calls", 1), ("query_tokens", q.size(-2))):
            target[key] = target.get(key, 0) + increment

    with torch.autocast(device_type=q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        qf, kf, vf = (value.float() for value in (q, k, v))
        parts = tuple(value.split(lengths, dim=-2) for value in (qf, kf, vf))
        repeats = q.size(1) // k.size(1)
        pieces = []
        for length, (qq, kk, vv) in zip(lengths, zip(*parts)):
            if repeats != 1:
                kk, vv = native._repeat_kv(kk, repeats), native._repeat_kv(vv, repeats)
            pieces.append(F.scaled_dot_product_attention(qq, kk, vv, is_causal=True))
            for target in targets:
                elements = q.size(1) * length ** 2
                target["doc_fragment_calls"] = target.get("doc_fragment_calls", 0) + 1
                target["fragment_score_elements"] = target.get("fragment_score_elements", 0) + elements
                target["max_fragment_score_elements"] = max(
                    target.get("max_fragment_score_elements", 0), elements)
        # Keep both reference concatenations, including its batch-one cat.
        batches = [torch.cat(pieces, dim=-2)]
        native._note_sdpa("math_fp32", torch.float32)
        return torch.cat(batches, dim=0).to(q.dtype)


def split_window_sdpa(q, k, v, window, doc_ids=None, extra_bias=None, *, stats=None):
    """Standalone candidate; only a covering, single-row document plan splits.

    Unsupported attention modes keep packed_window_sdpa's exact path. The
    function installs no patches; use it directly for isolated comparisons.
    """
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape
            or q.size(0) != k.size(0) or q.size(-1) != k.size(-1)
            or k.size(1) < 1 or q.size(1) % k.size(1)):
        raise ValueError("invalid Q/K/V shape or GQA heads")
    if any(value.dtype != q.dtype or value.device != q.device for value in (k, v)):
        raise ValueError("Q/K/V must share device and dtype")
    width = int(window)
    if (doc_ids is None or extra_bias is not None or width < q.size(-2)
            or k.size(-2) != q.size(-2)):
        _fallback_note(stats, None, "non_dense_or_extra_bias")
        return packed.packed_window_sdpa(q, k, v, window, doc_ids, extra_bias, stats=stats)
    if doc_ids.shape != (q.size(0), q.size(-2)):
        raise ValueError("doc_ids must have shape [B,S]")
    return _execute(q, k, v, doc_ids, packed._ORIGINAL_WINDOW_SDPA, window,
                    extra_bias, stats)


@dataclass
class Installation:
    previous: Callable
    replacement: Callable | None
    stats: dict = field(default_factory=new_stats)
    removed: bool = False

    def report(self) -> dict:
        return {**self.stats,
                "split_fallback_reasons": dict(self.stats["split_fallback_reasons"])}

    def remove(self) -> None:
        if self.removed:
            return
        if packed._dense_document_sdpa is not self.replacement:
            raise RuntimeError("split attention contexts must be removed in reverse order")
        packed._dense_document_sdpa = self.previous
        self.removed = True


def install_split_attention() -> Installation:
    """Process-local helper patch; enter inside the packed attention context.

    The caller must serialize context installation/removal across threads.
    No native attention method, parameter, optimizer or planner is replaced.
    """
    previous = packed._dense_document_sdpa
    installation = Installation(previous, None)

    def replacement(q, k, v, docs, fallback, window, extra_bias, stats):
        return _execute(q, k, v, docs, fallback, window, extra_bias, stats,
                        previous=previous, observer=installation.stats)

    installation.replacement = replacement
    packed._dense_document_sdpa = replacement
    return installation


@contextmanager
def split_attention_context():
    installation = install_split_attention()
    try:
        yield installation
    finally:
        installation.remove()
