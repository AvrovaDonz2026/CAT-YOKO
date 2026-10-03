"""Opt-in length-bucketed FP32 document attention; no production patch on import.

Enter this context inside packed_attention_context. Only the packed helper is
replaced, so native window/cross patches remain owned by their existing context.
Padding is on the right: causal real queries can never attend to padded keys.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from typing import Callable
import weakref

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import cat_yoko.attention as native
from operators.rocm import packed_attention as packed

_ORIGINAL_DENSE = packed._dense_document_sdpa
_BUCKET_CACHE: dict[tuple, tuple] = {}


@dataclass(frozen=True)
class BucketConfig:
    max_padding_ratio: float = 1.25
    max_bucket_size: int = 16
    # Per-head padded score count; cap multi-document batches to bound memory.
    max_bucket_score_elements: int = 4 << 20
    long_document_fraction: float = 0.75

    def __post_init__(self):
        if not math.isfinite(self.max_padding_ratio) or self.max_padding_ratio < 1:
            raise ValueError("max_padding_ratio must be finite and >= 1")
        if self.max_bucket_size < 2 or self.max_bucket_score_elements < 1:
            raise ValueError("bucket size must be >= 2 and score budget positive")
        if not math.isfinite(self.long_document_fraction) or not 0 < self.long_document_fraction <= 1:
            raise ValueError("long_document_fraction must be in (0,1]")


def new_stats() -> dict:
    return {**packed.new_stats(), "bucketed_calls": 0, "bucketed_fallback_calls": 0,
            "bucket_plan_builds": 0, "bucket_sdpa_calls": 0,
            "batched_sdpa_calls": 0, "singleton_sdpa_calls": 0,
            "bucket_fragments": 0, "raw_score_elements": 0,
            "padded_score_elements": 0, "max_bucket_size": 0,
            "max_bucket_length": 0, "max_padding_ratio": 1.0,
            "bucket_fallback_reasons": {}}


def clear_bucket_plan_cache() -> None:
    _BUCKET_CACHE.clear()


def _targets(stats, observer):
    return [value for i, value in enumerate((stats, observer))
            if value is not None and (i == 0 or value is not stats)]


def _fallback_note(stats, observer, reason):
    for target in _targets(stats, observer):
        target["bucketed_fallback_calls"] = target.get("bucketed_fallback_calls", 0) + 1
        reasons = target.setdefault("bucket_fallback_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1


def build_bucket_plan(document_plan, seq: int, config: BucketConfig):
    """Return original-order records, length-sorted buckets, and fallback reason.

    Each bucket independently respects the padding/score budget; consequently
    the entire plan also stays within the configured padding ratio. Singleton
    fragments use unpadded SDPA and do not incur extra square work.
    """
    if document_plan is None:
        return (), (), "noncontiguous_document_ids"
    records = tuple((batch, start, end) for batch, fragments in enumerate(document_plan)
                    for start, end in fragments)
    if not records or any(end <= start for _, start, end in records):
        return records, (), "empty_fragment"
    if len(records) == 1:
        return records, (), "single_document"
    if max(end - start for _, start, end in records) >= seq * config.long_document_fraction:
        return records, (), "long_document"
    order = sorted(range(len(records)), key=lambda index: records[index][2] - records[index][1])
    buckets, current, raw = [], [], 0
    for index in order:
        length = records[index][2] - records[index][1]
        count, proposed_raw = len(current) + 1, raw + length ** 2
        padded = count * length ** 2
        if current and (count > config.max_bucket_size
                        or padded > config.max_bucket_score_elements
                        or padded > proposed_raw * config.max_padding_ratio):
            buckets.append(tuple(current))
            current, raw = [], 0
        current.append(index)
        raw += length ** 2
    if current:
        buckets.append(tuple(current))
    if not any(len(bucket) > 1 for bucket in buckets):
        return records, tuple(buckets), "no_batchable_fragments"
    return records, tuple(buckets), None


def _bucket_plan(docs, document_plan, config, stats, observer):
    try:
        version = docs._version
    except RuntimeError:
        version = None
    key = (id(docs), config)
    entry = _BUCKET_CACHE.get(key)
    if version is not None and entry is not None and entry[0]() is docs and entry[1] == version:
        return entry[2]
    for target in _targets(stats, observer):
        target["bucket_plan_builds"] = target.get("bucket_plan_builds", 0) + 1
    result = build_bucket_plan(document_plan, int(docs.size(-1)), config)
    if version is not None:
        def remove(reference, cache_key=key):
            current = _BUCKET_CACHE.get(cache_key)
            if current is not None and current[0] is reference:
                _BUCKET_CACHE.pop(cache_key, None)
        _BUCKET_CACHE[key] = (weakref.ref(docs, remove), version, result)
    return result


def _execute(q, k, v, docs, fallback, window, extra_bias, stats, *, config,
             previous=_ORIGINAL_DENSE, observer=None):
    document_plan = packed._document_plan(docs, stats)
    records, buckets, reason = _bucket_plan(docs, document_plan, config, stats, observer)
    if reason is not None:
        _fallback_note(stats, observer, reason)
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    heads, repeats = q.size(1), q.size(1) // k.size(1)
    raw = heads * sum((end - start) ** 2 for _, start, end in records)
    padded = heads * sum(len(bucket) * max(records[i][2] - records[i][1] for i in bucket) ** 2
                         for bucket in buckets)
    for target in _targets(stats, observer):
        for key, count in (("bucketed_calls", 1), ("bucket_sdpa_calls", len(buckets)),
                           ("batched_sdpa_calls", sum(len(b) > 1 for b in buckets)),
                           ("singleton_sdpa_calls", sum(len(b) == 1 for b in buckets)),
                           ("bucket_fragments", len(records)), ("raw_score_elements", raw),
                           ("padded_score_elements", padded)):
            target[key] = target.get(key, 0) + count
        target["max_bucket_size"] = max(target.get("max_bucket_size", 0), max(map(len, buckets)))
        target["max_bucket_length"] = max(target.get("max_bucket_length", 0),
                                           max(end - start for _, start, end in records))
        target["max_padding_ratio"] = max(target.get("max_padding_ratio", 1), padded / raw)
    if stats is not None:
        stats["optimized_calls"] += 1
        stats["dense_packed_calls"] += 1
        stats["query_tokens"] += q.size(0) * q.size(-2)
    # This single cast per input preserves the original BF16 gradient boundary.
    # Padding/stacking/slicing remain differentiable and never reset RoPE.
    with torch.autocast(device_type=q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        floats = tuple(value.float() for value in (q, k, v))
        outputs = [None] * len(records)
        for bucket in buckets:
            length = max(records[index][2] - records[index][1] for index in bucket)
            batched = []
            for value in floats:
                pieces = []
                for index in bucket:
                    batch, start, end = records[index]
                    piece = value[batch:batch + 1, :, start:end, :]
                    if end - start != length:
                        piece = F.pad(piece, (0, 0, 0, length - (end - start)))
                    pieces.append(piece)
                batched.append(torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0])
            qq, kk, vv = batched
            if repeats != 1:
                kk, vv = native._repeat_kv(kk, repeats), native._repeat_kv(vv, repeats)
            result = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            for slot, index in enumerate(bucket):
                size = records[index][2] - records[index][1]
                outputs[index] = result[slot:slot + 1, :, :size, :]
            if stats is not None:
                elements = heads * len(bucket) * length ** 2
                stats["doc_fragment_calls"] += 1  # One real SDPA call, not each padded item.
                stats["fragment_score_elements"] += elements
                stats["max_fragment_score_elements"] = max(stats["max_fragment_score_elements"], elements)
        batches, offset = [], 0
        for fragments in document_plan:
            count = len(fragments)
            pieces = outputs[offset:offset + count]
            batches.append(torch.cat(pieces, dim=-2) if count > 1 else pieces[0])
            offset += count
        output = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
        native._note_sdpa("math_fp32", torch.float32)
        return output.to(q.dtype)


def bucketed_window_sdpa(q, k, v, window, doc_ids=None, extra_bias=None, *,
                         config: BucketConfig | None = None, stats: dict | None = None):
    """Standalone dense-packed candidate; retain existing packed fallbacks."""
    config = config or BucketConfig()
    if doc_ids is None or extra_bias is not None or window < q.size(-2):
        _fallback_note(stats, None, "non_dense_or_extra_bias")
        return packed.packed_window_sdpa(q, k, v, window, doc_ids, extra_bias, stats=stats)
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or k.size(0) != q.size(0)
            or k.size(-2) != q.size(-2) or k.size(-1) != q.size(-1)
            or q.size(1) % k.size(1) or doc_ids.shape != (q.size(0), q.size(-2))):
        raise ValueError("invalid dense packed Q/K/V or document shape")
    if any(value.device != q.device or value.dtype != q.dtype for value in (k, v)):
        raise ValueError("Q/K/V must share dtype and device")
    return _execute(q, k, v, doc_ids, packed._ORIGINAL_WINDOW_SDPA, window, extra_bias,
                    stats, config=config)


@dataclass
class Installation:
    previous: Callable
    replacement: Callable
    config: BucketConfig
    stats: dict = field(default_factory=new_stats)
    removed: bool = False

    def report(self):
        return {**self.stats, "bucket_fallback_reasons": dict(self.stats["bucket_fallback_reasons"])}

    def remove(self):
        if self.removed:
            return
        if packed._dense_document_sdpa is not self.replacement:
            raise RuntimeError("bucketed attention contexts must be removed in reverse order")
        packed._dense_document_sdpa = self.previous
        self.removed = True


def install_bucketed_attention(*, config: BucketConfig | None = None):
    config = config or BucketConfig()
    previous = packed._dense_document_sdpa
    installation = Installation(previous, None, config)

    def replacement(q, k, v, docs, fallback, window, extra_bias, stats):
        return _execute(q, k, v, docs, fallback, window, extra_bias, stats,
                        config=config, previous=previous, observer=installation.stats)

    installation.replacement = replacement
    packed._dense_document_sdpa = replacement
    return installation


@contextmanager
def bucketed_attention_context(*, config: BucketConfig | None = None):
    installation = install_bucketed_attention(config=config)
    try:
        yield installation
    finally:
        installation.remove()
