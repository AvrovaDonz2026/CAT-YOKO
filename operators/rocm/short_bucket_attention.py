"""Isolated short-document FP32 attention; no patches are installed on import.

The executor receives its plan directly and never replaces another module's
planner or executor globals. The optional installation changes only the packed
helper, process-locally and in LIFO order; distinct concurrent thread contexts
are unsupported. Prefer the standalone function for operator benchmarks.
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
from operators.rocm.bucketed_attention import BucketConfig, build_bucket_plan

_BASE_DENSE = packed._dense_document_sdpa
_PLAN_CACHE: dict[tuple, tuple] = {}


@dataclass(frozen=True)
class ShortBucketConfig:
    max_length: int = 256
    min_members: int = 4
    max_padding_ratio: float = 1.10
    max_bucket_size: int = 16
    max_bucket_score_elements: int = 4 << 20

    def __post_init__(self):
        if self.max_length < 1 or not 2 <= self.min_members <= self.max_bucket_size:
            raise ValueError("positive max_length and 2 <= min_members <= max_bucket_size required")
        if not math.isfinite(self.max_padding_ratio) or self.max_padding_ratio < 1:
            raise ValueError("max_padding_ratio must be finite and >= 1")
        if self.max_bucket_score_elements < 1:
            raise ValueError("max_bucket_score_elements must be positive")


def new_stats():
    return {**packed.new_stats(), "short_bucketed_calls": 0, "short_fallback_calls": 0,
            "short_plan_builds": 0, "short_batched_sdpa_calls": 0,
            "short_singleton_sdpa_calls": 0, "short_batched_fragments": 0,
            "short_batched_tokens": 0, "short_padded_score_elements": 0,
            "short_raw_score_elements": 0, "short_max_padding_ratio": 1.0,
            "short_fallback_reasons": {}}


def clear_short_plan_cache():
    _PLAN_CACHE.clear()


def build_short_bucket_plan(document_plan, seq, config: ShortBucketConfig):
    """Filter tight short groups; leave every other fragment a singleton.

    Build the existing length plan on short fragments alone. Its virtual batch
    is used only to group lengths; original batch/position records remain intact.
    The planner's long-document gate is inapplicable to this filtered domain.
    """
    if document_plan is None:
        return (), (), "noncontiguous_document_ids"
    records = tuple((b, start, end) for b, fragments in enumerate(document_plan)
                    for start, end in fragments)
    if not records or any(end <= start for _, start, end in records):
        return records, (), "empty_fragment"
    short = tuple(i for i, (_, start, end) in enumerate(records)
                  if end - start <= config.max_length)
    virtual = (tuple((records[i][1], records[i][2]) for i in short),)
    broad_config = BucketConfig(max_padding_ratio=config.max_padding_ratio,
        max_bucket_size=config.max_bucket_size,
        max_bucket_score_elements=config.max_bucket_score_elements,
        long_document_fraction=1.0)
    _, groups, _ = build_bucket_plan(virtual, max(seq, config.max_length) + 1, broad_config)
    selected = [tuple(short[i] for i in group) for group in groups
                if len(group) >= config.min_members]
    if not selected:
        return records, (), "no_qualifying_short_bucket"
    merged = {i for group in selected for i in group}
    singletons = [(i,) for i in range(len(records)) if i not in merged]
    return records, tuple(selected + singletons), None


def _targets(stats, observer):
    return [value for i, value in enumerate((stats, observer))
            if value is not None and (i == 0 or value is not stats)]


def _plan(docs, document_plan, config, stats, observer):
    try:
        version = docs._version
    except RuntimeError:
        version = None
    key = (id(docs), config)
    entry = _PLAN_CACHE.get(key)
    if version is not None and entry is not None and entry[0]() is docs and entry[1] == version:
        return entry[2]
    for target in _targets(stats, observer):
        target["short_plan_builds"] = target.get("short_plan_builds", 0) + 1
    result = build_short_bucket_plan(document_plan, int(docs.size(-1)), config)
    if version is not None:
        def remove(reference, cache_key=key):
            current = _PLAN_CACHE.get(cache_key)
            if current is not None and current[0] is reference:
                _PLAN_CACHE.pop(cache_key, None)
        _PLAN_CACHE[key] = (weakref.ref(docs, remove), version, result)
    return result


def _fallback_note(stats, observer, reason):
    for target in _targets(stats, observer):
        target["short_fallback_calls"] = target.get("short_fallback_calls", 0) + 1
        reasons = target.setdefault("short_fallback_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1


def _execute(q, k, v, docs, fallback, window, extra_bias, stats, *, config,
             previous=_BASE_DENSE, observer=None):
    document_plan = packed._document_plan(docs, stats)
    records, groups, reason = _plan(docs, document_plan, config, stats, observer)
    if reason is not None:
        _fallback_note(stats, observer, reason)
        return previous(q, k, v, docs, fallback, window, extra_bias, stats)
    heads, repeats = q.size(1), q.size(1) // k.size(1)
    batched = [group for group in groups if len(group) > 1]
    raw = sum((records[i][2] - records[i][1]) ** 2 for group in batched for i in group)
    padded = sum(len(group) * max(records[i][2] - records[i][1] for i in group) ** 2 for group in batched)
    for target in _targets(stats, observer):
        for key, increment in (("short_bucketed_calls", 1), ("optimized_calls", 1),
                ("dense_packed_calls", 1), ("query_tokens", q.size(0) * q.size(-2)),
                ("short_batched_sdpa_calls", len(batched)),
                ("short_singleton_sdpa_calls", len(groups) - len(batched)),
                ("short_batched_fragments", sum(map(len, batched))),
                ("short_batched_tokens", sum(records[i][2] - records[i][1] for g in batched for i in g)),
                ("short_padded_score_elements", heads * padded),
                ("short_raw_score_elements", heads * raw)):
            target[key] = target.get(key, 0) + increment
        target["short_max_padding_ratio"] = max(target.get("short_max_padding_ratio", 1), padded / raw)
    # The complete inputs have one cast each; all fragment gradients accumulate
    # in FP32 before crossing back to the original BF16 leaf dtype.
    with torch.autocast(device_type=q.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        floats = tuple(value.float() for value in (q, k, v))
        outputs = [None] * len(records)
        for group in groups:
            length = max(records[i][2] - records[i][1] for i in group)
            values = []
            for value in floats:
                pieces = []
                for index in group:
                    batch, start, end = records[index]
                    piece = value[batch:batch + 1, :, start:end, :]
                    if end - start != length:
                        piece = F.pad(piece, (0, 0, 0, length - (end - start)))
                    pieces.append(piece)
                values.append(torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0])
            qq, kk, vv = values
            if repeats != 1:
                kk, vv = native._repeat_kv(kk, repeats), native._repeat_kv(vv, repeats)
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
            for slot, index in enumerate(group):
                size = records[index][2] - records[index][1]
                outputs[index] = out[slot:slot + 1, :, :size, :]
            for target in _targets(stats, observer):
                elements = heads * len(group) * length ** 2
                target["doc_fragment_calls"] = target.get("doc_fragment_calls", 0) + 1
                target["fragment_score_elements"] = target.get("fragment_score_elements", 0) + elements
                target["max_fragment_score_elements"] = max(target.get("max_fragment_score_elements", 0), elements)
        batches, offset = [], 0
        for fragments in document_plan:
            pieces = outputs[offset:offset + len(fragments)]
            batches.append(torch.cat(pieces, dim=-2) if len(pieces) > 1 else pieces[0])
            offset += len(fragments)
        output = torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]
        native._note_sdpa("math_fp32", torch.float32)
        return output.to(q.dtype)


def short_bucket_window_sdpa(q, k, v, window, doc_ids=None, extra_bias=None, *,
                             config=None, stats=None):
    config = config or ShortBucketConfig()
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape
            or q.size(0) != k.size(0) or q.size(-1) != k.size(-1)
            or k.size(1) < 1 or q.size(1) % k.size(1)):
        raise ValueError("invalid Q/K/V shape or GQA heads")
    if any(value.dtype != q.dtype or value.device != q.device for value in (k, v)):
        raise ValueError("Q/K/V must share device and dtype")
    if doc_ids is None or extra_bias is not None or window < q.size(-2) or k.size(-2) != q.size(-2):
        _fallback_note(stats, None, "non_dense_or_extra_bias")
        return packed.packed_window_sdpa(q, k, v, window, doc_ids, extra_bias, stats=stats)
    if doc_ids.shape != (q.size(0), q.size(-2)):
        raise ValueError("doc_ids must have shape [B,S]")
    return _execute(q, k, v, doc_ids, packed._ORIGINAL_WINDOW_SDPA, window, extra_bias,
                    stats, config=config)


@dataclass
class Installation:
    previous: Callable
    replacement: Callable
    config: ShortBucketConfig
    stats: dict = field(default_factory=new_stats)
    removed: bool = False

    def report(self):
        return {**self.stats, "short_fallback_reasons": dict(self.stats["short_fallback_reasons"])}

    def remove(self):
        if self.removed:
            return
        if packed._dense_document_sdpa is not self.replacement:
            raise RuntimeError("short bucket contexts must be removed in reverse order")
        packed._dense_document_sdpa = self.previous
        self.removed = True


def install_short_bucket_attention(*, config=None):
    """Optional process-local helper patch; enter inside packed context.

    This never patches bucketed_attention's planner/executor or native SDPA.
    Callers must serialize context installation/removal across threads.
    """
    config = config or ShortBucketConfig()
    previous = packed._dense_document_sdpa
    installation = Installation(previous, None, config)

    def replacement(q, k, v, docs, fallback, window, extra_bias, stats):
        return _execute(q, k, v, docs, fallback, window, extra_bias, stats,
                        config=config, previous=previous, observer=installation.stats)

    installation.replacement = replacement
    packed._dense_document_sdpa = replacement
    return installation


@contextmanager
def short_bucket_attention_context(*, config=None):
    installation = install_short_bucket_attention(config=config)
    try:
        yield installation
    finally:
        installation.remove()
