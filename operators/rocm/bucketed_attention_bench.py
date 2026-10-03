#!/usr/bin/env python3
"""Compare packed and bucketed FP32 attention, including cold document plans.

No model updates or production installation. The independent dense oracle
checks output and all Q/K/V gradients before warm/cold GPU timings are recorded.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import cat_yoko.attention as native
from operators.rocm import packed_attention as packed
from operators.rocm.attention_bench import _emit, _errors, _is_oom, _measure
from operators.rocm.bucketed_attention import (
    BucketConfig, bucketed_window_sdpa, clear_bucket_plan_cache, new_stats,
)
from operators.rocm.gpu_wait import wait_for_gpu_idle
from operators.rocm.packed_attention_bench import (
    TOLERANCES, dense_oracle, input_values, load_documents, pid_snapshot, snapshot,
)


CASES = ("one_long", "six_medium", "many_short", "uneven", "real")


def case_lengths(case: str, seq: int):
    if case == "one_long":
        return [seq]
    if case == "many_short":
        return [64] * (seq // 64) + ([seq % 64] if seq % 64 else [])
    weights = [660, 668, 675, 690, 698, 705] if case == "six_medium" else [44, 8, 4, 2, 2, 2, 1, 1]
    if case not in {"six_medium", "uneven"}:
        raise ValueError("unknown synthetic case")
    lengths = [max(1, seq * value // sum(weights)) for value in weights]
    lengths[-1] += seq - sum(lengths)
    if min(lengths) <= 0:
        raise ValueError("sequence is too short for document fixture")
    return lengths


def fixture_documents(case, args):
    if case == "real":
        docs, info = load_documents(args.data, seq=args.seq_len, batch=args.batch,
            row=args.data_row, eos_id=args.eos_id, synthetic_doc_len=64)
    else:
        lengths = case_lengths(case, args.seq_len)
        row = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        docs, info = row.repeat(args.batch, 1), {"kind": "synthetic", "lengths": lengths}
    info["document_boundaries"] = int((docs[:, 1:] != docs[:, :-1]).sum())
    return docs, info


def clear_plans():
    packed.clear_doc_plan_cache()
    clear_bucket_plan_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES[:-1]))
    parser.add_argument("--layouts", choices=("packed_qkv", "cross_cache"), nargs="+",
                        default=["packed_qkv", "cross_cache"])
    parser.add_argument("--dtypes", choices=tuple(TOLERANCES), nargs="+", default=["bf16"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-row", type=int, default=0)
    parser.add_argument("--eos-id", type=int, default=1)
    parser.add_argument("--max-padding-ratio", type=float, default=1.25)
    parser.add_argument("--max-bucket-size", type=int, default=16)
    parser.add_argument("--max-bucket-score-elements", type=int, default=4 << 20)
    parser.add_argument("--long-document-fraction", type=float, default=0.75)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-merge-heads", action="store_true",
                        help="omit output layout conversion; default includes the o_proj input layout")
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--gpu-wait-max-seconds", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 64 <= args.seq_len <= 4096 or args.window < args.seq_len:
        parser.error("this dense-packed benchmark needs seq-len in [64,4096] and window >= seq-len")
    if min(args.batch, args.repeats) < 1 or args.warmup < 0 or args.data_row < 0:
        parser.error("invalid batch, repeats, warmup or row")
    try:
        config = BucketConfig(args.max_padding_ratio, args.max_bucket_size,
                              args.max_bucket_score_elements, args.long_document_fraction)
    except ValueError as error:
        parser.error(str(error))
    cases = list(dict.fromkeys(args.cases))
    if args.data is not None and "real" not in cases:
        cases.append("real")
    if "real" in cases and (args.data is None or not args.data.is_file()):
        parser.error("real fixture requires an existing --data packed bin")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.wait_gpu_idle:
        wait_for_gpu_idle(max_wait_seconds=args.gpu_wait_max_seconds,
                          on_event=lambda event: _emit(event, args.output))
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP runtime is required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    _emit({"event": "environment", "torch": torch.__version__, "hip": torch.version.hip,
           "gpu": torch.cuda.get_device_name(), "shape": [args.batch, 16, args.seq_len, 128],
           "kv_heads": 2, "window": args.window, "config": asdict(config),
           "tolerances": TOLERANCES, "merge_heads_included": not args.skip_merge_heads,
           "scope": "QKV cast, padding, bucket copies, GQA repetition, SDPA, output assembly and optional merge_heads; projections excluded",
           "cold_scope": "clear document/bucket caches per invocation; includes GPU-to-CPU plan copy, Python scan and bucket planning",
           "fixture_creation_in_timer": False, "wait_does_not_reserve_gpu": True,
           **pid_snapshot()}, args.output)
    counts = {"passed": 0, "failed": 0, "errors": 0, "bucketed_exercised": 0}
    for case in cases:
        docs_cpu, provenance = fixture_documents(case, args)
        docs = docs_cpu.to("cuda")
        attention_docs = docs if provenance["document_boundaries"] else None
        for dtype_name in args.dtypes:
            dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
            for layout in args.layouts:
                values = input_values(args.seq_len, args.batch, dtype, layout, args.seed)
                dy = torch.randn(values[0].shape, device="cuda", dtype=dtype,
                                 generator=torch.Generator(device="cuda").manual_seed(args.seed + 10000))
                dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
                if not args.skip_merge_heads:
                    dy = native.merge_heads(dy)

                def finish(result):
                    return result if args.skip_merge_heads else native.merge_heads(result)

                base = {"event": "candidate", "case": case, "dtype": dtype_name, "layout": layout,
                        "documents": provenance, "input_strides": [list(v.stride()) for v in values],
                        "dy_stride": list(dy.stride()), "gpu_processes_before": pid_snapshot()}
                try:
                    reference, reference_grads = snapshot(
                        lambda q, k, v: finish(dense_oracle(q, k, v, args.window, docs)), values, dy)
                except Exception as error:
                    _emit({**base, "candidate": "oracle", "status": "error", "error": str(error),
                           "oom": _is_oom(error)}, args.output)
                    counts["errors"] += 1
                    continue
                for name in ("packed", "bucketed"):
                    clear_plans()
                    stats = new_stats()
                    function = packed.packed_window_sdpa if name == "packed" else bucketed_window_sdpa

                    def operation(q, k, v):
                        kwargs = {"stats": stats}
                        if name == "bucketed":
                            kwargs["config"] = config
                        return finish(function(q, k, v, args.window, attention_docs, **kwargs))

                    row = {**base, "candidate": name}
                    try:
                        actual, grads = snapshot(operation, values, dy)
                        row["output"] = _errors(actual, reference, TOLERANCES[dtype_name])
                        row["gradients"] = {label: _errors(a, b, TOLERANCES[dtype_name])
                                            for label, a, b in zip(("dq", "dk", "dv"), grads, reference_grads)}
                        passed = row["output"]["pass"] and all(g["pass"] for g in row["gradients"].values())
                        row["status"] = "pass" if passed else "numerical_failure"
                        row["correctness_path_counts"] = json.loads(json.dumps(stats))
                        if name == "bucketed" and stats["bucketed_calls"]:
                            counts["bucketed_exercised"] += 1
                        del actual, grads
                        if passed:
                            for plan_mode in ("warm", "cold"):
                                def forward():
                                    if plan_mode == "cold":
                                        clear_plans()
                                    with torch.no_grad():
                                        return operation(*values)

                                def forward_backward():
                                    if plan_mode == "cold":
                                        clear_plans()
                                    qkv = tuple(v.detach().requires_grad_() for v in values)
                                    return torch.autograd.grad(operation(*qkv), qkv, dy)

                                row[plan_mode + "_forward"] = _measure(forward, args.warmup, args.repeats)
                                gc.collect()
                                torch.cuda.empty_cache()
                                row[plan_mode + "_forward_backward"] = _measure(forward_backward, args.warmup, args.repeats)
                        counts["passed" if passed else "failed"] += 1
                    except Exception as error:
                        row.update(status="error", error=str(error), oom=_is_oom(error))
                        counts["errors"] += 1
                    row["gpu_processes_after"] = pid_snapshot()
                    _emit(row, args.output)
                    gc.collect()
                    torch.cuda.empty_cache()
                del values, dy, reference, reference_grads
        del docs
    _emit({"event": "summary", **counts}, args.output)
    return 1 if counts["failed"] or counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
