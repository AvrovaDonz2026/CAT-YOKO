#!/usr/bin/env python3
"""Isolated packed-window correctness and GPU timing; never patches training.

The native packed FP32 MATH path is timed beside the opt-in tiled candidate.
An independent explicit dense FP32 oracle checks output and every input gradient.
Use --wait-gpu-idle to observe KFD processes without suspending any process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

import cat_yoko.attention as native
from operators.rocm.attention_bench import _emit, _errors, _is_oom, _measure
from operators.rocm.gpu_wait import other_gpu_pids, wait_for_gpu_idle
from operators.rocm.packed_attention import packed_window_sdpa, new_stats


TOLERANCES = {
    "bf16": {"atol": 0.02, "rtol": 0.02, "relative_l2": 0.01},
    "fp32": {"atol": 3e-5, "rtol": 3e-5, "relative_l2": 3e-5},
}


def load_documents(path: Path | None, *, seq: int, batch: int, row: int, eos_id: int | None,
                   synthetic_doc_len: int) -> tuple[torch.Tensor, dict]:
    if path is None:
        docs = torch.arange(seq).div(synthetic_doc_len, rounding_mode="floor").repeat(batch, 1)
        for b in range(1, batch):
            docs[b] = (torch.arange(seq) + b * synthetic_doc_len // 2).div(synthetic_doc_len, rounding_mode="floor")
        return docs, {"kind": "synthetic_doc_ids", "document_length": synthetic_doc_len}
    meta_path = path.with_name(path.name + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    eos = int(eos_id) if eos_id is not None else meta.get("eos_id")
    if eos is None:
        raise ValueError("--eos-id or packed sidecar eos_id is required with --data")
    pack_width = int(meta.get("seq_len", seq))
    if seq > pack_width:
        raise ValueError("requested sequence exceeds the source packed row width")
    records = []
    with path.open("rb") as stream:
        for b in range(batch):
            stream.seek((row + b) * pack_width * 4)
            chunk = stream.read(seq * 4)
            if len(chunk) != seq * 4:
                raise ValueError(f"packed data has no complete row {row + b}")
            records.append(struct.unpack("<" + "i" * seq, chunk))
    ids = torch.tensor(records, dtype=torch.long)
    hits = (ids == int(eos)).long()
    docs = hits.cumsum(-1) - hits
    return docs, {"kind": "packed_bin", "path": str(path), "row": row, "eos_id": int(eos),
                  "source_pack_width": pack_width,
                  "sidecar_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest() if meta_path.exists() else None}


def input_values(seq, batch, dtype, layout, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = [torch.randn(batch, heads, seq, 128, device="cuda", dtype=dtype, generator=generator)
              for heads in (16, 2, 2)]
    if layout == "bshd":
        values = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in values]
    elif layout == "packed_qkv":
        packed = torch.cat([value.transpose(1, 2).reshape(batch, seq, -1) for value in values], -1)
        values = [part.reshape(batch, seq, heads, 128).transpose(1, 2)
                  for part, heads in zip(packed.split((2048, 256, 256), -1), (16, 2, 2))]
    elif layout == "cross_cache":
        q, k, v = values
        q, k = (value.transpose(1, 2).contiguous().transpose(1, 2) for value in (q, k))
        cache = torch.cat((values[1].transpose(1, 2).reshape(batch, seq, 256),
                           v.transpose(1, 2).reshape(batch, seq, 256)), -1)
        v = cache[..., 256:].reshape(batch, seq, 2, 128).transpose(1, 2)
        values = [q, k, v]
    return tuple(value.detach() for value in values)


def dense_oracle(q, k, v, window, docs):
    with torch.autocast("cuda", enabled=False):
        repeats = q.size(1) // k.size(1)
        qf, kf, vf = q.float(), k.float().repeat_interleave(repeats, 1), v.float().repeat_interleave(repeats, 1)
        scores = qf @ kf.transpose(-1, -2) / math.sqrt(q.size(-1))
        qi = torch.arange(q.size(-2), device=q.device)[:, None]
        ki = torch.arange(k.size(-2), device=q.device)[None, :]
        keep = ((ki <= qi) & (qi - ki < window))[None] & (docs[:, :, None] == docs[:, None, :])
        probabilities = scores.masked_fill(~keep[:, None], torch.finfo(torch.float32).min).softmax(-1)
        return (probabilities @ vf).to(q.dtype)


def snapshot(operation, values, dy):
    qkv = tuple(value.detach().requires_grad_() for value in values)
    result = operation(*qkv)
    gradients = torch.autograd.grad(result, qkv, dy)
    # Move captures off GPU before creating another dense S² graph.
    output = result.detach().cpu()
    grads = tuple(gradient.detach().cpu() for gradient in gradients)
    del result, gradients, qkv
    gc.collect()
    torch.cuda.empty_cache()
    return output, grads


def pid_snapshot():
    try:
        return {"other_gpu_pids": other_gpu_pids()}
    except RuntimeError as error:
        return {"other_gpu_pids": None, "pid_observation_error": str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096])
    parser.add_argument("--windows", nargs="+", type=int, default=[8192, 32])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--dtypes", nargs="+", choices=tuple(TOLERANCES), default=["bf16"])
    parser.add_argument("--layouts", nargs="+", choices=("bhsd", "bshd", "packed_qkv", "cross_cache"),
                        default=["packed_qkv"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--data-row", type=int, default=0)
    parser.add_argument("--eos-id", type=int)
    parser.add_argument("--synthetic-doc-len", type=int, default=377)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--gpu-wait-max-seconds", type=float)
    args = parser.parse_args(argv)
    if (min(args.seq_lens) < 2 or max(args.seq_lens) > 4096 or min(args.windows) < 1
            or min(args.batch, args.tile_size, args.synthetic_doc_len, args.repeats) < 1
            or args.warmup < 0 or args.data_row < 0):
        parser.error("invalid shape, tile, row, warmup or repeat settings")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.wait_gpu_idle:
        wait_for_gpu_idle(max_wait_seconds=args.gpu_wait_max_seconds,
                          on_event=lambda event: _emit(event, args.output))
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP GPU is required for this benchmark")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    _emit({"event": "environment", "torch": torch.__version__, "hip": torch.version.hip,
           "gpu": torch.cuda.get_device_name(), "shape": [args.batch, 16, "seq", 128], "kv_heads": 2,
           "tile_size": args.tile_size, "tolerances": TOLERANCES,
           "reference": "independent explicit dense FP32 masked attention",
           "baseline": "unmodified native _window_sdpa with original input/mask dtype (ROCm uses FP32 MATH)",
           "candidate": "opt-in FP32 document fragments / short-window tiles",
           "timing_scope": "casts, layout copies, masks and GQA repetition included; synthetic QKV values; same doc plan cached across repeats",
           "wait_does_not_reserve_gpu": True, **pid_snapshot()}, args.output)
    counts = {"passed": 0, "failed": 0, "errors": 0}
    for seq in sorted(set(args.seq_lens)):
        docs_cpu, provenance = load_documents(args.data, seq=seq, batch=args.batch, row=args.data_row,
                        eos_id=args.eos_id, synthetic_doc_len=args.synthetic_doc_len)
        docs = docs_cpu.to("cuda")
        boundaries = int((docs_cpu[..., 1:] != docs_cpu[..., :-1]).sum())
        # Production collapses an entirely single-document batch to None.
        # Preserve that gate instead of timing an unnecessary document mask.
        attention_docs = docs if boundaries else None
        for dtype_name in args.dtypes:
            dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
            for layout in args.layouts:
                values = input_values(seq, args.batch, dtype, layout, args.seed + seq)
                dy = torch.randn(values[0].shape, device="cuda", dtype=dtype,
                                 generator=torch.Generator(device="cuda").manual_seed(args.seed + seq + 10000))
                dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
                for window in args.windows:
                    base = {"event": "candidate", "seq": seq, "window": window, "dtype": dtype_name,
                            "layout": layout, "input_strides": [list(value.stride()) for value in values],
                            "dy_stride": list(dy.stride()), "document_boundaries": boundaries,
                            "documents": provenance, "gpu_processes_before": pid_snapshot()}
                    oracle = lambda q, k, v: dense_oracle(q, k, v, window, docs)
                    try:
                        reference, reference_grads = snapshot(oracle, values, dy)
                    except Exception as error:
                        _emit({**base, "candidate": "independent_oracle", "status": "error",
                               "oom": _is_oom(error), "error": str(error)}, args.output)
                        counts["errors"] += 1
                        continue
                    for name in ("native", "packed_attention"):
                        stats = new_stats()
                        operation = ((lambda q, k, v: native._window_sdpa(q, k, v, window, attention_docs)) if name == "native"
                                     else (lambda q, k, v: packed_window_sdpa(q, k, v, window, attention_docs,
                                               tile_size=args.tile_size, stats=stats)))
                        row = {**base, "candidate": name}
                        try:
                            actual, grads = snapshot(operation, values, dy)
                            row["output"] = _errors(actual, reference, TOLERANCES[dtype_name])
                            row["gradients"] = {label: _errors(a, b, TOLERANCES[dtype_name])
                                                for label, a, b in zip(("dq", "dk", "dv"), grads, reference_grads)}
                            passed = row["output"]["pass"] and all(g["pass"] for g in row["gradients"].values())
                            row["status"] = "pass" if passed else "numerical_failure"
                            row["correctness_path_counts"] = dict(stats)
                            del actual, grads
                            if passed:
                                def forward():
                                    with torch.no_grad():
                                        return operation(*values)

                                def forward_backward():
                                    qkv = tuple(value.detach().requires_grad_() for value in values)
                                    result = operation(*qkv)
                                    return torch.autograd.grad(result, qkv, dy)

                                row["forward"] = _measure(forward, args.warmup, args.repeats)
                                gc.collect()
                                torch.cuda.empty_cache()
                                row["forward_backward"] = _measure(forward_backward, args.warmup, args.repeats)
                            counts["passed" if passed else "failed"] += 1
                        except Exception as error:
                            row.update(status="error", error=str(error), oom=_is_oom(error))
                            counts["errors"] += 1
                        row["gpu_processes_after"] = pid_snapshot()
                        _emit(row, args.output)
                        gc.collect()
                        torch.cuda.empty_cache()
                    del reference, reference_grads
                del values, dy
        del docs
    _emit({"event": "summary", **counts}, args.output)
    return 1 if counts["failed"] or counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
