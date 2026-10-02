#!/usr/bin/env python3
"""Standalone ROCm causal GQA correctness and timing ledger; imports only torch.

Default shape is the real 12B attention: batch=1, Q heads=16, KV heads=2,
head_dim=128. Production attention and any running training are untouched.
Use a fresh process so AOTriton's experimental architecture flag is effective:

    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python operators/rocm/attention_bench.py
    python operators/rocm/attention_bench.py --include-4096 --output attention.jsonl

Timing is meaningful when other GPU work is idle. Both wall and GPU event
times include KV repetition, casts, gradient layout fixes and chunking.
Production's FP32 MATH path is the baseline. An independent checkpointed
query-chunk oracle validates causal offsets without retaining an S² graph.
The same quantized input values and upstream gradient values are used for every
candidate; contiguous and output-stride gradients are checked separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
# Fixed before observing candidate results. L2 gates catch widespread errors
# even when individual near-zero elements fall under the absolute allowance.
TOLERANCES = {
    "bf16": {"atol": 0.02, "rtol": 0.02, "relative_l2": 0.01},
    "fp16": {"atol": 0.004, "rtol": 0.01, "relative_l2": 0.003},
}
CANDIDATES = (
    "math_fp32", "flash_gqa", "flash_repeat_kv", "flash_gqa_contiguous_grad",
    "math_chunked",
)


def _emit(record: dict, output: Path | None) -> None:
    line = json.dumps(record, separators=(",", ":"), allow_nan=False)
    print(line, flush=True)
    if output is not None:
        with output.open("a") as stream:
            stream.write(line + "\n")


def _inputs(seq: int, dtype: torch.dtype, layout: str, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Logical values stay identical across layouts, so stride comparisons are
    # not confounded by drawing the RNG sequence in a different dimension order.
    values = [torch.randn(1, heads, seq, 128, device="cuda", dtype=dtype, generator=generator)
              for heads in (16, 2, 2)]
    if layout == "packed_qkv":
        # Fused Q/K/V linear output, before QK norm / RoPE materialization.
        packed = torch.cat([value.transpose(1, 2).reshape(1, seq, heads * 128)
                            for value, heads in zip(values, (16, 2, 2))], dim=-1)
        splits = packed.split((16 * 128, 2 * 128, 2 * 128), dim=-1)
        return tuple(part.view(1, seq, heads, 128).transpose(1, 2).detach().requires_grad_()
                     for part, heads in zip(splits, (16, 2, 2)))
    if layout == "bshd":
        values = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in values]
    return tuple(value.detach().requires_grad_() for value in values)


def _math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                             is_causal=True, enable_gqa=True).to(q.dtype)


def _chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int,
             *, manual: bool) -> torch.Tensor:
    """Exact full-context causal attention, with global query offsets.

    FP32 conversion happens once so KV gradients from chunks sum in FP32.
    Checkpointing each chunk avoids retaining all S² softmax matrices in
    backward. Every query still sees all earlier keys, including other chunks.
    """
    qf, kf, vf = q.float(), k.float(), v.float()
    n_rep = q.size(1) // k.size(1)
    outputs = []
    for start in range(0, q.size(-2), chunk_size):
        end = min(start + chunk_size, q.size(-2))

        def _one(qq, kk, vv, offset=start):
            with torch.autocast("cuda", enabled=False):
                qi = torch.arange(offset, offset + qq.size(-2), device=qq.device)
                ki = torch.arange(kk.size(-2), device=kk.device)
                keep = ki.unsqueeze(0) <= qi.unsqueeze(1)
                if manual:
                    kr = kk.repeat_interleave(n_rep, dim=1)
                    vr = vv.repeat_interleave(n_rep, dim=1)
                    scores = (qq / math.sqrt(qq.size(-1))) @ kr.transpose(-2, -1)
                    probabilities = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1)
                    return probabilities @ vr
                with sdpa_kernel(SDPBackend.MATH):
                    return F.scaled_dot_product_attention(qq, kk, vv, attn_mask=keep,
                                                         enable_gqa=True)

        pieces = (qf[..., start:end, :], kf[..., :end, :], vf[..., :end, :])
        if torch.is_grad_enabled() and any(value.requires_grad for value in pieces):
            part = checkpoint(_one, *pieces, use_reentrant=False, preserve_rng_state=False)
        else:
            part = _one(*pieces)
        outputs.append(part)
    return torch.cat(outputs, dim=-2).to(q.dtype)


def _candidate(name: str, chunk_size: int) -> Callable:
    if name == "math_fp32":
        return _math
    if name == "math_chunked":
        return lambda q, k, v: _chunked(q, k, v, chunk_size, manual=False)

    def _flash(q, k, v):
        if name == "flash_repeat_kv":
            repeats = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   enable_gqa=name != "flash_repeat_kv")
        if name == "flash_gqa_contiguous_grad" and output.requires_grad:
            output.register_hook(lambda dy: dy.contiguous())
        return output

    return _flash


def _errors(actual: torch.Tensor, reference: torch.Tensor, tolerance: dict) -> dict:
    af, rf = actual.detach().float(), reference.detach().float()
    finite = bool(torch.isfinite(af).all() and torch.isfinite(rf).all())
    if not finite:
        return {"finite": False, "max_abs": None, "relative_l2": None, "pass": False}
    difference = af - rf
    max_abs = float(difference.abs().max())
    relative_l2 = float(difference.norm() / rf.norm().clamp_min(1e-20))
    elementwise = bool((difference.abs() <= tolerance["atol"] + tolerance["rtol"] * rf.abs()).all())
    return {"finite": True, "max_abs": max_abs, "relative_l2": relative_l2,
            "elementwise_pass": elementwise,
            "pass": elementwise and relative_l2 <= tolerance["relative_l2"]}


def _layout_dy(values: torch.Tensor, output: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "contiguous":
        return values.contiguous()
    # Preserve values, dtype and exact output strides, unlike a second RNG draw.
    dy = torch.empty_strided(output.shape, output.stride(), device=output.device, dtype=values.dtype)
    return dy.copy_(values)


def _measure(operation: Callable, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        result = operation()
        del result
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        start.record()
        result = operation()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - tick) * 1000.0)
        gpu.append(start.elapsed_time(end))
        del result
    return {"wall_ms": statistics.median(wall), "gpu_ms": statistics.median(gpu),
            "gpu_ms_min": min(gpu), "repeats": repeats,
            "baseline_mib": baseline / 2**20,
            "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
            "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20}


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--include-4096", action="store_true")
    parser.add_argument("--dtypes", nargs="+", choices=tuple(DTYPES), default=list(DTYPES))
    parser.add_argument("--layouts", nargs="+", choices=("bhsd", "bshd", "packed_qkv"), default=["bhsd", "bshd"])
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--dy-layouts", nargs="+", choices=("contiguous", "output"), default=["contiguous", "output"])
    parser.add_argument("--reference-chunk-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--benchmark-failed", action="store_true",
                        help="Time numerically failing candidates too; they remain ineligible.")
    args = parser.parse_args(argv)
    if min(args.seq_lens) < 2 or max(args.seq_lens) > 4096:
        parser.error("seq-lens must be in [2, 4096]")
    if min(args.reference_chunk_size, args.chunk_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("chunk sizes/repeats must be positive and warmup nonnegative")
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP GPU is required")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reference FP32 GEMMs must not use reduced precision acceleration.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    seq_lens = sorted(set(args.seq_lens + ([4096] if args.include_4096 else [])))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    _emit({"event": "environment", "torch": torch.__version__, "hip": torch.version.hip,
           "gpu": torch.cuda.get_device_name(), "shape": {"batch": 1, "q_heads": 16, "kv_heads": 2, "head_dim": 128},
           "free_mib_at_start": free_bytes / 2**20, "total_mib": total_bytes / 2**20,
           "experimental_aotriton": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
           "fa_library": str(torch.backends.cuda.preferred_rocm_fa_library()) if torch.version.hip else None,
           "reference": "production FP32 MATH SDPA with output dtype cast",
           "independent_oracle": "checkpointed explicit FP32 causal attention",
           "reference_chunk_size": args.reference_chunk_size, "candidate_chunk_size": args.chunk_size,
           "tolerances": TOLERANCES, "warmup": args.warmup, "repeats": args.repeats,
           "other_gpu_workloads_can_affect_timing": True}, args.output)
    counts = {"rows": 0, "eligible": 0, "failed": 0, "unsupported": 0, "errors": 0}
    for seq in seq_lens:
        for dtype_name in args.dtypes:
            for layout in args.layouts:
                inputs = _inputs(seq, DTYPES[dtype_name], layout, args.seed + seq)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + seq + 100_000)
                upstream = torch.randn(inputs[0].shape, device="cuda", dtype=inputs[0].dtype, generator=generator)
                reference = _math(*inputs)
                reference_grads = torch.autograd.grad(reference, inputs, upstream)
                reference = reference.detach()
                reference_grads = tuple(value.detach() for value in reference_grads)
                oracle = _chunked(*inputs, args.reference_chunk_size, manual=True)
                oracle_grads = torch.autograd.grad(oracle, inputs, upstream)
                oracle_checks = {"output": _errors(oracle, reference, TOLERANCES[dtype_name])}
                oracle_checks.update({key: _errors(value, ref, TOLERANCES[dtype_name])
                                      for key, value, ref in zip(("dq", "dk", "dv"), oracle_grads, reference_grads)})
                oracle_pass = all(value["pass"] for value in oracle_checks.values())
                _emit({"event": "reference_validation", "seq": seq, "dtype": dtype_name,
                       "layout": layout, "pass": oracle_pass, "checks": oracle_checks}, args.output)
                del oracle, oracle_grads
                if not oracle_pass:
                    raise RuntimeError("Independent query-chunk reference disagrees with FP32 MATH baseline")
                for name in args.candidates:
                    counts["rows"] += 1
                    row = {"event": "candidate", "seq": seq, "dtype": dtype_name, "layout": layout,
                           "candidate": name, "input_strides": [list(value.stride()) for value in inputs]}
                    operation = _candidate(name, args.chunk_size)
                    result, saved_dys, actual_grads = None, {}, None
                    try:
                        result = operation(*inputs)
                        row["output_stride"] = list(result.stride())
                        row["output"] = _errors(result, reference, TOLERANCES[dtype_name])
                        gradients = {}
                        saved_dys = {}
                        for index, dy_layout in enumerate(args.dy_layouts):
                            dy = _layout_dy(upstream, result, dy_layout)
                            saved_dys[dy_layout] = dy
                            actual_grads = torch.autograd.grad(result, inputs, dy,
                                                              retain_graph=index < len(args.dy_layouts) - 1)
                            metrics = {key: _errors(value, ref, TOLERANCES[dtype_name])
                                       for key, value, ref in zip(("dq", "dk", "dv"), actual_grads, reference_grads)}
                            metrics["dy_stride"] = list(dy.stride())
                            metrics["pass"] = all(metrics[key]["pass"] for key in ("dq", "dk", "dv"))
                            gradients[dy_layout] = metrics
                            del actual_grads
                        row["gradients"] = gradients
                        row["eligible_dy_layouts"] = [key for key, metrics in gradients.items()
                                                     if row["output"]["pass"] and metrics["pass"]]
                        row["eligible"] = len(row["eligible_dy_layouts"]) == len(args.dy_layouts)
                        row["status"] = "pass" if row["eligible"] else "numerical_failure"
                        counts["eligible" if row["eligible"] else "failed"] += 1
                        del result

                        def _forward():
                            with torch.no_grad():
                                return operation(*inputs)

                        if row["eligible_dy_layouts"] or args.benchmark_failed:
                            row["forward"] = _measure(_forward, args.warmup, args.repeats)
                            row["forward_backward"] = {}
                            for dy_layout in args.dy_layouts:
                                if dy_layout not in row["eligible_dy_layouts"] and not args.benchmark_failed:
                                    continue

                                def _forward_backward(dy=saved_dys[dy_layout]):
                                    output = operation(*inputs)
                                    return torch.autograd.grad(output, inputs, dy)

                                row["forward_backward"][dy_layout] = _measure(_forward_backward, args.warmup, args.repeats)
                        del saved_dys
                    except (RuntimeError, NotImplementedError) as exc:
                        row["status"] = "oom" if _is_oom(exc) else "execution_error"
                        if "no available kernel" in str(exc).lower() or "no viable backend" in str(exc).lower():
                            row["status"] = "unsupported"
                            counts["unsupported"] += 1
                        else:
                            counts["errors"] += 1
                        row["eligible"] = False
                        row["error"] = str(exc)[:1000]
                        # Illegal access can poison the HIP context. Do not keep
                        # running and misclassify all later candidates as failing.
                        if any(phrase in str(exc).lower() for phrase in ("illegal memory", "device-side assert")):
                            _emit(row, args.output)
                            raise
                    finally:
                        result, saved_dys, actual_grads = None, {}, None
                    _emit(row, args.output)
                    gc.collect()
                    torch.cuda.empty_cache()
                del inputs, reference, reference_grads, upstream
                gc.collect()
                torch.cuda.empty_cache()
    _emit({"event": "summary", **counts}, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
