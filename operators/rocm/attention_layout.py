#!/usr/bin/env python3
"""Isolated ROCm attention layout, Efficient and hybrid-backward experiments.

No production imports or edits. Uses the FP32 SDPA baseline and fixed error
gates from attention_bench.py; the only third-party dependency is torch.

    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python operators/rocm/attention_layout.py
    python operators/rocm/attention_layout.py --seq-lens 4096 --layouts cross_cache --repeats 3

The hybrid calls Flash only for forward, then recomputes FP32 math in backward.
Its gradient is the FP32 reference derivative, not the exact derivative of
rounded Flash output. It is an experimental candidate, not a production
recommendation. Query chunks keep full causal context with global offsets.
Copying inputs, repeating KV and recomputation are included in every timing.
Run with other GPU computation paused; resident allocations are reported.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention_bench import DTYPES, TOLERANCES, _chunked, _emit, _errors, _inputs, _is_oom, _layout_dy, _math, _measure


CANDIDATES = (
    "math_fp32", "math_grouped_q", "efficient_gqa", "efficient_repeat_kv",
    "efficient_repeat_kv_contiguous", "hybrid_flash_math", "hybrid_flash_chunked",
    "hybrid_flash_math_contiguous",
)
LAYOUTS = ("bhsd", "bshd", "packed_qkv", "cross_cache")


def _layout_inputs(seq: int, dtype: torch.dtype, layout: str, seed: int) -> tuple[torch.Tensor, ...]:
    if layout != "cross_cache":
        return _inputs(seq, dtype, layout, seed)
    q, k, v = _inputs(seq, dtype, "bhsd", seed)
    # Cross attention cache is projected as [B,S,2*KV*D], split into K and V.
    # Q/K RMSNorm and RoPE materialize dense per-tensor storage. V is untouched
    # by RoPE and retains the fused-cache row stride. Preserve identical values
    # while representing that mixed layout, without importing production code.
    with torch.no_grad():
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        cache = torch.cat((k.transpose(1, 2).reshape(1, seq, 256),
                           v.transpose(1, 2).reshape(1, seq, 256)), dim=-1)
        ks, vs = cache.split((256, 256), dim=-1)
        k = ks.view(1, seq, 2, 128).contiguous().transpose(1, 2)
        v = vs.view(1, seq, 2, 128).transpose(1, 2)
    return tuple(value.detach().requires_grad_() for value in (q, k, v))


def _native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, backend: SDPBackend,
            repeat_kv: bool, copy_inputs: bool = False) -> torch.Tensor:
    if copy_inputs:
        q, k, v = (value.contiguous() for value in (q, k, v))
    if repeat_kv:
        repeats = q.size(1) // k.size(1)
        k, v = (value.repeat_interleave(repeats, dim=1) for value in (k, v))
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             enable_gqa=not repeat_kv)


class _FlashForwardMathBackward(torch.autograd.Function):
    """Bypass the experimentally incorrect ROCm Flash dQ kernel entirely."""

    @staticmethod
    def forward(ctx, q, k, v, chunk_size: int, copy_inputs: bool):
        ctx.save_for_backward(q, k, v)
        ctx.chunk_size = chunk_size
        # autograd.Function.forward already runs without recording a graph.
        with torch.no_grad():
            return _native(q, k, v, backend=SDPBackend.FLASH_ATTENTION,
                           repeat_kv=False, copy_inputs=copy_inputs)

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved_tensors
        needed = ctx.needs_input_grad[:3]
        if not any(needed):
            return None, None, None, None, None
        values = tuple(value.detach().requires_grad_(need) for value, need in zip(saved, needed))
        with torch.enable_grad(), torch.autocast("cuda", enabled=False):
            if ctx.chunk_size:
                output = _chunked(*values, ctx.chunk_size, manual=False)
            else:
                output = _math(*values)
            active = tuple(value for value, need in zip(values, needed) if need)
            gradients = iter(torch.autograd.grad(output, active, dy, create_graph=False))
        result = tuple(next(gradients) if need else None for need in needed)
        return (*result, None, None)


def flash_forward_math_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                *, chunk_size: int = 0, copy_inputs: bool = False) -> torch.Tensor:
    return _FlashForwardMathBackward.apply(q, k, v, chunk_size, copy_inputs)


def grouped_q_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """One FP32 bmm per KV head, merging its query heads into rows.

    This avoids explicit KV head repetition. The full attention score matrix
    still has B*Q_heads*S*S elements; this candidate reduces KV traffic only.
    """
    batch, heads, seq, hd = q.shape
    kv_heads, k_len = k.size(1), k.size(-2)
    repeats = heads // kv_heads
    with torch.autocast("cuda", enabled=False):
        qf, kf, vf = (value.float() for value in (q, k, v))
        # Match MATH SDPA's split scaling between Q and K for stability.
        root_scale = hd ** -0.25
        qm = (qf * root_scale).reshape(batch * kv_heads, repeats * seq, hd)
        km = (kf * root_scale).reshape(batch * kv_heads, k_len, hd)
        vm = vf.reshape(batch * kv_heads, k_len, hd)
        scores = torch.bmm(qm, km.transpose(1, 2)).view(batch, kv_heads, repeats, seq, k_len)
        keep = torch.arange(k_len, device=q.device).unsqueeze(0) <= torch.arange(seq, device=q.device).unsqueeze(1)
        scores = scores.masked_fill(~keep, float("-inf"))
        probabilities = scores.softmax(dim=-1).view(batch * kv_heads, repeats * seq, k_len)
        output = torch.bmm(probabilities, vm).view(batch, heads, seq, hd)
    return output.to(q.dtype)


def _operation(name: str, chunk_size: int) -> Callable:
    if name == "math_fp32":
        return _math
    if name == "math_grouped_q":
        return grouped_q_math
    if name.startswith("efficient"):
        return lambda q, k, v: _native(q, k, v, backend=SDPBackend.EFFICIENT_ATTENTION,
                                      repeat_kv=name != "efficient_gqa",
                                      copy_inputs=name.endswith("_contiguous"))
    return lambda q, k, v: flash_forward_math_backward(
        q, k, v, chunk_size=chunk_size if name == "hybrid_flash_chunked" else 0,
        copy_inputs=name.endswith("_contiguous"),
    )


def _diagnostics(name: str, inputs: tuple[torch.Tensor, ...]) -> dict:
    if not name.startswith("efficient"):
        return {}
    q, k, v = inputs
    if name.endswith("_contiguous"):
        q, k, v = (value.contiguous() for value in inputs)
    if name != "efficient_gqa":
        repeat = q.size(1) // k.size(1)
        k, v = (value.repeat_interleave(repeat, dim=1) for value in (k, v))
    params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, True, name == "efficient_gqa")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        available = torch.backends.cuda.can_use_efficient_attention(params, debug=True)
    return {"can_use_efficient_attention": bool(available),
            "kernel_input_strides": [list(value.stride()) for value in (q, k, v)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--include-4096", action="store_true")
    parser.add_argument("--dtypes", nargs="+", choices=tuple(DTYPES), default=["bf16"])
    parser.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=["bhsd", "bshd", "cross_cache"])
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--dy-layouts", nargs="+", choices=("contiguous", "output"), default=["contiguous", "output"])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.seq_lens) < 2 or max(args.seq_lens) > 4096:
        parser.error("seq-lens must be in [2,4096]")
    if min(args.chunk_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("chunk-size/repeats must be positive, warmup nonnegative")
    if not torch.cuda.is_available():
        parser.error("a CUDA/HIP GPU is required")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    free, total = torch.cuda.mem_get_info()
    _emit({"event": "environment", "experiment": "attention_layout", "torch": torch.__version__,
           "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(),
           "experimental_aotriton": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", ""),
           "free_mib": free / 2**20, "total_mib": total / 2**20,
           "shape": [1, 16, "seq", 128], "kv_heads": 2, "chunk_size": args.chunk_size,
           "tolerances": TOLERANCES, "hybrid_gradient": "FP32 MATH surrogate derivative",
           "copy_and_recompute_included_in_timings": True}, args.output)
    counts = {"rows": 0, "eligible": 0, "numerical_failure": 0, "unsupported": 0, "errors": 0}
    for seq in sorted(set(args.seq_lens + ([4096] if args.include_4096 else []))):
        for dtype_name in args.dtypes:
            for layout in args.layouts:
                inputs = _layout_inputs(seq, DTYPES[dtype_name], layout, args.seed + seq)
                generator = torch.Generator(device="cuda").manual_seed(args.seed + seq + 100_000)
                upstream = torch.randn(inputs[0].shape, device="cuda", dtype=inputs[0].dtype, generator=generator)
                reference = _math(*inputs)
                reference_grads = tuple(value.detach() for value in torch.autograd.grad(reference, inputs, upstream))
                reference = reference.detach()
                for name in args.candidates:
                    counts["rows"] += 1
                    row = {"event": "candidate", "experiment": "attention_layout", "seq": seq,
                           "dtype": dtype_name, "layout": layout, "candidate": name,
                           "input_strides": [list(value.stride()) for value in inputs]}
                    operation = _operation(name, args.chunk_size)
                    result, actual_grads, dys = None, None, {}
                    try:
                        with torch.no_grad():
                            row.update(_diagnostics(name, inputs))
                        result = operation(*inputs)
                        row["output_stride"] = list(result.stride())
                        row["output"] = _errors(result, reference, TOLERANCES[dtype_name])
                        row["gradients"] = {}
                        for index, dy_layout in enumerate(args.dy_layouts):
                            dy = _layout_dy(upstream, result, dy_layout)
                            dys[dy_layout] = dy
                            actual_grads = torch.autograd.grad(result, inputs, dy,
                                                              retain_graph=index < len(args.dy_layouts) - 1)
                            metrics = {key: _errors(value, ref, TOLERANCES[dtype_name])
                                       for key, value, ref in zip(("dq", "dk", "dv"), actual_grads, reference_grads)}
                            metrics["dy_stride"] = list(dy.stride())
                            metrics["pass"] = all(metrics[key]["pass"] for key in ("dq", "dk", "dv"))
                            row["gradients"][dy_layout] = metrics
                            actual_grads = None
                        eligible_layouts = [key for key, value in row["gradients"].items()
                                            if row["output"]["pass"] and value["pass"]]
                        row["eligible_dy_layouts"] = eligible_layouts
                        row["eligible"] = len(eligible_layouts) == len(args.dy_layouts)
                        row["status"] = "pass" if row["eligible"] else "numerical_failure"
                        if name.startswith("hybrid"):
                            row["experimental_surrogate_backward"] = True
                        result = None
                        if eligible_layouts:

                            def _forward():
                                with torch.no_grad():
                                    return operation(*inputs)

                            row["forward"] = _measure(_forward, args.warmup, args.repeats)
                            row["forward_backward"] = {}
                            for dy_layout in eligible_layouts:

                                def _both(dy=dys[dy_layout]):
                                    value = operation(*inputs)
                                    return torch.autograd.grad(value, inputs, dy)

                                row["forward_backward"][dy_layout] = _measure(_both, args.warmup, args.repeats)
                        counts["eligible" if row["eligible"] else "numerical_failure"] += 1
                    except (RuntimeError, NotImplementedError) as exc:
                        message = str(exc).lower()
                        row["status"] = "oom" if _is_oom(exc) else "execution_error"
                        if "no available kernel" in message or "no viable backend" in message:
                            row["status"] = "unsupported"
                        row["eligible"] = False
                        row["error"] = str(exc)[:1000]
                        counts["unsupported" if row["status"] == "unsupported" else "errors"] += 1
                        if "illegal memory" in message or "device-side assert" in message:
                            _emit(row, args.output)
                            raise
                    finally:
                        result, actual_grads, dys = None, None, {}
                    _emit(row, args.output)
                    gc.collect()
                    torch.cuda.empty_cache()
                del inputs, reference, reference_grads, upstream
                gc.collect()
                torch.cuda.empty_cache()
    _emit({"event": "summary", "experiment": "attention_layout", **counts}, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
