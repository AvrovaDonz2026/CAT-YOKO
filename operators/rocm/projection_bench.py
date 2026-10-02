#!/usr/bin/env python3
"""Standalone BF16 projection and offload-copy benchmarks at CAT-YOKO shapes.

No production modules are changed. Frozen QKV and shared SwiGLU gate/up
projections compare separate GEMMs, concatenation on each call, and an
already packed weight. Copy timings distinguish pageable from pinned CPU
storage and reused GPU destinations from allocation on each visit.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    delta = (actual.detach().float() - expected.detach().float())
    denom = expected.detach().float().norm().clamp_min(1e-12)
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.08, rtol=0.04)
        close = True
    except AssertionError:
        close = False
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / denom),
        "close": close,
    }


def timed(fn: Callable, device: torch.device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    event_ms, wall_ms = [], []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - begin) * 1000)
        event_ms.append(start.elapsed_time(end))
        del result
    return {
        "event_ms_median": statistics.median(event_ms),
        "wall_ms_median": statistics.median(wall_ms),
        "repeats": repeats,
    }


def projection_case(
    name: str, outputs: tuple[int, ...], tokens: int,
    device: torch.device, warmup: int, repeats: int,
    cpu_weights: list[torch.Tensor] | None = None,
) -> dict:
    hidden = 2048
    # A frozen Parameter's input gradient is still needed in the B0 decoder.
    if cpu_weights is None:
        cpu_weights = [torch.randn(n, hidden, dtype=torch.bfloat16).mul_(0.02) for n in outputs]
    cpu_weights = [weight.to(dtype=torch.bfloat16, device="cpu").contiguous() for weight in cpu_weights]
    for weight, output in zip(cpu_weights, outputs, strict=True):
        if weight.shape != (output, hidden):
            raise ValueError(f"unexpected {name} weight shape {tuple(weight.shape)}")
    gpu_weights = [weight.to(device) for weight in cpu_weights]
    packed_cpu = torch.cat(cpu_weights, dim=0).contiguous()
    packed_gpu = torch.cat(gpu_weights, dim=0).contiguous()
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16, requires_grad=True)

    def separate(inp):
        return torch.cat([F.linear(inp, weight) for weight in gpu_weights], dim=-1)

    def concat_each_call(inp):
        return F.linear(inp, torch.cat(gpu_weights, dim=0))

    def cached_concat(inp):
        return F.linear(inp, packed_gpu)

    # An fp32 graph is a reference for finite differences between BF16
    # GEMM shapes; all variants additionally compare to the same BF16 baseline.
    reference_x = x.detach().float().requires_grad_(True)
    reference_w = [weight.float() for weight in gpu_weights]
    reference_y = torch.cat([F.linear(reference_x, weight) for weight in reference_w], dim=-1)
    upstream = torch.randn_like(reference_y, dtype=torch.bfloat16).contiguous()
    reference_dx = torch.autograd.grad(reference_y, reference_x, upstream.float())[0]
    baseline_y = separate(x)
    baseline_dx = torch.autograd.grad(baseline_y, x, upstream)[0]
    rows = []
    for variant, fn in (
        ("separate_gemms", separate),
        ("gpu_cat_each_call", concat_each_call),
        ("gpu_cat_cached", cached_concat),
    ):
        y = fn(x)
        dx = torch.autograd.grad(y, x, upstream)[0]
        row = {
            "variant": variant,
            "forward_vs_separate": error_stats(y, baseline_y),
            "input_grad_vs_separate": error_stats(dx, baseline_dx),
            "forward_vs_fp32": error_stats(y, reference_y),
            "input_grad_vs_fp32": error_stats(dx, reference_dx),
        }

        def forward_only(fn=fn):
            with torch.no_grad():
                return fn(x)

        def forward_backward(fn=fn):
            out = fn(x)
            return torch.autograd.grad(out, x, upstream)[0]

        row["forward"] = timed(forward_only, device, warmup, repeats)
        row["forward_and_input_grad"] = timed(forward_backward, device, warmup, repeats)
        rows.append(row)
        del y, dx

    del reference_y, reference_dx, reference_x, reference_w, baseline_y, baseline_dx
    torch.cuda.empty_cache()
    byte_count = sum(weight.numel() * weight.element_size() for weight in cpu_weights)
    copy_rows = []

    def add_copy(label: str, fn: Callable, *, pinned: bool, reusable: bool):
        row = {"variant": label, "pinned_source": pinned, "reused_destination": reusable}
        row.update(timed(fn, device, warmup, repeats))
        row["payload_bytes"] = byte_count
        row["payload_gb_per_s_wall"] = byte_count / (row["wall_ms_median"] * 1e6)
        copy_rows.append(row)

    destination_parts = [torch.empty_like(weight, device=device) for weight in cpu_weights]
    destination_packed = torch.empty_like(packed_cpu, device=device)

    def pageable_parts_reuse():
        for source, dest in zip(cpu_weights, destination_parts):
            dest.copy_(source, non_blocking=False)
        return destination_parts

    def pageable_packed_reuse():
        destination_packed.copy_(packed_cpu, non_blocking=False)
        return destination_packed

    def legacy_visit():
        # Current offload drops native GEMM caches on every module move:
        # transfer independent Parameter weights, then rebuild the GPU cat.
        transferred = [weight.to(device) for weight in cpu_weights]
        return torch.cat(transferred, dim=0)

    def packed_visit():
        return packed_cpu.to(device)

    add_copy("pageable_parts_copy_reused", pageable_parts_reuse, pinned=False, reusable=True)
    add_copy("pageable_packed_copy_reused", pageable_packed_reuse, pinned=False, reusable=True)
    add_copy("pageable_parts_allocate_and_gpu_cat", legacy_visit, pinned=False, reusable=False)
    add_copy("pageable_packed_allocate", packed_visit, pinned=False, reusable=False)
    pinned_cpu = packed_cpu.pin_memory()

    def pinned_packed_reuse():
        destination_packed.copy_(pinned_cpu, non_blocking=True)
        return destination_packed

    add_copy("pinned_packed_copy_reused", pinned_packed_reuse, pinned=True, reusable=True)

    visit_rows = []
    for label, fetch in (("offload_parts_then_gpu_cat", legacy_visit), ("packed_pinned_reuse", pinned_packed_reuse)):
        def visit_forward_backward(fetch=fetch):
            weight = fetch()
            out = F.linear(x, weight)
            return torch.autograd.grad(out, x, upstream)[0]
        visit_rows.append({
            "variant": label,
            **timed(visit_forward_backward, device, warmup, repeats),
        })

    # Verify that the proposed CPU packed representation carries the exact
    # same BF16 values before applying any transport or lifetime change.
    torch.testing.assert_close(pinned_cpu, torch.cat(cpu_weights), rtol=0, atol=0)
    packed_forward = F.linear(x, pinned_packed_reuse())
    packed_dx = torch.autograd.grad(packed_forward, x, upstream)[0]
    cached_forward = cached_concat(x)
    cached_dx = torch.autograd.grad(cached_forward, x, upstream)[0]
    transfer_parity = {
        "forward": error_stats(packed_forward, cached_forward),
        "input_grad": error_stats(packed_dx, cached_dx),
    }
    return {
        "case": name, "tokens": tokens, "input_features": hidden,
        "output_features": list(outputs), "weight_mib": byte_count / 1024**2,
        "projections": rows, "host_to_gpu": copy_rows,
        "visit_forward_and_input_grad": visit_rows,
        "packed_transfer_parity": transfer_parity,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base", type=Path, help="optional MiniCPM5 local dir or model.safetensors")
    parser.add_argument("--layer", type=int, default=16, help="base decoder source layer, default 16")
    args = parser.parse_args(argv)
    if args.tokens <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("tokens and repeats must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("a CUDA or ROCm PyTorch GPU runtime is required")
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    base_weights = None
    if args.base is not None:
        from safetensors import safe_open

        path = args.base / "model.safetensors" if args.base.is_dir() else args.base
        prefix = f"model.layers.{args.layer}."
        with safe_open(str(path), framework="pt", device="cpu") as source:
            qkv = [
                source.get_tensor(prefix + "self_attn." + name + ".weight").clone()
                for name in ("q_proj", "k_proj", "v_proj")
            ]
            # CAT-YOKO B0 decoder: each 2048-wide expert copies the leading
            # rows of the base 6144-wide FFN, with its published upcycle scale.
            scale = (21 / 10) ** (1 / 3)
            gate_up = [
                (source.get_tensor(prefix + "mlp." + name + ".weight")[:2048] / scale).clone()
                for name in ("gate_proj", "up_proj")
            ]
        base_weights = {"qkv": qkv, "shared_swiglu_gate_up": gate_up}
    cases = []
    for name, outputs in (("qkv", (2048, 256, 256)), ("shared_swiglu_gate_up", (2048, 2048))):
        weights = base_weights[name] if base_weights is not None else None
        cases.append(projection_case(name, outputs, args.tokens, device, args.warmup, args.repeats, weights))
        gc.collect()
        torch.cuda.empty_cache()
    correct = all(
        row[key]["close"]
        for case in cases
        for row in case["projections"]
        for key in ("forward_vs_separate", "input_grad_vs_separate", "forward_vs_fp32", "input_grad_vs_fp32")
    ) and all(
        stats["close"] for case in cases for stats in case["packed_transfer_parity"].values()
    )
    result = {
        "ok": correct, "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "dtype": "bf16", "seed": args.seed,
        "weight_source": str(args.base) if args.base is not None else "synthetic",
        "base_layer": args.layer if args.base is not None else None,
        "cases": cases,
        "notes": [
            "Frozen projections require input gradients, but no weight gradients.",
            "CUDA/HIP events report device time; wall timings include allocation and host dispatch.",
            "Pinned packing is prepared once outside timed visits; benchmarks do not overlap transfers.",
            "Weights have production BF16 matrix shapes; the optional base reads only selected tensors.",
            "Reused packed buffers may only be overwritten after autograd releases their saved weights.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ok": correct, "out": str(args.out), "gpu": result["gpu"]}), flush=True)
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
