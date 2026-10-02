#!/usr/bin/env python3
"""Experimental identical frozen expert collapse; PyTorch is the only dependency.

This does not patch CAT-YOKO. ``FrozenIdenticalMoE(moe)`` accepts its native
BF16/FP32 MoE/SwiGLU layout, verifies byte-identical routed weights on CPU once,
and preserves input gradients and normalized router gates. Expert updates or
trainable/quantized expert weights are rejected. Shared experts are unchanged.

Run on ROCm/CUDA:
  python operators/rocm/frozen_moe.py --device cuda --json results.json
The defaults use the actual repository MoE as the batched reference, published
2048-wide, 2048-intermediate, 20-expert shapes, 64/256/1024 tokens, top-k 7/10,
and FP32 storage plus BF16 storage/autocast. Use ``--reference standalone``
outside the repository to select the independent reference. Use
``--tokens 64,256,1024,4096`` to include the longer sequence. CPU smoke test:
  python operators/rocm/frozen_moe.py --device cpu --hidden 32 --intermediate 24 \
      --experts 4 --topk 2 --tokens 16 --warmup 0 --iterations 1

Use ``--base /path/to/minicpm5 --layer 16 --modes bf16 --topk 10 --tokens 4096``
to validate actual base-checkpoint upcycled frozen decoder weights. This optional
loader additionally needs safetensors; it reads only the chosen MLP slices.

BF16 reassociates GEMM/backward reductions, so numerical equality is approximate;
the CLI reports output/input-gradient errors and exits nonzero on tolerance
failure. This optimization is invalid once initially copied experts diverge.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _expert_signature(experts) -> tuple:
    return tuple(
        (id(p), p._version, p.requires_grad, p.dtype, tuple(p.shape))
        for expert in experts for p in expert.parameters()
    )


def _routing(moe, flat: torch.Tensor, token_ids: torch.Tensor | None):
    if moe.hash_route and token_ids is not None:
        indices = ((token_ids.reshape(-1).to(torch.int64) * 2654435761)
                   .remainder(moe.n_routed)).unsqueeze(1)
        return torch.ones(flat.shape[0], 1, dtype=flat.dtype, device=flat.device), indices
    # Match production router math, including ambient BF16 autocast behavior.
    logits = F.linear(flat.float(), moe.router.weight.float())
    affinity = torch.sqrt(F.softplus(logits))
    probs = torch.softmax(affinity + moe.e_score_correction_bias.float(), dim=-1)
    values, indices = torch.topk(probs, moe.top_k, dim=-1)
    gates = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return gates, indices


def _shared_output(moe, flat: torch.Tensor) -> torch.Tensor:
    value = moe.shared[0](flat)
    for expert in moe.shared[1:]:
        value = value + expert(flat)
    return value


class FrozenIdenticalMoE:
    """A pure forward operator that keeps original MoE routing statistics intact.

    Construct after freezing/wrapping/loading the checkpoint. Validation costs
    CPU copies of all routed weights once; it is not part of timed forwards.
    Runtime checks reject normal in-place updates and re-enabling gradients.
    Do not modify parameters through ``.data`` after validation: PyTorch does
    not increment their version counters for those unsupported updates.
    """

    def __init__(self, moe):
        if not moe.experts or not moe.shared:
            raise ValueError("requires routed and shared experts")
        if any(p.requires_grad for p in moe.parameters()):
            raise ValueError("identical expert collapse requires a fully frozen MoE")
        if len(moe.experts) != moe.n_routed:
            raise ValueError("expert count does not match router")
        if not 1 <= moe.top_k <= moe.n_routed:
            raise ValueError("invalid top-k")
        projections = ("gate_proj", "up_proj", "down_proj")
        for expert in (*moe.experts, *moe.shared):
            for name in projections:
                layer = getattr(expert, name, None)
                if type(layer) is not nn.Linear or layer.bias is not None:
                    raise ValueError("requires native bias-free SwiGLU linears")
        reference = moe.experts[0]
        for name in projections:
            weight = getattr(reference, name).weight
            raw = weight.detach().cpu().contiguous().view(torch.uint8)
            for expert in moe.experts[1:]:
                other = getattr(expert, name).weight
                if other.shape != weight.shape or other.dtype != weight.dtype:
                    raise ValueError(f"nonidentical routed expert {name} shape/dtype")
                if not torch.equal(raw, other.detach().cpu().contiguous().view(torch.uint8)):
                    raise ValueError(f"routed expert {name} weights are not byte-identical")
        self.moe = moe
        self._signature = _expert_signature(moe.experts)

    def __call__(self, x: torch.Tensor, token_ids: torch.Tensor | None = None):
        moe = self.moe
        if _expert_signature(moe.experts) != self._signature:
            raise RuntimeError("routed experts changed; recreate and revalidate the operator")
        if any(p.requires_grad for p in moe.parameters()):
            raise RuntimeError("identical expert collapse cannot train MoE parameters")
        flat = x.reshape(-1, x.shape[-1])
        gates, _ = _routing(moe, flat, token_ids)
        # Production casts each gate to the expert output dtype before its
        # weighted reduction. Preserve that gate quantization and sum; a
        # literal 1 would drop both its rounding and router input gradients.
        expert = moe.experts[0](flat)
        gate_sum = gates.to(expert.dtype).sum(dim=-1, keepdim=True, dtype=flat.dtype)
        routed = expert * gate_sum
        shared = _shared_output(moe, flat)
        routed = routed.to(dtype=shared.dtype, device=shared.device)
        return (shared + routed).reshape_as(x)


class _SwiGLU(nn.Module):
    """Standalone equivalent of native CAT-YOKO's fused gate/up SwiGLU."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self._gu = None

    def forward(self, x):
        signature = tuple((id(p), p._version, p.device, p.dtype)
                          for p in (self.gate_proj.weight, self.up_proj.weight))
        if self._gu is None or self._gu[0] != signature:
            self._gu = (signature, torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0))
        g, u = F.linear(x, self._gu[1]).chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class _BenchmarkMoE(nn.Module):
    """Production-style padded batched dispatch; no repo imports or side effects."""

    def __init__(self, hidden: int, intermediate: int, count: int):
        super().__init__()
        self.n_routed = count
        self.top_k = min(7, count)
        self.hash_route = False
        self.shared = nn.ModuleList([_SwiGLU(hidden, intermediate)])
        self.experts = nn.ModuleList(_SwiGLU(hidden, intermediate) for _ in range(count))
        for expert in self.experts[1:]:
            expert.load_state_dict(self.experts[0].state_dict())
        self.router = nn.Linear(hidden, count, bias=False)
        self.register_buffer("e_score_correction_bias", torch.randn(count) * 0.01)
        self.requires_grad_(False)
        self._packed = None

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None):
        flat = x.reshape(-1, x.shape[-1])
        shared = _shared_output(self, flat)
        gates, topi = _routing(self, flat, token_ids)
        count_tokens, d = flat.shape
        token_ids_flat = torch.arange(count_tokens, device=x.device).unsqueeze(1)
        token_ids_flat = token_ids_flat.expand_as(topi).reshape(-1)
        expert_ids = topi.reshape(-1)
        order = expert_ids.argsort()
        expert_sorted = expert_ids[order]
        token_sorted = token_ids_flat[order]
        gate_sorted = gates.reshape(-1)[order]
        counts = torch.bincount(expert_sorted, minlength=self.n_routed)
        max_n = int(counts.max().item())
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts[:-1], dim=0)
        local = torch.arange(order.numel(), device=x.device) - starts[expert_sorted]
        x_pad = flat.new_zeros(self.n_routed, max_n, d)
        x_pad[expert_sorted, local] = flat[token_sorted]
        if self._packed is None:
            gu = torch.stack([torch.cat((e.gate_proj.weight, e.up_proj.weight), dim=0)
                              for e in self.experts])
            down = torch.stack([e.down_proj.weight for e in self.experts])
            self._packed = (gu, down)
        gu, down = self._packed
        projected = torch.bmm(x_pad, gu.transpose(1, 2))
        g, u = projected.chunk(2, dim=-1)
        outputs = torch.bmm(F.silu(g) * u, down.transpose(1, 2))
        sorted_output = outputs[expert_sorted, local]
        sorted_output = sorted_output * gate_sorted.unsqueeze(-1).to(sorted_output.dtype)
        routed = torch.zeros_like(flat)
        routed.index_add_(0, token_sorted, sorted_output.to(routed.dtype))
        return (shared + routed.to(shared.dtype)).reshape_as(x)


def _autocast(device: torch.device, mode: str):
    return (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if mode == "bf16" else nullcontext())


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _errors(actual: torch.Tensor, expected: torch.Tensor):
    difference = actual.float() - expected.float()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    }


def _measure(fn, source, probe, mode, backward, warmup, iterations):
    device = source.device

    def run():
        value = source.detach().requires_grad_(backward)
        with _autocast(device, mode):
            result = fn(value)
            if backward:
                (result.float() * probe).sum().backward()
        return result, value.grad

    for _ in range(warmup):
        run()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iterations):
        run()
    _sync(device)
    milliseconds = (time.perf_counter() - start) * 1000 / iterations
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"ms": milliseconds, "peak_allocated_bytes": peak}


def _guard_selfcheck():
    model = _BenchmarkMoE(8, 8, 3)
    op = FrozenIdenticalMoE(model)
    x = torch.randn(1, 4, 8, requires_grad=True)
    # Hash routing also collapses: its single selected gate is exactly one.
    model.hash_route = True
    ids = torch.arange(4).unsqueeze(0)
    torch.testing.assert_close(op(x, ids), model(x, ids), atol=1e-6, rtol=1e-5)
    model.experts[1].gate_proj.weight.requires_grad_(True)
    try:
        op(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("trainable expert guard failed")
    model.experts[1].gate_proj.weight.requires_grad_(False)
    with torch.no_grad():
        model.experts[1].gate_proj.weight.add_(0.01)
    try:
        op(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed expert guard failed")
    try:
        FrozenIdenticalMoE(model)
    except ValueError:
        pass
    else:
        raise AssertionError("nonidentical expert guard failed")


def _make_benchmark_model(args):
    if args.reference == "standalone":
        return _BenchmarkMoE(args.hidden, args.intermediate, args.experts)
    # The actual repository reference still has no third-party dependencies
    # beyond PyTorch. The independent equivalent is available for portability.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cat_yoko.config import CATYokoConfig
    from cat_yoko.moe import MoE

    cfg = replace(CATYokoConfig.tiny(), hidden_size=args.hidden,
                  moe_intermediate_size=args.intermediate, n_shared=1)
    model = MoE(cfg, args.experts, min(7, args.experts))
    for expert in model.experts[1:]:
        expert.load_state_dict(model.experts[0].state_dict())
    with torch.no_grad():
        model.e_score_correction_bias.normal_(std=0.01)
    model.requires_grad_(False)
    model.grouped_experts = False
    model.eval()
    if args.base:
        _load_base_experts(model, args)
    return model


def _load_base_experts(model, args):
    """Read only the chosen MiniCPM5 MLP slices and reproduce CPU upcycling."""
    from safetensors import safe_open

    path = args.base.expanduser().resolve()
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors files found at {path}")
    projections = ("gate_proj", "up_proj", "down_proj")
    remaining = set(projections)
    source = {}
    provenance = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in tuple(remaining):
                candidates = (f"model.layers.{args.layer}.mlp.{name}.weight",
                              f"layers.{args.layer}.mlp.{name}.weight")
                key = next((key for key in candidates if key in available), None)
                if key is None:
                    continue
                view = handle.get_slice(key)
                full_shape = tuple(view.get_shape())
                required = ((args.hidden, 6144) if name == "down_proj"
                            else (6144, args.hidden))
                if full_shape != required:
                    raise ValueError(f"{key}: expected MiniCPM5 {required}, got {full_shape}")
                rows, cols = getattr(model.experts[0], name).weight.shape
                source[name] = view[:rows, :cols]
                provenance[name] = {"file": str(file), "key": key,
                                    "source_shape": full_shape,
                                    "slice_shape": tuple(source[name].shape),
                                    "source_dtype": str(source[name].dtype)}
                remaining.remove(name)
        if not remaining:
            break
    if remaining:
        raise ValueError(f"base checkpoint missing layer {args.layer} MLP projections: {sorted(remaining)}")
    # Match upcycle.py: divide in the source dtype before copying into the
    # destination master. All shared and routed experts receive the same slice.
    scale = ((len(model.shared) + model.n_routed) / 10) ** (1.0 / 3.0)
    with torch.no_grad():
        for name in projections:
            scaled = source[name] / scale
            for expert in (*model.shared, *model.experts):
                getattr(expert, name).weight.copy_(scaled)
    args._base_info = {"path": str(path), "layer": args.layer,
                       "upcycle_topk": 10, "scale": scale,
                       "projections": provenance,
                       "router": "frozen seed-2026 random router; not checkpoint router"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=20)
    parser.add_argument("--tokens", default="64,256,1024")
    parser.add_argument("--topk", default="7,10")
    parser.add_argument("--modes", default="fp32,bf16")
    parser.add_argument("--reference", choices=("native", "standalone"), default="native")
    parser.add_argument("--base", type=Path,
                        help="MiniCPM5 safetensors file/directory; reconstruct frozen decoder experts")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    modes = args.modes.split(",")
    if any(mode not in ("fp32", "bf16") for mode in modes):
        parser.error("modes must be fp32,bf16")
    if args.base and (args.reference != "native" or args.topk != "10" or
                      args.hidden != 2048 or args.intermediate != 2048 or args.experts != 20):
        parser.error("--base requires native reference, --topk 10, and published 2048/2048/20 dimensions")
    _guard_selfcheck()
    torch.manual_seed(2026)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Float32 tests must not silently become low precision TF32 tests.
        torch.backends.cuda.matmul.allow_tf32 = False
    guard_seconds = {}
    rows = []
    for mode in modes:
        # Match production storage/accumulation as well as GEMM autocast.
        # Create and validate after converting; .to(dtype) changes signatures.
        dtype = torch.float32 if mode == "fp32" else torch.bfloat16
        torch.manual_seed(2026)
        model = _make_benchmark_model(args).to(device=device, dtype=dtype)
        construction_start = time.perf_counter()
        collapsed = FrozenIdenticalMoE(model)
        guard_seconds[mode] = time.perf_counter() - construction_start
        for topk in map(int, args.topk.split(",")):
            if not 1 <= topk <= args.experts:
                parser.error("topk must be in [1, experts]")
            model.top_k = topk
            for tokens in map(int, args.tokens.split(",")):
                if tokens < 1:
                    parser.error("token counts must be positive")
                source = torch.randn(1, tokens, args.hidden, device=device, dtype=dtype)
                probe = torch.randn(source.shape, device=device, dtype=torch.float32)
                value_ref = source.detach().requires_grad_(True)
                value_op = source.detach().requires_grad_(True)
                with _autocast(device, mode):
                    y_ref = model(value_ref)
                    y_op = collapsed(value_op)
                (y_ref.float() * probe).sum().backward()
                (y_op.float() * probe).sum().backward()
                output_error = _errors(y_op, y_ref)
                grad_error = _errors(value_op.grad, value_ref.grad)
                # Explicit thresholds bound BF16 reassociation separately.
                limit = 1e-4 if mode == "fp32" else 0.015
                passed = output_error["relative_l2"] <= limit and grad_error["relative_l2"] <= limit
                finite = all(torch.isfinite(t).all().item()
                             for t in (y_ref, y_op, value_ref.grad, value_op.grad))
                row = {
                    "tokens": tokens, "topk": topk, "mode": mode,
                    "weight_dtype": str(dtype), "input_dtype": str(source.dtype),
                    "output_error": output_error, "input_grad_error": grad_error,
                    "relative_l2_limit": limit, "passed": bool(passed and finite),
                }
                del y_ref, y_op, value_ref, value_op
                for backward in (False, True):
                    name = "forward_backward" if backward else "forward"
                    baseline = _measure(model, source, probe, mode, backward, args.warmup, args.iterations)
                    optimized = _measure(collapsed, source, probe, mode, backward, args.warmup, args.iterations)
                    row[name] = {"batched": baseline, "collapsed": optimized,
                                 "speedup": baseline["ms"] / optimized["ms"]}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del source, probe
        del collapsed, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hidden": args.hidden, "intermediate": args.intermediate, "experts": args.experts,
        "reference": args.reference,
        "base_checkpoint": getattr(args, "_base_info", None),
        "cpu_byte_equality_guard_seconds": guard_seconds,
        "all_passed": all(row["passed"] for row in rows), "results": rows,
        "limitations": [
            "Only native bias-free SwiGLU with fully frozen, byte-identical routed experts.",
            "Validation must follow checkpoint loading; never mutate weights through .data afterward.",
            "BF16 reduction order changes; parity is numerical, not bitwise.",
            "Peak allocated memory includes both benchmark operators and cached packed baseline weights.",
            "Pure prototype leaves original MoE aux/load statistics untouched and is not installed in the model.",
        ],
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "cases": len(rows)}), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
