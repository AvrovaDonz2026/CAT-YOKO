#!/usr/bin/env python3
"""Probe native BF16 MoE index_add reproducibility and deterministic support.

No model or checkpoint is loaded. Input rows, normalized top-k gates, and
expert-sorted token indices are fixed across every repeat/policy. Defaults
match 4096 tokens, hidden 2048, 20 experts, top-k 10. Both identical-expert
and independent-expert output cases are measured. PyTorch is the only
dependency. This file never enables an optimization or changes production.

  python operators/rocm/probe_index_add.py --device cuda --json result.json
  python operators/rocm/probe_index_add.py --device cpu --tokens 32 --hidden 16 \
      --experts 4 --topk 2 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def errors(actual, expected):
    difference = actual.float() - expected.float()
    return {"bitwise_equal": bool(torch.equal(actual, expected)),
            "max_abs": difference.abs().max().item(),
            "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=20)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cases", default="identical,independent")
    parser.add_argument("--modes", default="default,deterministic")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if min(args.tokens, args.hidden, args.experts) < 1 or not 1 <= args.topk <= args.experts:
        parser.error("positive dimensions and topk in [1,experts] are required")
    if args.repeats < 2:
        parser.error("repeats must be at least 2")
    cases, modes = args.cases.split(","), args.modes.split(",")
    if any(case not in ("identical", "independent") for case in cases):
        parser.error("cases must be identical,independent")
    if any(mode not in ("default", "deterministic") for mode in modes):
        parser.error("modes must be default,deterministic")
    device = torch.device(args.device)
    original = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.manual_seed(args.seed)
    # Build routing once on CPU, so policy changes never regenerate inputs.
    scores = torch.softmax(torch.randn(args.tokens, args.experts), dim=-1)
    gates, expert_ids = torch.topk(scores, args.topk, dim=-1)
    gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    order = expert_ids.reshape(-1).argsort()
    tokens = torch.arange(args.tokens).unsqueeze(1).expand_as(expert_ids).reshape(-1)
    token_sorted = tokens[order].to(device)
    gates_sorted = gates.reshape(-1)[order].to(device=device, dtype=torch.bfloat16)
    report = {"torch": torch.__version__, "hip": torch.version.hip,
              "device": str(device), "dtype": "bfloat16", "tokens": args.tokens,
              "hidden": args.hidden, "experts": args.experts, "topk": args.topk,
              "repeats": args.repeats, "seed": args.seed,
              "original_deterministic_algorithms": original,
              "results": [], "note": "Only native weighted index_add, with immutable inputs; no optimizer updates."}
    try:
        for case in cases:
            if case == "identical":
                expert_output = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
                rows = expert_output.index_select(0, token_sorted)
                del expert_output
            else:
                rows = torch.randn(args.tokens * args.topk, args.hidden,
                                   device=device, dtype=torch.bfloat16)
            rows = rows * gates_sorted.unsqueeze(-1)
            default_reference = None
            for mode in modes:
                torch.use_deterministic_algorithms(mode == "deterministic", warn_only=False)
                first = None
                measurements = []
                try:
                    for repeat in range(args.repeats):
                        out = torch.zeros(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
                        sync(device)
                        begin = time.perf_counter()
                        out.index_add_(0, token_sorted, rows)
                        sync(device)
                        milliseconds = (time.perf_counter() - begin) * 1000
                        if first is None:
                            first = out.clone()
                        measurements.append({"repeat": repeat, "ms": milliseconds,
                                             **errors(out, first)})
                        del out
                    row = {"case": case, "mode": mode, "supported": True,
                           "all_repeats_bitwise_equal": all(x["bitwise_equal"] for x in measurements),
                           "max_repeat_relative_l2": max(x["relative_l2"] for x in measurements),
                           "max_repeat_abs": max(x["max_abs"] for x in measurements),
                           "measurements": measurements}
                    if mode == "default":
                        default_reference = first
                    elif default_reference is not None:
                        row["versus_default"] = errors(first, default_reference)
                except RuntimeError as exc:
                    row = {"case": case, "mode": mode, "supported": False,
                           "error": str(exc), "completed_repeats": len(measurements)}
                report["results"].append(row)
                print(json.dumps(row, allow_nan=False), flush=True)
            del rows, default_reference
    finally:
        torch.use_deterministic_algorithms(original, warn_only=warn_only)
    report["restored_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    report["restored_deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"supported": all(r["supported"] for r in report["results"]),
                      "restored_deterministic_algorithms": report["restored_deterministic_algorithms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
