#!/usr/bin/env python3
"""Check cached CPU Adam byte parity and synchronized optimizer-only wall time.

No production installation, process control, or full-training speedup claim.
Every timing segment restores the same checkpoint and repeats identical BF16
gradients. Cache construction is reported separately from later updates.
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from operators.rocm import gpu_adam_bench as reference
from operators.rocm.cpu_adam_cached import use_cached_cpu_adam
from operators.rocm.gpu_adam_streamed_bench import digest, validate_source


def strict_compare(native, candidate, names, expected_step):
    result = reference.compare_optimizers(native, candidate, names)
    result["gate"]["fp32_moments"] = "byte equality required after every step"
    counters = [int(state["step"]) for optimizer in (native, candidate) for state in optimizer.state.values()]
    result["counters_match_source_delta"] = len(counters) == 2 * len(names) and all(step == expected_step for step in counters)
    result["pass"] = bool(result["pass"] and result["all_weights_bitwise"] and result["all_moments_bitwise"]
                          and result["counters_match_source_delta"])
    return result


def timed_update(optimizer, parameters, templates, device):
    reference.assign_gradients(parameters, templates)
    reference.synchronize(device)
    initial = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    optimizer.step()
    reference.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("optimizer completion wall time must be finite and positive")
    result = {"wall_ms": elapsed}
    if device.type == "cuda":
        result.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                      incremental_peak_mib=(torch.cuda.max_memory_allocated(device) - initial) / 2**20)
    return result


def probe(checkpoint, *, device="cuda", steps=5, seed=20261004, expected_parameters=132,
          on_report=lambda report: None):
    if steps < 5:
        raise ValueError("at least five consecutive updates are required")
    weights, records = validate_source(checkpoint, expected_parameters)
    source_step = int(records[0][2]["step"])
    device = torch.device(device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("only CUDA/HIP or explicit CPU tests are supported")
    if device.type == "cuda":
        required = sum(value.numel() * value.element_size() for value in weights.values()) * 4
        required += max(value.numel() for value in weights.values()) * 16 + (512 << 20)
        free, total = torch.cuda.mem_get_info(device)
        if free < required:
            raise ValueError(f"insufficient GPU memory: require {required} bytes, free {free}")
    native, candidate, names, native_params, candidate_params = reference.build_optimizers(
        weights, checkpoint["optimizer"], device)
    identities = [id(parameter) for parameter in candidate_params]
    report = {"status": "running", "arithmetic_variant": "cached-cpu-fp32", "device": str(device),
              "steps": steps, "expected_parameters": expected_parameters, "seed": seed, "parity": [],
              "persistent_fp32_master": False, "whole_training_speedup_measured": False,
              "scope": "optimizer step including D2H, CPU math and completed H2D; gradient creation/assignment, comparisons and resets excluded",
              "timing_segment_order": ["native_before", "cached", "native_after"],
              "checkpoint_moments_cpu_fp32": False}
    on_report(report)

    def gradients(step):
        return list(reference.seeded_gradients(native_params, seed=seed + step, scale=1e-4))

    @torch.no_grad()
    def reset(optimizer, parameters):
        for name, parameter in zip(names, parameters):
            parameter.copy_(weights[name])
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        reference.synchronize(device)
        gc.collect()

    def segment(optimizer, parameters):
        samples = []
        for step in range(steps):
            templates = gradients(step)
            samples.append(timed_update(optimizer, parameters, templates, device))
            del templates
        return {"wall_ms_median": statistics.median(row["wall_ms"] for row in samples),
                "steady_wall_ms_median": statistics.median(row["wall_ms"] for row in samples[1:]),
                "samples": samples}

    with use_cached_cpu_adam(allow_cpu=device.type == "cpu", optimizers=[candidate]) as installation:
        for step in range(steps):
            templates = gradients(step)
            reference.assign_gradients(native_params, templates)
            reference.assign_gradients(candidate_params, templates)
            native.step()
            candidate.step()
            reference.synchronize(device)
            result = strict_compare(native, candidate, names, source_step + step + 1)
            result["invocation_step"] = step + 1
            report["parity"].append(result)
            on_report(report)
            if not result["pass"]:
                report.update(status="parity_failure", installation=installation.report())
                on_report(report)
                return report
            del templates
        reset(native, native_params)
        report["native_before"] = segment(native, native_params)
        reset(candidate, candidate_params)  # load explicitly invalidates every BF16 cache entry.
        report["cached"] = segment(candidate, candidate_params)
        cached_samples = report["cached"]["samples"]
        report["cached"].update(first_cache_build_wall_ms=cached_samples[0]["wall_ms"],
                               steady_wall_ms_median=statistics.median(row["wall_ms"] for row in cached_samples[1:]),
                               steady_updates=steps - 1)
        report["timing_terminal_before"] = strict_compare(native, candidate, names, source_step + steps)
        if not report["timing_terminal_before"]["pass"]:
            report.update(status="timing_parity_failure", installation=installation.report())
            on_report(report)
            return report
        reset(native, native_params)
        report["native_after"] = segment(native, native_params)
        report["terminal"] = strict_compare(native, candidate, names, source_step + steps)
        snapshot = candidate.state_dict()
        moments = [value for state in snapshot["state"].values() for key, value in state.items()
                   if key in ("exp_avg", "exp_avg_sq")]
        exported = len(snapshot["state"]) == expected_parameters and len(moments) == 2 * expected_parameters and all(
            torch.is_tensor(value) and value.device.type == "cpu" and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()) for value in moments)
        identities_preserved = identities == [id(parameter) for parameter in candidate_params]
        report.update(status="complete" if report["terminal"]["pass"] and exported and identities_preserved else "terminal_failure",
                      checkpoint_moments_cpu_fp32=exported, exported_moments_checked=len(moments),
                      installation=installation.report(), parameter_identities_preserved=identities_preserved)
        reference_median = min(report[key]["steady_wall_ms_median"] for key in ("native_before", "native_after"))
        report["optimizer_only_conservative_speedup"] = reference_median / report["cached"]["steady_wall_ms_median"]
        # Publish terminal status only after the CLI has verified source integrity.
        on_report({**report, "status": "verification_pending"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    if args.steps < 5:
        parser.error("steps must be at least five")
    if args.json.exists():
        parser.error("refusing to overwrite an existing optimizer report")
    resume = reference.resolve_resume_path(args.resume).resolve()
    if args.json.resolve() == resume or args.json.resolve().is_relative_to(resume.parent):
        parser.error("report must be separate from the source checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/HIP runtime unavailable")
    source_sha = digest(resume)
    checkpoint = reference.load_checkpoint(resume, map_location="cpu")
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), *(root / "operators/rocm" / name for name in
             ("cpu_adam_cached.py", "gpu_adam_bench.py", "gpu_adam_streamed_bench.py", "gpu_adam.py", "gpu_adam_ordered.py")),
             root / "cat_yoko/optim.py", root / "cat_yoko/checkpoint.py"]
    metadata = {"source_checkpoint": str(resume), "source_sha256": source_sha,
                "source_step": (checkpoint.get("extra") or {}).get("step"),
                "file_sha256": {str(path.relative_to(root)): digest(path) for path in files}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {}

    def save(value):
        report.clear()
        report.update(value, **metadata)
        reference.write_report(args.json, report)

    try:
        result = probe(checkpoint, device=args.device, steps=args.steps, seed=args.seed, on_report=save)
        if digest(resume) != source_sha or any(digest(root / name) != value for name, value in metadata["file_sha256"].items()):
            raise ValueError("source checkpoint or implementation changed during the experiment")
        result["source_integrity_verified"] = True
        save(result)
        return 0 if result["status"] == "complete" else 1
    except BaseException as error:
        save({**report, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
