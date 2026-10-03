#!/usr/bin/env python3
"""Numerically probe ordered Adam one parameter at a time; no speed claims."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from cat_yoko.checkpoint import load_checkpoint, resolve_resume_path
from cat_yoko.optim import CPUOffloadAdamW
from operators.rocm.gpu_adam_bench import compare_optimizers, write_report
from operators.rocm.gpu_adam_ordered import use_ordered_gpu_fp32_adam


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def validate_source(checkpoint, expected_parameters=132):
    extra = checkpoint.get("extra") or {}
    if checkpoint.get("kind") != "trainable" or extra.get("phase") != "B0" or extra.get("name") != "CAT-YOKO-12B":
        raise ValueError("expected native B0 12B overlay")
    weights, optimizer = checkpoint.get("trainable") or {}, checkpoint.get("optimizer") or {}
    if len(weights) != expected_parameters or any(not torch.is_tensor(p) or p.device.type != "cpu"
            or p.dtype != torch.bfloat16 or not bool(torch.isfinite(p).all()) for p in weights.values()):
        raise ValueError("expected finite CPU BF16 source parameters")
    def no_decay(name, value):
        parts = name.split(".")
        return value.ndim < 2 or parts[-1] == "bias" or any("norm" in part for part in parts) or "router" in parts
    names = [[name for name, p in weights.items() if not no_decay(name, p)],
             [name for name, p in weights.items() if no_decay(name, p)]]
    groups, states = optimizer.get("param_groups") or [], optimizer.get("state") or {}
    identifiers = [identifier for group in groups for identifier in group.get("params", [])]
    if (len(groups) != 2 or len(identifiers) != len(set(identifiers)) or set(identifiers) != set(states)
            or len(identifiers) != expected_parameters or any(len(group["params"]) != len(group_names) for group, group_names in zip(groups, names))):
        raise ValueError("source Adam parameter IDs/decay group mapping differ")
    records, counters = [], set()
    for group, group_names in zip(groups, names):
        if any(not math.isfinite(float(group.get(key, 0))) for key in ("lr", "eps", "weight_decay")):
            raise ValueError("invalid source optimizer hyperparameter")
        betas = group.get("betas", ())
        if len(betas) != 2 or any(not 0 <= value < 1 for value in betas):
            raise ValueError("invalid source Adam betas")
        for identifier, name in zip(group["params"], group_names):
            state = states[identifier]
            counter = state.get("step")
            counter = counter.item() if torch.is_tensor(counter) else counter
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or not isinstance(counter, (float, int)) or not math.isfinite(counter) or int(counter) != counter or counter <= 0:
                raise ValueError("invalid source Adam counter/state keys")
            counters.add(int(counter))
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != weights[name].shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("source moment must be finite CPU FP32 with matching shape")
                if key == "exp_avg_sq" and not bool((value >= 0).all()):
                    raise ValueError("negative source squared moment")
            records.append((name, group, state))
    if len(counters) != 1:
        raise ValueError("source Adam counters disagree")
    return weights, records


def probe(checkpoint, *, device="cuda", steps=5, seed=20261003, expected_parameters=132, on_report=lambda report: None):
    weights, records = validate_source(checkpoint, expected_parameters)
    device = torch.device(device)
    rows = [{"invocation_step": step + 1, "pass": True, "parameters": 0, "all_weights_bitwise": True,
             "all_moments_bitwise": True, "failed_names": [], "tensors": {},
             "gate": {"fp32_moment_atol": 1e-8, "fp32_moment_rtol": 1e-6}} for step in range(steps)]
    report = {"status": "running", "arithmetic_variant": "ordered-fp32", "mode": "per_parameter",
              "device": str(device), "steps": steps, "expected_parameters": expected_parameters,
              "parity": rows, "parameters_checked": 0, "persistent_fp32_master": False,
              "whole_training_speedup_measured": False, "timing_measured": False,
              "checkpoint_moments_cpu_fp32": False, "exported_moments_checked": 0, "max_parameter_peak_bytes": 0}
    on_report(report)
    for index, (name, group, source_state) in enumerate(records):
        # Four bytes per moment or scratch element; leave headroom for live
        # parameters, comparisons and pointwise temporaries before allocating.
        if device.type == "cuda" and weights[name].numel() * 40 > 256 << 20:
            raise ValueError("individual parameter exceeds the 256 MiB probe budget")
        native_p, candidate_p = nn.Parameter(weights[name].clone()), nn.Parameter(weights[name].to(device).clone())
        options = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        native = CPUOffloadAdamW([dict(options, params=[native_p])], state_dtype=torch.float32, retain_state=True)
        candidate = CPUOffloadAdamW([dict(options, params=[candidate_p])], state_dtype=torch.float32, retain_state=True)
        single = {"state": {0: source_state}, "param_groups": [dict(options, params=[0])]}
        native.load_state_dict(copy.deepcopy(single))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            initial = torch.cuda.memory_allocated(device)
        with use_ordered_gpu_fp32_adam(allow_cpu=device.type == "cpu", optimizers=[candidate]):
            candidate.load_state_dict(copy.deepcopy(single))
            for step in range(steps):
                gradient = torch.randn(native_p.shape, generator=torch.Generator().manual_seed(seed + index * steps + step)).mul_(1e-4).to(torch.bfloat16)
                native_p.grad, candidate_p.grad = gradient.clone(), gradient.to(device).clone()
                native.step()
                candidate.step()
                result = compare_optimizers(native, candidate, [name])
                row = rows[step]
                row["tensors"].update(result["tensors"])
                row["parameters"] += 1
                for key in ("pass", "all_weights_bitwise", "all_moments_bitwise"):
                    row[key] = row[key] and result[key]
                row["failed_names"].extend(result["failed_names"])
                if not result["pass"]:
                    report.update(status="parity_failure", failed_parameter=name, failed_invocation_step=step + 1,
                                  parameters_checked=index, incomplete_parameter=True)
                    on_report(report)
                    return report
            snapshot = candidate.state_dict()
            for key in ("exp_avg", "exp_avg_sq"):
                value = snapshot["state"][0][key]
                if value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != candidate_p.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError("ordered probe did not export finite CPU FP32 moments")
                report["exported_moments_checked"] += 1
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device) - initial + candidate_p.numel() * candidate_p.element_size()
                report["max_parameter_peak_bytes"] = max(report["max_parameter_peak_bytes"], peak)
                if peak > 256 << 20:
                    raise ValueError("ordered probe exceeded 256 MiB incremental peak")
        report["parameters_checked"] = index + 1
        on_report(report)
        del native, candidate, native_p, candidate_p, snapshot, gradient, single
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report.update(status="complete", terminal=rows[-1], incomplete_parameter=False, checkpoint_moments_cpu_fp32=True)
    on_report(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu", action="store_true", help="CPU-only numerical tests")
    args = parser.parse_args(argv)
    resume = resolve_resume_path(args.resume).resolve()
    if args.json.resolve() == resume or args.json.resolve().is_relative_to(resume.parent):
        parser.error("report must be separate from source checkpoint")
    device = "cpu" if args.cpu else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/HIP runtime unavailable")
    source_sha = digest(resume)
    checkpoint = load_checkpoint(resume, map_location="cpu")
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), *(root / "operators/rocm" / name for name in
             ("gpu_adam_ordered.py", "gpu_adam.py", "gpu_adam_bench.py")),
             root / "cat_yoko/optim.py", root / "cat_yoko/checkpoint.py"]
    metadata = {"source_checkpoint": str(resume), "source_sha256": source_sha, "source_step": checkpoint["extra"]["step"],
                "ordered_source_sha256": {str(path.relative_to(root)): digest(path) for path in files}}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    def save(report):
        write_report(args.json, {**report, **metadata})
    result = probe(checkpoint, device=device, on_report=save)
    if digest(resume) != source_sha:
        raise ValueError("source checkpoint changed during streamed numerical probe")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
