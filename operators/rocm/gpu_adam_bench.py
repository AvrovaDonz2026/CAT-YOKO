#!/usr/bin/env python3
"""Check multi-step Adam arithmetic and time only optimizer updates.

Native and candidate receive identical pre-generated BF16 gradients. This is
an optimizer benchmark, not a full-model speedup measurement. Source weights,
moments and data are never changed; outputs go into a separate JSON file.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn

from cat_yoko.checkpoint import load_checkpoint, resolve_resume_path
from cat_yoko.optim import CPUOffloadAdamW
from operators.rocm.gpu_adam import use_gpu_fp32_adam
from operators.rocm.gpu_wait import wait_for_gpu_idle


def tensor_error(actual, expected, *, atol, rtol):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
    bitwise = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    if not finite:
        return {"finite": False, "bitwise": bitwise, "relative_l2": None, "max_abs": None, "within_tolerance": False}
    delta = actual.double() - expected.double()
    denominator = float(expected.double().square().sum())
    error = float(delta.square().sum())
    relative = math.sqrt(error / max(denominator, 1e-60))
    return {"finite": True, "bitwise": bitwise, "relative_l2": relative,
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            "within_tolerance": bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))}


def compare_optimizers(native, candidate, names, *, moment_atol=1e-8, moment_rtol=1e-6):
    rows, failures = {}, []
    native_parameters = [p for group in native.param_groups for p in group["params"]]
    candidate_parameters = [p for group in candidate.param_groups for p in group["params"]]
    if len(names) != len(native_parameters) or len(names) != len(candidate_parameters):
        raise ValueError("optimizer parameter/name mappings differ")
    for name, reference, actual in zip(names, native_parameters, candidate_parameters):
        weights = tensor_error(actual, reference, atol=0, rtol=0)
        moments = {key: tensor_error(candidate.state[actual][key], native.state[reference][key],
                                    atol=moment_atol, rtol=moment_rtol) for key in ("exp_avg", "exp_avg_sq")}
        steps_match = candidate.state[actual]["step"] == native.state[reference]["step"]
        passed = weights["finite"] and weights["bitwise"] and steps_match and all(
            row["finite"] and row["within_tolerance"] for row in moments.values())
        rows[name] = {"weights": weights, "moments": moments, "steps_match": steps_match, "pass": passed}
        if not passed:
            failures.append(name)
    return {"pass": not failures, "parameters": len(rows), "failed_names": failures, "tensors": rows,
            "gate": {"bf16_weights": "bitwise equality required after every step",
                     "fp32_moment_atol": moment_atol, "fp32_moment_rtol": moment_rtol},
            "all_weights_bitwise": all(row["weights"]["bitwise"] for row in rows.values()),
            "all_moments_bitwise": all(value["bitwise"] for row in rows.values() for value in row["moments"].values())}


def seeded_gradients(parameters, *, seed, scale):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for parameter in parameters:
        source = torch.randn(parameter.shape, generator=generator, dtype=torch.float32).mul_(scale)
        yield source.to(dtype=torch.bfloat16).to(parameter.device)


def assign_gradients(parameters, templates):
    for parameter, template in zip(parameters, templates):
        parameter.grad = template.clone()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(optimizer, parameters, templates, *, device, warmup, repeats):
    for _ in range(warmup):
        assign_gradients(parameters, templates)
        optimizer.step()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        assign_gradients(parameters, templates)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            initial = torch.cuda.memory_allocated(device)
        begin = time.perf_counter()
        optimizer.step()
        synchronize(device)
        row = {"wall_ms": (time.perf_counter() - begin) * 1000}
        if device.type == "cuda":
            row.update(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                       incremental_peak_mib=(torch.cuda.max_memory_allocated(device) - initial) / 2**20)
        samples.append(row)
    return {"wall_ms_median": statistics.median(row["wall_ms"] for row in samples), "samples": samples,
            "scope": "optimizer update and completion sync; gradient generation/restore, checkpoint I/O and model excluded"}


def build_optimizers(weights, state, device):
    native_params = {name: nn.Parameter(value.to(device=device).clone()) for name, value in weights.items()}
    candidate_params = {name: nn.Parameter(value.to(device=device).clone()) for name, value in weights.items()}
    def no_decay(name, value):
        parts = name.split(".")
        return value.ndim < 2 or parts[-1] == "bias" or any("norm" in part for part in parts) or "router" in parts
    names = [name for name,value in weights.items() if not no_decay(name,value)]
    names += [name for name,value in weights.items() if no_decay(name,value)]
    decay_count = sum(not no_decay(name,value) for name,value in weights.items())
    groups = [names[:decay_count], names[decay_count:]]
    def optimizer(parameters):
        values = [{"params": [parameters[name] for name in group], "weight_decay": decay}
                  for group, decay in zip(groups, (0.1, 0.0)) if group]
        return CPUOffloadAdamW(values, lr=1e-4, state_dtype=torch.float32, retain_state=True)
    native, candidate = optimizer(native_params), optimizer(candidate_params)
    if state is not None:
        # CPU loader can alias incoming states, so isolate the native copy.
        native.load_state_dict(copy.deepcopy(state))
        candidate.load_state_dict(copy.deepcopy(state))
    return native, candidate, names, [native_params[name] for name in names], [candidate_params[name] for name in names]


def write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--grad-scale", type=float, default=1e-4)
    parser.add_argument("--moment-atol", type=float, default=1e-8)
    parser.add_argument("--moment-rtol", type=float, default=1e-6)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--gpu-idle-max-wait", type=float, default=3600)
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("steps/repeats must be positive and warmup nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.grad_scale, args.moment_atol, args.moment_rtol, args.gpu_idle_max_wait)):
        parser.error("scale, tolerance and GPU wait must be finite and positive")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be CPU or CUDA/HIP")
    state, source_info = None, {"source": "small synthetic CPU-test shapes"}
    if args.resume is not None:
        resume = resolve_resume_path(args.resume).resolve()
        if args.json.resolve() == resume or args.json.resolve().is_relative_to(resume.parent):
            parser.error("JSON output must be separate from the source checkpoint")
        checkpoint = load_checkpoint(resume, map_location="cpu")
        extra = checkpoint.get("extra") or {}
        weights, state = checkpoint.get("trainable") or {}, checkpoint.get("optimizer")
        if checkpoint.get("kind") != "trainable" or extra.get("phase") != "B0" or extra.get("name") != "CAT-YOKO-12B":
            parser.error("resume must be a native B0 12B overlay")
        if len(weights) != 132 or any(value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()) for value in weights.values()):
            parser.error("expected 132 finite BF16 trainable tensors")
        ids = [parameter for group in (state or {}).get("param_groups", []) for parameter in group["params"]]
        if len(ids) != len(set(ids)) or len(ids) != 132 or set(ids) != set((state or {}).get("state", {})):
            parser.error("source must preserve all 132 native Adam states")
        def no_decay(name, value):
            parts = name.split(".")
            return value.ndim < 2 or parts[-1] == "bias" or any("norm" in part for part in parts) or "router" in parts
        ordered_weights = [[value for name,value in weights.items() if not no_decay(name,value)],
                           [value for name,value in weights.items() if no_decay(name,value)]]
        groups = state["param_groups"]
        if len(groups) != 2 or any(len(group["params"]) != len(values) for group,values in zip(groups,ordered_weights)):
            parser.error("source native decay groups do not match the model")
        shapes = {identifier: value.shape for group,values in zip(groups,ordered_weights)
                  for identifier,value in zip(group["params"],values)}
        counters = set()
        for identifier, values in state["state"].items():
            step = values.get("step")
            step = step.item() if torch.is_tensor(step) else step
            if not isinstance(step, (int, float)) or not math.isfinite(step) or step < 1 or int(step) != step:
                parser.error("source Adam step counters must be positive integers")
            counters.add(int(step))
            for key in ("exp_avg", "exp_avg_sq"):
                value = values.get(key)
                if not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    parser.error("source Adam moments must be finite CPU FP32")
                if value.shape != shapes[identifier]:
                    parser.error("source Adam moment shapes do not match the model")
                if key == "exp_avg_sq" and not bool((value >= 0).all()):
                    parser.error("source squared moments must be nonnegative")
        if len(counters) != 1:
            parser.error("source Adam step counters disagree")
        source_info = {"source_checkpoint": str(resume), "source_step": extra["step"],
                       "source_bytes": resume.stat().st_size}
        digest = hashlib.sha256()
        with resume.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        source_info["source_sha256"] = digest.hexdigest()
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        weights = {"projection.weight": torch.randn(32, 24, generator=generator).to(torch.bfloat16),
                   "norm.weight": torch.ones(24, dtype=torch.bfloat16)}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    if args.wait_gpu_idle and device.type == "cuda":
        wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/HIP runtime is unavailable")
    source_root = Path(__file__).resolve().parents[2]
    optimizer_sources = ("operators/rocm/gpu_adam.py", "operators/rocm/gpu_adam_bench.py",
                         "cat_yoko/optim.py")
    report = {**source_info, "status": "running", "torch": torch.__version__, "hip": torch.version.hip,
              "device": str(device), "steps": args.steps, "seed": args.seed, "whole_training_speedup_measured": False,
              "persistent_fp32_master": False, "parity": [], "arithmetic_variant": "native-order",
              "optimizer_source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                          for name in optimizer_sources}}
    write_report(args.json, report)
    native, candidate, names, native_params, candidate_params = build_optimizers(weights, state, device)
    del state, weights
    gc.collect()
    with use_gpu_fp32_adam(allow_cpu=device.type == "cpu", optimizers=[candidate]) as installation:
        for step in range(args.steps):
            templates = list(seeded_gradients(native_params, seed=args.seed + step, scale=args.grad_scale))
            assign_gradients(native_params, templates)
            assign_gradients(candidate_params, templates)
            native.step()
            candidate.step()
            result = compare_optimizers(native, candidate, names, moment_atol=args.moment_atol, moment_rtol=args.moment_rtol)
            result["invocation_step"] = step + 1
            report["parity"].append(result)
            write_report(args.json, report)
            if not result["pass"]:
                report.update(status="parity_failure", installation=installation.report())
                write_report(args.json, report)
                return 1
        native_timing = measure(native, native_params, templates, device=device, warmup=args.warmup, repeats=args.repeats)
        candidate_timing = measure(candidate, candidate_params, templates, device=device, warmup=args.warmup, repeats=args.repeats)
        terminal = compare_optimizers(native, candidate, names, moment_atol=args.moment_atol, moment_rtol=args.moment_rtol)
        snapshot = candidate.state_dict()
        exported = all(value.device.type == "cpu" and value.dtype == torch.float32
                       for state in snapshot["state"].values() for key,value in state.items() if key in ("exp_avg", "exp_avg_sq"))
        report.update(status="complete" if terminal["pass"] and exported else "terminal_failure",
                      native=native_timing, candidate=candidate_timing, terminal=terminal,
                      checkpoint_moments_cpu_fp32=exported, installation=installation.report(),
                      optimizer_wall_speedup=native_timing["wall_ms_median"] / candidate_timing["wall_ms_median"])
        write_report(args.json, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
