#!/usr/bin/env python3
"""Compare native and batched gradient clipping on a real B0 overlay's shapes.

Fresh seeded BF16 gradients are restored outside every timed invocation. Wall
timing includes CPU synchronization; HIP events include gaps caused by scalar
readbacks. Frozen model blocks, Adam, and data loading are excluded, so this
benchmark makes no whole-training throughput claim. It does no optimizer updates
and never saves or changes the input checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from cat_yoko.checkpoint import is_trainable_ckpt, load_checkpoint, resolve_resume_path
from cat_yoko.offload import clip_grad_norm_mixed as native_clip
from operators.rocm.grad_norm import clip_grad_norm_mixed as batched_clip
from operators.rocm.gpu_wait import wait_for_gpu_idle


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reset_gradients(parameters: list[nn.Parameter], templates: list[torch.Tensor]) -> None:
    for parameter, template in zip(parameters, templates):
        if parameter.grad is None:
            parameter.grad = template.clone()
        else:
            parameter.grad.copy_(template)


def parity(parameters, templates, names, threshold: float) -> dict:
    reset_gradients(parameters, templates)
    expected_norm = native_clip(parameters, threshold)
    reference = [parameter.grad.detach().cpu().clone() for parameter in parameters]
    reset_gradients(parameters, templates)
    actual_norm = batched_clip(parameters, threshold)
    mismatch = []
    for name, parameter, expected in zip(names, parameters, reference):
        actual = parameter.grad.detach().cpu()
        if not torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)):
            mismatch.append(name)
    return {"threshold": threshold, "native_norm": expected_norm, "batched_norm": actual_norm,
            "norm_exact": actual_norm == expected_norm, "gradient_bitwise": not mismatch,
            "gradient_tensors": len(reference), "failed_gradient_names": mismatch,
            "pass": actual_norm == expected_norm and not mismatch}


def measure(function, parameters, templates, threshold: float, device: torch.device,
            warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        reset_gradients(parameters, templates)
        function(parameters, threshold)
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        # Copy restoration and its synchronization are deliberately outside the
        # timer. This keeps repeated clipping from measuring changed gradients.
        reset_gradients(parameters, templates)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        norm = function(parameters, threshold)
        end.record()
        torch.cuda.synchronize(device)
        samples.append({"wall_ms": (time.perf_counter() - begin) * 1000,
                        "event_ms": start.elapsed_time(end), "norm": norm,
                        "incremental_peak_mib": (torch.cuda.max_memory_allocated(device) - before) / 1024**2,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2})
    return {"wall_ms_median": statistics.median(row["wall_ms"] for row in samples),
            "event_ms_median": statistics.median(row["event_ms"] for row in samples),
            "max_incremental_peak_mib": max(row["incremental_peak_mib"] for row in samples),
            "samples": samples}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--grad-scale", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--gpu-idle-max-wait", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be nonnegative; repeats must be positive")
    if not math.isfinite(args.grad_scale) or args.grad_scale <= 0:
        parser.error("grad-scale must be finite and positive")
    if args.gpu_idle_max_wait is not None and (not math.isfinite(args.gpu_idle_max_wait) or args.gpu_idle_max_wait <= 0):
        parser.error("gpu-idle-max-wait must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("device must be cuda or cuda:N")
    resume = resolve_resume_path(args.resume)
    if args.json.resolve() == resume.resolve() or args.json.resolve().is_relative_to(resume.parent.resolve()):
        parser.error("write benchmark JSON outside the input checkpoint directory")
    checkpoint = load_checkpoint(resume, map_location="cpu")
    extra = checkpoint.get("extra") or {}
    if (not is_trainable_ckpt(checkpoint) or extra.get("phase") != "B0"
            or extra.get("name") != "CAT-YOKO-12B"):
        parser.error("resume must be a native B0 trainable overlay")
    weights = checkpoint["trainable"]
    if len(weights) != 132 or any(value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all())
                                  for value in weights.values()):
        parser.error("expected 132 finite native BF16 B0 trainable tensors")
    report = {"status": "waiting" if args.wait_gpu_idle else "starting",
              "source_checkpoint": str(resume.resolve()), "source_sha256": file_digest(resume),
              "source_step": int(extra["step"]), "trainable_tensors": len(weights),
              "trainable_parameters": sum(value.numel() for value in weights.values()),
              "seed": args.seed, "gradient_generation": "CPU seeded FP32 normal, scaled, rounded to BF16",
              "grad_scale": args.grad_scale, "warmup": args.warmup, "repeats": args.repeats,
              "shape_dtype": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for name, value in weights.items()},
              "whole_training_speedup_measured": False, "events": []}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.json, report)

    def record(event):
        report["events"].append(event)
        write_report(args.json, report)
        print(json.dumps(event), flush=True)

    if args.wait_gpu_idle:
        wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait, on_event=record)
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("a ROCm PyTorch runtime is required")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    report.update(status="running", torch=torch.__version__, hip=torch.version.hip,
                  gpu=torch.cuda.get_device_name(device), device=str(device))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    names, parameters, templates = [], [], []
    for name, weight in weights.items():
        names.append(name)
        parameters.append(nn.Parameter(weight.to(device)))
        template = torch.randn(weight.shape, generator=generator, dtype=torch.float32)
        template.mul_(args.grad_scale)
        templates.append(template.to(dtype=torch.bfloat16).to(device))
    del checkpoint, weights, template
    gc.collect()
    reset_gradients(parameters, templates)
    uncut_norm = native_clip(parameters, float("inf"))
    if not math.isfinite(uncut_norm) or uncut_norm <= 0:
        raise ValueError("seeded gradients must have a finite positive global norm")
    cases = {"clipped": uncut_norm * 0.25, "unclipped": uncut_norm * 2.0}
    report["cases"] = {}
    for case, threshold in cases.items():
        equality = parity(parameters, templates, names, threshold)
        report["cases"][case] = {"parity": equality}
        record({"event": "parity", "case": case, **equality})
        if not equality["pass"]:
            report["status"] = "parity_failure"
            write_report(args.json, report)
            return 1
        native = measure(native_clip, parameters, templates, threshold, device, args.warmup, args.repeats)
        batched = measure(batched_clip, parameters, templates, threshold, device, args.warmup, args.repeats)
        report["cases"][case].update(native=native, batched=batched,
                                    wall_speedup=native["wall_ms_median"] / batched["wall_ms_median"])
        record({"event": "timing", "case": case,
                "native_wall_ms": native["wall_ms_median"], "batched_wall_ms": batched["wall_ms_median"],
                "native_event_ms": native["event_ms_median"], "batched_event_ms": batched["event_ms_median"]})
    report["status"] = "complete"
    report["notes"] = ["Norm kernels and gradient multiplication remain native and unfused.",
                       "Timing excludes gradient restoration, checkpoint loading, Adam, model forward/backward, and data.",
                       "Idle observation does not reserve the GPU or prove exclusive use throughout timing.",
                       "Event elapsed time includes GPU idle gaps while the host reads native scalar norms."]
    write_report(args.json, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
