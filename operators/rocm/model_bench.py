#!/usr/bin/env python3
"""Compare full B0 arithmetic, then optionally train a resident MoE layout.

Build one native BF16 model on CPU, reconstruct its frozen backbone, overlay
the supplied B0 checkpoint, and record baseline losses and all 132 trainable
gradients using the production block-offload path. Replace only byte-identical
frozen experts in that same model, put the candidate model on GPU, and repeat
the same batches. Bounded training starts only after every declared parity
check passes. Outputs and overlays stay in an independent experiment directory.

The parity calculation does no optimizer updates. Resident training retains
CPU Adam and uses global gradient clipping; the previous offload run used
per-block clipping. This difference is recorded; production is not patched.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("CAT_YOKO_TE_NVFP4", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from cat_yoko.attention import reset_sdpa_counts, sdpa_counts
from cat_yoko.checkpoint import (
    is_trainable_ckpt, load_checkpoint, load_trainable_state, resolve_resume_path,
)
from cat_yoko.config import CATYokoConfig
from cat_yoko.data import DummyStream
from cat_yoko.freeze import apply_freeze, gate_schedule, set_gate
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.moe import MoE
from cat_yoko.offload import move_module, set_after_block_backward
from cat_yoko.optim import trim_host_allocator
from cat_yoko.trainer import Trainer, build_model, configure_cuda, seed_all
from cat_yoko.upcycle import upcycle_from_minicpm
from operators.rocm.compact_model import compact_frozen_moes
from operators.rocm.gpu_wait import wait_for_gpu_idle


def emit(report_path: Path, report: dict, event: dict) -> None:
    report.setdefault("events", []).append(event)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    display = {key: value for key, value in event.items() if key != "gradients"}
    print(json.dumps(display, allow_nan=False), flush=True)


@contextmanager
def parity_determinism(enabled: bool):
    """Require deterministic parity kernels, then restore the caller's policy."""
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def source_code_version(layout: str) -> dict:
    """Record content hashes even when the remote checkout has no Git metadata."""
    root = Path(__file__).resolve().parents[2]
    module = "compact_model.py" if layout == "compact" else "shared_storage_moe.py"
    paths = (Path(__file__).resolve(), root / "operators" / "rocm" / module,
             root / "cat_yoko" / "moe.py")
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=5, check=False)
        revision = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {"git_commit": revision, "file_sha256": hashes}


def install_moe_layout(model, layout: str) -> dict:
    if layout == "compact":
        return compact_frozen_moes(model, phase="B0")
    if layout == "shared-storage":
        from operators.rocm.shared_storage_moe import share_frozen_moe_storage

        return share_frozen_moe_storage(model, phase="B0")
    raise ValueError(f"unsupported MoE layout: {layout}")


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.detach().to(device="cpu", dtype=torch.float32)
    expected = expected.detach().to(device="cpu", dtype=torch.float32)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"finite": False, "max_abs": None, "relative_l2": None,
                "error_square_sum": None, "reference_square_sum": None}
    difference = actual - expected
    err_sq = float(difference.double().square().sum())
    ref_sq = float(expected.double().square().sum())
    return {
        "finite": True, "max_abs": float(difference.abs().max()),
        "relative_l2": math.sqrt(err_sq / max(ref_sq, 1e-60)),
        "error_square_sum": err_sq, "reference_square_sum": ref_sq,
    }


def capture(model, batch: dict, *, device: torch.device, offload: bool) -> tuple[dict, dict]:
    """Loss+backward with no clipping or optimizer callback; retain CPU snapshots."""
    model.zero_grad(set_to_none=True)
    set_after_block_backward(None)
    model.offload_blocks = offload
    model.offload_encoder = False
    model.grad_checkpoint = True
    model.return_logits = False
    model.train()
    reset_sdpa_counts()
    sampled_hidden = []
    seq_len = int(batch["input_ids"].shape[-1])
    positions = torch.linspace(0, seq_len - 1, min(8, seq_len), device=device).long().unique()

    def final_norm_hook(_module, _inputs, output):
        sampled_hidden.append(output.detach().index_select(1, positions).to("cpu"))

    hook = model.norm.register_forward_hook(final_norm_hook)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mib = torch.cuda.memory_allocated(device) / 1024**2
    begin = time.perf_counter()
    try:
        # Free HBM differs between layouts. Hold the CE GEMM chunk constant
        # for arithmetic parity so that the comparison isolates compaction.
        with patch("cat_yoko.loss._ce_chunk_tokens", return_value=min(seq_len - 1, 512)), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**batch)
            loss = output["loss"]
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begin
        row = {
            "seq_len": seq_len, "offload_blocks": offload,
            "loss": float(loss.detach()), "nll": float(output["nll"].detach()),
            "aux": float(output["aux"].detach()), "elapsed_s": elapsed,
            "tok_s": int(batch["input_ids"].numel()) / elapsed,
            "baseline_mib": baseline_mib,
            "peak_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "parity_ce_chunk_tokens": min(seq_len - 1, 512),
            "sdpa_counts": sdpa_counts(),
        }
        if len(sampled_hidden) != 1:
            raise RuntimeError(f"expected one final norm, saw {len(sampled_hidden)}")
        hidden = sampled_hidden[0]
        vocab = int(model.lm_head.weight.shape[0])
        selected_vocabulary = torch.linspace(0, vocab - 1, 256, device=device).long().unique()
        # Selected logits avoid allocating [4096,130560] solely for reporting.
        with torch.no_grad():
            logits = F.linear(
                hidden.to(device), model.lm_head.weight.index_select(0, selected_vocabulary)
            ) / model.logit_scale
        gradients = {}
        missing = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing.append(name)
            else:
                gradients[name] = parameter.grad.detach().to("cpu").clone()
        if missing:
            raise RuntimeError(f"missing B0 gradients: {missing}")
        row["gradient_tensors"] = len(gradients)
        snapshot = {"gradients": gradients, "selected_logits": logits.to("cpu"), "hidden": hidden}
        del output, loss, logits
        return row, snapshot
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        # Frozen B0 bias is never updated, but pending parity loads would
        # contaminate the first continuation step's utilization log.
        for module in model.modules():
            if isinstance(module, MoE):
                module.last_load = None
                module._load_n = 0
                module.last_aux = None
        gc.collect()
        torch.cuda.empty_cache()


def compare(reference: dict, candidate: dict, base_row: dict, row: dict, args) -> dict:
    if set(reference["gradients"]) != set(candidate["gradients"]):
        raise RuntimeError("candidate model changed trainable gradient names")
    errors, total_error, total_reference = {}, 0.0, 0.0
    for name, gradient in candidate["gradients"].items():
        error = tensor_error(gradient, reference["gradients"][name])
        error["pass"] = error["finite"] and error["relative_l2"] <= args.grad_relative_l2
        if error["finite"]:
            total_error += error["error_square_sum"]
            total_reference += error["reference_square_sum"]
        errors[name] = error
    global_error = math.sqrt(total_error / max(total_reference, 1e-60))
    logits = tensor_error(candidate["selected_logits"], reference["selected_logits"])
    hidden = tensor_error(candidate["hidden"], reference["hidden"])
    loss_abs = abs(row["loss"] - base_row["loss"])
    passed = (
        math.isfinite(loss_abs) and loss_abs <= args.loss_atol
        and global_error <= args.grad_relative_l2
        and all(error["pass"] for error in errors.values())
        and logits["finite"] and logits["relative_l2"] <= args.output_relative_l2
        and hidden["finite"] and hidden["relative_l2"] <= args.output_relative_l2
    )
    return {
        "event": "parity", "seq_len": row["seq_len"], "pass": passed,
        "moe_layout": getattr(args, "moe_layout", "compact"),
        "loss_abs": loss_abs, "nll_abs": abs(row["nll"] - base_row["nll"]),
        "global_gradient_relative_l2": global_error,
        "selected_logits": logits, "final_hidden": hidden,
        "failed_gradients": [name for name, error in errors.items() if not error["pass"]],
        "gradients": errors,
        "baseline": base_row, "candidate": row,
        # Keep the existing compact report key for consumers of earlier runs.
        **({"compact": row} if getattr(args, "moe_layout", "compact") == "compact" else {}),
        "forward_backward_speedup": base_row["elapsed_s"] / row["elapsed_s"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moe-layout", choices=("compact", "shared-storage"), default="compact",
                        help="compact changes expert arithmetic; shared-storage retains native dispatch")
    parser.add_argument("--parity-seqs", default="64,256,4096")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--run-steps", type=int, default=10)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--deterministic-parity", action="store_true",
                        help="require deterministic kernels for all reference/candidate parity; restore policy before training")
    parser.add_argument("--reference-repeat", action="store_true",
                        help="repeat the native 4096-token reference before installing the layout")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--wait-gpu-idle", action="store_true",
                        help="after CPU setup, wait for other KFD GPU processes to exit")
    parser.add_argument("--gpu-idle-max-wait", type=float, default=None,
                        help="maximum idle wait in seconds; default waits without a limit")
    parser.add_argument("--loss-atol", type=float, default=0.02)
    parser.add_argument("--grad-relative-l2", type=float, default=0.05)
    parser.add_argument("--output-relative-l2", type=float, default=0.05)
    args = parser.parse_args(argv)
    sequences = [int(part) for part in args.parity_seqs.split(",")]
    if not sequences or any(seq < 2 or seq > 4096 for seq in sequences):
        parser.error("parity-seqs must contain lengths in [2,4096]")
    if args.reference_repeat and 4096 not in sequences:
        parser.error("--reference-repeat requires 4096 in --parity-seqs")
    if args.run_steps <= 0 or args.seq_len < 2 or args.seq_len > 4096:
        parser.error("run-steps must be positive; seq-len must be in [2,4096]")
    if min(args.loss_atol, args.grad_relative_l2, args.output_relative_l2) <= 0:
        parser.error("parity tolerances must be positive")
    if args.gpu_idle_max_wait is not None and args.gpu_idle_max_wait <= 0:
        parser.error("gpu-idle-max-wait must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("a ROCm PyTorch GPU runtime is required")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("device must be cuda or cuda:N under ROCm")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    resume = resolve_resume_path(args.resume)
    if args.out.resolve() == resume.parent.resolve():
        parser.error("use an output directory separate from the source checkpoint")
    checkpoint = load_checkpoint(resume, map_location="cpu")
    extra = checkpoint.get("extra") or {}
    if not is_trainable_ckpt(checkpoint) or extra.get("phase") != "B0" or extra.get("name") != "CAT-YOKO-12B":
        parser.error("resume must be a CAT-YOKO-12B B0 trainable overlay")
    cfg = replace(CATYokoConfig(**extra["cfg"]), use_nvfp4=False, use_fp8=False)
    seed = int(extra.get("seed", 0))
    torch.set_num_threads(max(args.cpu_threads, 1))
    seed_all(seed)
    configure_cuda()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "parity.json"
    report = {
        "status": "running", "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "source_checkpoint": str(resume), "source_step": int(extra["step"]),
        "source_tokens_in_phase": float(extra["tokens_in_phase"]),
        "dtype": "bf16", "parity_sequences": sequences,
        "moe_layout": args.moe_layout,
        "deterministic_parity": args.deterministic_parity,
        "reference_repeat": args.reference_repeat,
        "original_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "original_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "source_code_version": source_code_version(args.moe_layout),
        "updates": 0,
        "wait_gpu_idle": args.wait_gpu_idle,
        "gpu_idle_max_wait_s": args.gpu_idle_max_wait,
        "thresholds": {"loss_atol": args.loss_atol, "gradient_relative_l2_per_tensor_and_global": args.grad_relative_l2,
                       "selected_output_relative_l2": args.output_relative_l2},
        "notes": [
            "One native CPU model is measured, receives the selected MoE layout in place, and moves to GPU.",
            "Parity uses the same source overlay and batch without optimizer updates or clipping.",
            ("Compaction reassociates BF16 expert reductions; all gradient tensors are reported."
             if args.moe_layout == "compact" else
             "Shared-storage retains native MoE dispatch and reduction order; stride-zero weight broadcasting is unvalidated until parity passes."),
            "Resident training retains CPU Adam; global clipping differs from baseline per-block clipping.",
            "Optimizer moments restart because the source B0 overlay contains only trainable weights.",
            "Flash remains disabled by the production ROCm math SDPA path.",
        ],
    }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    emit(report_path, report, {"event": "preflight", "free_gpu_mib": free_bytes / 1024**2,
                               "total_gpu_mib": total_bytes / 1024**2})
    with parity_determinism(args.deterministic_parity):
        report["parity_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        begin = time.perf_counter()
        model = build_model(cfg, "cpu", dtype="bf16")
        source = load_minicpm_state(args.base)
        upcycle_from_minicpm(model, source, cfg)
        del source
        load_trainable_state(model, checkpoint["trainable"])
        del checkpoint
        apply_freeze(model, "B0")
        trainable_parameters = {name: parameter for name, parameter in model.named_parameters()
                                if parameter.requires_grad}
        trainable_names = set(trainable_parameters)
        if len(trainable_names) != 132:
            raise RuntimeError(f"expected 132 B0 trainables, got {len(trainable_names)}")
        trim_host_allocator()
        emit(report_path, report, {"event": "model_built", "elapsed_s": time.perf_counter() - begin,
                                   "trainable_tensors": len(trainable_names)})
        if args.wait_gpu_idle:
            report["status"] = "waiting_for_gpu"
            wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait,
                              on_event=lambda event: emit(report_path, report, event))
            report["status"] = "running"
        for name, module in model.named_children():
            if name not in {"encoder", "decoder"}:
                module.to(device)
        gate = gate_schedule("B0", min((float(extra["tokens_in_phase"]) + 1) / 8e9, 1.0))
        set_gate(model, gate)
        report["parity_gate"] = gate
        references = {}
        batches = {}
        reference_stable = True
        for seq in sequences:
            stream = DummyStream(cfg.vocab_size, seq, seed)
            if extra.get("stream") is not None:
                stream.load_state_dict(extra["stream"])
            batch = stream.batch(1, str(device))
            row, snapshot = capture(model, batch, device=device, offload=True)
            references[seq] = (row, snapshot)
            batches[seq] = batch
            emit(report_path, report, {"event": "baseline", **row})
            if args.reference_repeat and seq == 4096:
                repeated_row, repeated_snapshot = capture(model, batch, device=device, offload=True)
                stability = compare(snapshot, repeated_snapshot, row, repeated_row, args)
                stability["event"] = "reference_repeat"
                stability["moe_layout"] = "native-reference-repeat"
                emit(report_path, report, stability)
                reference_stable = reference_stable and stability["pass"]
                del repeated_snapshot
                trim_host_allocator()
        for block in list(model.encoder) + list(model.decoder):
            move_module(block, "cpu")
        layout_report = install_moe_layout(model, args.moe_layout)
        trainable_after = {name: parameter for name, parameter in model.named_parameters()
                           if parameter.requires_grad}
        if set(trainable_after) != trainable_names or any(
            trainable_after[name] is not parameter for name, parameter in trainable_parameters.items()
        ):
            raise RuntimeError("MoE layout installer changed the native 132 B0 trainable names or identities")
        trim_host_allocator()
        torch.cuda.empty_cache()
        emit(report_path, report, {"event": "moe_layout_installed", "moe_layout": args.moe_layout,
                                   **layout_report})
        model.to(device)
        passed = reference_stable
        for seq in sequences:
            base_row, reference = references.pop(seq)
            row, candidate = capture(model, batches.pop(seq), device=device, offload=False)
            parity = compare(reference, candidate, base_row, row, args)
            emit(report_path, report, parity)
            passed = passed and parity["pass"]
            del reference, candidate
            trim_host_allocator()
        report["status"] = "parity_pass" if passed else "parity_failure"
        emit(report_path, report, {"event": "parity_complete", "pass": passed})
    emit(report_path, report, {
        "event": "parity_determinism_restored",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    })
    if not passed:
        return 1
    if args.parity_only:
        return 0
    training_dir = args.out / "train"
    training_start = time.perf_counter()
    result = Trainer(
        cfg, "B0", str(device), reuse_model=model, resume=resume,
        run_steps=args.run_steps, tokens=8e9, seq_len=args.seq_len,
        micro_batch=1, accum=1, dtype="bf16", seed=seed,
        grad_ckpt=True, offload_encoder=False, offload_blocks=False, optim_cpu=True,
        save_dir=training_dir, save_every=args.save_every, save_keep=args.keep_last,
        save_full=False, save_trainable=True, save_optim=False,
        log_every=1, log_path=training_dir / "metrics.jsonl",
    ).run()
    report["status"] = "training_complete"
    report["updates"] = result.step - int(extra["step"])
    emit(report_path, report, {
        "event": "training_complete", "step": result.step, "nll": result.nll,
        "tokens_seen": result.tokens_seen, "peak_mib": result.peak_mib,
        "updates": result.step - int(extra["step"]),
        "training_elapsed_s": time.perf_counter() - training_start,
        "save_dir": str(training_dir), "optimizer_restored": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
