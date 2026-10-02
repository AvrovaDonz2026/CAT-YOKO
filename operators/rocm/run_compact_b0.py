#!/usr/bin/env python3
"""Experimental resident B0 continuation after compact-model parity validation.

The native BF16 model is built/upcycled on CPU, its 132-tensor B0 overlay is
loaded, and byte-identical frozen routed experts are compacted before GPU
placement. CPU Adam moments, the 8B-token curriculum, and trainable overlay
keys are retained. Resident training uses global clipping instead of the
baseline offload path's per-block clipping. This remains DummyStream training,
so this launcher makes no language-quality claim. Full checkpoints are not
saved; reconstruct the native model to continue into B1/B2.

Use model_bench.py to validate the full graph before using this launcher.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CAT_YOKO_TE_NVFP4", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cat_yoko.checkpoint import (
    is_trainable_ckpt, load_checkpoint, load_trainable_state, resolve_resume_path,
)
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.optim import trim_host_allocator
from cat_yoko.trainer import Trainer, build_model, seed_all
from cat_yoko.upcycle import upcycle_from_minicpm
from operators.rocm.compact_model import compact_frozen_moes
from operators.rocm.gpu_wait import wait_for_gpu_idle


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--run-steps", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--wait-gpu-idle", action="store_true",
                        help="after CPU setup, wait for other KFD GPU processes to exit")
    parser.add_argument("--gpu-idle-max-wait", type=float, default=None,
                        help="maximum idle wait in seconds; default waits without a limit")
    args = parser.parse_args(argv)
    if args.run_steps <= 0 or not 2 <= args.seq_len <= 4096:
        parser.error("run-steps must be positive; seq-len must be in [2,4096]")
    if args.save_every <= 0 or args.keep_last <= 0 or args.cpu_threads <= 0:
        parser.error("save-every, keep-last, and cpu-threads must be positive")
    if args.gpu_idle_max_wait is not None and args.gpu_idle_max_wait <= 0:
        parser.error("gpu-idle-max-wait must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("this launcher requires a ROCm PyTorch GPU runtime")
    if not (args.base / "model.safetensors").is_file():
        parser.error("base must contain the verified MiniCPM5 model.safetensors")
    resume = resolve_resume_path(args.resume)
    if resume.parent.resolve() == args.save_dir.resolve():
        parser.error("use a separate output directory to preserve the source overlay")
    checkpoint = load_checkpoint(resume, map_location="cpu")
    extra = checkpoint.get("extra") or {}
    if not is_trainable_ckpt(checkpoint) or extra.get("phase") != "B0" or extra.get("name") != "CAT-YOKO-12B":
        parser.error("resume must be a CAT-YOKO-12B B0 trainable overlay; B1/B2 are unsupported")
    cfg = replace(CATYokoConfig(**extra["cfg"]), use_nvfp4=False, use_fp8=False)
    seed = int(extra.get("seed", 0))
    source_step = int(extra["step"])
    source_tokens = float(extra["tokens_in_phase"])
    source_seen = float(extra.get("tokens_seen", source_tokens))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.save_dir / f"compact_manifest_from_{source_step}.json"
    manifest = {
        "status": "initializing", "experimental": True,
        "layout": "compact-identical-frozen-B0-MoE-resident",
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0), "base": str(args.base.resolve()),
        "source_checkpoint": str(resume.resolve()), "source_phase": "B0",
        "source_step": source_step, "source_tokens_in_phase": source_tokens,
        "source_tokens_seen": source_seen,
        "requested_updates": args.run_steps, "seq_len": args.seq_len,
        "phase_token_budget": 8e9, "seed": seed, "dtype": "bf16",
        "optimizer": "CPUOffloadAdamW", "optimizer_restored": False,
        "save_full": False, "save_trainable": True, "keep_last": args.keep_last,
        "offload_blocks": False, "offload_encoder": False,
        "grad_checkpoint": True, "gradient_clipping": "global",
        "stream_kind": "DummyStream",
        "wait_gpu_idle": args.wait_gpu_idle,
        "gpu_idle_max_wait_s": args.gpu_idle_max_wait,
        "notes": [
            "Only frozen, byte-identical routed experts are compacted; B0 overlay keys remain native.",
            "This launcher assumes full model parity was checked with model_bench.py.",
            "CPU Adam moments restart because the source overlay contains no optimizer state.",
            "Global clipping differs from the previous block-offload path's per-block clipping.",
            "BF16 expert reduction reassociation is approximate, not bitwise identical.",
            "DummyStream continuation is not evidence of improved language quality.",
            "B1/B2 require reconstruction of the original native model; full compact weights are not published.",
        ],
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps({"event": "start", **{key: manifest[key] for key in (
        "source_step", "source_tokens_in_phase", "requested_updates", "seq_len", "layout",
    )}}), flush=True)
    begin = time.perf_counter()
    torch.set_num_threads(args.cpu_threads)
    seed_all(seed)
    model = build_model(cfg, "cpu", dtype="bf16")
    source = load_minicpm_state(args.base)
    upcycle_from_minicpm(model, source, cfg)
    del source
    load_trainable_state(model, checkpoint["trainable"])
    del checkpoint
    apply_freeze(model, "B0")
    overlay_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(overlay_keys) != 132:
        raise RuntimeError(f"expected 132 native B0 trainables, got {len(overlay_keys)}")
    compact_report = compact_frozen_moes(model, phase="B0")
    gc.collect()
    trim_host_allocator()
    manifest.update(status="compacted", compact=compact_report,
                    initialization_elapsed_s=time.perf_counter() - begin)
    write_manifest(manifest_path, manifest)
    print(json.dumps({"event": "compacted", **compact_report}), flush=True)
    if args.wait_gpu_idle:
        manifest["status"] = "waiting_for_gpu"

        def report_wait(event: dict) -> None:
            manifest["gpu_wait"] = event
            write_manifest(manifest_path, manifest)
            print(json.dumps(event), flush=True)

        wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait, on_event=report_wait)
    model.to("cuda")
    manifest["status"] = "training"
    write_manifest(manifest_path, manifest)
    training_begin = time.perf_counter()
    result = Trainer(
        cfg, "B0", "cuda", reuse_model=model, resume=resume,
        run_steps=args.run_steps, tokens=8e9, seq_len=args.seq_len,
        micro_batch=1, accum=1, dtype="bf16", seed=seed,
        grad_ckpt=True, offload_encoder=False, offload_blocks=False, optim_cpu=True,
        save_dir=args.save_dir, save_every=args.save_every, save_keep=args.keep_last,
        save_full=False, save_trainable=True, save_optim=False,
        log_every=1, log_path=args.save_dir / "metrics.jsonl",
    ).run()
    summary = {
        "status": "complete", "experimental": True, "phase": result.phase,
        "step": result.step, "nll": result.nll, "tokens_seen": result.tokens_seen,
        "peak_mib": result.peak_mib, "source_step": source_step,
        "updates": result.step - source_step, "added_tokens": result.tokens_seen - source_seen,
        "elapsed_s": time.perf_counter() - begin,
        "training_elapsed_s": time.perf_counter() - training_begin,
        "seq_len": args.seq_len, "dtype": "bf16", "optimizer_restored": False,
        "stream_kind": "DummyStream", "layout": manifest["layout"],
    }
    manifest.update(summary)
    write_manifest(manifest_path, manifest)
    write_manifest(args.save_dir / "result.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
