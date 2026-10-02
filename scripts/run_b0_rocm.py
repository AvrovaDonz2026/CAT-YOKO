#!/usr/bin/env python3
"""Resume a B0 overlay on a small ROCm GPU using CPU initialization/offload.

Frozen weights come from the original MiniCPM5 base; only the B0 trainable
overlay is saved. The additional-step bound leaves the 8B token schedule
intact. Run in a separate save directory from the Hub source checkpoint.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.checkpoint import peek_checkpoint_extra, resolve_resume_path
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.optim import trim_host_allocator
from cat_yoko.trainer import Trainer, build_model, seed_all
from cat_yoko.upcycle import upcycle_from_minicpm


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
    args = parser.parse_args(argv)
    if args.run_steps <= 0 or args.seq_len < 2 or args.seq_len > 4096:
        parser.error("run-steps must be positive and seq-len must be in [2, 4096]")
    if not torch.version.hip or not torch.cuda.is_available():
        parser.error("this launcher requires a ROCm PyTorch GPU runtime")
    if not (args.base / "model.safetensors").is_file():
        parser.error("base must contain the verified MiniCPM5 model.safetensors")
    resume = resolve_resume_path(args.resume)
    if resume.parent.resolve() == args.save_dir.resolve():
        parser.error("use a separate output directory to preserve the source overlay")
    extra = peek_checkpoint_extra(resume)
    if not extra or extra.get("phase") != "B0" or extra.get("name") != "CAT-YOKO-12B":
        parser.error("resume must be a CAT-YOKO-12B B0 checkpoint")
    torch.set_num_threads(max(args.cpu_threads, 1))
    seed_all(int(extra.get("seed", 0)))
    cfg = replace(
        CATYokoConfig(**extra["cfg"]),
        use_nvfp4=False,
        use_fp8=False,
        use_kda=bool(extra.get("use_kda", False)),
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(json.dumps({
        "event": "start", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "hip": torch.version.hip,
        "resume": str(resume), "source_step": extra.get("step"),
        "source_tokens": extra.get("tokens_in_phase"),
        "seq_len": args.seq_len, "run_steps": args.run_steps,
        "precision": "bf16", "offload": "all transformer blocks and Adam",
    }), flush=True)
    # Build directly in BF16 on the host; never allocate the 12B graph in VRAM.
    model = build_model(cfg, "cpu", dtype="bf16")
    print(f"CPU model built in {time.time() - start:.1f}s", flush=True)
    source = load_minicpm_state(args.base)
    upcycle_from_minicpm(model, source, cfg)
    del source
    gc.collect()
    trim_host_allocator()
    apply_freeze(model, "B0")
    # Blocks stay on the host. Trainer reloads one block at a time for forward
    # and backward; embedding/cache/final head stay on the GPU.
    for name, module in model.named_children():
        if name not in {"encoder", "decoder"}:
            module.to("cuda")
    print(f"Base upcycled and staged in {time.time() - start:.1f}s", flush=True)
    trainer = Trainer(
        cfg, "B0", "cuda", reuse_model=model, resume=resume,
        run_steps=args.run_steps, tokens=8e9, seq_len=args.seq_len,
        micro_batch=1, accum=1, dtype="bf16", seed=int(extra.get("seed", 0)),
        grad_ckpt=True, offload_encoder=False, offload_blocks=True, optim_cpu=True,
        save_dir=args.save_dir, save_every=args.save_every, save_keep=args.keep_last,
        save_full=False, save_trainable=True, save_optim=False,
        log_every=1, log_path=args.save_dir / "metrics.jsonl",
    )
    result = trainer.run()
    summary = {
        "step": result.step, "nll": result.nll,
        "tokens_seen": result.tokens_seen, "peak_mib": result.peak_mib,
        "phase": result.phase, "elapsed_s": time.time() - start,
        "source_step": int(extra["step"]),
        "updates": result.step - int(extra["step"]),
        "added_tokens": result.tokens_seen - float(extra.get("tokens_seen", 0)),
        "seq_len": args.seq_len, "precision": "bf16",
        "stream_kind": "DummyStream", "optimizer_restored": False,
    }
    (args.save_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
