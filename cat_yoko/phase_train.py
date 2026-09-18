"""Published C1 B0 / B1 / B2 CLIs.

Usage::

    python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
    python3 -m cat_yoko.b1 --resume checkpoints/b0 --save-dir checkpoints/b1
    python3 -m cat_yoko.b2 --resume checkpoints/b1 --save-dir checkpoints/b2

``--try`` is the 32GB path: seq=64, 32 optimizer steps, trainable.pt overlay (Hub, not GitHub).
Without ``--try`` the envelope is the published 8 / 27 / 15B tokens (H100-scale).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cat_yoko.phases import (
    PHASES,
    PUBLISHED_SAVE_EVERY,
    PUBLISHED_SEQ,
    TIGHT_GPU_SEQ,
    TRY_STEPS,
    spec as phase_spec,
)
from cat_yoko.recipe import MINICPM5_HF
from cat_yoko.train import TIGHT_12B_GPU_GIB, cuda_runtime_available, main as train_main


def _tight_gpu() -> bool:
    if not cuda_runtime_available():
        return False
    import torch

    gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return gib < TIGHT_12B_GPU_GIB


def build_phase_argv(phase: str, argv: list[str] | None = None) -> list[str]:
    ph = phase_spec(phase)
    p = argparse.ArgumentParser(
        prog=f"cat_yoko.{phase.lower()}",
        description=f"CAT-YOKO C1 {phase}: {ph.notes}",
    )
    p.add_argument(
        "--try",
        action="store_true",
        dest="try_run",
        help=f"32GB path: {TRY_STEPS} steps, seq={TIGHT_GPU_SEQ}, trainable.pt overlay only",
    )
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--tokens", type=float, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints") / phase.lower())
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--upcycle-hf", nargs="?", const=MINICPM5_HF, default=None)
    p.add_argument("--upcycle", type=Path, default=None)
    p.add_argument("--dummy-upcycle", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-last", type=int, default=2)
    p.add_argument(
        "--offload-encoder",
        action="store_true",
        help="force encoder CPU offload (B0/B1 spec default)",
    )
    p.add_argument(
        "--no-offload-encoder",
        action="store_true",
        help="keep encoder on GPU (83GiB 6000D published B0)",
    )
    p.add_argument("--offload-blocks", action="store_true")
    p.add_argument("--no-offload-blocks", action="store_true")
    p.add_argument("--optim-cpu", action="store_true")
    p.add_argument("--no-optim-cpu", action="store_true")
    p.add_argument("--log-every", type=int, default=None)
    args, rest = p.parse_known_args(argv)
    if args.offload_encoder and args.no_offload_encoder:
        p.error("pick one of --offload-encoder / --no-offload-encoder")
    if args.offload_blocks and args.no_offload_blocks:
        p.error("pick one of --offload-blocks / --no-offload-blocks")
    if args.optim_cpu and args.no_optim_cpu:
        p.error("pick one of --optim-cpu / --no-optim-cpu")

    out: list[str] = [
        "--config",
        "12b",
        "--phase",
        phase,
        "--device",
        args.device,
        "--dtype",
        "bf16",
        "--grad-ckpt",
        "--seed",
        str(args.seed),
        "--save-dir",
        str(args.save_dir),
        "--keep-last",
        str(args.keep_last),
        "--no-save-full",
        "--save-trainable",
        "--no-save-optim",
        "--micro-batch",
        "1",
        "--accum",
        "1",
    ]
    if args.no_offload_encoder:
        out.append("--no-offload-encoder")
    elif args.offload_encoder or ph.offload_encoder:
        out.append("--offload-encoder")
    if args.no_offload_blocks:
        out.append("--no-offload-blocks")
    elif args.offload_blocks or ph.offload_blocks:
        out.append("--offload-blocks")
    if args.no_optim_cpu:
        out.append("--no-optim-cpu")
    elif args.optim_cpu or ph.optim_cpu:
        out.append("--optim-cpu")
    tight = str(args.device).startswith("cuda") and _tight_gpu()
    if tight and not args.try_run and args.steps is None and args.tokens is None:
        p.error(
            f"{phase} published envelope is {ph.tokens:.0e} tokens; a "
            f"<{int(TIGHT_12B_GPU_GIB)}GiB GPU cannot finish it. "
            "Pass --try (32 steps, seq=64, trainable.pt overlay) "
            "or explicit --steps / --tokens."
        )
    seq = args.seq_len
    if seq is None and (args.try_run or tight):
        seq = TIGHT_GPU_SEQ
    elif seq is None and not args.try_run:
        seq = PUBLISHED_SEQ
    if seq is not None:
        out.extend(["--seq-len", str(seq)])
    if args.try_run:
        steps = TRY_STEPS if args.steps is None else args.steps
        out.extend(["--steps", str(steps)])
        every = 8 if args.save_every is None else args.save_every
        out.extend(["--save-every", str(every)])
    else:
        if args.steps is not None:
            out.extend(["--steps", str(args.steps)])
        elif args.tokens is not None:
            out.extend(["--tokens", str(args.tokens)])
        else:
            out.extend(["--tokens", str(ph.tokens)])
        every = PUBLISHED_SAVE_EVERY if args.save_every is None else args.save_every
        out.extend(["--save-every", str(every)])
    if args.log_every is not None:
        out.extend(["--log-every", str(args.log_every)])
    if args.data is not None:
        out.extend(["--data", str(args.data)])
    log = args.log if args.log is not None else args.save_dir / "metrics.jsonl"
    out.extend(["--log", str(log)])
    if args.resume is not None:
        out.extend(["--resume", str(args.resume)])
    n_up = sum(
        [
            bool(args.dummy_upcycle),
            args.upcycle is not None,
            args.upcycle_hf is not None,
        ]
    )
    if n_up > 1:
        p.error("pick one of --dummy-upcycle / --upcycle / --upcycle-hf")
    if args.dummy_upcycle:
        out.append("--dummy-upcycle")
    elif args.upcycle is not None:
        out.extend(["--upcycle", str(args.upcycle)])
    elif args.upcycle_hf is not None:
        out.extend(["--upcycle-hf", str(args.upcycle_hf)])
    elif args.try_run and (args.resume is None or phase in {"B1", "B2"}):
        # --try without --upcycle* uses dummy MiniCPM5 weights. B1/B2 overlay
        # resume is MiniCPM5 (or dummy) upcycle + load_trainable_state: the
        # previous overlay has no encoder/embed, so skipping upcycle leaves
        # them random. AutoDL scripts pass --upcycle-hf when MiniCPM5 is local.
        out.append("--dummy-upcycle")
    out.extend(rest)
    return out


def run_phase(phase: str, argv: list[str] | None = None) -> int:
    if phase not in PHASES:
        raise KeyError(phase)
    built = build_phase_argv(phase, argv)
    print(f"{phase} argv: {' '.join(built)}", flush=True)
    return train_main(built)


def main_b0(argv: list[str] | None = None) -> int:
    return run_phase("B0", argv)


def main_b1(argv: list[str] | None = None) -> int:
    return run_phase("B1", argv)


def main_b2(argv: list[str] | None = None) -> int:
    return run_phase("B2", argv)
