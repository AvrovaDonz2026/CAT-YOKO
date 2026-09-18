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

from cat_yoko.phases import PHASES, TIGHT_GPU_SEQ, TRY_STEPS, spec as phase_spec
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
    args, rest = p.parse_known_args(argv)

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
    if ph.offload_encoder:
        out.append("--offload-encoder")
    if ph.offload_blocks:
        out.append("--offload-blocks")
    if ph.optim_cpu:
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
        if args.save_every is not None:
            out.extend(["--save-every", str(args.save_every)])
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
    elif args.resume is None and args.try_run:
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
