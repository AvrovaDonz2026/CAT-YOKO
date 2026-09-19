"""Published C1 B0 / B1 / B2 plus Phase C–G CLIs.

Usage::

    python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
    python3 -m cat_yoko.b1 --resume checkpoints/b0 --save-dir checkpoints/b1
    python3 -m cat_yoko.b2 --resume checkpoints/b1 --save-dir checkpoints/b2
    python3 -m cat_yoko.c --try --stage indexer --resume checkpoints/b2
    python3 -m cat_yoko.c --try --chain
    python3 -m cat_yoko.d --try --stage 8k
    python3 -m cat_yoko.d --try --chain
    python3 -m cat_yoko.e --try
    python3 -m cat_yoko.f --try
    python3 -m cat_yoko.g --try --algo grpo

``--try`` is the 32GB path: seq=64, 32 optimizer steps, trainable.pt overlay (Hub, not GitHub).
Without ``--try`` the envelope is the published token / step budget.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cat_yoko.phases import (
    C_STAGES,
    D_CHAIN,
    D_STAGES,
    G_ALGOS,
    PHASES,
    PUBLISHED_SAVE_EVERY,
    TIGHT_GPU_SEQ,
    TRY_STEPS,
    c_chain,
    resolve_phase_spec,
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
    p.add_argument("--eval-data", type=Path, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints") / phase.lower())
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--upcycle-hf", nargs="?", const=MINICPM5_HF, default=None)
    p.add_argument("--upcycle", type=Path, default=None)
    p.add_argument("--dummy-upcycle", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--use-kda",
        action="store_true",
        help="implement 3:1 KDA in the graph (Phase B still window; C lights C-kda)",
    )
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
    p.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="12b published default is 1. B200 can try 2 (~90GiB free at seq=4096).",
    )
    p.add_argument(
        "--grad-ckpt",
        action="store_true",
        dest="grad_ckpt",
        default=None,
        help="activation checkpoint (12b default on)",
    )
    p.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        dest="no_grad_ckpt",
        help="keep activations; B200 192GiB default in run_b0_full_b200.sh",
    )
    args, rest = p.parse_known_args(argv)
    ph = resolve_phase_spec(phase, use_kda=bool(args.use_kda)) or ph
    if args.offload_encoder and args.no_offload_encoder:
        p.error("pick one of --offload-encoder / --no-offload-encoder")
    if args.offload_blocks and args.no_offload_blocks:
        p.error("pick one of --offload-blocks / --no-offload-blocks")
    if args.optim_cpu and args.no_optim_cpu:
        p.error("pick one of --optim-cpu / --no-optim-cpu")
    if args.grad_ckpt and args.no_grad_ckpt:
        p.error("pick one of --grad-ckpt / --no-grad-ckpt")

    out: list[str] = [
        "--config",
        "12b",
        "--phase",
        phase,
        "--device",
        args.device,
        "--dtype",
        "bf16",
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
        str(max(int(args.micro_batch), 1)),
        "--accum",
        "1",
    ]
    if args.use_kda:
        out.append("--use-kda")
    if args.no_grad_ckpt:
        out.append("--no-grad-ckpt")
    else:
        out.append("--grad-ckpt")
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
        seq = ph.seq_len
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
        elif ph.tokens and ph.tokens > 0:
            out.extend(["--tokens", str(ph.tokens)])
        else:
            out.extend(["--steps", str(int(ph.default_steps or TRY_STEPS))])
        every = PUBLISHED_SAVE_EVERY if args.save_every is None else args.save_every
        out.extend(["--save-every", str(every)])
    if args.log_every is not None:
        out.extend(["--log-every", str(args.log_every)])
    if args.data is not None:
        out.extend(["--data", str(args.data)])
    if args.eval_data is not None:
        out.extend(["--eval-data", str(args.eval_data)])
    if args.eval_every is not None:
        out.extend(["--eval-every", str(args.eval_every)])
    if args.eval_batches is not None:
        out.extend(["--eval-batches", str(args.eval_batches)])
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
    elif args.try_run and (args.resume is None or phase != "B0"):
        # --try without --upcycle* uses dummy MiniCPM5 weights. Overlay
        # resume is MiniCPM5 (or dummy) upcycle + load_trainable_state: the
        # previous overlay may have no encoder/embed, so skipping upcycle
        # leaves them random. AutoDL scripts pass --upcycle-hf when MiniCPM5 is local.
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


def _peel_flag(argv: list[str] | None, flag: str, default: str) -> tuple[str, list[str]]:
    rest = list(argv or [])
    if flag in rest:
        i = rest.index(flag)
        if i + 1 >= len(rest):
            raise SystemExit(f"{flag} needs a value")
        val = rest[i + 1]
        del rest[i : i + 2]
        return val, rest
    return default, rest


def _strip_bool(argv: list[str] | None, flag: str) -> tuple[bool, list[str]]:
    rest = list(argv or [])
    if flag in rest:
        rest.remove(flag)
        return True, rest
    return False, rest


def _cli_chain(phases: tuple[str, ...], argv: list[str] | None) -> int:
    """Sequential CLI: each stage writes ``save_dir/{phase}`` and the next resumes it."""
    rest = list(argv or [])
    save, rest = _peel_flag(rest, "--save-dir", str(Path("checkpoints") / phases[0].split("-")[0].lower()))
    resume, rest = _peel_flag(rest, "--resume", "")
    prev = resume or None
    root = Path(save)
    rc = 0
    for phase in phases:
        extra = list(rest)
        extra.extend(["--save-dir", str(root / phase)])
        if prev:
            extra.extend(["--resume", prev])
        rc = run_phase(phase, extra)
        if rc:
            return rc
        prev = str(root / phase)
    return rc


def _argv_implemented_kda(cli: bool, argv: list[str]) -> bool:
    from cat_yoko.checkpoint import peek_checkpoint_extra
    from cat_yoko.kda import resolve_implemented_kda

    extra = None
    if "--resume" in argv:
        i = argv.index("--resume")
        if i + 1 < len(argv):
            extra = peek_checkpoint_extra(argv[i + 1])
    return resolve_implemented_kda(cli=cli, extra=extra)


def _inherit_kda_argv(argv: list[str] | None) -> tuple[bool, list[str]]:
    use_kda, rest = _strip_bool(argv, "--use-kda")
    try:
        use_kda = _argv_implemented_kda(use_kda, rest)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if use_kda:
        rest = ["--use-kda", *rest]
    return use_kda, rest


def main_c(argv: list[str] | None = None) -> int:
    chain, rest = _strip_bool(argv, "--chain")
    use_kda, rest = _inherit_kda_argv(rest)
    if chain:
        if "--stage" in rest:
            raise SystemExit(
                "--chain runs indexer→topk→hca→win "
                "(or kda→index→topk→hca→win after B implements --use-kda); "
                "do not pass --stage"
            )
        return _cli_chain(c_chain(use_kda=use_kda), rest)
    stage, rest = _peel_flag(rest, "--stage", "indexer")
    if stage not in C_STAGES:
        raise SystemExit(f"unknown C --stage {stage}; choose {sorted(C_STAGES)}")
    return run_phase(C_STAGES[stage], rest)


def main_d(argv: list[str] | None = None) -> int:
    chain, rest = _strip_bool(argv, "--chain")
    _, rest = _inherit_kda_argv(rest)
    if chain:
        if "--stage" in rest:
            raise SystemExit("--chain runs 8k→32k→128k; do not pass --stage")
        return _cli_chain(D_CHAIN, rest)
    stage, rest = _peel_flag(rest, "--stage", "8k")
    if stage not in D_STAGES:
        raise SystemExit(f"unknown D --stage {stage}; choose {sorted(D_STAGES)}")
    return run_phase(D_STAGES[stage], rest)


def main_e(argv: list[str] | None = None) -> int:
    _, rest = _inherit_kda_argv(argv)
    return run_phase("E", rest)


def main_f(argv: list[str] | None = None) -> int:
    _, rest = _inherit_kda_argv(argv)
    return run_phase("F", rest)


def main_g(argv: list[str] | None = None) -> int:
    algo, rest = _peel_flag(argv, "--algo", "grpo")
    if algo not in G_ALGOS:
        raise SystemExit(f"unknown G --algo {algo}; choose {sorted(G_ALGOS)}")
    return run_phase(G_ALGOS[algo], rest)
