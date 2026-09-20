"""12B C1 trainer CLI. Tiny configs run on CPU; 12B uses --meta unless you have the RAM/GPU."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cat_yoko.config import CATYokoConfig, C1_SPLIT
from cat_yoko.data import resolve_eos
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.parallel import ParallelPlan, validate_parallel
from cat_yoko.phases import PHASES, resolve_phase_spec
from cat_yoko.recipe import MINICPM5_HF, assert_minicpm5_id
from cat_yoko.teacher import DummyTeacher, load_teacher
from cat_yoko.trainer import Trainer, build_model, print_meta, run_c1_chain, train_loop
from cat_yoko.upcycle import dummy_minicpm_state

TIGHT_12B_GPU_GIB = 40.0


def cuda_runtime_available() -> bool:
    """Import-safe CUDA probe so 12B CLI can refuse before ``build_model``."""
    import torch

    return bool(torch.cuda.is_available())


def twelve_b_cli_errors(
    *,
    seq_len: int | None,
    teacher_hf: bool,
    gpu_gib: float,
) -> list[str]:
    """Foot-guns on a 32GB card: default seq=4096 and MiniCPM5 teacher+student."""
    errs: list[str] = []
    if gpu_gib >= TIGHT_12B_GPU_GIB:
        return errs
    if seq_len is None:
        errs.append(
            f"12b on a {gpu_gib:.1f}GiB GPU needs --seq-len 64 (default 4096 OOMs activations); "
            "RTX 4080 SUPER peaks were measured at seq=64"
        )
    if teacher_hf:
        errs.append(
            f"12b + MiniCPM5 teacher needs >{TIGHT_12B_GPU_GIB:.0f}GiB GPU "
            f"(student ~23–28GiB + teacher ~5GiB); this card is {gpu_gib:.1f}GiB. "
            "Drop --teacher-hf."
        )
    return errs


def _tokens_offset(phase: str, explicit: float | None, *, use_kda: bool = False) -> float:
    if explicit is not None:
        return explicit
    ph = resolve_phase_spec(phase, use_kda=use_kda)
    if ph is not None:
        return float(ph.tokens_offset)
    if phase == "B0":
        return 0.0
    if phase == "B1":
        return C1_SPLIT["B0"]
    return C1_SPLIT["B0"] + C1_SPLIT["B1"]


def resolve_12b_accum(
    accum: int,
    *,
    tokens: float | None,
    offload_blocks: bool | None,
    phase: str,
    c1: bool,
) -> int:
    """Clamp 12B accum so B2/C1 block offload never auto-expands to 4M tokens.

    ``accum==0`` means auto. The offload path (explicit ``--offload-blocks`` or
    B2 / ``--c1`` default) uses 1 unless the user passed ``--accum >= 1``
    together with ``--no-offload-blocks``. Raising means both offload and
    accum>1, which B2 per-layer Adam cannot do.
    """
    offload_will = offload_blocks is True or (
        offload_blocks is None
        and (
            c1
            or phase == "B2"
            or bool(PHASES.get(phase) and PHASES[phase].offload_blocks)
        )
    )
    if offload_will:
        if accum > 1:
            raise ValueError(
                "B2 offload cannot 4M-token batch (accum>1); "
                "use --accum 1, or --no-offload-blocks for the 4M-token batch"
            )
        return 1 if accum <= 0 else accum
    if accum == 0 and tokens is None:
        return 1
    return accum


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cat-yoko-train", description="CAT-YOKO-12B C1 trainer")
    p.add_argument("--config", choices=["12b", "tiny"], default="tiny")
    p.add_argument("--phase", choices=sorted(PHASES), default="B0")
    p.add_argument(
        "--use-kda",
        action="store_true",
        help="implement 3:1 KDA in the graph (Phase B still window; C lights C-kda)",
    )
    p.add_argument("--steps", type=int, default=None, help="optimizer steps (tiny default 3)")
    p.add_argument("--tokens", type=float, default=None, help="phase token budget (overrides C1 split if set)")
    p.add_argument("--tokens-offset", type=float, default=None, help="global tokens already seen (WSD)")
    p.add_argument("--micro-batch", type=int, default=None, help="default 2 (tiny) / 1 (12b)")
    p.add_argument("--accum", type=int, default=0, help="0 = auto from global_batch_tokens")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=None, help="packed .bin (mmap) or jsonl tokens")
    p.add_argument("--eval-data", type=Path, default=None)
    p.add_argument("--eos", type=int, default=None, help="document break id for packed .bin")
    p.add_argument("--meta", action="store_true", help="12B param count on meta device")
    p.add_argument("--upcycle", type=Path, default=None, help="MiniCPM5-2B state_dict (.pt) or local HF dir")
    p.add_argument(
        "--upcycle-hf",
        nargs="?",
        const=MINICPM5_HF,
        default=None,
        help="MiniCPM5-2B Hub id or HF dir (default openbmb/MiniCPM5-2B-Base)",
    )
    p.add_argument("--dummy-upcycle", action="store_true")
    p.add_argument("--teacher", type=Path, default=None, help="pickled nn.Module teacher")
    p.add_argument(
        "--teacher-hf",
        nargs="?",
        const=MINICPM5_HF,
        default=None,
        help="MiniCPM5-2B teacher Hub id / HF dir for logit KD",
    )
    p.add_argument("--dummy-teacher", action="store_true", help="logit KD against a dummy teacher")
    p.add_argument("--fsdp", action="store_true", help="FSDP (torch backend; requires dist init)")
    p.add_argument("--ddp", action="store_true", help="DDP (also auto when WORLD_SIZE>1)")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="checkpoint file, or directory (latest.pt, then trainable.pt, then newest step_*)",
    )
    p.add_argument("--log", type=Path, default=None, help="jsonl metrics path")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=2, help="micro-batches per eval tick")
    p.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="override cfg seq_len; packed .bin is re-windowed when sidecar width differs",
    )
    p.add_argument("--grad-ckpt", action="store_true", help="activation checkpoint encoder/decoder blocks")
    p.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="disable activation checkpoint (B200 192GiB; 12b default is on)",
    )
    p.add_argument(
        "--c1-smoke",
        action="store_true",
        help="B0→B1→B2 on the same weights (ignores --phase); writes --save-dir/{B0,B1,B2}",
    )
    p.add_argument(
        "--c1",
        action="store_true",
        help="alias of --c1-smoke",
    )
    p.add_argument("--save-optim", action="store_true", help="store Adam moments in checkpoints")
    p.add_argument(
        "--no-save-optim",
        action="store_true",
        help="weights-only checkpoints (default for 12B)",
    )
    p.add_argument(
        "--keep-last",
        type=int,
        default=0,
        help="keep only N step_*.pt files (0 = keep all); latest.pt is always kept",
    )
    p.add_argument(
        "--save-full",
        action="store_true",
        help="write 23GiB full graphs (not for GitHub; default on for tiny)",
    )
    p.add_argument(
        "--no-save-full",
        action="store_true",
        help="skip full 23GiB graphs (default for 12B)",
    )
    p.add_argument(
        "--save-trainable",
        action="store_true",
        help="write trainable.pt overlay (default for 12B; B0 ~0.44GiB, Hub)",
    )
    p.add_argument(
        "--no-save-trainable",
        action="store_true",
        help="do not write trainable.pt",
    )
    p.add_argument("--offload-encoder", action="store_true", help="force B0/B1 encoder CPU offload")
    p.add_argument("--no-offload-encoder", action="store_true", help="disable encoder CPU offload")
    p.add_argument("--offload-blocks", action="store_true", help="force per-block CPU offload (B2)")
    p.add_argument("--no-offload-blocks", action="store_true", help="disable per-block CPU offload")
    p.add_argument("--optim-cpu", action="store_true", help="force AdamW moments on CPU")
    p.add_argument("--no-optim-cpu", action="store_true", help="keep AdamW moments on the param device")
    p.add_argument(
        "--backend",
        choices=["torch", "megatron", "deepspeed"],
        default="torch",
        help="torch = in-repo trainer; megatron = NVIDIA Megatron-LM hook; "
        "deepspeed = ZeRO (optional extra)",
    )
    p.add_argument(
        "--zero",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="DeepSpeed ZeRO stage (default 2 with --backend deepspeed; "
        "--zero-offload-param forces 3)",
    )
    p.add_argument(
        "--zero-offload",
        action="store_true",
        help="ZeRO-Offload: optimizer states on CPU (DeepSpeed, not native --optim-cpu)",
    )
    p.add_argument(
        "--zero-offload-param",
        action="store_true",
        help="ZeRO-3 param CPU offload; implies stage 3 + optimizer offload. 3090 48GiB path",
    )
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--cp", type=int, default=1)
    p.add_argument("--sequence-parallel", action="store_true")
    p.add_argument(
        "--dump-megatron",
        action="store_true",
        help="print Megatron TransformerConfig mapping JSON and exit",
    )
    p.add_argument(
        "--dump-deepspeed",
        action="store_true",
        help="print DeepSpeed ZeRO config JSON and exit (no deepspeed install)",
    )
    args = p.parse_args(argv)
    if args.backend == "megatron" and args.fsdp:
        p.error("--fsdp is the torch path; Megatron uses its own DDP/FSDP")
    if args.backend == "deepspeed" and (args.fsdp or args.ddp):
        p.error("--fsdp/--ddp is the torch path; DeepSpeed ZeRO cannot wrap DDP/FSDP")
    if args.backend == "megatron" and (
        args.zero is not None or args.zero_offload or args.zero_offload_param or args.dump_deepspeed
    ):
        p.error("ZeRO is --backend deepspeed, not megatron")
    if args.dump_megatron and args.dump_deepspeed:
        p.error("pick one of --dump-megatron / --dump-deepspeed")
    if args.c1_smoke or args.c1:
        if args.backend == "deepspeed" or args.zero_offload or args.zero_offload_param:
            p.error(
                "--c1-smoke rebuilds freeze each phase; DeepSpeed ZeRO cannot re-wrap "
                "in-process. Run B0/B1/B2 separately with --resume overlay"
            )
    zero_requested = (
        args.backend == "deepspeed"
        or args.zero is not None
        or args.zero_offload
        or args.zero_offload_param
    )
    if zero_requested and args.backend == "torch" and not args.dump_deepspeed:
        p.error("ZeRO needs --backend deepspeed (or --dump-deepspeed)")
    if args.zero_offload_param:
        args.zero = 3
        args.zero_offload = True
    elif args.backend == "deepspeed" and args.zero is None:
        args.zero = 2
    cfg = CATYokoConfig.tiny() if args.config == "tiny" else CATYokoConfig.middle_12b()
    if args.use_kda:
        cfg = replace(cfg, use_kda=True)
    plan = ParallelPlan(
        tensor_parallel=args.tp,
        pipeline_parallel=args.pp,
        expert_parallel=args.ep,
        context_parallel=args.cp,
        sequence_parallel=args.sequence_parallel,
    )
    validate_parallel(cfg, plan)
    if args.dump_megatron or args.backend == "megatron":
        from cat_yoko.megatron.provider import dump_mapping, run_pretrain

        if args.dump_megatron:
            dump_mapping(cfg, plan, args.phase)
            return 0
        return run_pretrain(cfg, plan, args.phase)
    if args.dump_deepspeed:
        from cat_yoko.deepspeed_zero import dump_zero_config

        dump_zero_config(
            stage=args.zero,
            offload_optimizer=bool(args.zero_offload or args.zero_offload_param),
            offload_param=bool(args.zero_offload_param),
            bf16=args.dtype == "bf16" or args.config == "12b",
            gradient_accumulation_steps=max(args.accum, 1),
            train_micro_batch_size_per_gpu=1 if args.config == "12b" else (args.micro_batch or 2),
            phase=args.phase,
        )
        return 0
    if args.meta:
        print_meta(cfg)
        return 0
    if args.offload_encoder and args.no_offload_encoder:
        p.error("pick one of --offload-encoder / --no-offload-encoder")
    if args.offload_blocks and args.no_offload_blocks:
        p.error("pick one of --offload-blocks / --no-offload-blocks")
    if args.optim_cpu and args.no_optim_cpu:
        p.error("pick one of --optim-cpu / --no-optim-cpu")
    if args.save_optim and args.no_save_optim:
        p.error("pick one of --save-optim / --no-save-optim")
    if args.save_full and args.no_save_full:
        p.error("pick one of --save-full / --no-save-full")
    if args.save_trainable and args.no_save_trainable:
        p.error("pick one of --save-trainable / --no-save-trainable")
    if args.grad_ckpt and args.no_grad_ckpt:
        p.error("pick one of --grad-ckpt / --no-grad-ckpt")
    args.c1_smoke = bool(args.c1_smoke or args.c1)
    if args.c1_smoke and args.resume is not None:
        p.error("--c1-smoke builds a fresh C1 chain; do not pass --resume")
    if args.resume is not None:
        from cat_yoko.checkpoint import peek_checkpoint_extra, resolve_resume_path
        from cat_yoko.kda import resolve_implemented_kda

        try:
            args.resume = resolve_resume_path(args.resume)
        except FileNotFoundError as exc:
            p.error(str(exc))
        try:
            args.use_kda = resolve_implemented_kda(
                cli=bool(args.use_kda), extra=peek_checkpoint_extra(args.resume)
            )
        except RuntimeError as exc:
            p.error(str(exc))
        if args.use_kda:
            cfg = replace(cfg, use_kda=True)
    if args.c1_smoke and args.tokens is not None:
        p.error("--c1-smoke is step-limited; do not pass --tokens")
    if args.config == "12b" and not str(args.device).startswith("cuda"):
        p.error(
            "12b training needs --device cuda --dtype bf16 "
            "(CPU is --meta / --dump-megatron / --dump-deepspeed only)"
        )
    if args.backend == "deepspeed" and not str(args.device).startswith("cuda"):
        p.error("DeepSpeed ZeRO needs --device cuda (CPU is --dump-deepspeed only)")
    if args.config == "12b" and str(args.device).startswith("cuda"):
        import torch

        if not cuda_runtime_available():
            p.error(
                "12b --device cuda requires a CUDA runtime "
                "(torch.cuda.is_available() is False); refusing to build"
            )
        gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
        for err in twelve_b_cli_errors(
            seq_len=args.seq_len,
            teacher_hf=args.teacher_hf is not None,
            gpu_gib=gpu_gib,
        ):
            p.error(err)
    if args.steps is None and args.tokens is None:
        if args.c1_smoke:
            args.steps = 1
        elif args.config == "tiny":
            args.steps = 3
        else:
            p.error("12b training needs --steps or --tokens (this VM cannot run the 8B-token B0 envelope)")
    if args.micro_batch is None:
        args.micro_batch = 1 if args.config == "12b" else 2
    offload_encoder = True if args.offload_encoder else (False if args.no_offload_encoder else None)
    offload_blocks = True if args.offload_blocks else (False if args.no_offload_blocks else None)
    optim_cpu = True if args.optim_cpu else (False if args.no_optim_cpu else None)
    if args.backend == "deepspeed":
        if offload_encoder is True or offload_blocks is True:
            p.error(
                "DeepSpeed ZeRO cannot mix native --offload-encoder/--offload-blocks; "
                "use --zero-offload-param"
            )
        if optim_cpu is True:
            p.error("DeepSpeed ZeRO cannot mix native --optim-cpu; use --zero-offload")
        offload_encoder = False
        offload_blocks = False
        optim_cpu = False
    save_optim = True if args.save_optim else (False if args.no_save_optim else None)
    save_full = True if args.save_full else (False if args.no_save_full else None)
    save_trainable = True if args.save_trainable else (False if args.no_save_trainable else None)
    if args.config == "12b":
        if args.dtype != "bf16":
            args.dtype = "bf16"
        if args.no_grad_ckpt:
            args.grad_ckpt = False
        else:
            args.grad_ckpt = True
        try:
            args.accum = resolve_12b_accum(
                args.accum,
                tokens=args.tokens,
                offload_blocks=offload_blocks,
                phase=args.phase,
                c1=args.c1_smoke,
            )
        except ValueError as exc:
            p.error(str(exc))
    n_up = sum(x is not None for x in (True if args.dummy_upcycle else None, args.upcycle, args.upcycle_hf))
    if n_up > 1:
        p.error("pick one of --dummy-upcycle / --upcycle / --upcycle-hf")
    n_t = sum(
        x is not None for x in (True if args.dummy_teacher else None, args.teacher, args.teacher_hf)
    )
    if n_t > 1:
        p.error("pick one of --dummy-teacher / --teacher / --teacher-hf")
    if args.upcycle_hf is not None:
        try:
            assert_minicpm5_id(str(args.upcycle_hf), kind="upcycle")
        except ValueError as exc:
            p.error(str(exc))
    if args.teacher_hf is not None:
        try:
            assert_minicpm5_id(str(args.teacher_hf), kind="teacher")
        except ValueError as exc:
            p.error(str(exc))
    src = None
    if args.dummy_upcycle:
        src = dummy_minicpm_state(cfg)
    elif args.upcycle is not None:
        src = load_minicpm_state(args.upcycle)
    elif args.upcycle_hf is not None:
        src = load_minicpm_state(args.upcycle_hf)
    teacher = None
    if args.dummy_teacher:
        teacher = DummyTeacher(cfg.vocab_size, cfg.hidden_size)
    elif args.teacher is not None:
        teacher = load_teacher(args.teacher, args.device)
    elif args.teacher_hf is not None:
        teacher = load_teacher(args.teacher_hf, args.device)
    eos = resolve_eos(args.data, args.eos)
    shared = dict(
        micro_batch=args.micro_batch,
        accum=args.accum,
        seed=args.seed,
        data=args.data,
        eval_data=args.eval_data,
        eos_id=eos,
        teacher=teacher,
        fsdp=args.fsdp,
        ddp=args.ddp,
        deepspeed=args.backend == "deepspeed",
        zero_stage=2 if args.zero is None else args.zero,
        zero_offload=bool(args.zero_offload),
        zero_offload_param=bool(args.zero_offload_param),
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        log_path=args.log,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        dtype=args.dtype,
        grad_ckpt=args.grad_ckpt,
        seq_len=args.seq_len,
        offload_encoder=offload_encoder,
        offload_blocks=offload_blocks,
        optim_cpu=optim_cpu,
        save_optim=save_optim,
        save_full=save_full,
        save_trainable=save_trainable,
        save_keep=args.keep_last,
    )
    if args.c1_smoke:
        run_c1_chain(
            cfg,
            args.device,
            steps=args.steps,
            upcycle_src=src,
            **shared,
        )
        return 0
    tr = Trainer(
        cfg,
        args.phase,
        args.device,
        steps=args.steps,
        tokens=args.tokens,
        upcycle_src=src,
        resume=args.resume,
        global_tokens_offset=_tokens_offset(
            args.phase, args.tokens_offset, use_kda=args.use_kda
        ),
        **shared,
    )
    tr.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
