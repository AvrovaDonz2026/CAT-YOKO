"""12B C1 trainer CLI. Tiny configs run on CPU; 12B uses --meta unless you have the RAM/GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cat_yoko.config import CATYokoConfig, C1_SPLIT
from cat_yoko.data import resolve_eos
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.parallel import ParallelPlan, validate_parallel
from cat_yoko.recipe import MINICPM_HF
from cat_yoko.teacher import DummyTeacher, load_teacher
from cat_yoko.trainer import Trainer, build_model, print_meta, run_c1_chain, train_loop
from cat_yoko.upcycle import dummy_minicpm_state


def _tokens_offset(phase: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    if phase == "B0":
        return 0.0
    if phase == "B1":
        return C1_SPLIT["B0"]
    return C1_SPLIT["B0"] + C1_SPLIT["B1"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAT-YOKO-12B C1 trainer")
    p.add_argument("--config", choices=["12b", "tiny"], default="tiny")
    p.add_argument("--phase", choices=["B0", "B1", "B2"], default="B0")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps (tiny default 3)")
    p.add_argument("--tokens", type=float, default=None, help="phase token budget (overrides C1 split if set)")
    p.add_argument("--tokens-offset", type=float, default=None, help="global tokens already seen (WSD)")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=0, help="0 = auto from global_batch_tokens")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=None, help="packed .bin (mmap) or jsonl tokens")
    p.add_argument("--eval-data", type=Path, default=None)
    p.add_argument("--eos", type=int, default=None, help="document break id for packed .bin")
    p.add_argument("--meta", action="store_true", help="12B param count on meta device")
    p.add_argument("--upcycle", type=Path, default=None, help="MiniCPM state_dict (.pt) or local HF dir")
    p.add_argument(
        "--upcycle-hf",
        nargs="?",
        const=MINICPM_HF,
        default=None,
        help="MiniCPM Hub id or HF dir (default openbmb/MiniCPM-2B-sft-bf16)",
    )
    p.add_argument("--dummy-upcycle", action="store_true")
    p.add_argument("--teacher", type=Path, default=None, help="pickled nn.Module teacher")
    p.add_argument(
        "--teacher-hf",
        nargs="?",
        const=MINICPM_HF,
        default=None,
        help="MiniCPM teacher Hub id / HF dir for logit KD",
    )
    p.add_argument("--dummy-teacher", action="store_true", help="logit KD against a dummy teacher")
    p.add_argument("--fsdp", action="store_true", help="FSDP (torch backend; requires dist init)")
    p.add_argument("--ddp", action="store_true", help="DDP (also auto when WORLD_SIZE>1)")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--log", type=Path, default=None, help="jsonl metrics path")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=None, help="override cfg seq_len; must match packed .bin")
    p.add_argument("--grad-ckpt", action="store_true", help="activation checkpoint encoder/decoder blocks")
    p.add_argument(
        "--c1-smoke",
        action="store_true",
        help="one step each of B0→B1→B2 on the same weights (ignores --phase)",
    )
    p.add_argument("--offload-encoder", action="store_true", help="force B0/B1 encoder CPU offload")
    p.add_argument("--no-offload-encoder", action="store_true", help="disable encoder CPU offload")
    p.add_argument("--offload-blocks", action="store_true", help="force per-block CPU offload (B2)")
    p.add_argument("--no-offload-blocks", action="store_true", help="disable per-block CPU offload")
    p.add_argument("--optim-cpu", action="store_true", help="force AdamW moments on CPU")
    p.add_argument("--no-optim-cpu", action="store_true", help="keep AdamW moments on the param device")
    p.add_argument(
        "--backend",
        choices=["torch", "megatron"],
        default="torch",
        help="torch = in-repo trainer; megatron = NVIDIA Megatron-LM hook",
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
    args = p.parse_args(argv)
    if args.backend == "megatron" and args.fsdp:
        p.error("--fsdp is the torch path; Megatron uses its own DDP/FSDP")
    cfg = CATYokoConfig.tiny() if args.config == "tiny" else CATYokoConfig.middle_12b()
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
    if args.meta:
        print_meta(cfg)
        return 0
    if args.offload_encoder and args.no_offload_encoder:
        p.error("pick one of --offload-encoder / --no-offload-encoder")
    if args.offload_blocks and args.no_offload_blocks:
        p.error("pick one of --offload-blocks / --no-offload-blocks")
    if args.optim_cpu and args.no_optim_cpu:
        p.error("pick one of --optim-cpu / --no-optim-cpu")
    if args.c1_smoke and args.resume is not None:
        p.error("--c1-smoke builds a fresh C1 chain; do not pass --resume")
    if args.c1_smoke and args.tokens is not None:
        p.error("--c1-smoke is step-limited; do not pass --tokens")
    if args.config == "12b" and not str(args.device).startswith("cuda"):
        p.error("12b training needs --device cuda --dtype bf16 (CPU is --meta / --dump-megatron only)")
    if args.steps is None and args.tokens is None:
        if args.c1_smoke:
            args.steps = 1
        elif args.config == "tiny":
            args.steps = 3
        else:
            p.error("12b training needs --steps or --tokens (this VM cannot run the 8B-token B0 envelope)")
    if args.config == "12b":
        if args.dtype != "bf16":
            args.dtype = "bf16"
        if not args.grad_ckpt:
            args.grad_ckpt = True
        if args.accum == 0 and args.tokens is None:
            # do not expand a smoke --steps run into the 4M-token global batch
            args.accum = 1
    offload_encoder = True if args.offload_encoder else (False if args.no_offload_encoder else None)
    offload_blocks = True if args.offload_blocks else (False if args.no_offload_blocks else None)
    optim_cpu = True if args.optim_cpu else (False if args.no_optim_cpu else None)
    n_up = sum(x is not None for x in (True if args.dummy_upcycle else None, args.upcycle, args.upcycle_hf))
    if n_up > 1:
        p.error("pick one of --dummy-upcycle / --upcycle / --upcycle-hf")
    n_t = sum(
        x is not None for x in (True if args.dummy_teacher else None, args.teacher, args.teacher_hf)
    )
    if n_t > 1:
        p.error("pick one of --dummy-teacher / --teacher / --teacher-hf")
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
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        log_path=args.log,
        eval_every=args.eval_every,
        dtype=args.dtype,
        grad_ckpt=args.grad_ckpt,
        seq_len=args.seq_len,
        offload_encoder=offload_encoder,
        offload_blocks=offload_blocks,
        optim_cpu=optim_cpu,
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
        global_tokens_offset=_tokens_offset(args.phase, args.tokens_offset),
        **shared,
    )
    tr.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
