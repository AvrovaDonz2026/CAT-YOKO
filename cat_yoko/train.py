"""12B C1 trainer. Tiny configs run on CPU; 12B uses --meta unless you have the RAM/GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from cat_yoko.config import CATYokoConfig
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze, gate_schedule, set_gate
from cat_yoko.fsdp import wrap_fsdp
from cat_yoko.loss import kd_kl, kd_weight
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.parallel import ParallelPlan, validate_parallel
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm


def build_model(cfg: CATYokoConfig, device: str) -> CATYokoForCausalLM:
    if device == "meta":
        with torch.device("meta"):
            model = CATYokoForCausalLM(cfg)
        return model
    model = CATYokoForCausalLM(cfg)
    return model.to(device)


def dummy_batch(cfg: CATYokoConfig, device: str, batch: int = 2) -> dict[str, torch.Tensor]:
    ids = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len), device=device)
    return {"input_ids": ids, "labels": ids.clone()}


def wsd_lr(step: int, tokens_seen: float, cfg: CATYokoConfig, phase: str) -> float:
    base = cfg.lr_b2 if phase == "B2" else cfg.lr
    if tokens_seen < cfg.warmup_tokens:
        return base * max(tokens_seen / cfg.warmup_tokens, 1e-3)
    return base


class DummyTeacher(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embed(input_ids))}


def train_loop(
    cfg: CATYokoConfig,
    phase: str,
    steps: int,
    device: str,
    *,
    upcycle_src: dict | None = None,
    teacher: nn.Module | None = None,
    fsdp: bool = False,
) -> float:
    model = build_model(cfg, device)
    if device == "meta":
        bd = model.param_breakdown()
        n = bd["total"]
        print(f"meta {cfg.name}: {n:,} params ({n / 1e9:.3f}B)")
        for k, v in bd.items():
            if k != "total":
                print(f"  {k:8s} {v / 1e9:.3f}B")
        return 0.0
    if upcycle_src is not None:
        upcycle_from_minicpm(model, upcycle_src, cfg)
    apply_freeze(model, phase)
    model = wrap_fsdp(model, enabled=fsdp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(
        trainable,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    model.train()
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    last = 0.0
    tokens_seen = 0.0
    use_fp8 = should_autocast(phase, cuda=device.startswith("cuda"), enabled=cfg.use_fp8)
    for step in range(steps):
        batch = dummy_batch(cfg, device)
        tokens_seen += batch["input_ids"].numel()
        set_gate(model, gate_schedule(phase, (step + 1) / steps))
        lr = wsd_lr(step, tokens_seen, cfg, phase)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        if use_fp8:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
        else:
            out = model(**batch)
        loss = out["loss"]
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(batch["input_ids"])["logits"]
            w = kd_weight(step, steps, cfg.kd_weight_start)
            if w > 0:
                loss = loss + w * kd_kl(
                    out["logits"][:, :-1],
                    t_logits[:, :-1],
                    cfg.kd_temperature,
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()
        last = float(out["nll"].detach())
        if step == 0 or step + 1 == steps:
            n_train = sum(p.numel() for p in trainable)
            print(
                f"{cfg.name} {phase} step {step + 1}/{steps} nll={last:.4f} "
                f"gate={float(model.decoder[0].gate):.3f} "
                f"trainable={n_train / 1e6:.2f}M lr={lr:.2e} fp8={use_fp8}"
            )
    return last


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAT-YOKO-12B C1 trainer")
    p.add_argument("--config", choices=["12b", "tiny"], default="tiny")
    p.add_argument("--phase", choices=["B0", "B1", "B2"], default="B0")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--meta", action="store_true", help="12B param count on meta device")
    p.add_argument("--upcycle", type=Path, default=None, help="MiniCPM state_dict path (.pt)")
    p.add_argument("--dummy-upcycle", action="store_true")
    p.add_argument("--dummy-teacher", action="store_true", help="logit KD against a dummy teacher")
    p.add_argument("--fsdp", action="store_true", help="wrap with FSDP (torch backend; requires dist init)")
    p.add_argument(
        "--backend",
        choices=["torch", "megatron"],
        default="torch",
        help="torch = in-repo reference graph; megatron = NVIDIA Megatron-LM hook (optional extra)",
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
    device = "meta" if args.meta else args.device
    src = None
    if args.dummy_upcycle:
        src = dummy_minicpm_state(cfg)
    elif args.upcycle is not None:
        src = torch.load(args.upcycle, map_location="cpu", weights_only=True)
    teacher = None
    if args.dummy_teacher and device != "meta":
        teacher = DummyTeacher(cfg.vocab_size, cfg.hidden_size).to(device)
    train_loop(
        cfg,
        args.phase,
        args.steps,
        device,
        upcycle_src=src,
        teacher=teacher,
        fsdp=args.fsdp,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
