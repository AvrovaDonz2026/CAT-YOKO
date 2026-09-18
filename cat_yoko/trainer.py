"""C1 Phase B trainer: packing, accum, WSD, freeze, checkpoint, DDP/FSDP."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cat_yoko.checkpoint import load_checkpoint, load_model_state, save_checkpoint
from cat_yoko.config import CATYokoConfig
from cat_yoko.data import open_stream
from cat_yoko.dist_util import barrier, init_distributed, is_rank0, wrap_distributed
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze, gate_schedule, set_gate
from cat_yoko.loss import kd_kl, kd_weight
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import build_optimizer, unwrap, wsd_lr
from cat_yoko.upcycle import upcycle_from_minicpm


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: CATYokoConfig, device: str) -> CATYokoForCausalLM:
    if device == "meta":
        with torch.device("meta"):
            return CATYokoForCausalLM(cfg)
    model = CATYokoForCausalLM(cfg)
    return model.to(device)


def print_meta(cfg: CATYokoConfig) -> None:
    model = build_model(cfg, "meta")
    bd = model.param_breakdown()
    n = bd["total"]
    print(f"meta {cfg.name}: {n:,} params ({n / 1e9:.3f}B)")
    for k, v in bd.items():
        if k != "total":
            print(f"  {k:8s} {v / 1e9:.3f}B")


def auto_accum(cfg: CATYokoConfig, micro_batch: int, world: int) -> int:
    per = micro_batch * cfg.seq_len * max(world, 1)
    return max(int(cfg.global_batch_tokens // per), 1)


@dataclass
class TrainResult:
    nll: float
    step: int
    tokens_seen: float


class Trainer:
    def __init__(
        self,
        cfg: CATYokoConfig,
        phase: str,
        device: str,
        *,
        steps: int | None = None,
        tokens: float | None = None,
        micro_batch: int = 2,
        accum: int = 1,
        seed: int = 0,
        data: Path | None = None,
        eval_data: Path | None = None,
        eos_id: int | None = None,
        upcycle_src: dict | None = None,
        teacher: nn.Module | None = None,
        fsdp: bool = False,
        ddp: bool = False,
        save_dir: Path | None = None,
        save_every: int = 0,
        resume: Path | None = None,
        log_every: int = 1,
        log_path: Path | None = None,
        eval_every: int = 0,
        dtype: str = "fp32",
        global_tokens_offset: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.phase = phase
        self.steps = steps
        self.tokens_target = tokens
        self.micro_batch = micro_batch
        self.seed = seed
        self.data = data
        self.eval_data = eval_data
        self.eos_id = eos_id
        self.upcycle_src = upcycle_src
        self.teacher = teacher
        self.fsdp = fsdp
        self.ddp = ddp
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = save_every
        self.resume = Path(resume) if resume else None
        self.log_every = max(log_every, 1)
        self.log_path = Path(log_path) if log_path else None
        self.eval_every = eval_every
        self.dtype = dtype
        self.global_tokens_offset = global_tokens_offset
        self.device, self.rank, self.world = init_distributed(device)
        self.accum = auto_accum(cfg, micro_batch, self.world) if accum <= 0 else max(accum, 1)

    def _cast(self, model: nn.Module) -> nn.Module:
        if self.dtype == "bf16":
            return model.to(dtype=torch.bfloat16)
        return model

    def _log(self, row: dict) -> None:
        if not is_rank0(self.rank):
            return
        line = (
            f"{row['name']} {row['phase']} step {row['step']}/{row['steps_or_inf']} "
            f"nll={row['nll']:.4f} gate={row['gate']:.3f} "
            f"trainable={row['trainable_m']:.2f}M lr={row['lr']:.2e} "
            f"fp8={row['fp8']} tok={row['tokens_seen']:.0f}"
        )
        print(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def _maybe_save(self, model: nn.Module, opt, extra: dict, tag: str) -> None:
        if self.save_dir is None or not is_rank0(self.rank):
            return
        save_checkpoint(
            self.save_dir / tag,
            model=model,
            optimizer=opt,
            extra=extra,
        )

    @torch.no_grad()
    def _eval_nll(self, model: nn.Module, batches: int = 2) -> float:
        if self.eval_data is None and batches <= 0:
            return float("nan")
        stream = open_stream(
            self.eval_data,
            self.cfg.vocab_size,
            self.cfg.seq_len,
            seed=self.seed + 1,
            eos_id=self.eos_id,
        )
        raw = unwrap(model)
        raw.eval()
        total = 0.0
        n = 0
        for _ in range(max(batches, 1)):
            batch = stream.batch(self.micro_batch, self.device)
            out = raw(**batch)
            total += float(out["nll"])
            n += 1
        raw.train()
        return total / max(n, 1)

    def run(self) -> TrainResult:
        seed_all(self.seed + self.rank)
        model = build_model(self.cfg, self.device)
        model = self._cast(model)
        if self.upcycle_src is not None:
            upcycle_from_minicpm(unwrap(model), self.upcycle_src, self.cfg)
        apply_freeze(unwrap(model), self.phase)
        model = wrap_distributed(unwrap(model), fsdp=self.fsdp, ddp=self.ddp)
        opt = build_optimizer(model, self.cfg)
        stream = open_stream(
            self.data,
            self.cfg.vocab_size,
            self.cfg.seq_len,
            seed=self.seed,
            eos_id=self.eos_id,
        )
        step = 0
        tokens_in_phase = 0.0
        tokens_seen = self.global_tokens_offset
        if self.resume is not None:
            ckpt = load_checkpoint(self.resume, map_location="cpu")
            load_model_state(model, ckpt["model"])
            if ckpt.get("optimizer") is not None:
                opt.load_state_dict(ckpt["optimizer"])
            extra = ckpt.get("extra") or {}
            step = int(extra.get("step", 0))
            tokens_in_phase = float(extra.get("tokens_in_phase", 0.0))
            tokens_seen = float(extra.get("tokens_seen", tokens_seen))
            self.phase = str(extra.get("phase", self.phase))
            apply_freeze(unwrap(model), self.phase)

        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        unwrap(model).train()
        use_fp8 = should_autocast(
            self.phase, cuda=self.device.startswith("cuda"), enabled=self.cfg.use_fp8
        )
        last = 0.0
        phase_budget = self.tokens_target
        max_steps = self.steps
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)

        while True:
            if max_steps is not None and step >= max_steps:
                break
            if phase_budget is not None and tokens_in_phase >= phase_budget:
                break
            progress = 0.0
            if max_steps:
                progress = (step + 1) / max_steps
            elif phase_budget:
                progress = min((tokens_in_phase + 1) / phase_budget, 1.0)
            set_gate(unwrap(model), gate_schedule(self.phase, progress))
            lr = wsd_lr(tokens_seen, self.cfg, self.phase)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            step_nll = 0.0
            step_loss = 0.0
            step_tokens = 0
            for _ in range(self.accum):
                batch = stream.batch(self.micro_batch, self.device)
                step_tokens += int(batch["input_ids"].numel())
                if use_fp8:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(**batch)
                        loss = out["loss"] / self.accum
                else:
                    out = model(**batch)
                    loss = out["loss"] / self.accum
                if self.teacher is not None:
                    with torch.no_grad():
                        t_logits = self.teacher(batch["input_ids"])["logits"]
                    w = kd_weight(
                        step,
                        max_steps or 1,
                        self.cfg.kd_weight_start,
                    )
                    if w > 0:
                        loss = loss + w * kd_kl(
                            out["logits"][:, :-1],
                            t_logits[:, :-1],
                            self.cfg.kd_temperature,
                        )
                loss.backward()
                step_nll += float(out["nll"].detach()) / self.accum
                step_loss += float(out["loss"].detach()) / self.accum
            torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip)
            opt.step()
            step += 1
            tokens_in_phase += step_tokens * self.world
            tokens_seen += step_tokens * self.world
            last = step_nll
            extra = {
                "phase": self.phase,
                "step": step,
                "tokens_in_phase": tokens_in_phase,
                "tokens_seen": tokens_seen,
                "gate": float(unwrap(model).decoder[0].gate),
                "name": self.cfg.name,
            }
            if step == 1 or step % self.log_every == 0 or (
                max_steps is not None and step == max_steps
            ):
                row = {
                    "name": self.cfg.name,
                    "phase": self.phase,
                    "step": step,
                    "steps_or_inf": max_steps if max_steps is not None else "-",
                    "nll": last,
                    "loss": step_loss,
                    "gate": extra["gate"],
                    "trainable_m": n_train / 1e6,
                    "lr": lr,
                    "fp8": use_fp8,
                    "tokens_seen": tokens_seen,
                    "tokens_in_phase": tokens_in_phase,
                }
                self._log(row)
            if self.eval_every and step % self.eval_every == 0:
                ev = self._eval_nll(model)
                if is_rank0(self.rank):
                    print(f"eval nll={ev:.4f}")
            if self.save_every and step % self.save_every == 0:
                self._maybe_save(model, opt, extra, f"step_{step}.pt")
            if max_steps is None and phase_budget is None:
                break
        extra = {
            "phase": self.phase,
            "step": step,
            "tokens_in_phase": tokens_in_phase,
            "tokens_seen": tokens_seen,
            "gate": float(unwrap(model).decoder[0].gate),
            "name": self.cfg.name,
        }
        self._maybe_save(model, opt, extra, "latest.pt")
        barrier()
        return TrainResult(nll=last, step=step, tokens_seen=tokens_seen)


def train_loop(
    cfg: CATYokoConfig,
    phase: str,
    steps: int,
    device: str,
    *,
    upcycle_src: dict | None = None,
    teacher: nn.Module | None = None,
    fsdp: bool = False,
    **kwargs,
) -> float:
    """Tiny / unit-test entry: one micro-batch per step, dummy data by default."""
    tr = Trainer(
        cfg,
        phase,
        device,
        steps=steps,
        micro_batch=int(kwargs.pop("micro_batch", 2)),
        accum=int(kwargs.pop("accum", 1)),
        upcycle_src=upcycle_src,
        teacher=teacher,
        fsdp=fsdp,
        **kwargs,
    )
    return tr.run().nll
