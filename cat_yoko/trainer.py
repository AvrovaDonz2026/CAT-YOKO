"""C1 Phase B trainer: packing, accum, WSD, freeze, checkpoint, DDP/FSDP."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from cat_yoko.checkpoint import load_checkpoint, load_model_state, load_optimizer_state, save_checkpoint
from cat_yoko.config import CATYokoConfig
from cat_yoko.data import open_stream, resolve_eos, resolve_seq_len, sidecar_meta
from cat_yoko.dist_util import barrier, init_distributed, is_rank0, wrap_distributed
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze, gate_schedule, set_gate
from cat_yoko.loss import kd_kl, kd_weight
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.offload import (
    auto_offload_flags,
    clip_grad_norm_mixed,
    move_module,
    set_after_block_backward,
)
from cat_yoko.optim import CPUOffloadAdamW, build_optimizer, plan_cpu_adam, trim_host_allocator, unwrap, wsd_lr
from cat_yoko.upcycle import upcycle_from_minicpm


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _as_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {dtype}")


def build_model(
    cfg: CATYokoConfig,
    device: str,
    dtype: str | torch.dtype | None = None,
) -> CATYokoForCausalLM:
    """Factory on ``device``. 12B must be built in bf16 on CUDA — never fp32-then-cast."""
    if device == "meta":
        with torch.device("meta"):
            return CATYokoForCausalLM(cfg)
    dt = _as_dtype(dtype)
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dt)
        with torch.device(device):
            return CATYokoForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)


def print_meta(cfg: CATYokoConfig) -> None:
    model = build_model(cfg, "meta")
    bd = model.param_breakdown()
    n = bd["total"]
    print(f"meta {cfg.name}: {n:,} params ({n / 1e9:.3f}B)")
    for k, v in bd.items():
        if k != "total":
            print(f"  {k:8s} {v / 1e9:.3f}B")


def auto_accum(
    cfg: CATYokoConfig, micro_batch: int, world: int, seq_len: int | None = None
) -> int:
    sl = cfg.seq_len if seq_len is None else seq_len
    per = micro_batch * sl * max(world, 1)
    return max(int(cfg.global_batch_tokens // per), 1)


@dataclass
class TrainResult:
    nll: float
    step: int
    tokens_seen: float
    phase: str = ""
    peak_mib: float = 0.0


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
        grad_ckpt: bool = False,
        seq_len: int | None = None,
        reuse_model: nn.Module | None = None,
        offload_encoder: bool | None = None,
        offload_blocks: bool | None = None,
        optim_cpu: bool | None = None,
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
        self.grad_ckpt = grad_ckpt
        self.reuse_model = reuse_model
        self.offload_encoder_arg = offload_encoder
        self.offload_blocks_arg = offload_blocks
        self.optim_cpu_arg = optim_cpu
        self.offload_encoder = False
        self.offload_blocks = False
        self.optim_cpu = False
        self.adam_state = "gpu"
        self.device, self.rank, self.world = init_distributed(device)
        if seq_len is not None:
            packed = sidecar_meta(data).get("seq_len") if data is not None else None
            if packed is not None and int(packed) != int(seq_len):
                raise ValueError(f"seq_len override {seq_len} != packed bin seq_len {packed}")
            self.seq_len = int(seq_len)
        else:
            self.seq_len = resolve_seq_len(data, cfg.seq_len)
        self.accum = (
            auto_accum(cfg, micro_batch, self.world, seq_len=self.seq_len)
            if accum <= 0
            else max(accum, 1)
        )

    def _cast(self, model: nn.Module) -> nn.Module:
        if self.dtype == "bf16":
            return model.to(dtype=torch.bfloat16)
        return model

    def _use_amp(self) -> bool:
        cuda = str(self.device).startswith("cuda")
        if should_autocast(self.phase, cuda=cuda, enabled=self.cfg.use_fp8):
            return True
        return cuda and self.dtype == "bf16"

    def _amp(self):
        if not self._use_amp():
            return nullcontext()
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    def _log(self, row: dict) -> None:
        if not is_rank0(self.rank):
            return
        line = (
            f"{row['name']} {row['phase']} step {row['step']}/{row['steps_or_inf']} "
            f"nll={row['nll']:.4f} aux={row['aux']:.4f} gate={row['gate']:.3f} "
            f"gn={row['grad_norm']:.2f} trainable={row['trainable_m']:.2f}M "
            f"lr={row['lr']:.2e} fp8={row['fp8']} tok={row['tokens_seen']:.0f} "
            f"tok/s={row['tok_s']:.0f} mem={row['mem_mib']:.0f}MiB"
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

    def _clip(self, model: nn.Module, trainable: list) -> float:
        if hasattr(model, "clip_grad_norm_") and type(model).__name__ == "FullyShardedDataParallel":
            gn = model.clip_grad_norm_(self.cfg.grad_clip)
        elif self.offload_blocks or self.optim_cpu:
            gn = clip_grad_norm_mixed(trainable, self.cfg.grad_clip)
        else:
            gn = torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip)
        return float(gn)

    def _mem_mib(self) -> float:
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024**2

    def _open(self, path: Path | None, seed: int):
        return open_stream(
            path,
            self.cfg.vocab_size,
            self.seq_len,
            seed=seed,
            eos_id=resolve_eos(path, self.eos_id),
            rank=self.rank,
            world=self.world,
        )

    @torch.no_grad()
    def _eval_nll(self, model: nn.Module, batches: int = 2) -> float:
        if self.eval_data is None and batches <= 0:
            return float("nan")
        stream = self._open(self.eval_data, self.seed + 1 + self.rank)
        raw = unwrap(model)
        raw.eval()
        total = 0.0
        n = 0
        for _ in range(max(batches, 1)):
            batch = stream.batch(self.micro_batch, self.device)
            with self._amp():
                out = raw(**batch)
            total += float(out["nll"])
            n += 1
        raw.train()
        return total / max(n, 1)

    def _resolve_offload(self) -> None:
        self.offload_encoder, self.offload_blocks, self.optim_cpu = auto_offload_flags(
            phase=self.phase,
            cfg_name=self.cfg.name,
            device=str(self.device),
            fsdp=self.fsdp,
            ddp=self.ddp,
            offload_encoder=self.offload_encoder_arg,
            offload_blocks=self.offload_blocks_arg,
            optim_cpu=self.optim_cpu_arg,
        )

    def _restore_rng(self, extra: dict) -> None:
        if extra.get("rng_torch") is not None:
            torch.set_rng_state(extra["rng_torch"].cpu())
        rng_cuda = extra.get("rng_cuda")
        if (
            rng_cuda is not None
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        ):
            try:
                torch.cuda.set_rng_state_all([t.cpu() for t in rng_cuda])
            except (RuntimeError, TypeError, ValueError):
                pass

    def _load_resume(self, model: nn.Module, opt, stream) -> tuple[int, float, float]:
        """Load weights from ``resume``. Same-phase = crash recovery; new phase = C1 handoff.

        Checkpoints map onto CPU so a 12B state_dict does not clone VRAM. Packed
        cursor and RNG always restore. Optimizer / step / tokens_in_phase only
        restore when CLI ``phase`` matches the checkpoint.
        """
        ckpt = load_checkpoint(self.resume, map_location="cpu")
        load_model_state(model, ckpt["model"])
        extra = ckpt.get("extra") or {}
        ckpt_phase = str(extra.get("phase", self.phase))
        same_phase = ckpt_phase == self.phase
        step = 0
        tokens_in_phase = 0.0
        tokens_seen = self.global_tokens_offset
        if same_phase:
            try:
                load_optimizer_state(opt, ckpt.get("optimizer"))
            except (ValueError, RuntimeError, KeyError):
                pass
            step = int(extra.get("step", 0))
            tokens_in_phase = float(extra.get("tokens_in_phase", 0.0))
            tokens_seen = float(extra.get("tokens_seen", tokens_seen))
        apply_freeze(unwrap(model), self.phase)
        self._resolve_offload()
        self._apply_runtime_flags(model)
        if extra.get("stream") is not None:
            stream.load_state_dict(extra["stream"])
        self._restore_rng(extra)
        del ckpt
        return step, tokens_in_phase, tokens_seen

    def _apply_runtime_flags(self, model: nn.Module) -> None:
        raw = unwrap(model)
        raw.grad_checkpoint = self.grad_ckpt
        raw.offload_encoder = self.offload_encoder
        raw.offload_blocks = self.offload_blocks
        if self.offload_blocks:
            for blk in list(raw.encoder) + list(raw.decoder):
                move_module(blk, "cpu")
        elif self.offload_encoder:
            move_module(raw.encoder, "cpu")

    def run(self) -> TrainResult:
        if self.cfg.name == "CAT-YOKO-12B" and not str(self.device).startswith("cuda"):
            raise RuntimeError("CAT-YOKO-12B weights need --device cuda --dtype bf16 (CPU is --meta only)")
        seed_all(self.seed + self.rank)
        configure_cuda()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._resolve_offload()
        if self.offload_blocks and self.accum > 1:
            raise RuntimeError(
                "B2 --offload-blocks Adams each layer during backward and cannot "
                "gradient-accumulate; use --accum 1, or ZeRO/multi-GPU for the 4M-token batch"
            )
        if self.reuse_model is not None:
            model = self.reuse_model
        else:
            model = build_model(self.cfg, self.device, dtype=self.dtype)
            if self.upcycle_src is not None:
                upcycle_from_minicpm(unwrap(model), self.upcycle_src, self.cfg)
        apply_freeze(unwrap(model), self.phase)
        self._apply_runtime_flags(model)
        if self.reuse_model is None:
            model = wrap_distributed(unwrap(model), fsdp=self.fsdp, ddp=self.ddp)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        adam_state = "gpu"
        state_dtype = torch.float32
        retain_state = True
        if self.optim_cpu:
            state_dtype, retain_state, adam_state = plan_cpu_adam(
                n_train, steps=self.steps
            )
        self.adam_state = adam_state
        if is_rank0(self.rank) and str(self.device).startswith("cuda") and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(
                f"built {self.cfg.name} on {self.device} dtype={self.dtype} "
                f"params={unwrap(model).param_count():,} alloc={alloc:.2f}GiB "
                f"grad_ckpt={self.grad_ckpt} seq={self.seq_len} "
                f"offload_enc={self.offload_encoder} offload_blocks={self.offload_blocks} "
                f"optim_cpu={self.optim_cpu} adam={adam_state} "
                f"trainable={n_train/1e6:.2f}M reuse={self.reuse_model is not None}",
                flush=True,
            )
        opt = build_optimizer(
            model,
            self.cfg,
            cpu_offload=self.optim_cpu,
            state_dtype=state_dtype,
            retain_state=retain_state,
        )
        stream = self._open(self.data, self.seed + self.rank)
        step = 0
        tokens_in_phase = 0.0
        tokens_seen = self.global_tokens_offset
        if self.resume is not None:
            step, tokens_in_phase, tokens_seen = self._load_resume(model, opt, stream)

        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        unwrap(model).train()
        use_fp8 = should_autocast(
            self.phase, cuda=str(self.device).startswith("cuda"), enabled=self.cfg.use_fp8
        )
        last = 0.0
        phase_budget = self.tokens_target
        max_steps = self.steps
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)

        gn_parts: list[float] = []
        if self.offload_blocks and isinstance(opt, CPUOffloadAdamW):
            clip = self.cfg.grad_clip

            def _on_block(blk):
                params = [p for p in blk.parameters() if p.grad is not None]
                gn_parts.append(clip_grad_norm_mixed(params, clip))
                opt.step_params(blk.parameters())

            set_after_block_backward(_on_block)
        try:
            while True:
                if max_steps is not None and step >= max_steps:
                    break
                if phase_budget is not None and tokens_in_phase >= phase_budget:
                    break
                gn_parts.clear()
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
                step_aux = 0.0
                step_tokens = 0
                t0 = time.perf_counter()
                for _ in range(self.accum):
                    batch = stream.batch(self.micro_batch, self.device)
                    step_tokens += int(batch["input_ids"].numel())
                    with self._amp():
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
                    unwrap(model).step_router_bias()
                    step_nll += float(out["nll"].detach()) / self.accum
                    step_loss += float(out["loss"].detach()) / self.accum
                    step_aux += float(out.get("aux", out["loss"].new_zeros(())).detach()) / self.accum
                if not math.isfinite(step_nll):
                    raise FloatingPointError(f"non-finite nll at step {step + 1}: {step_nll}")
                if self.offload_blocks:
                    leftover = [p for p in trainable if p.grad is not None]
                    if leftover:
                        gn_parts.append(clip_grad_norm_mixed(leftover, self.cfg.grad_clip))
                    grad_norm = math.sqrt(sum(g * g for g in gn_parts)) if gn_parts else 0.0
                else:
                    grad_norm = self._clip(model, trainable)
                opt.step()
                dt = max(time.perf_counter() - t0, 1e-9)
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
                    "seq_len": self.seq_len,
                    "stream": stream.state_dict(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
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
                        "aux": step_aux,
                        "gate": extra["gate"],
                        "grad_norm": grad_norm,
                        "trainable_m": n_train / 1e6,
                        "lr": lr,
                        "fp8": use_fp8,
                        "tokens_seen": tokens_seen,
                        "tokens_in_phase": tokens_in_phase,
                        "tok_s": (step_tokens * self.world) / dt,
                        "mem_mib": self._mem_mib(),
                        "seq_len": self.seq_len,
                        "grad_ckpt": self.grad_ckpt,
                        "offload_encoder": self.offload_encoder,
                        "offload_blocks": self.offload_blocks,
                        "optim_cpu": self.optim_cpu,
                        "adam": self.adam_state,
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
                "seq_len": self.seq_len,
                "stream": stream.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            self._maybe_save(model, opt, extra, "latest.pt")
            barrier()
            peak = self._mem_mib()
            del opt
            if self.optim_cpu or self.offload_blocks or self.offload_encoder:
                trim_host_allocator()
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return TrainResult(
                nll=last, step=step, tokens_seen=tokens_seen, phase=self.phase, peak_mib=peak
            )
        finally:
            set_after_block_backward(None)


def run_c1_chain(
    cfg: CATYokoConfig,
    device: str,
    *,
    steps: int = 1,
    reuse_model: nn.Module | None = None,
    **kwargs,
) -> dict[str, TrainResult]:
    """One (or N) optimizer step of B0, then B1, then B2 on the same weights."""
    dtype = kwargs.pop("dtype", "bf16" if str(device).startswith("cuda") else "fp32")
    if reuse_model is None:
        reuse_model = build_model(cfg, device, dtype=dtype)
        upcycle_src = kwargs.pop("upcycle_src", None)
        if upcycle_src is not None:
            upcycle_from_minicpm(unwrap(reuse_model), upcycle_src, cfg)
    else:
        kwargs.pop("upcycle_src", None)
    out: dict[str, TrainResult] = {}
    offset = float(kwargs.pop("global_tokens_offset", 0.0))
    for phase in ("B0", "B1", "B2"):
        tr = Trainer(
            cfg,
            phase,
            device,
            steps=steps,
            dtype=dtype,
            reuse_model=reuse_model,
            global_tokens_offset=offset,
            **kwargs,
        )
        out[phase] = tr.run()
        offset = out[phase].tokens_seen
        if str(device).startswith("cuda") and torch.cuda.is_available():
            trim_host_allocator()
            torch.cuda.empty_cache()
    return out


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
