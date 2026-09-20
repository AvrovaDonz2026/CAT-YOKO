"""C1 Phase B trainer: packing, accum, WSD, freeze, checkpoint, DDP/FSDP."""

from __future__ import annotations

import json
import math
import os
import random
import time

# Before importing torch: ZeRO param-offload H2D must share the GPU with GEMM.
# DeepSpeed/NCCL often pins CUDA_DEVICE_MAX_CONNECTIONS=1, which serializes
# memcpy behind compute and shows up as GPU util dropping to 0%.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from cat_yoko.checkpoint import (
    cleanup_save_tmp,
    is_trainable_ckpt,
    load_checkpoint,
    load_model_state,
    load_optimizer_state,
    load_trainable_state,
    peek_checkpoint_extra,
    prune_step_checkpoints,
    publish_latest,
    resolve_resume_path,
    save_checkpoint,
    save_trainable_checkpoint,
)
from cat_yoko.config import CATYokoConfig
from cat_yoko.data import open_stream, resolve_eos, resolve_seq_len, sidecar_meta
from cat_yoko.dist_util import (
    allreduce_router_loads,
    barrier,
    backward_sync_ctx,
    init_distributed,
    is_rank0,
    reduce_mean,
    reduce_sum,
    wrap_distributed,
)
from cat_yoko.nvfp4 import low_prec_enabled, should_autocast
from cat_yoko.nvfp4_linear import (
    apply_nvfp4,
    nvfp4_module_names,
    te_available,
    te_nvfp4_linear_enabled,
)
from cat_yoko.nvfp4_hw import compute_family, prefer_te_linear
from cat_yoko.freeze import apply_freeze, gate_schedule, set_gate
from cat_yoko.indexer import ensure_indexers, phase_needs_indexer, set_align_indexer, set_sparse_mode
from cat_yoko.loss import kd_kl, kd_weight, safe_ppl
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import grouped_mm_available, moe_utilization
from cat_yoko.offload import (
    auto_offload_flags,
    clip_grad_norm_mixed,
    move_module,
    set_after_block_backward,
)
from cat_yoko.optim import CPUOffloadAdamW, build_optimizer, plan_cpu_adam, trim_host_allocator, unwrap, wsd_lr
from cat_yoko.phases import c_chain, resolve_phase_spec
from cat_yoko.upcycle import upcycle_from_minicpm


def enable_expandable_segments() -> str:
    """Set before the CUDA caching allocator starts. B1 is ~28.4/32GiB."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def quiet_inductor() -> None:
    """Stop torch.compile from forking 32 inductor workers that steal CPU from ZeRO H2D.

    3090 ZeRO-3 param offload needs the host to feed PCIe. An idle compile pool
    still shows up as GPU util dropping to 0 between steps.
    """
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    try:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
    except Exception:
        pass
    try:
        import torch._inductor.config as inductor_cfg

        inductor_cfg.compile_threads = 1
    except Exception:
        pass


def _json_safe(obj):
    """Strict-JSON form: non-finite floats become null; nested dict/list walked."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def token_mean_nll(weighted: float, n_valid: float) -> float:
    """``sum(nll * n_valid) / sum(n_valid)``. Empty ranks contribute ``(0, 0)``, not nan."""
    if n_valid <= 0 or not math.isfinite(weighted):
        return float("nan")
    return weighted / n_valid


def _host_step_stats(
    nll_w: torch.Tensor,
    n_valid: torch.Tensor,
    loss: torch.Tensor,
    aux: torch.Tensor,
) -> tuple[float, float, float, float]:
    """One D2H for nll/n_valid/loss/aux instead of four ``.item()`` syncs."""
    packed = torch.stack(
        (
            nll_w.detach().float().reshape(()),
            n_valid.detach().float().reshape(()),
            loss.detach().float().reshape(()),
            aux.detach().float().reshape(()),
        )
    )
    vals = packed.cpu().tolist()
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


enable_expandable_segments()
quiet_inductor()


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda() -> None:
    enable_expandable_segments()
    quiet_inductor()
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    for _name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
        _fn = getattr(torch.backends.cuda, _name, None)
        if callable(_fn):
            _fn(True)


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
    stream: dict | None = None


class Trainer:
    def __init__(
        self,
        cfg: CATYokoConfig,
        phase: str,
        device: str,
        *,
        steps: int | None = None,
        more_steps: int | None = None,
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
        deepspeed: bool = False,
        zero_stage: int = 2,
        zero_offload: bool = False,
        zero_offload_param: bool = False,
        save_dir: Path | None = None,
        save_every: int = 0,
        resume: Path | None = None,
        log_every: int = 1,
        log_path: Path | None = None,
        eval_every: int = 0,
        eval_batches: int = 2,
        dtype: str = "fp32",
        global_tokens_offset: float = 0.0,
        grad_ckpt: bool = False,
        seq_len: int | None = None,
        reuse_model: nn.Module | None = None,
        offload_encoder: bool | None = None,
        offload_blocks: bool | None = None,
        optim_cpu: bool | None = None,
        save_optim: bool | None = None,
        save_keep: int = 0,
        initial_stream: dict | None = None,
        save_full: bool | None = None,
        save_trainable: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.phase = phase
        self.steps = steps
        self.more_steps = int(more_steps) if more_steps is not None else None
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
        self.deepspeed = bool(deepspeed)
        self.zero_offload_param = bool(zero_offload_param)
        self.zero_offload = bool(zero_offload or zero_offload_param)
        if self.zero_offload_param:
            self.zero_stage = 3
        elif self.deepspeed:
            self.zero_stage = int(zero_stage or 2)
        else:
            self.zero_stage = 0
        if self.deepspeed and (self.fsdp or self.ddp):
            raise ValueError("DeepSpeed ZeRO cannot mix --fsdp/--ddp")
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_every = save_every
        self.resume = Path(resume) if resume else None
        if self.resume is not None:
            self.resume = resolve_resume_path(self.resume)
        self.log_every = max(log_every, 1)
        self.log_path = Path(log_path) if log_path else None
        self._prefetch_batch: dict | None = None
        self.eval_every = eval_every
        self.eval_batches = max(int(eval_batches), 1)
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
        self.save_optim_arg = save_optim
        self.save_keep = max(int(save_keep), 0)
        self.initial_stream = initial_stream
        self.save_full_arg = save_full
        self.save_trainable_arg = save_trainable
        self._trainable_names: set[str] = set()
        self.device, self.rank, self.world = init_distributed(
            device, force=bool(fsdp or self.deepspeed)
        )
        if seq_len is not None:
            packed = sidecar_meta(data).get("seq_len") if data is not None else None
            self.seq_len = int(seq_len)
            if packed is not None and int(packed) != self.seq_len:
                print(
                    f"re-window packed bin sidecar seq_len={int(packed)} → {self.seq_len}",
                    flush=True,
                )
        else:
            self.seq_len = resolve_seq_len(data, cfg.seq_len)
        self.accum = (
            auto_accum(cfg, micro_batch, self.world, seq_len=self.seq_len)
            if accum <= 0
            else max(accum, 1)
        )
        self.nvfp4_n = 0
        ph = resolve_phase_spec(phase, use_kda=bool(getattr(cfg, "use_kda", False)))
        self.loss_mode = ph.loss if ph is not None else "ce"
        self.phase_sparse = ph.sparse if ph is not None else "window"
        self.phase_align = bool(ph.align_indexer) if ph is not None else False

    def _apply_nvfp4(self, model: nn.Module) -> int:
        """After freeze, before DDP. QKV/O Linears only; SDPA stays fp32."""
        apply_nvfp4(
            unwrap(model),
            self.phase,
            enabled=bool(getattr(self.cfg, "use_nvfp4", False)),
        )
        self.nvfp4_n = len(nvfp4_module_names(unwrap(model)))
        return self.nvfp4_n

    def _cast(self, model: nn.Module) -> nn.Module:
        if self.dtype == "bf16":
            return model.to(dtype=torch.bfloat16)
        return model

    def _use_amp(self) -> bool:
        cuda = str(self.device).startswith("cuda")
        if should_autocast(self.phase, cuda=cuda, enabled=low_prec_enabled(self.cfg)):
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
        ppl = row.get("ppl")
        ppl_s = f"{ppl:.1f}" if isinstance(ppl, (int, float)) and math.isfinite(ppl) else "-"
        line = (
            f"{row['name']} {row['phase']} step {row['step']}/{row['steps_or_inf']} "
            f"nll={row['nll']:.4f} ppl={ppl_s} aux={row['aux']:.4f} "
            f"gate={row['gate']:.3f} gn={row['grad_norm']:.2f} "
            f"moe_cv={row.get('moe_cv', 0):.2f} trainable={row['trainable_m']:.2f}M "
            f"lr={row['lr']:.2e} fp8={row['fp8']} nvfp4={row.get('nvfp4', False)} "
            f"tok={row['tokens_seen']:.0f} "
            f"tok/s={row['tok_s']:.0f} mem={row['mem_mib']:.0f}MiB"
        )
        rec = row.get("indexer_recall")
        if isinstance(rec, (int, float)) and math.isfinite(rec):
            line += f" rec={rec:.3f}"
        rew = row.get("rl_reward")
        if isinstance(rew, (int, float)) and math.isfinite(rew):
            line += f" rew={rew:.3f}"
        print(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(_json_safe(row), default=str, allow_nan=False) + "\n")

    def _attach_phase_modules(self, model: nn.Module) -> None:
        """Indexers + sparse flags. Phase B no-ops (no indexer modules)."""
        raw = unwrap(model)
        if phase_needs_indexer(self.phase) or self.phase_sparse in {"topk", "hca"}:
            ensure_indexers(raw, self.cfg)
        set_align_indexer(raw, self.phase_align)
        set_sparse_mode(raw, self.phase_sparse)

    def _maybe_save(self, model: nn.Module, opt, extra: dict, tag: str) -> None:
        if self.save_dir is None:
            return
        from cat_yoko.deepspeed_zero import (
            gathered_state_dict,
            gathered_trainable_state_dict,
            is_deepspeed_engine,
            zero_stage_of,
        )

        trainable_sd = None
        full_sd = None
        if is_deepspeed_engine(model) and zero_stage_of(model) >= 3:
            # All ranks must enter the ZeRO-3 gather. Rank-0-only would deadlock.
            if self.save_trainable:
                trainable_sd = gathered_trainable_state_dict(
                    model, names=self._trainable_names or None
                )
            if self.save_full:
                full_sd = gathered_state_dict(model)
        if not is_rank0(self.rank):
            return
        if self.save_trainable:
            if tag == "latest.pt":
                step = extra.get("step")
                step_path = (
                    self.save_dir / f"trainable_step_{int(step)}.pt" if step else None
                )
                dest = self.save_dir / "trainable.pt"
                if step_path is not None and step_path.is_file():
                    publish_latest(step_path, dest)
                else:
                    save_trainable_checkpoint(
                        dest, model=model, extra=extra, state=trainable_sd
                    )
            elif tag.startswith("step_"):
                dest = self.save_dir / f"trainable_{tag}"
                save_trainable_checkpoint(
                    dest, model=model, extra=extra, state=trainable_sd
                )
                # Rental GPUs die mid-envelope. Point trainable.pt at this
                # step so --resume save_dir works before the final latest.pt.
                publish_latest(dest, self.save_dir / "trainable.pt")
        if self.save_full:
            dest = self.save_dir / tag
            if tag == "latest.pt":
                step = extra.get("step")
                step_path = self.save_dir / f"step_{int(step)}.pt" if step else None
                if step_path is not None and step_path.is_file():
                    publish_latest(step_path, dest)
                    return
            save_checkpoint(
                dest,
                model=model,
                optimizer=opt,
                extra=extra,
                save_optimizer=self.save_optim,
                model_state=full_sd,
            )
        if tag.startswith("step_") and self.save_keep:
            prune_step_checkpoints(self.save_dir, self.save_keep)

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
        # DummyStream already offsets by rank; do not add rank onto ``seed`` again.
        return open_stream(
            path,
            self.cfg.vocab_size,
            self.seq_len,
            seed=seed,
            eos_id=resolve_eos(path, self.eos_id),
            rank=self.rank,
            world=self.world,
            response_only=self.loss_mode == "sft",
            needle=self.phase.startswith(("D", "G")) or self.loss_mode in {"grpo", "dpo"},
        )

    @torch.no_grad()
    def _eval_nll_stats(self, model: nn.Module, batches: int | None = None) -> tuple[float, float]:
        """Local ``(sum nll*n_valid, sum n_valid)``. Empty / no eval_data → ``(0, 0)``."""
        n_batches = self.eval_batches if batches is None else batches
        if self.eval_data is None:
            return 0.0, 0.0
        stream = self._open(self.eval_data, self.seed + 1)
        was_train = model.training
        model.eval()
        weighted = 0.0
        n_valid_total = 0.0
        try:
            for _ in range(max(n_batches, 1)):
                batch = stream.batch(self.micro_batch, self.device)
                with self._amp():
                    out = model(**batch)
                nv = out.get("n_valid")
                if nv is None:
                    continue
                n_valid_f = float(nv.detach() if torch.is_tensor(nv) else nv)
                if n_valid_f <= 0:
                    continue
                weighted += float(out["nll"]) * n_valid_f
                n_valid_total += n_valid_f
        finally:
            if was_train:
                model.train()
        return weighted, n_valid_total

    def _eval_nll(self, model: nn.Module, batches: int | None = None) -> float:
        """Token-mean NLL: ``sum(nll * n_valid) / sum(n_valid)``. Skip empty batches."""
        return token_mean_nll(*self._eval_nll_stats(model, batches))

    def _allreduce_token_nll(self, weighted: float, n_valid: float) -> float:
        """DDP token-mean. Empty ranks send ``(0, 0)`` so they cannot poison with nan."""
        w = reduce_sum(weighted, device=str(self.device), world=self.world)
        n = reduce_sum(n_valid, device=str(self.device), world=self.world)
        return token_mean_nll(w, n)

    def _resolve_offload(self) -> None:
        self.offload_encoder, self.offload_blocks, self.optim_cpu = auto_offload_flags(
            phase=self.phase,
            cfg_name=self.cfg.name,
            device=str(self.device),
            fsdp=self.fsdp,
            ddp=self.ddp,
            deepspeed=self.deepspeed,
            offload_encoder=self.offload_encoder_arg,
            offload_blocks=self.offload_blocks_arg,
            optim_cpu=self.optim_cpu_arg,
        )

    def _restore_rng(self, extra: dict) -> None:
        if extra.get("rng_py") is not None:
            try:
                random.setstate(extra["rng_py"])
            except (TypeError, ValueError):
                pass
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
        extra = ckpt.get("extra") or {}
        if is_trainable_ckpt(ckpt):
            n_ov = len(ckpt["trainable"])
            load_trainable_state(model, ckpt["trainable"])
            if is_rank0(self.rank):
                # B1: MiniCPM5/dummy upcycle already copied encoder+embed;
                # this overlay is B0 new-modules (cache/cross) or B1 decoder.
                print(
                    f"overlay resume {self.resume}: {n_ov} tensors "
                    f"ckpt_phase={extra.get('phase')} cli_phase={self.phase}",
                    flush=True,
                )
        else:
            load_model_state(model, ckpt["model"])
        ckpt_phase = str(extra.get("phase", self.phase))
        same_phase = ckpt_phase == self.phase
        step = 0
        tokens_in_phase = 0.0
        tokens_seen = self.global_tokens_offset
        if same_phase:
            if opt is not None:
                try:
                    load_optimizer_state(opt, ckpt.get("optimizer"))
                except (ValueError, RuntimeError, KeyError):
                    pass
            step = int(extra.get("step", 0))
            tokens_in_phase = float(extra.get("tokens_in_phase", 0.0))
            tokens_seen = float(extra.get("tokens_seen", tokens_seen))
        apply_freeze(unwrap(model), self.phase)
        self._attach_phase_modules(model)
        self._apply_nvfp4(model)
        self._resolve_offload()
        self._apply_runtime_flags(model)
        if extra.get("stream") is not None:
            try:
                stream.load_state_dict(extra["stream"])
            except ValueError:
                # Packed vs jsonl / DDP stride mismatch: keep the new stream
                # at shard start (published: kind mismatch skips cursor).
                pass
        self._restore_rng(extra)
        del ckpt
        return step, tokens_in_phase, tokens_seen

    def _inherit_implemented_kda(self) -> None:
        """B overlay owns the 3:1 graph. C inherits; C cannot implement late."""
        from cat_yoko.kda import resolve_implemented_kda

        extra = peek_checkpoint_extra(self.resume) if self.resume is not None else None
        want = resolve_implemented_kda(
            cli=bool(getattr(self.cfg, "use_kda", False)), extra=extra
        )
        if want == bool(getattr(self.cfg, "use_kda", False)):
            return
        self.cfg = replace(self.cfg, use_kda=want)
        ph = resolve_phase_spec(self.phase, use_kda=want)
        if ph is None:
            return
        self.loss_mode = ph.loss
        self.phase_sparse = ph.sparse
        self.phase_align = bool(ph.align_indexer)

    def _extra(
        self,
        model: nn.Module,
        step: int,
        tokens_in_phase: float,
        tokens_seen: float,
        stream,
        *,
        gate: float | None = None,
        stream_state: dict | None = None,
        include_rng: bool = True,
    ) -> dict:
        """Checkpoint / log metadata. CUDA RNG snapshot syncs the default stream.

        Call on save steps (and the final latest.pt). Do not call every step:
        ``get_rng_state_all`` plus ``float(gate)`` used to stall the GPU after
        every optimizer step.
        """
        if gate is None:
            gate = float(unwrap(model).decoder[0].gate)
        extra = {
            "phase": self.phase,
            "step": step,
            "tokens_in_phase": tokens_in_phase,
            "tokens_seen": tokens_seen,
            "gate": float(gate),
            "name": self.cfg.name,
            "seq_len": self.seq_len,
            "seed": self.seed,
            "cfg": asdict(self.cfg),
            "use_kda": bool(getattr(self.cfg, "use_kda", False)),
            "sparse": self.phase_sparse,
            "stream": stream_state if stream_state is not None else stream.state_dict(),
        }
        if include_rng:
            extra["rng_py"] = random.getstate()
            extra["rng_torch"] = torch.get_rng_state()
            extra["rng_cuda"] = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
        else:
            extra["rng_py"] = None
            extra["rng_torch"] = None
            extra["rng_cuda"] = None
        return extra

    def _forward_loss(self, model: nn.Module, batch: dict) -> dict[str, torch.Tensor]:
        """CE / indexer KL / SFT / GRPO / DPO. Always returns loss, nll, n_valid, aux."""
        mode = self.loss_mode
        if mode == "indexer_kl":
            out = model(
                input_ids=batch["input_ids"],
                doc_ids=batch.get("doc_ids"),
                labels=None,
            )
            if "indexer_kl" not in out:
                raise RuntimeError(
                    f"{self.phase} indexer_kl missing; ensure_indexers + align_indexer"
                )
            kl = out["indexer_kl"]
            out["loss"] = kl
            out["nll"] = kl
            out["n_valid"] = kl.new_ones(())
            out.setdefault("aux", kl.new_zeros(()))
            return out
        if mode == "grpo":
            from cat_yoko.rl import grpo_step_loss, sample_completions

            prompt_len = max(self.seq_len // 2, 1)
            max_new = min(int(self.cfg.grpo_max_new), max(self.seq_len - prompt_len, 1))
            group = max(int(self.cfg.grpo_group), 1)
            prompt = batch["input_ids"][:, :prompt_len]
            prompts = prompt.repeat_interleave(group, dim=0)
            with torch.no_grad():
                full = sample_completions(unwrap(model), prompts, max_new)
            completions = full[:, prompt_len:]
            loss, reward = grpo_step_loss(
                unwrap(model),
                prompts,
                completions,
                group=group,
                prompt_len=prompt_len,
            )
            z = loss.reshape(())
            return {
                "loss": z,
                "nll": z,
                "n_valid": z.new_ones(()) * prompts.size(0),
                "aux": z.new_zeros(()),
                "rl_reward": reward.reshape(()),
            }
        if mode == "dpo":
            from cat_yoko.rl import dpo_step_loss

            ids = batch["input_ids"]
            plen = max(self.seq_len // 2, 1)
            loss = dpo_step_loss(
                unwrap(model),
                ids,
                ids.roll(1, dims=0),
                prompt_len=plen,
                beta=float(self.cfg.dpo_beta),
            )
            z = loss.reshape(())
            return {
                "loss": z,
                "nll": z,
                "n_valid": z.new_ones(()) * ids.size(0),
                "aux": z.new_zeros(()),
            }
        return model(**batch)

    def _apply_runtime_flags(self, model: nn.Module) -> None:
        raw = unwrap(model)
        raw.grad_checkpoint = self.grad_ckpt
        raw.offload_encoder = self.offload_encoder
        raw.offload_blocks = self.offload_blocks
        # Chunked lm_head+CE unless KD needs the full student logit tensor.
        raw.return_logits = self.teacher is not None or self.loss_mode in {"grpo", "dpo"}
        if self.offload_blocks:
            for blk in list(raw.encoder) + list(raw.decoder):
                move_module(blk, "cpu")
        elif self.offload_encoder:
            move_module(raw.encoder, "cpu")

    def _print_built(self, model: nn.Module, n_train: int, adam_state: str) -> None:
        """Rank-0 CUDA banner after the graph exists. Includes allocator conf."""
        raw = unwrap(model)
        alloc = 0.0
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
        alloc_conf = enable_expandable_segments()
        print(
            f"built {self.cfg.name} on {self.device} dtype={self.dtype} "
            f"params={raw.param_count():,} alloc={alloc:.2f}GiB "
            f"grad_ckpt={self.grad_ckpt} seq={self.seq_len} "
            f"offload_enc={self.offload_encoder} offload_blocks={self.offload_blocks} "
            f"optim_cpu={self.optim_cpu} adam={adam_state} "
            f"zero={self.zero_stage} ds={self.deepspeed} "
            f"zero_offload={self.zero_offload} zero_offload_param={self.zero_offload_param} "
            f"trainable={n_train/1e6:.2f}M reuse={self.reuse_model is not None} "
            f"nvfp4={bool(getattr(self.cfg, 'use_nvfp4', False))} "
            f"nvfp4_n={self.nvfp4_n} nvfp4_family={compute_family()} "
            f"te_linear={prefer_te_linear()} grouped_mm={grouped_mm_available()} "
            f"te={te_available()} te_nvfp4={te_nvfp4_linear_enabled()} "
            f"return_logits={bool(getattr(raw, 'return_logits', True))} "
            f"sparse={self.phase_sparse} loss={self.loss_mode} "
            f"PYTORCH_CUDA_ALLOC_CONF={alloc_conf}",
            flush=True,
        )

    def _wrap_and_optim(self, model: nn.Module):
        """Freeze/NVFP4 already applied. DeepSpeed initialize, else DDP/FSDP."""
        from cat_yoko.deepspeed_zero import is_zero_partitioned, wrap_deepspeed, zero_config

        raw = unwrap(model)
        if is_zero_partitioned(raw):
            raise RuntimeError(
                "cannot re-wrap a ZeRO-3 partitioned module; "
                "run B0/B1/B2 as separate processes with --resume overlay"
            )
        self._trainable_names = {
            n for n, p in raw.named_parameters() if p.requires_grad
        }
        n_train = sum(p.numel() for n, p in raw.named_parameters() if p.requires_grad)
        if self.deepspeed:
            if not str(self.device).startswith("cuda"):
                raise RuntimeError("DeepSpeed ZeRO needs --device cuda")
            adam_state = "ds-cpu" if self.zero_offload else "ds"
            self.adam_state = adam_state
            # Fused DeepSpeedCPUAdam keeps the two decay groups. Torch AdamW on
            # ZeRO-3 CPU shards is ~1s/step and shows up as GPU util 0%.
            opt = build_optimizer(
                raw, self.cfg, cpu_offload=False, cpu_adam_fast=self.zero_offload
            )
            if type(opt).__name__ == "DeepSpeedCPUAdam":
                adam_state = "ds-cpuadam"
                self.adam_state = adam_state
            elif self.zero_offload:
                print(
                    "DeepSpeedCPUAdam unavailable (need ninja + cpu_adam op); "
                    "torch AdamW on ZeRO CPU shards",
                    flush=True,
                )
            cfg = zero_config(
                stage=self.zero_stage,
                offload_optimizer=self.zero_offload,
                offload_param=self.zero_offload_param,
                bf16=self.dtype == "bf16",
                # Trainer owns the micro-loop (loss / accum). DS GAS=1.
                gradient_accumulation_steps=1,
                gradient_clipping=float(self.cfg.grad_clip),
                train_micro_batch_size_per_gpu=self.micro_batch,
            )
            model, opt = wrap_deepspeed(raw, opt, cfg)
            return model, opt, n_train, adam_state
        model = wrap_distributed(raw, fsdp=self.fsdp, ddp=self.ddp)
        adam_state = "gpu"
        state_dtype = torch.float32
        retain_state = True
        if self.optim_cpu:
            state_dtype, retain_state, adam_state = plan_cpu_adam(
                n_train, steps=self.steps
            )
        self.adam_state = adam_state
        opt = build_optimizer(
            model,
            self.cfg,
            cpu_offload=self.optim_cpu,
            state_dtype=state_dtype,
            retain_state=retain_state,
        )
        return model, opt, n_train, adam_state

    def _begin_step_peak(self) -> None:
        """Start peak tracking after freeze/offload, not at 12B ``build_model``.

        Isolated B2 otherwise reports the full 22.82GiB graph that existed
        only before per-block CPU offload. ``--c1`` B2 otherwise reports B1's
        leftover decoder (~14.7GiB). B0/B1 still capture encoder-on-GPU
        forward because that copy-back happens after this reset.
        """
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def run(self) -> TrainResult:
        if self.cfg.name == "CAT-YOKO-12B" and not str(self.device).startswith("cuda"):
            raise RuntimeError("CAT-YOKO-12B weights need --device cuda --dtype bf16 (CPU is --meta only)")
        seed_all(self.seed + self.rank)
        configure_cuda()
        if self.phase_align:
            # Indexer KL is a side tensor on the block; activation checkpoint
            # would drop it from the autograd graph.
            self.grad_ckpt = False
        self._resolve_offload()
        if self.save_optim_arg is None:
            self.save_optim = self.cfg.name != "CAT-YOKO-12B"
        else:
            self.save_optim = bool(self.save_optim_arg)
        if self.save_full_arg is None:
            self.save_full = self.cfg.name != "CAT-YOKO-12B"
        else:
            self.save_full = bool(self.save_full_arg)
        if self.save_trainable_arg is None:
            self.save_trainable = self.cfg.name == "CAT-YOKO-12B"
        else:
            self.save_trainable = bool(self.save_trainable_arg)
        if self.save_dir is not None and not self.save_full and not self.save_trainable:
            raise RuntimeError("need --save-full and/or --save-trainable")
        if self.save_dir is not None:
            cleanup_save_tmp(self.save_dir)
            if self.cfg.name == "CAT-YOKO-12B":
                root = self.save_dir.resolve()
                if root == Path("/tmp") or Path("/tmp") in root.parents:
                    print(
                        f"warning: 12B checkpoints under {root} often fill the overlay; "
                        "prefer a large volume (e.g. /root/autodl-tmp)",
                        flush=True,
                    )
        if self.offload_blocks and self.accum > 1:
            raise RuntimeError(
                "B2 --offload-blocks Adams each layer during backward and cannot "
                "gradient-accumulate; use --accum 1, or ZeRO/multi-GPU for the 4M-token batch"
            )
        self._inherit_implemented_kda()
        if self.reuse_model is not None:
            model = unwrap(self.reuse_model)
        else:
            model = build_model(self.cfg, self.device, dtype=self.dtype)
            if self.upcycle_src is not None:
                upcycle_from_minicpm(unwrap(model), self.upcycle_src, self.cfg)
                del self.upcycle_src
        self._attach_phase_modules(model)
        apply_freeze(unwrap(model), self.phase)
        self._attach_phase_modules(model)
        self._apply_nvfp4(model)
        self._apply_runtime_flags(model)
        stream = self._open(self.data, self.seed)
        step = 0
        tokens_in_phase = 0.0
        tokens_seen = self.global_tokens_offset
        if self.deepspeed:
            if self.resume is not None:
                step, tokens_in_phase, tokens_seen = self._load_resume(
                    unwrap(model), None, stream
                )
            elif self.initial_stream is not None:
                stream.load_state_dict(self.initial_stream)
        model, opt, n_train, adam_state = self._wrap_and_optim(model)
        if is_rank0(self.rank) and str(self.device).startswith("cuda") and torch.cuda.is_available():
            self._print_built(unwrap(model), n_train, adam_state)
            if (
                self.save_dir is not None
                and self.cfg.name == "CAT-YOKO-12B"
                and adam_state in {"fp16", "fp32"}
            ):
                print(
                    "warning: 12B checkpoint adds ~23GiB host tensors on top of CPU Adam "
                    "moments; --steps 1 or omit --save-dir on a 62GiB cgroup",
                    flush=True,
                )
        if not self.deepspeed:
            if self.resume is not None:
                step, tokens_in_phase, tokens_seen = self._load_resume(model, opt, stream)
            elif self.initial_stream is not None:
                stream.load_state_dict(self.initial_stream)

        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        unwrap(model).train()
        self._nll_ema = None
        use_fp8 = should_autocast(
            self.phase,
            cuda=str(self.device).startswith("cuda"),
            enabled=low_prec_enabled(self.cfg),
        )
        use_nvfp4 = bool(getattr(self.cfg, "use_nvfp4", False))
        last = 0.0
        phase_budget = self.tokens_target
        max_steps = self.steps
        if self.more_steps is not None:
            max_steps = int(step) + int(self.more_steps)
        if not self.deepspeed:
            trainable = [p for p in model.parameters() if p.requires_grad]
            n_train = sum(p.numel() for p in trainable)
        else:
            trainable = []

        gn_parts: list[float] = []
        if self.offload_blocks and isinstance(opt, CPUOffloadAdamW):
            clip = self.cfg.grad_clip

            def _on_block(blk):
                params = [p for p in blk.parameters() if p.grad is not None]
                gn_parts.append(clip_grad_norm_mixed(params, clip))
                opt.step_params(blk.parameters())

            set_after_block_backward(_on_block)
        self._begin_step_peak()
        try:
            while True:
                if max_steps is not None and step >= max_steps:
                    break
                if phase_budget is not None and tokens_in_phase >= phase_budget:
                    break
                gn_parts.clear()
                moe_stats: dict[str, float] = {
                    "moe_cv": 0.0,
                    "moe_max": 0.0,
                    "moe_min": 0.0,
                    "moe_layers": 0,
                }
                progress = 0.0
                if phase_budget:
                    progress = min((tokens_in_phase + 1) / phase_budget, 1.0)
                elif max_steps:
                    progress = (step + 1) / max_steps
                gate_val = gate_schedule(self.phase, progress)
                set_gate(unwrap(model), gate_val)
                lr = wsd_lr(
                    tokens_seen,
                    self.cfg,
                    self.phase,
                    tokens_in_phase=tokens_in_phase,
                    phase_budget=phase_budget,
                )
                lr_opt = getattr(model, "optimizer", None) if self.deepspeed else opt
                if lr_opt is None:
                    lr_opt = opt
                for g in lr_opt.param_groups:
                    g["lr"] = lr
                if self.deepspeed:
                    zfn = getattr(model, "zero_grad", None)
                    if callable(zfn):
                        zfn()
                else:
                    opt.zero_grad(set_to_none=True)
                step_nll_w = 0.0
                step_loss = 0.0
                step_aux = 0.0
                step_tokens = 0
                step_n_valid = 0.0
                kd_w = 0.0
                step_recall = None
                step_reward = None
                if self.teacher is not None:
                    kd_w = kd_weight(
                        step,
                        max_steps,
                        self.cfg.kd_weight_start,
                        tokens_in_phase=tokens_in_phase,
                        phase_budget=phase_budget,
                    )
                t0 = time.perf_counter()
                stats_nll_w = stats_n_valid = stats_loss = stats_aux = None
                for micro_i in range(self.accum):
                    batch = self._prefetch_batch
                    self._prefetch_batch = None
                    if batch is None:
                        batch = stream.batch(self.micro_batch, self.device)
                    step_tokens += int(batch["input_ids"].numel())
                    last_micro = micro_i == self.accum - 1
                    with backward_sync_ctx(model, last_micro=last_micro, world=self.world):
                        with self._amp():
                            out = self._forward_loss(model, batch)
                            loss = out["loss"] / self.accum
                        if (
                            self.teacher is not None
                            and kd_w > 0
                            and self.loss_mode in {"ce", "sft"}
                            and "logits" in out
                        ):
                            with torch.no_grad():
                                t_logits = self.teacher(batch["input_ids"])["logits"]
                            shift_labels = batch["labels"][:, 1:]
                            loss = loss + kd_w * kd_kl(
                                out["logits"][:, :-1],
                                t_logits[:, :-1],
                                self.cfg.kd_temperature,
                                ignore=shift_labels,
                            )
                        if self.deepspeed:
                            model.backward(loss)
                        else:
                            loss.backward()
                    z = out["loss"].detach().reshape(()).float()
                    if stats_nll_w is None:
                        stats_nll_w = z.new_zeros(())
                        stats_n_valid = z.new_zeros(())
                        stats_loss = z.new_zeros(())
                        stats_aux = z.new_zeros(())
                    n_valid = out.get("n_valid")
                    n_valid_t = (
                        n_valid.detach().reshape(()).float()
                        if torch.is_tensor(n_valid)
                        else z.new_zeros(())
                    )
                    nll_t = out["nll"].detach().reshape(()).float()
                    stats_nll_w = stats_nll_w + torch.where(n_valid_t > 0, nll_t * n_valid_t, z.new_zeros(()))
                    stats_n_valid = stats_n_valid + n_valid_t
                    stats_loss = stats_loss + z / self.accum
                    aux = out.get("aux")
                    aux_t = aux.detach().reshape(()).float() if torch.is_tensor(aux) else z.new_zeros(())
                    stats_aux = stats_aux + aux_t / self.accum
                    rec = out.get("indexer_recall")
                    if torch.is_tensor(rec):
                        step_recall = rec.detach().float().reshape(())
                    rew = out.get("rl_reward")
                    if torch.is_tensor(rew):
                        step_reward = rew.detach().float().reshape(())
                allreduce_router_loads(model, device=str(self.device), world=self.world)
                next_step = step + 1
                will_log = next_step == 1 or next_step % self.log_every == 0 or (
                    max_steps is not None and next_step == max_steps
                )
                will_save = bool(self.save_every and next_step % self.save_every == 0)
                tokens_after = tokens_in_phase + step_tokens * self.world
                will_stop = (
                    (max_steps is not None and next_step >= max_steps)
                    or (phase_budget is not None and tokens_after >= phase_budget)
                    or (max_steps is None and phase_budget is None)
                )
                # Snapshot the stream before prefetch so resume does not skip the
                # already-copied next batch. H2D of the next DummyStream batch
                # overlaps leftover backward kernels.
                stream_snap = stream.state_dict() if will_save else None
                if not will_stop:
                    self._prefetch_batch = stream.batch(self.micro_batch, self.device)
                if will_log:
                    moe_stats = moe_utilization(unwrap(model))
                unwrap(model).step_router_bias()
                # Host D2H / DS grad-norm stall the CUDA pipeline. Only take them
                # on log steps so ZeRO prefetch can keep feeding the GPU.
                step_nll = last
                step_nll_w = step_n_valid = step_loss = step_aux = 0.0
                if will_log:
                    if stats_nll_w is None:
                        step_nll_w = step_n_valid = step_loss = step_aux = 0.0
                    else:
                        step_nll_w, step_n_valid, step_loss, step_aux = _host_step_stats(
                            stats_nll_w, stats_n_valid, stats_loss, stats_aux
                        )
                    step_nll_w = reduce_sum(step_nll_w, device=str(self.device), world=self.world)
                    step_n_valid = reduce_sum(step_n_valid, device=str(self.device), world=self.world)
                    step_nll = token_mean_nll(step_nll_w, step_n_valid)
                    step_loss = reduce_mean(step_loss, device=str(self.device), world=self.world)
                    step_aux = reduce_mean(step_aux, device=str(self.device), world=self.world)
                    if not math.isfinite(step_nll):
                        raise FloatingPointError(f"non-finite nll at step {step + 1}: {step_nll}")
                    spike = float(getattr(self.cfg, "nll_spike_factor", 0.0) or 0.0)
                    if spike > 0 and step >= 7 and getattr(self, "_nll_ema", None) is not None:
                        ema = float(self._nll_ema)
                        if step_nll > spike * max(ema, 1e-3):
                            raise RuntimeError(
                                f"nll spike {step_nll:.4f} > {spike:g}× ema {ema:.4f} "
                                f"at {self.phase} step {step + 1}"
                            )
                    if getattr(self, "_nll_ema", None) is None:
                        self._nll_ema = step_nll
                    else:
                        self._nll_ema = 0.9 * float(self._nll_ema) + 0.1 * step_nll
                grad_norm = 0.0
                if self.deepspeed:
                    if will_log:
                        gn_fn = getattr(model, "get_global_grad_norm", None)
                        gn_val = gn_fn() if callable(gn_fn) else None
                        try:
                            grad_norm = float(gn_val) if gn_val is not None else 0.0
                        except (TypeError, ValueError):
                            grad_norm = 0.0
                    model.step()
                elif self.offload_blocks:
                    leftover = [p for p in trainable if p.grad is not None]
                    if leftover:
                        gn_parts.append(clip_grad_norm_mixed(leftover, self.cfg.grad_clip))
                    grad_norm = math.sqrt(sum(g * g for g in gn_parts)) if gn_parts else 0.0
                    opt.step()
                else:
                    grad_norm = self._clip(model, trainable)
                    opt.step()
                dt = max(time.perf_counter() - t0, 1e-9)
                step += 1
                tokens_in_phase += step_tokens * self.world
                tokens_seen += step_tokens * self.world
                if will_log:
                    last = step_nll
                extra = None
                if will_save:
                    extra = self._extra(
                        model,
                        step,
                        tokens_in_phase,
                        tokens_seen,
                        stream,
                        gate=gate_val,
                        stream_state=stream_snap,
                        include_rng=True,
                    )
                if will_log:
                    row = {
                        "name": self.cfg.name,
                        "phase": self.phase,
                        "step": step,
                        "steps_or_inf": max_steps if max_steps is not None else "-",
                        "nll": last,
                        "ppl": safe_ppl(last),
                        "loss": step_loss,
                        "aux": step_aux,
                        "gate": gate_val,
                        "grad_norm": grad_norm,
                        "trainable_m": n_train / 1e6,
                        "lr": lr,
                        "fp8": use_fp8,
                        "nvfp4": use_nvfp4,
                        "nvfp4_n": self.nvfp4_n,
                        "grouped_mm": grouped_mm_available(),
                        "te_nvfp4": te_nvfp4_linear_enabled(),
                        "nvfp4_family": compute_family(),
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
                        "backend": "deepspeed" if self.deepspeed else "torch",
                        "zero": self.zero_stage,
                        "zero_offload": self.zero_offload,
                        "zero_offload_param": self.zero_offload_param,
                        "kd_w": kd_w,
                        "world": self.world,
                        "accum": self.accum,
                        "n_valid": step_n_valid,
                        "loss_mode": self.loss_mode,
                        "sparse": self.phase_sparse,
                        **moe_stats,
                    }
                    if step_recall is not None:
                        row["indexer_recall"] = float(step_recall.cpu())
                    if step_reward is not None:
                        row["rl_reward"] = float(step_reward.cpu())
                    if self.eval_every and self.eval_data is not None and step % self.eval_every == 0:
                        ev = self._allreduce_token_nll(*self._eval_nll_stats(model))
                        row["eval_nll"] = ev
                        row["eval_ppl"] = safe_ppl(ev)
                    self._log(row)
                elif self.eval_every and self.eval_data is not None and step % self.eval_every == 0:
                    ev = self._allreduce_token_nll(*self._eval_nll_stats(model))
                    if is_rank0(self.rank):
                        print(f"eval nll={ev:.4f} ppl={safe_ppl(ev) or '-'}")
                if will_save:
                    barrier()
                    self._maybe_save(model, opt, extra, f"step_{step}.pt")
                    barrier()
                if max_steps is None and phase_budget is None:
                    break
            extra = self._extra(model, step, tokens_in_phase, tokens_seen, stream)
            barrier()
            self._maybe_save(model, opt, extra, "latest.pt")
            barrier()
            peak = self._mem_mib()
            del opt
            if self.optim_cpu or self.offload_blocks or self.offload_encoder:
                trim_host_allocator()
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return TrainResult(
                nll=last,
                step=step,
                tokens_seen=tokens_seen,
                phase=self.phase,
                peak_mib=peak,
                stream=stream.state_dict(),
            )
        finally:
            set_after_block_backward(None)


def run_phase_chain(
    cfg: CATYokoConfig,
    device: str,
    phases: tuple[str, ...] | list[str],
    *,
    steps: int = 1,
    reuse_model: nn.Module | None = None,
    **kwargs,
) -> dict[str, TrainResult]:
    """Run ``phases`` on the same weights. Packed cursor continues.

    ``save_dir`` becomes ``save_dir/{phase}/``. Cross-phase resume still works
    as ``--phase NEXT --resume save_dir/PREV``.
    """
    dtype = kwargs.pop("dtype", "bf16" if str(device).startswith("cuda") else "fp32")
    save_root = kwargs.pop("save_dir", None)
    stream_state = kwargs.pop("initial_stream", None)
    if reuse_model is None:
        reuse_model = build_model(cfg, device, dtype=dtype)
        upcycle_src = kwargs.pop("upcycle_src", None)
        if upcycle_src is not None:
            upcycle_from_minicpm(unwrap(reuse_model), upcycle_src, cfg)
            del upcycle_src
    else:
        kwargs.pop("upcycle_src", None)
    out: dict[str, TrainResult] = {}
    offset = float(kwargs.pop("global_tokens_offset", 0.0))
    for phase in phases:
        phase_dir = Path(save_root) / phase if save_root is not None else None
        tr = Trainer(
            cfg,
            phase,
            device,
            steps=steps,
            dtype=dtype,
            reuse_model=reuse_model,
            global_tokens_offset=offset,
            save_dir=phase_dir,
            initial_stream=stream_state,
            **kwargs,
        )
        out[phase] = tr.run()
        offset = out[phase].tokens_seen
        stream_state = out[phase].stream
        if str(device).startswith("cuda") and torch.cuda.is_available():
            trim_host_allocator()
            torch.cuda.empty_cache()
    return out


def run_c1_chain(
    cfg: CATYokoConfig,
    device: str,
    *,
    steps: int = 1,
    reuse_model: nn.Module | None = None,
    **kwargs,
) -> dict[str, TrainResult]:
    """B0 then B1 then B2 on the same weights."""
    return run_phase_chain(
        cfg, device, ("B0", "B1", "B2"), steps=steps, reuse_model=reuse_model, **kwargs
    )


def run_c_chain(
    cfg: CATYokoConfig,
    device: str,
    *,
    steps: int = 1,
    reuse_model: nn.Module | None = None,
    **kwargs,
) -> dict[str, TrainResult]:
    return run_phase_chain(
        cfg,
        device,
        c_chain(use_kda=bool(getattr(cfg, "use_kda", False))),
        steps=steps,
        reuse_model=reuse_model,
        **kwargs,
    )


def run_d_chain(
    cfg: CATYokoConfig,
    device: str,
    *,
    steps: int = 1,
    reuse_model: nn.Module | None = None,
    **kwargs,
) -> dict[str, TrainResult]:
    from cat_yoko.phases import D_CHAIN

    return run_phase_chain(
        cfg, device, D_CHAIN, steps=steps, reuse_model=reuse_model, **kwargs
    )


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
