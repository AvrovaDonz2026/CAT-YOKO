"""DeepSpeed ZeRO backend. Optional extra; CI does not install DeepSpeed.

Not Megatron EP/TP. Not a CSA kernel. Overlay save gathers ZeRO-3 16-bit
trainable weights only (not frozen 12B); do not ``--save-full``. Native encoder/block/Adam CPU offload stays
on the torch path — ZeRO owns partitioning when this backend is on.

Single-GPU ZeRO-1/2 does **not** shard frozen 12B weights. A 48GiB Ampere
card needs ZeRO-3 + optimizer CPU offload + param CPU offload.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from typing import Any

from torch import nn

from cat_yoko.optim import unwrap

DEEPSPEED = "https://github.com/microsoft/DeepSpeed"

_IMPORT_HINT = (
    "DeepSpeed backend is reserved but not installed.\n"
    f"  pip install 'cat-yoko[deepspeed]'   # or: pip install deepspeed\n"
    f"Upstream: {DEEPSPEED}\n"
    "3090 48GiB: --backend deepspeed --zero 3 --zero-offload --zero-offload-param. "
    "ZeRO-1/2 on one GPU does not shard the frozen 12B weights."
)

AMPERE_48GIB_ARGV = (
    "--backend",
    "deepspeed",
    "--zero",
    "3",
    "--zero-offload",
    "--zero-offload-param",
    "--no-offload-encoder",
    "--no-offload-blocks",
    "--no-optim-cpu",
)

NOTES = (
    "Freeze + NVFP4 wrap before deepspeed.initialize.",
    "Do not wrap DDP/FSDP around the engine.",
    "Disable native --offload-encoder/--offload-blocks/--optim-cpu.",
    "ZeRO-3 overlay: GatheredParameters on requires_grad only. Do not --save-full.",
    "Single-GPU ZeRO-1/2 does not shard 12B frozen weights; 48GiB needs ZeRO-3 + param CPU offload.",
    "Not Megatron EP/TP. Not a CSA kernel. Not a 50B download.",
    "C1 chain: run B0/B1/B2 as separate processes with --resume; do not --c1 in-process.",
    "ZeRO fits memory. It does not make the 8e9 B0 envelope sane on one 3090.",
    "Single-process python -m does not need the deepspeed launcher; seed LOCAL_RANK=0.",
    "Warm up ZeRO-3 with one dummy backward (no Adam step) so the first timed step is not the allgather-trace pass.",
)


class DeepSpeedNotInstalled(ImportError):
    pass


class DeepSpeedBackendError(RuntimeError):
    """Installed DeepSpeed cannot serve this request (CPU, C1 reuse, gather)."""


def import_deepspeed() -> Any:
    try:
        import deepspeed as ds  # noqa: F401
    except ImportError as exc:
        raise DeepSpeedNotInstalled(_IMPORT_HINT) from exc
    return ds


def resolve_zero_stage(stage: int | None, *, offload_param: bool = False) -> int:
    """Param CPU offload is ZeRO-3 only. Default stage is 2."""
    if offload_param:
        return 3
    if stage is None:
        return 2
    stage_i = int(stage)
    if stage_i not in {1, 2, 3}:
        raise ValueError(f"ZeRO stage must be 1, 2, or 3; got {stage}")
    return stage_i


def zero_config(
    *,
    stage: int | None = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    bf16: bool = True,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    train_micro_batch_size_per_gpu: int = 1,
    overlap_comm: bool = True,
) -> dict[str, Any]:
    """JSON-serializable DeepSpeed config. Does not import DeepSpeed."""
    stage_i = resolve_zero_stage(stage, offload_param=offload_param)
    offload_optimizer = bool(offload_optimizer or offload_param)
    zero: dict[str, Any] = {
        "stage": stage_i,
        "overlap_comm": bool(overlap_comm),
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "round_robin_gradients": True,
    }
    if offload_optimizer:
        zero["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 8,
        }
    if offload_param:
        # Extra pinned buffers so the next MoE layer can H2D while GEMM runs.
        zero["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 8,
        }
    if stage_i == 3:
        zero["stage3_gather_16bit_weights_on_model_save"] = True
        # Params smaller than this stay gathered. 5e6 would keep every
        # 2048×2048 Q/O resident; on 3090 ZeRO-3+offload that OOM'd at
        # post-step persistent all_gather (47.34/47.41 GiB). Keep 1e6 so
        # Q/O (4.19e6) still stream. Experts 12.6e6 stay offloaded.
        zero["stage3_param_persistence_threshold"] = 1_000_000
        # Default max_live=1e9 (~2GiB). One frozen MoE layer is ~0.75e9 plus
        # a 0.5e9 prefetch bucket, so the default silently dropped prefetch
        # and GPU util fell to 0% for ~1s every step.
        zero["stage3_max_live_parameters"] = 2_000_000_000
        zero["stage3_max_reuse_distance"] = 2_000_000_000
        # 50e6 (~100MiB bf16) starved Ampere: a frozen MoE layer is ~1.5GiB
        # experts and GPU util dropped to 0 waiting on PCIe. DeepSpeed default
        # is 5e8 (~1GiB bf16). 3090 ZeRO-3+offload leaves ~3GiB headroom
        # (measured ~45.6/49.1GiB with the old 50e6 bucket).
        zero["stage3_prefetch_bucket_size"] = 500_000_000
        zero["reduce_bucket_size"] = 500_000_000
    cfg: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": int(max(train_micro_batch_size_per_gpu, 1)),
        "gradient_accumulation_steps": int(max(gradient_accumulation_steps, 1)),
        "gradient_clipping": float(gradient_clipping),
        "zero_optimization": zero,
        "zero_allow_untested_optimizer": True,
        # Trainer passes DeepSpeedCPUAdam (two param groups) when available.
        # Do not let ZeRO flatten them into a single decay group.
        "zero_force_ds_cpu_optimizer": False,
        "steps_per_print": 2_147_483_647,
        "wall_clock_breakdown": False,
        "prescale_gradients": False,
        # Model already has --grad-ckpt. Do not enable DS activation checkpointing.
        "activation_checkpointing": {"partition_activations": False},
        "bf16": {"enabled": bool(bf16)},
        "fp16": {"enabled": False},
    }
    return cfg


def dump_zero_config(
    *,
    stage: int | None = 2,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    bf16: bool = True,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    train_micro_batch_size_per_gpu: int = 1,
    phase: str = "B0",
    file=None,
) -> dict[str, Any]:
    """Print the ZeRO mapping JSON (no DeepSpeed import) and return it."""
    config = zero_config(
        stage=stage,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
        bf16=bf16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clipping=gradient_clipping,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
    )
    payload = {
        "upstream": DEEPSPEED,
        "backend": "deepspeed",
        "phase": phase,
        "zero": config,
        "ampere_48gib_argv": list(AMPERE_48GIB_ARGV),
        "notes": list(NOTES),
    }
    json.dump(payload, file or sys.stdout, indent=2)
    (file or sys.stdout).write("\n")
    return payload


def is_deepspeed_engine(model: nn.Module) -> bool:
    name = type(model).__name__
    if name in {"DeepSpeedEngine", "PipelineEngine"}:
        return True
    mod = type(model).__module__
    return "deepspeed" in mod and hasattr(model, "backward") and hasattr(model, "module")


def zero_stage_of(model: nn.Module) -> int:
    if not is_deepspeed_engine(model):
        return 0
    fn = getattr(model, "zero_optimization_stage", None)
    if callable(fn):
        try:
            return int(fn())
        except Exception:
            return 0
    return 0


def is_zero_partitioned(model: nn.Module) -> bool:
    """True after ZeRO-3 has replaced Parameters with partitioned tensors."""
    for p in unwrap(model).parameters():
        if hasattr(p, "ds_id"):
            return True
    return False


def seed_single_process_rank_env() -> None:
    """DeepSpeed asserts ``LOCAL_RANK`` even without the deepspeed launcher.

    ``python -m cat_yoko.b0 --backend deepspeed`` on one 3090 is world=1.
    Do not require ``deepspeed --num_gpus 1``. Leave already-set launcher env.
    """
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")
    if "MASTER_PORT" not in os.environ:
        from cat_yoko.dist_util import free_tcp_port

        os.environ["MASTER_PORT"] = str(free_tcp_port())


def wrap_deepspeed(
    model: nn.Module,
    optimizer,
    config: dict[str, Any],
    *,
    dist_init_required: bool | None = None,
):
    """``deepspeed.initialize`` after freeze + NVFP4. Do not wrap DDP/FSDP first."""
    if is_deepspeed_engine(model):
        raise DeepSpeedBackendError("model is already a DeepSpeed engine")
    raw = unwrap(model)
    if is_zero_partitioned(raw):
        raise DeepSpeedBackendError(
            "cannot re-wrap a ZeRO-3 partitioned module; "
            "run B0/B1/B2 as separate processes with --resume overlay"
        )
    seed_single_process_rank_env()
    ds = import_deepspeed()
    import torch.distributed as dist

    if dist_init_required is None:
        dist_init_required = not (dist.is_available() and dist.is_initialized())
    engine, opt, _, _ = ds.initialize(
        model=raw,
        optimizer=optimizer,
        config=config,
        dist_init_required=dist_init_required,
    )
    return engine, opt


def warmup_zero3(engine: nn.Module, loss) -> bool:
    """Record the ZeRO-3 allgather trace with one dummy backward.

    Does **not** ``engine.step()``: no Adam, no overlay write, no token
    accounting. Caller restores RNG. False if ``engine`` is not DeepSpeed.
    """
    if not is_deepspeed_engine(engine):
        return False
    engine.backward(loss)
    zfn = getattr(engine, "zero_grad", None)
    if callable(zfn):
        zfn()
    return True


def _dist_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _cpu_sd(state: dict[str, Any] | None) -> dict[str, Any] | None:
    if state is None:
        return None
    import torch

    out: dict[str, Any] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().contiguous().cpu()
        else:
            out[key] = value
    return out


def gathered_state_dict(model: nn.Module) -> dict[str, Any] | None:
    """All ranks must enter on ZeRO-3. Rank 0 returns a CPU 16-bit dict.

    Non-rank-0 typically gets ``None``. Callers that are not a DeepSpeed
    engine, or are ZeRO-1/2 (full params on each rank), get ``None`` so the
    torch ``state_dict`` path can run on rank 0 only.
    """
    if not is_deepspeed_engine(model) or zero_stage_of(model) < 3:
        return None
    fn = getattr(model, "_zero3_consolidated_16bit_state_dict", None)
    if not callable(fn):
        raise DeepSpeedBackendError(
            "ZeRO-3 overlay/full gather needs consolidate 16-bit weights"
        )
    try:
        state = fn(exclude_frozen_parameters=False)
    except TypeError:
        state = fn()
    return _cpu_sd(state) if isinstance(state, dict) else state


def gathered_trainable_state_dict(
    model: nn.Module,
    names: set[str] | frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """ZeRO-3: all ranks enter; rank 0 returns the Hub overlay tensors.

    ``None`` means “not ZeRO-3; use ``trainable_state_dict`` on rank 0”.
    An empty dict is the non-rank-0 ZeRO-3 result (still participated).

    Do **not** call the full-model ZeRO consolidate helper. That walk still
    ``GatheredParameters`` every layer (frozen 12B included) and only skips the
    CPU copy. On 3090 that is ~6s of SM 0% every ``SAVE_EVERY`` while PCIe
    streams experts that the overlay never stores.
    """
    if not is_deepspeed_engine(model) or zero_stage_of(model) < 3:
        return None
    ds = import_deepspeed()
    gp = getattr(getattr(ds, "zero", None), "GatheredParameters", None)
    if not callable(gp):
        raise DeepSpeedBackendError("ZeRO-3 overlay gather needs GatheredParameters")
    named = [(n, p) for n, p in unwrap(model).named_parameters() if p.requires_grad]
    if names is not None:
        want = set(names)
        named = [(n, p) for n, p in named if n in want]
    params = [p for _, p in named]
    out: dict[str, Any] = {}
    ctx = gp(params, modifier_rank=0) if params else nullcontext()
    with ctx:
        if _dist_rank() == 0:
            for key, param in named:
                out[key] = param.detach().contiguous().cpu()
    return out
