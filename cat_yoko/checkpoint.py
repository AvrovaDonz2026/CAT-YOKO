"""Checkpoint save/load. Rank-0 full state; FSDP uses FULL_STATE_DICT when wrapped."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from cat_yoko.optim import unwrap


def _is_fsdp(model: nn.Module) -> bool:
    return type(model).__name__ == "FullyShardedDataParallel"


def _tensor_cpu(t: torch.Tensor) -> torch.Tensor:
    """Host copy so torch.save does not clone CUDA storage (12B ~23GiB)."""
    return t.detach().contiguous().cpu()


def _cpu_copy(obj: Any) -> Any:
    """Copy tensors onto CPU without mutating the live object graph.

    ``Optimizer.state_dict()`` aliases live moment tensors; in-place ``.cpu()``
    would yank GPU AdamW state off device. CPUOffloadAdamW moments are already
    on host, so ``.cpu()`` is a no-op on storage.
    """
    if torch.is_tensor(obj):
        return _tensor_cpu(obj)
    if isinstance(obj, dict):
        return {k: _cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_copy(v) for v in obj)
    return obj


def model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return {
        k: v.detach().contiguous().cpu() if torch.is_tensor(v) else v
        for k, v in unwrap(model).state_dict().items()
    }


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if _is_fsdp(model):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            model.load_state_dict(state)
        return
    unwrap(model).load_state_dict(state)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    extra: dict[str, Any],
    save_optimizer: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_sd = model_state_dict(model)
    # FSDP already offloads via FullStateDictConfig; still pin every tensor on CPU.
    model_sd = _cpu_copy(model_sd)
    opt_sd = None
    if save_optimizer and optimizer is not None:
        opt_sd = _cpu_copy(optimizer.state_dict())
    payload = {
        "model": model_sd,
        "optimizer": opt_sd,
        "extra": extra,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def prune_step_checkpoints(save_dir: Path, keep: int) -> None:
    """Keep the newest ``step_*.pt`` files; ``latest.pt`` is not touched."""
    if keep <= 0:
        return
    files = []
    for p in Path(save_dir).glob("step_*.pt"):
        try:
            files.append((int(p.stem.split("_", 1)[1]), p))
        except (IndexError, ValueError):
            continue
    files.sort()
    for _, old in files[:-keep]:
        old.unlink(missing_ok=True)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_optimizer_state(optimizer: Optimizer | None, state: dict | None) -> None:
    """Load Adam state. Checkpoints are read on CPU so 12B does not double VRAM.

    GPU AdamW moments are then copied onto each param's device. CPU-offload
    AdamW keeps moments on host (see ``CPUOffloadAdamW.load_state_dict``).
    """
    if optimizer is None or not state:
        return
    optimizer.load_state_dict(state)
    from cat_yoko.optim import CPUOffloadAdamW

    if isinstance(optimizer, CPUOffloadAdamW):
        return
    for p, st in optimizer.state.items():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=p.device, dtype=v.dtype)
