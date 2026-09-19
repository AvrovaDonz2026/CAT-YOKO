"""Checkpoint save/load. Rank-0 full state; FSDP uses FULL_STATE_DICT when wrapped."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from cat_yoko.optim import unwrap

_SAVE_MARGIN = 1 << 30  # 1 GiB headroom so 12B pickle does not fill the volume mid-write.


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


def _nbytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(v) for v in obj)
    return 0


def cleanup_save_tmp(save_dir: Path | None) -> None:
    """Drop leftover ``*.pt.tmp`` from a crashed 12B ``torch.save``."""
    if save_dir is None:
        return
    root = Path(save_dir)
    if not root.is_dir():
        return
    for p in root.glob("*.pt.tmp"):
        p.unlink(missing_ok=True)


def newest_step_checkpoint(save_dir: Path) -> Path | None:
    files: list[tuple[int, Path]] = []
    for p in Path(save_dir).glob("step_*.pt"):
        try:
            files.append((int(p.stem.split("_", 1)[1]), p))
        except (IndexError, ValueError):
            continue
    if not files:
        return None
    files.sort()
    return files[-1][1]


def newest_trainable_checkpoint(save_dir: Path) -> Path | None:
    files: list[tuple[int, Path]] = []
    for p in Path(save_dir).glob("trainable_step_*.pt"):
        try:
            files.append((int(p.stem.rsplit("_", 1)[1]), p))
        except (IndexError, ValueError):
            continue
    if not files:
        return None
    files.sort()
    return files[-1][1]


def resolve_resume_path(path: Path | str) -> Path:
    """File as-is; directory prefers full ``latest.pt``, then newest overlay step.

    ``trainable.pt`` may be a stale hardlink (live 6000D B0 wrote
    ``trainable_step_N`` without republishing the pointer). Prefer the
    highest ``trainable_step_*.pt`` so operator restarts keep tokens_in_phase.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        latest = path / "latest.pt"
        if latest.is_file():
            return latest
        step = newest_trainable_checkpoint(path) or newest_step_checkpoint(path)
        if step is not None:
            return step
        trainable = path / "trainable.pt"
        if trainable.is_file():
            return trainable
        raise FileNotFoundError(f"no latest.pt / trainable.pt / step_*.pt in {path}")
    raise FileNotFoundError(str(path))


def require_host_bytes(nbytes: int, *, what: str) -> None:
    """Fail before the 23GiB CPU copy if the cgroup cannot hold it."""
    if nbytes <= 0:
        return
    from cat_yoko.optim import host_memory_limit_bytes, host_memory_used_bytes

    limit = host_memory_limit_bytes()
    if limit is None:
        return
    used = host_memory_used_bytes()
    extra = (2 << 30) if nbytes >= 8 * (1 << 30) else (64 << 20)
    if used + nbytes + extra <= limit:
        return
    raise OSError(
        f"not enough host RAM for {what}: need {(nbytes + extra) / 2**30:.1f}GiB extra "
        f"(cgroup used {used / 2**30:.1f} / limit {limit / 2**30:.1f}GiB). "
        "12B save is ~23GiB CPU tensors; do not combine with B1 CPU Adam moments. "
        "Use --no-save-optim (default) and --steps 1 on a 62GiB cgroup."
    )


def _live_param_nbytes(model: nn.Module) -> int:
    n = 0
    raw = unwrap(model)
    for p in raw.parameters():
        n += int(p.numel() * p.element_size())
    for b in raw.buffers():
        n += int(b.numel() * b.element_size())
    return n


def require_free_bytes(directory: Path, nbytes: int, *, what: str) -> None:
    """Fail before ``torch.save`` if the volume cannot hold ``nbytes`` + 1 GiB."""
    if nbytes <= 0:
        return
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    need = nbytes + _SAVE_MARGIN
    if free >= need:
        return
    raise OSError(
        f"not enough disk for {what}: need {need / 2**30:.1f}GiB "
        f"(payload {nbytes / 2**30:.1f}GiB + 1GiB), have {free / 2**30:.1f}GiB free in {directory}. "
        "12B checkpoints belong on a large volume (e.g. /root/autodl-tmp), not /tmp. "
        "If step_N.pt already exists, resume from that file or the save directory."
    )


def publish_latest(step_path: Path, latest_path: Path | None = None) -> Path:
    """Point ``latest.pt`` at ``step_N.pt`` without a second 23GiB ``torch.save``.

    Same-directory hardlink when the FS allows it (nlink=2, no extra bytes).
    Copy only if ``os.link`` fails. Unlink a leftover ``latest.pt.tmp`` first.
    """
    step_path = Path(step_path)
    if not step_path.is_file():
        raise FileNotFoundError(str(step_path))
    latest = Path(latest_path) if latest_path is not None else step_path.parent / "latest.pt"
    tmp = latest.with_name(latest.name + ".tmp")
    tmp.unlink(missing_ok=True)
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_file() and step_path.is_file() and latest.samefile(step_path):
                return latest
        except OSError:
            pass
        latest.unlink()
    try:
        os.link(step_path, latest)
        return latest
    except OSError:
        require_free_bytes(latest.parent, step_path.stat().st_size, what=str(latest))
        shutil.copy2(step_path, latest)
        return latest


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


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """B0/B1 Hub overlay: only ``requires_grad`` params (B0 ≈ 219M, ~0.44GiB bf16)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in unwrap(model).named_parameters():
        if p.requires_grad:
            out[name] = _tensor_cpu(p)
    return out


def is_trainable_ckpt(ckpt: dict[str, Any]) -> bool:
    return str(ckpt.get("kind", "")) == "trainable" or (
        isinstance(ckpt.get("trainable"), dict) and "model" not in ckpt
    )


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Overlay saved new-module / unfrozen weights. Frozen tensors stay as-is."""
    raw = unwrap(model)
    current = raw.state_dict()
    mapped = {k: v for k, v in state.items() if k in current}
    raw.load_state_dict(mapped, strict=False)


def save_trainable_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    extra: dict[str, Any],
) -> None:
    """Trainable overlay for HuggingFace Hub. Resume = MiniCPM5 upcycle + this file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = trainable_state_dict(model)
    payload = {
        "kind": "trainable",
        "trainable": sd,
        "extra": extra,
        "n_tensors": len(sd),
        "nbytes": _nbytes(sd),
    }
    require_host_bytes(_nbytes(sd), what=str(path))
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    require_free_bytes(path.parent, _nbytes(payload), what=str(path))
    torch.save(payload, tmp)
    tmp.replace(path)


def shard_state_dict(
    state: dict[str, torch.Tensor],
    *,
    max_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    """Split a full 12B dict into Hub-sized pieces (≤4GiB)."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    size = 0
    for key, tensor in state.items():
        n = _nbytes(tensor)
        if cur and size + n > max_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[key] = tensor
        size += n
    if cur:
        shards.append(cur)
    return shards or [{}]


def save_sharded_checkpoint(
    directory: Path,
    *,
    model: nn.Module,
    extra: dict[str, Any],
    max_bytes: int | None = None,
) -> Path:
    """Write ``shard-00000.pt`` … plus ``manifest.json``. B1/B2 full graphs."""
    from cat_yoko.phases import HUB_SHARD_MAX_BYTES

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cap = int(max_bytes) if max_bytes is not None else HUB_SHARD_MAX_BYTES
    sd = _cpu_copy(model_state_dict(model))
    shards = shard_state_dict(sd, max_bytes=cap)
    names: list[str] = []
    for i, piece in enumerate(shards):
        name = f"shard-{i:05d}.pt"
        dest = directory / name
        payload = {"kind": "shard", "index": i, "model": piece}
        require_host_bytes(_nbytes(piece), what=str(dest))
        require_free_bytes(directory, _nbytes(payload), what=str(dest))
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.unlink(missing_ok=True)
        torch.save(payload, tmp)
        tmp.replace(dest)
        names.append(name)
    import json

    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"kind": "sharded", "shards": names, "n_shards": len(names), "extra": extra},
            default=str,
        )
        + "\n"
    )
    return manifest


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
    est = _live_param_nbytes(model)
    if save_optimizer and optimizer is not None:
        est *= 2
    require_host_bytes(est, what=str(path))
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
    tmp.unlink(missing_ok=True)
    require_free_bytes(path.parent, _nbytes(payload), what=str(path))
    torch.save(payload, tmp)
    tmp.replace(path)


def _latest_inode(save_dir: Path) -> int | None:
    """Inode of ``latest.pt`` when it is a real file. Dangling links are ignored."""
    latest = Path(save_dir) / "latest.pt"
    try:
        # exists() is False for a dangling symlink; do not follow it.
        if not latest.exists() or not latest.is_file():
            return None
        return latest.stat().st_ino
    except OSError:
        return None


def prune_named_checkpoints(
    save_dir: Path,
    keep: int,
    *,
    glob: str,
    latest_name: str,
) -> None:
    if keep <= 0:
        return
    save_dir = Path(save_dir)
    latest = save_dir / latest_name
    latest_ino: int | None = None
    try:
        if latest.exists() and latest.is_file():
            latest_ino = latest.stat().st_ino
    except OSError:
        latest_ino = None
    files: list[tuple[int, Path]] = []
    for p in save_dir.glob(glob):
        try:
            files.append((int(p.stem.rsplit("_", 1)[1]), p))
        except (IndexError, ValueError):
            continue
    files.sort()
    for _, old in files[:-keep]:
        try:
            if latest_ino is not None and old.stat().st_ino == latest_ino:
                continue
        except OSError:
            pass
        old.unlink(missing_ok=True)


def prune_step_checkpoints(save_dir: Path, keep: int) -> None:
    """Keep the newest ``step_*.pt`` files; ``latest.pt`` is not touched.

    Never unlink a ``step_*.pt`` that shares an inode with ``latest.pt``
    (hardlink). POSIX unlink of one name in an nlink=2 pair would leave
    the bytes, but the step name must still stay so ``latest.pt`` is not
    the only remaining link. A copy of ``latest.pt`` (separate inode) does
    not protect the step file. A dangling ``latest.pt`` is ignored.
    """
    prune_named_checkpoints(save_dir, keep, glob="step_*.pt", latest_name="latest.pt")
    prune_named_checkpoints(
        save_dir, keep, glob="trainable_step_*.pt", latest_name="trainable.pt"
    )


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
