"""torch.distributed init, DDP / FSDP wrap, rank helpers."""

from __future__ import annotations

import os
import socket
from contextlib import nullcontext

import torch
from torch import nn

from cat_yoko.fsdp import wrap_fsdp
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import unwrap


def distributed_requested() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def pick_backend(device: str) -> str:
    """NCCL only when every rank can own a distinct GPU; else gloo (1-GPU DDP)."""
    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    world = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    n_gpus = torch.cuda.device_count() if want_cuda else 0
    if want_cuda and n_gpus >= world:
        return "nccl"
    return "gloo"


def init_distributed(device: str, *, force: bool = False) -> tuple[str, int, int]:
    """Return (device, rank, world). ``force`` inits a 1-rank group for FSDP."""
    if not distributed_requested() and not force:
        return device, 0, 1
    import torch.distributed as dist

    if not dist.is_initialized():
        if not distributed_requested() and force:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(free_tcp_port()))
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(pick_backend(device))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if want_cuda:
        n_gpus = max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local % n_gpus)
        device = f"cuda:{local % n_gpus}"
    return device, rank, world


def wrap_distributed(
    model: CATYokoForCausalLM,
    *,
    fsdp: bool,
    ddp: bool,
) -> nn.Module:
    """Wrap after ``apply_freeze`` so C1 B0→B1 rebuilds the reducer."""
    model = unwrap(model)
    if fsdp:
        return wrap_fsdp(model, enabled=True)
    if not (ddp or distributed_requested()):
        return model
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DDP requires torch.distributed to be initialized")
    if dist.get_world_size() <= 1:
        return model
    device_ids = None
    if model.embed.weight.is_cuda:
        idx = model.embed.weight.device.index
        if idx is None:
            idx = torch.cuda.current_device()
        if torch.cuda.device_count() >= dist.get_world_size():
            device_ids = [idx]
    return DDP(model, device_ids=device_ids, find_unused_parameters=True)


def backward_sync_ctx(model: nn.Module, *, last_micro: bool, world: int):
    """DDP/FSDP ``no_sync`` on every accum step except the last."""
    if last_micro or world <= 1 or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: float, *, device: str, world: int) -> float:
    """Average a scalar across DDP ranks. No-op when world==1."""
    if world <= 1:
        return float(value)
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    dev = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    t = torch.tensor([float(value)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / world
