"""torch.distributed init, DDP / FSDP wrap, rank helpers."""

from __future__ import annotations

import os

import torch
from torch import nn

from cat_yoko.fsdp import wrap_fsdp
from cat_yoko.model import CATYokoForCausalLM


def distributed_requested() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed(device: str) -> tuple[str, int, int]:
    """Return (device, rank, world). Initializes the process group when WORLD_SIZE>1."""
    if not distributed_requested():
        return device, 0, 1
    import torch.distributed as dist

    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not dist.is_initialized():
        backend = "nccl" if want_cuda else "gloo"
        dist.init_process_group(backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    if want_cuda:
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    return device, rank, world


def wrap_distributed(
    model: CATYokoForCausalLM,
    *,
    fsdp: bool,
    ddp: bool,
) -> nn.Module:
    if fsdp:
        return wrap_fsdp(model, enabled=True)
    if ddp or distributed_requested():
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        if not dist.is_initialized():
            raise RuntimeError("DDP requires torch.distributed to be initialized")
        device_ids = [model.embed.weight.device.index] if model.embed.weight.is_cuda else None
        return DDP(model, device_ids=device_ids, find_unused_parameters=True)
    return model


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
