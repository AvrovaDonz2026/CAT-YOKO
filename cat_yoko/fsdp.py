"""Optional FSDP wrap. Single-process training leaves the model unwrapped unless --fsdp."""

from __future__ import annotations

from functools import partial

from torch import nn

from cat_yoko.blocks import DecoderBlock, EncoderBlock
from cat_yoko.model import CATYokoForCausalLM


def wrap_fsdp(model: CATYokoForCausalLM, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("FSDP requires torch.distributed to be initialized")
    policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={EncoderBlock, DecoderBlock},
    )
    device_id = None
    mixed = None
    p = next(model.parameters(), None)
    if p is not None and p.is_cuda:
        idx = p.device.index
        device_id = idx if idx is not None else torch.cuda.current_device()
        from torch.distributed.fsdp import MixedPrecision

        if p.dtype == torch.float32:
            mixed = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
    from torch.distributed.fsdp import ShardingStrategy

    strategy = (
        ShardingStrategy.NO_SHARD if dist.get_world_size() == 1 else ShardingStrategy.FULL_SHARD
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=device_id,
        use_orig_params=True,
        mixed_precision=mixed,
        sync_module_states=True,
        sharding_strategy=strategy,
    )
