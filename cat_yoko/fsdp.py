"""Optional FSDP wrap. Single-process training leaves the model unwrapped."""

from __future__ import annotations

from functools import partial

from torch import nn

from cat_yoko.blocks import DecoderBlock, EncoderBlock
from cat_yoko.model import CATYokoForCausalLM


def wrap_fsdp(model: CATYokoForCausalLM, *, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("FSDP requires torch.distributed to be initialized")
    policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={EncoderBlock, DecoderBlock},
    )
    return FSDP(model, auto_wrap_policy=policy)
