"""Megatron-LM / Megatron-Core adapter (optional backend).

Upstream: https://github.com/NVIDIA/Megatron-LM

This package does **not** vendor Megatron and does **not** import it at
package import time. The reference graph remains ``cat_yoko.model``.

YOCO is not a Megatron ``GPTModel``: two stacks, different top-k, a single
global cache, scheduled gated cross-attn, GQA, and untied MiniCPM5 embeddings.
Mapping emits two ``TransformerConfig`` dicts plus a ``yoco`` extras block that a
custom ``MegatronModule`` must implement.

Install later with ``pip install 'cat-yoko[megatron]'`` or from the NVIDIA repo.
"""

from cat_yoko.megatron.mapping import (
    MEGATRON_LM,
    MEGATRON_CUSTOM_SURFACE,
    megatron_blueprint,
    transformer_config_dict,
)

__all__ = [
    "MEGATRON_LM",
    "MEGATRON_CUSTOM_SURFACE",
    "megatron_blueprint",
    "transformer_config_dict",
]
