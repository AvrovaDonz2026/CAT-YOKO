"""Layer-spec hook for a future MegatronModule (Phase B window, Phase C CSA)."""

from __future__ import annotations

from cat_yoko.config import CATYokoConfig, encoder_layer_kind
from cat_yoko.megatron.provider import MegatronBackendNotReady, import_megatron_core


def encoder_layer_kinds(cfg: CATYokoConfig) -> list[str]:
    return [encoder_layer_kind(i, cfg.encoder_layers) for i in range(cfg.encoder_layers)]


def encoder_module_spec(cfg: CATYokoConfig, *, phase: str = "B0") -> list:
    """Placeholder: one ModuleSpec per encoder layer.

    Phase B: every layer is sliding-window MHA even when the kind tag is csa/hca.
    Phase C: swap csa/hca indices for Megatron CSA specs (ratios 0/4/128).
    """
    import_megatron_core()
    kinds = encoder_layer_kinds(cfg)
    raise MegatronBackendNotReady(
        f"encoder ModuleSpec not built (phase={phase}, kinds={kinds}). "
        "Keep the torch EncoderBlock as the numerical oracle."
    )


def decoder_module_spec(cfg: CATYokoConfig, *, phase: str = "B0") -> list:
    import_megatron_core()
    raise MegatronBackendNotReady(
        f"decoder ModuleSpec not built (phase={phase}, layers={cfg.decoder_layers}, "
        f"hash_moe={cfg.hash_moe_decoder_layers}). Cross-attn + global cache are custom."
    )
