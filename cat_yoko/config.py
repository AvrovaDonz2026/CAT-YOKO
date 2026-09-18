"""Frozen 12B recipe. See docs/FROZEN_SPEC.md."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CATYokoConfig:
    name: str = "CAT-YOKO-12B"
    vocab_size: int = 122_753
    hidden_size: int = 2304
    num_heads: int = 36
    encoder_layers: int = 16
    decoder_layers: int = 24
    dense_intermediate_size: int = 5760
    moe_intermediate_size: int = 2048
    n_shared: int = 1
    n_routed_enc: int = 17
    n_routed_dec: int = 17
    top_k_enc: int = 6
    top_k_dec: int = 8
    first_dense: bool = False
    n_win: int = 8192
    compress_m: int = 4
    compress_m_hca: int = 128
    index_topk: int = 256
    hash_moe_decoder_layers: int = 2
    scale_emb: float = 12.0
    dim_model_base: int = 256
    scale_depth: float = 1.4
    base_layers: int = 40
    rms_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 10_000.0
    seq_len: int = 4096
    lr: float = 1e-4
    lr_b2: float = 3e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_tokens: float = 5e8
    global_batch_tokens: int = 4_194_304  # 1024 * 4096
    router_z_loss: float = 1e-4
    seq_balance_loss: float = 1e-3
    use_muon: bool = False
    use_fp8: bool = True
    attention_backend: str = "window"  # Phase B; "csa" is Phase C
    kd_temperature: float = 2.0
    kd_weight_start: float = 0.5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def residual_scale(self) -> float:
        return self.scale_depth / math.sqrt(self.base_layers)

    @property
    def logit_scale(self) -> float:
        return self.hidden_size / self.dim_model_base

    @property
    def n_routed(self) -> tuple[int, int]:
        return self.n_routed_enc, self.n_routed_dec

    def expert_count(self, stack: str) -> tuple[int, int, int]:
        """shared, routed, top_k for encoder or decoder."""
        if stack == "encoder":
            return self.n_shared, self.n_routed_enc, self.top_k_enc
        if stack == "decoder":
            return self.n_shared, self.n_routed_dec, self.top_k_dec
        raise ValueError(stack)

    @classmethod
    def middle_12b(cls) -> CATYokoConfig:
        return cls()

    @classmethod
    def tiny(cls) -> CATYokoConfig:
        return replace(
            cls(),
            name="tiny",
            vocab_size=128,
            hidden_size=64,
            num_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            dense_intermediate_size=128,
            moe_intermediate_size=32,
            n_shared=1,
            n_routed_enc=4,
            n_routed_dec=4,
            top_k_enc=2,
            top_k_dec=2,
            n_win=8,
            hash_moe_decoder_layers=1,
            seq_len=16,
            lr=3e-4,
            use_fp8=False,
            global_batch_tokens=128,
        )


def encoder_layer_kind(index: int) -> str:
    """Phase C labels. Phase B still runs sliding-window MHA on every encoder layer."""
    if index < 2:
        return "sliding"
    return "csa" if (index - 2) % 2 == 0 else "hca"


# Phase B C1 split (tokens).
C1_SPLIT = {"B0": 8e9, "B1": 27e9, "B2": 15e9}
FP8_KEEP_HIGH_PREC = (
    "embed",
    "lm_head",
    "norm",
    "router",
    "gate",
    "indexer",
    "qk_norm",
)
