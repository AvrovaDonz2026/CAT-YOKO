"""Frozen 12B recipe. See docs/FROZEN_SPEC.md.

Base checkpoint: Apache-2.0 MiniCPM5-2B (Llama GQA), not MiniCPM-2B-sft-bf16.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CATYokoConfig:
    name: str = "CAT-YOKO-12B"
    vocab_size: int = 130_560
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 2
    encoder_layers: int = 16
    decoder_layers: int = 26
    dense_intermediate_size: int = 6144
    moe_intermediate_size: int = 2048
    n_shared: int = 1
    n_routed_enc: int = 20
    n_routed_dec: int = 20
    top_k_enc: int = 7
    top_k_dec: int = 10
    first_dense: bool = False
    n_win: int = 8192
    compress_m: int = 4
    compress_m_hca: int = 128
    index_topk: int = 256
    hash_moe_decoder_layers: int = 2
    # MiniCPM5 is Llama. μP fields exist only so a mistaken MiniCPM-2B copy
    # cannot silently divide logits; published 12B keeps use_mup=False.
    use_mup: bool = False
    scale_emb: float = 1.0
    dim_model_base: int = 2048
    scale_depth: float = 1.0
    base_layers: int = 42
    tie_embeddings: bool = False
    rms_eps: float = 1e-6
    qk_norm: bool = True
    rope_theta: float = 5_000_000.0
    seq_len: int = 4096
    indexer_dim: int = 64
    grpo_group: int = 4
    grpo_max_new: int = 32
    dpo_beta: float = 0.1
    wsd_decay_min_ratio: float = 0.01
    nll_spike_factor: float = 8.0
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
    use_nvfp4: bool = True  # published compute dtype; Nvfp4Linear wrap
    use_fp8: bool = True  # Hopper/Ada fallback placeholder
    attention_backend: str = "window"  # Phase B; "csa" is Phase C
    kd_temperature: float = 2.0
    kd_weight_start: float = 0.5

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def residual_scale(self) -> float:
        if not self.use_mup:
            return 1.0
        import math

        return self.scale_depth / math.sqrt(self.base_layers)

    @property
    def embed_scale(self) -> float:
        return float(self.scale_emb) if self.use_mup else 1.0

    @property
    def logit_scale(self) -> float:
        if not self.use_mup:
            return 1.0
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
            num_kv_heads=2,
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
            compress_m_hca=4,
            hash_moe_decoder_layers=1,
            dim_model_base=64,
            seq_len=16,
            indexer_dim=16,
            grpo_group=2,
            grpo_max_new=4,
            lr=3e-4,
            use_nvfp4=False,
            use_fp8=False,
            global_batch_tokens=128,
            # Tiny is test-only. Match MiniCPM5 RMS; keep short-rope for seq_len=16.
            rope_theta=10_000.0,
            rms_eps=1e-6,
        )


def encoder_layer_kind(index: int, n_layers: int = 16) -> str:
    """Phase C labels. Phase B still runs sliding-window GQA on every encoder layer.

    12B (16 layers) is 2 sliding + 7 CSA + 7 HCA. Tiny graphs that cannot
    hold that schedule keep a sliding bootstrap then CSA/HCA as they fit,
    so C-index has an encoder CSA indexer to train.
    """
    n = int(n_layers)
    i = int(index)
    if n >= 16:
        if i < 2:
            return "sliding"
        return "csa" if (i - 2) % 2 == 0 else "hca"
    if i == 0:
        return "sliding"
    return "csa" if (i - 1) % 2 == 0 else "hca"


# Phase B C1 split (tokens).
C1_SPLIT = {"B0": 8e9, "B1": 27e9, "B2": 15e9}
# Must-high-prec ops. Everything else that is a linear GEMM is NVFP4
# (B1/B2 student; frozen-encoder forward from B0). lm_head is a GEMM.
KEEP_HIGH_PREC = (
    "embed",
    "rms_norm",
    "qk_norm",
    "router",
    "gate",
    "indexer",
    "attn_softmax",
)
FP8_KEEP_HIGH_PREC = KEEP_HIGH_PREC
NVFP4_KEEP_HIGH_PREC = KEEP_HIGH_PREC
NVFP4_GEMM_SLOTS = (
    "moe_expert",
    "attn_qkv",
    "attn_o",
    "cross_q",
    "cross_o",
    "cache_kv",
    "lm_head",
    "frozen_encoder_linear",
)
