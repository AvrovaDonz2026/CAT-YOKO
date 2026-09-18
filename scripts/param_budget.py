#!/usr/bin/env python3
"""CAT-YOKO parameter budget + middle-tier theoretical verification.

Causal Encoder-Decoder (YOCO-style) MoE built on Apache-2.0 MiniCPM5-2B
(Llama GQA), not MiniCPM-2B-sft-bf16.

Default target (compute-budget middle tier): total ~12.25B, Encoder(input)
active ~2.03B, Decoder(output) active ~4.33B.

Note: total params ~ memory/storage; TRAINING FLOPs ~ ACTIVE params x tokens.
Cutting total does not cut training cost -- cutting ACTIVE does.

The published attention term is MiniCPM5 GQA (16 Q / 2 KV). A CSA/HCA
sensitivity estimate is available via ``--attn csa_mqa64``; it does not
change the published middle-tier spec.

Freeze-curriculum (Phase B) is **frozen as C1**: MoE both stacks at
Phase A, freeze encoder in B0/B1, short joint B2. Delayed encoder MoE
is a sensitivity check only (``--curriculum``); it does not change the
published recipe. Do not franken-merge two LMs.

Phase B **wall-clock** is **frozen as C1+NVFP4**: C1 token split × mixed
NVFP4 (B0 student bf16; B1/B2 allowed linear GEMMs + frozen-encoder
forward at 2.0× vs bf16). Target GPU is RTX PRO 6000 / 6000D Blackwell.
Joint bf16 is the 100% baseline only. C1+FP8 (1.5×) is the Hopper/Ada
fallback. Peak 4× is sensitivity. NVFP4 does not change Kaplan 6NT.

"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---- MiniCPM5-2B base (openbmb/MiniCPM5-2B config.json) ----
D = 2048
V = 130_560
N_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = D // N_HEADS  # 128
KV_DIM = N_KV_HEADS * HEAD_DIM  # 256
DENSE_INT = 6144  # MiniCPM5 dense SwiGLU intermediate (6144/2048 = 3)
MOE_INT = 2048
SCALE_EMB = 1.0
DIM_MODEL_BASE = 2048
SCALE_DEPTH = 1.0
BASE_LAYERS = 42
TIE_EMBEDDINGS = False
LE_DEFAULT = 16
LD_DEFAULT = 26

EMB = V * D
LM_HEAD = 0 if TIE_EMBEDDINGS else V * D
EXPERT = 3 * D * MOE_INT  # SwiGLU expert (gate, up, down)
DENSE_FFN = 3 * D * DENSE_INT
MHA = 4 * D * D
GQA_SELF = 2 * D * D + 2 * D * KV_DIM  # Q,O + K,V
CROSS_QO = 2 * D * D  # decoder cross: Q,O only; K/V from the YOCO cache
# YOCO cache projections (W_K, W_V) at encoder top. d_kv = n_kv * head_dim.
# Not in the published 12.25B stack total (~1M).
CACHE_PROJ = 2 * D * KV_DIM
# Mixed-precision AdamW footprint (bytes / parameter). No fp32 master copy.
ADAM_STATE_BYTES = 8  # m, v in fp32
GRAD_BYTES = 2  # bf16 gradients
WEIGHT_BYTES = 2  # bf16 weights
TRAINABLE_FOOTPRINT = WEIGHT_BYTES + GRAD_BYTES + ADAM_STATE_BYTES  # 12
FROZEN_FOOTPRINT = WEIGHT_BYTES  # 2
DEFAULT_CURRICULUM_SPLIT = (8e9, 27e9, 15e9)  # B0 + B1 + B2 = 50B
# C1 = both stacks MoE at Phase A, freeze encoder in B0/B1 (FROZEN spec).
# Delayed encoder MoE (encoder_dense=True) is sensitivity only.
DEFAULT_DELAYED_ENCODER_MOE = False
# Published Phase B wall-clock. Joint bf16 = 100% baseline only.
# Applying 2.0× to B0 student, or claiming peak 4×, is sensitivity.
FROZEN_WALLCLOCK = "C1+NVFP4"

# Hardware for GPU-hour estimates (plan §15.1): 40% MFU.
H100_BF16_PEAK = 9.89e14  # H100 SXM bf16 tensor-core peak
H100_FP8_PEAK = 1.979e15  # H100 SXM FP8 tensor-core peak (~2× bf16)
H100_BF16_EFF = 4.0e14  # ~989 TFLOPS peak * 0.40
H100_FP8_EFF_SAME_MFU = H100_FP8_PEAK * 0.40  # 7.916e14 if MFU holds
A100_BF16_EFF = 1.25e14  # ~312 TFLOPS peak * 0.40
SECONDS_PER_HOUR = 3600.0
# RTX PRO 6000 Blackwell Server Edition (NVIDIA datasheet). 6000D is the
# same-family China SKU; hours stay in H100-h because BF16 peak matches.
RTX_PRO_6000_BF16_PEAK = 1.0e15
RTX_PRO_6000_FP8_PEAK = 2.0e15
RTX_PRO_6000_FP4_PEAK = 4.0e15
RTX_PRO_6000_MEM_GIB = 96
RTX_PRO_6000_BANDWIDTH = 1.597e12
# Wall-clock speedups. Dtype does not change Kaplan 6NT; it raises effective FLOPS.
FP8_SPEEDUP_CONSERVATIVE = 1.5  # Hopper/Ada fallback; 12B kernel/comm overhead
FP8_SPEEDUP_PEAK = H100_FP8_PEAK / H100_BF16_PEAK  # ≈2.0
NVFP4_SPEEDUP_CONSERVATIVE = 2.0  # vs bf16; 1.33× over published FP8 1.5×
NVFP4_SPEEDUP_PEAK = RTX_PRO_6000_FP4_PEAK / RTX_PRO_6000_BF16_PEAK  # 4.0
NVFP4_VS_FP8_NVIDIA = (1.31, 1.73)  # MaxText GB200/GB300 Llama 8B/405B
# Modules that stay high precision under the frozen NVFP4 policy.
# lm_head is a GEMM and is not on this list.
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

# KV recipes used in plan §16 (bytes of cache content, not allocator padding).
MLA_LATENT = 512 + 64  # DeepSeek-V3-style: kv_lora_rank + qk_rope_head_dim
GQA2_KV_DIM = 2 * KV_DIM  # MiniCPM5 2 KV heads, K and V
MHA_KV_DIM = 2 * D  # full MHA KV (not the published GQA graph)
MQA64_KV_DIM = 2 * HEAD_DIM  # single KV head at MiniCPM5 head_dim


@dataclass(frozen=True)
class Tier:
    name: str
    key: str
    ns_e: int
    tk_e: int
    nr_e: int
    ns_d: int
    tk_d: int
    nr_d: int
    le: int = 16
    ld: int = 26


TIERS: dict[str, Tier] = {
    # Same 1+20 experts on both stacks (882 slots). Tiers only change top-k.
    "low": Tier("省算力档", "low", ns_e=1, tk_e=4, nr_e=20, ns_d=1, tk_d=6, nr_d=20),
    "middle": Tier("默认（中间档）", "middle", ns_e=1, tk_e=7, nr_e=20, ns_d=1, tk_d=10, nr_d=20),
    "near_dense": Tier("近-dense 档", "near_dense", ns_e=1, tk_e=12, nr_e=20, ns_d=1, tk_d=16, nr_d=20),
}

DEFAULT_TIER = "middle"


@dataclass(frozen=True)
class AttnAccounting:
    name: str
    self_attn: int
    cross_attn: int
    note: str


def _csa_mqa64_self_attn(is_csa: bool) -> int:
    """MiniCPM5-native CSA/HCA parameter estimate (MQA, head_dim=128).

    Scaled from DeepSeek-V4 (arXiv 2606.19348 §2.3) onto MiniCPM5 dims rather
    than copying V4's head_dim=512, which would be oversized at d=2048.

    Shared pieces: LoRA-Q (W_DQ, W_UQ), grouped output, sliding-window KV,
    CSA/HCA compressor. CSA adds a Lightning Indexer; HCA does not.
    """
    q_lora = 512
    kv_dim = HEAD_DIM  # MQA-128
    o_groups = 4  # 16 heads / 4
    o_lora = 512
    index_n_heads = 16
    index_head_dim = 64
    m = 4 if is_csa else 128

    compressor = 4 * D * kv_dim + 2 * m * kv_dim if is_csa else 2 * D * kv_dim + m * kv_dim
    q_down = D * q_lora
    q_up = q_lora * N_HEADS * HEAD_DIM
    win_kv = 2 * D * kv_dim
    grouped_o = (
        o_groups * (HEAD_DIM * (N_HEADS // o_groups)) * o_lora
        + o_lora * o_groups * D
    )
    sink = N_HEADS
    indexer = 0
    if is_csa:
        idx_comp = 4 * D * index_head_dim + 2 * m * index_head_dim
        idx_q_up = q_lora * index_n_heads * index_head_dim
        idx_w = D * index_n_heads
        indexer = idx_comp + idx_q_up + idx_w
    return compressor + q_down + q_up + win_kv + grouped_o + sink + indexer


def attn_accounting(kind: str) -> AttnAccounting:
    if kind in {"gqa", "placeholder"}:
        return AttnAccounting(
            name="GQA 16/2",
            self_attn=GQA_SELF,
            cross_attn=CROSS_QO,
            note="MiniCPM5 GQA; Phase B window; YOCO cache supplies K/V",
        )
    if kind == "csa_mqa64":
        # Interleave CSA:HCA ≈ 1:1; average the two layer types.
        avg_self = (_csa_mqa64_self_attn(True) + _csa_mqa64_self_attn(False)) // 2
        return AttnAccounting(
            name="CSA/HCA MQA-128 (sensitivity)",
            self_attn=avg_self,
            cross_attn=CROSS_QO,
            note="MiniCPM5-native CSA/HCA estimate; does not change published spec",
        )
    raise ValueError(f"unknown attn accounting: {kind}")


@dataclass(frozen=True)
class StackBudget:
    layers: int
    ns: int
    tk: int
    nr: int
    moe_layers: int
    dense_layers: int
    self_attn: int
    extra_attn: int  # cross-attn (decoder) or 0
    ffn_active: int
    ffn_total: int
    attn_total: int
    stack_total: int  # no embedding
    active_no_emb: int


@dataclass(frozen=True)
class ModelBudget:
    tier: Tier
    attn: AttnAccounting
    first_dense: bool
    emb: int
    lm_head: int
    enc: StackBudget
    dec: StackBudget
    total: int
    enc_active: int  # includes input embed (plan convention: per-token stack)
    dec_active: int  # includes untied lm_head
    fwd_active: int  # embed + lm_head counted once each (one full forward)

    @property
    def expert_slots(self) -> int:
        return self.enc.moe_layers * (self.enc.ns + self.enc.nr) + self.dec.moe_layers * (
            self.dec.ns + self.dec.nr
        )

    @property
    def sparsity_enc(self) -> float:
        return (self.enc.ns + self.enc.tk) / (self.enc.ns + self.enc.nr)

    @property
    def sparsity_dec(self) -> float:
        return (self.dec.ns + self.dec.tk) / (self.dec.ns + self.dec.nr)

    @property
    def attn_frac(self) -> float:
        return (self.enc.attn_total + self.dec.attn_total) / self.total

    @property
    def moe_frac(self) -> float:
        return (self.enc.ffn_total + self.dec.ffn_total) / self.total


def _stack(
    layers: int,
    ns: int,
    tk: int,
    nr: int,
    self_attn: int,
    extra_attn: int,
    first_dense: bool,
) -> StackBudget:
    dense_layers = 1 if first_dense else 0
    moe_layers = layers - dense_layers
    if moe_layers < 0:
        raise ValueError("first_dense requires at least 1 layer")
    ffn_active = dense_layers * DENSE_FFN + moe_layers * (ns + tk) * EXPERT
    ffn_total = dense_layers * DENSE_FFN + moe_layers * (ns + nr) * EXPERT
    attn_total = layers * (self_attn + extra_attn)
    stack_total = attn_total + ffn_total
    active_no_emb = attn_total + ffn_active
    return StackBudget(
        layers=layers,
        ns=ns,
        tk=tk,
        nr=nr,
        moe_layers=moe_layers,
        dense_layers=dense_layers,
        self_attn=self_attn,
        extra_attn=extra_attn,
        ffn_active=ffn_active,
        ffn_total=ffn_total,
        attn_total=attn_total,
        stack_total=stack_total,
        active_no_emb=active_no_emb,
    )


def compute_budget(
    tier: Tier,
    *,
    attn: AttnAccounting | None = None,
    first_dense: bool = False,
) -> ModelBudget:
    attn = attn or attn_accounting("gqa")
    enc = _stack(tier.le, tier.ns_e, tier.tk_e, tier.nr_e, attn.self_attn, 0, first_dense)
    dec = _stack(
        tier.ld, tier.ns_d, tier.tk_d, tier.nr_d, attn.self_attn, attn.cross_attn, first_dense
    )
    total = EMB + LM_HEAD + enc.stack_total + dec.stack_total
    enc_active = EMB + enc.active_no_emb
    dec_active = (LM_HEAD or EMB) + dec.active_no_emb
    fwd_active = EMB + LM_HEAD + enc.active_no_emb + dec.active_no_emb
    return ModelBudget(
        tier=tier,
        attn=attn,
        first_dense=first_dense,
        emb=EMB,
        lm_head=LM_HEAD,
        enc=enc,
        dec=dec,
        total=total,
        enc_active=enc_active,
        dec_active=dec_active,
        fwd_active=fwd_active,
    )


def b(x: float, digits: int = 2) -> str:
    return f"{x / 1e9:.{digits}f}B"


def m(x: float, digits: int = 2) -> str:
    return f"{x / 1e6:.{digits}f}M"


def gb(nbytes: float, digits: int = 2) -> str:
    """Decimal GB (1e9 bytes), matching plan §16 (369 GB, 1.15 GB, …)."""
    return f"{nbytes / 1e9:.{digits}f} GB"


def training_flops(n_active: int, tokens: float) -> float:
    """Kaplan-style 6 N T (forward 2NT + backward 4NT), matmul-dominated."""
    return 6.0 * n_active * tokens


def gpu_hours(flops: float, effective_flops: float) -> float:
    return flops / (effective_flops * SECONDS_PER_HOUR)


@dataclass(frozen=True)
class StagedRecipe:
    """One training recipe's FLOPs over a token budget (matmul / 6NT-style)."""

    name: str
    flops: float
    tokens: float
    note: str = ""

    def h100_h(self) -> float:
        return gpu_hours(self.flops, H100_BF16_EFF)


def _n_parts(budget: ModelBudget) -> tuple[int, int, int, int]:
    """emb, encoder-no-emb, decoder-no-emb, new-module proxy.

    New modules = 26× cross-attn Q/O + cache W_K/W_V. Cache proj is ~1M
    and does not change published 12.25B at two decimals.
    """
    n_emb = budget.emb
    n_e = budget.enc.active_no_emb
    n_d = budget.dec.active_no_emb
    n_new = n_new_modules(budget)
    return n_emb, n_e, n_d, n_new


def n_new_modules(budget: ModelBudget) -> int:
    """Cross-attn (all decoder layers) + cache projections W_K, W_V."""
    return budget.dec.layers * budget.attn.cross_attn + CACHE_PROJ


def encoder_active_no_emb(budget: ModelBudget, *, dense: bool = False) -> int:
    """Encoder activation excluding embedding.

    ``dense=True`` is delayed-encoder-MoE (C2): MiniCPM5 SwiGLU on all 16
    layers, same self-attn accounting as the MoE budget. Dense FFN is fully
    active, so this equals encoder stack total.
    """
    if not dense:
        return budget.enc.active_no_emb
    return budget.enc.layers * (budget.attn.self_attn + DENSE_FFN)


def encoder_stack_total(budget: ModelBudget, *, dense: bool = False) -> int:
    """Encoder stored params excluding embedding (all experts, not top-k)."""
    if not dense:
        return budget.enc.stack_total
    return encoder_active_no_emb(budget, dense=True)


def flops_joint(budget: ModelBudget, tokens: float) -> float:
    """Both stacks trainable (emb counted once)."""
    return 6.0 * budget.fwd_active * tokens


def flops_encoder_lm(budget: ModelBudget, tokens: float) -> float:
    """Train encoder as a standalone causal LM (loss on X^{Le}, no decoder)."""
    n_emb, n_e, _, _ = _n_parts(budget)
    return 6.0 * (n_emb + n_e) * tokens


def flops_freeze_encoder(
    budget: ModelBudget, tokens: float, *, encoder_dense: bool = False
) -> float:
    """YOCO LM loss, encoder frozen (fwd only), decoder+cross-attn trainable.

    Backward stops at the global cache: encoder has no weight or activation grads.
    ``encoder_dense`` is delayed-encoder-MoE (C2): encoder FFN is MiniCPM5 dense.
    """
    n_emb, _, n_d, _ = _n_parts(budget)
    n_e = encoder_active_no_emb(budget, dense=encoder_dense)
    return (2.0 * (n_emb + n_e) + 6.0 * n_d) * tokens


def flops_new_modules(
    budget: ModelBudget, tokens: float, *, encoder_dense: bool = False
) -> float:
    """Freeze inherited MiniCPM5 stacks; train cross-attn + cache proj.

    Forward still runs both stacks. Activation backward runs through the decoder
    (chain rule to the new residual branch) but not the encoder.
    """
    n_emb, _, n_d, n_new = _n_parts(budget)
    n_e = encoder_active_no_emb(budget, dense=encoder_dense)
    fwd = 2.0 * (n_emb + n_e + n_d)
    bwd_w = 2.0 * n_new
    bwd_act = 2.0 * n_d
    return (fwd + bwd_w + bwd_act) * tokens


def _scale_split(
    tokens: float, split: tuple[float, float, float] = DEFAULT_CURRICULUM_SPLIT
) -> tuple[float, float, float]:
    base = split[0] + split[1] + split[2]
    if abs(base - tokens) <= 1.0:
        return split
    return (split[0] / base * tokens, split[1] / base * tokens, split[2] / base * tokens)


def flops_curriculum(
    budget: ModelBudget,
    tokens: float = 50e9,
    *,
    split: tuple[float, float, float] = DEFAULT_CURRICULUM_SPLIT,
    delayed: bool = False,
) -> float:
    """B0 new-modules + B1 freeze-enc + B2 joint. B2 always uses MoE encoder.

    ``delayed=True`` (C2): MiniCPM5-dense encoder in B0/B1; virtual-group
    encoder MoE starts at B2. ``delayed=False`` (C1): encoder already MoE,
    frozen through B0/B1. If ``split`` does not sum to ``tokens`` it is
    scaled (keeps the 8:27:15 ratio).
    """
    t0, t1, t2 = _scale_split(tokens, split)
    return (
        flops_new_modules(budget, t0, encoder_dense=delayed)
        + flops_freeze_encoder(budget, t1, encoder_dense=delayed)
        + flops_joint(budget, t2)
    )


def staged_recipes(budget: ModelBudget, tokens: float = 50e9) -> list[StagedRecipe]:
    """Compare independent-merge vs freeze-curriculum vs joint, same 50B envelope."""
    half = tokens / 2.0
    independent = (
        flops_encoder_lm(budget, tokens)
        + flops_freeze_encoder(budget, tokens)
        + flops_joint(budget, 0.4 * tokens)
    )
    return [
        StagedRecipe("joint both stacks", flops_joint(budget, tokens), tokens, "baseline"),
        StagedRecipe(
            "freeze-enc, train-dec (all tokens)",
            flops_freeze_encoder(budget, tokens),
            tokens,
            "saves enc backward; write/read don't co-adapt",
        ),
        StagedRecipe(
            "freeze-enc dense (delayed, all tokens)",
            flops_freeze_encoder(budget, tokens, encoder_dense=True),
            tokens,
            "C2 encoder FFN = MiniCPM5 dense; still no co-adapt",
        ),
        StagedRecipe(
            "new-modules only (cross-attn)",
            flops_new_modules(budget, tokens),
            tokens,
            "Phase A warmup / Phase C indexer-style",
        ),
        StagedRecipe(
            "unfreeze curriculum 8+27+15B",
            flops_curriculum(budget, tokens, delayed=DEFAULT_DELAYED_ENCODER_MOE),
            tokens,
            "C1 (frozen spec): MoE both stacks, freeze enc in B0/B1",
        ),
        StagedRecipe(
            "delayed-enc-MoE curriculum 8+27+15B",
            flops_curriculum(budget, tokens, delayed=True),
            tokens,
            "sensitivity: dense encoder in B0/B1 (not the frozen spec)",
        ),
        StagedRecipe(
            "independent enc LM + freeze-enc dec + 20B stitch",
            independent,
            tokens + tokens + 0.4 * tokens,
            "same 50B per stack then merge — usually MORE FLOPs",
        ),
        StagedRecipe(
            "enc-only LM then joint on half+half",
            flops_encoder_lm(budget, half) + flops_joint(budget, half),
            tokens,
            "saves only if half joint tokens suffice — quality bet",
        ),
    ]


@dataclass(frozen=True)
class FreezeBoundary:
    """What may receive gradients in one curriculum phase.

    Names are module groups, not parameter counts. Counts live in
    ``phase_param_counts``. Untied-embedding policy is part of the boundary
    because MiniCPM5 does not share E as LM head (Theorem E still freezes
    the input table in B0/B1).
    """

    phase: str
    frozen: tuple[str, ...]
    trainable: tuple[str, ...]
    detach_at_cache: bool
    tied_emb: str
    gate: str
    note: str = ""


# Default freeze-curriculum (ULMFiT-style). MiniCPM5 is untied: freeze the
# input table with the encoder; B0 also freezes lm_head; B1 trains lm_head.
# Training the input table while the encoder is frozen is still forbidden.
FREEZE_BOUNDARIES: tuple[FreezeBoundary, ...] = (
    FreezeBoundary(
        phase="B0",
        frozen=("encoder", "decoder_backbone", "embed", "lm_head"),
        trainable=("cross_attn", "cache_proj", "cross_ln", "gate"),
        detach_at_cache=True,
        tied_emb="freeze_embed_and_head",
        gate="0→0.3",
        note="new modules only; encoder FFN = frozen MoE copies (C1)",
    ),
    FreezeBoundary(
        phase="B1",
        frozen=("encoder", "embed"),
        trainable=("decoder_self_attn", "decoder_moe", "cross_attn", "cache_proj", "lm_head", "gate"),
        detach_at_cache=True,
        tied_emb="freeze_embed_train_head",
        gate="→1",
        note="unfreeze reader + lm_head; writer + input E stay MiniCPM5-16",
    ),
    FreezeBoundary(
        phase="B2",
        frozen=(),
        trainable=("encoder", "decoder", "embed", "lm_head", "cross_attn", "cache_proj"),
        detach_at_cache=False,
        tied_emb="train_untied",
        gate="1",
        note="short joint; unfreeze the already-MoE encoder",
    ),
)


# (label, t0, t1, t2, valid). Invalid = B2=0, write/read never co-adapt.
CURRICULUM_SPLITS: tuple[tuple[str, float, float, float, bool], ...] = (
    ("default 8+27+15", 8e9, 27e9, 15e9, True),
    ("short B2 5+35+10", 5e9, 35e9, 10e9, True),
    ("long B2 10+20+20", 10e9, 20e9, 20e9, True),
    ("short B0 5+25+20", 5e9, 25e9, 20e9, True),
    ("skip B0 0+35+15", 0e9, 35e9, 15e9, True),
    ("no B2 8+42+0", 8e9, 42e9, 0e9, False),
)


def phase_param_counts(
    budget: ModelBudget, phase: str, *, delayed: bool
) -> tuple[int, int]:
    """(n_trainable, n_frozen) stored params, not active-FLOP params.

    Decoder ``stack_total`` includes cross-attn. B0 moves that block into
    the trainable set; cache proj is never inside either stack total.
    B2 always stores the MoE encoder (C2 upcycles at the B2 boundary).
    """
    n_emb = budget.emb
    n_head = budget.lm_head
    n_cross = budget.dec.layers * budget.attn.cross_attn
    n_dec = budget.dec.stack_total
    if phase == "B2":
        n_enc = budget.enc.stack_total
    else:
        n_enc = encoder_stack_total(budget, dense=delayed)
    if phase == "B0":
        train = n_new_modules(budget)
        frozen = n_emb + n_head + n_enc + (n_dec - n_cross)
        return train, frozen
    if phase == "B1":
        return n_dec + CACHE_PROJ + n_head, n_emb + n_enc
    if phase == "B2":
        return n_emb + n_head + n_enc + n_dec + CACHE_PROJ, 0
    raise ValueError(f"unknown phase {phase}")


def optimizer_state_bytes(n_trainable: int) -> int:
    return n_trainable * ADAM_STATE_BYTES


def param_footprint_bytes(n_trainable: int, n_frozen: int) -> int:
    return n_trainable * TRAINABLE_FOOTPRINT + n_frozen * FROZEN_FOOTPRINT


def activation_keep_frac() -> float:
    """Layer-count model: detach drops encoder activations (16/40)."""
    return TIERS["middle"].ld / (TIERS["middle"].le + TIERS["middle"].ld)


def curriculum_split_table(
    budget: ModelBudget, *, delayed: bool
) -> list[tuple[str, float, float, bool]]:
    """label, flops, vs-joint, valid."""
    joint = flops_joint(budget, 50e9)
    rows = []
    for label, t0, t1, t2, valid in CURRICULUM_SPLITS:
        flops = flops_curriculum(
            budget, t0 + t1 + t2, split=(t0, t1, t2), delayed=delayed
        )
        rows.append((label, flops, flops / joint, valid))
    return rows


def claims_curriculum(budget: ModelBudget) -> list[Claim]:
    """Freeze-curriculum ledger (Theorems D/E, C1 vs C2, memory, splits)."""
    joint = flops_joint(budget, 50e9)
    c1 = flops_curriculum(budget, 50e9, delayed=False)
    c2 = flops_curriculum(budget, 50e9, delayed=True)
    n_e_moe = encoder_active_no_emb(budget, dense=False)
    n_e_dense = encoder_active_no_emb(budget, dense=True)
    b1_tr, b1_fr = phase_param_counts(budget, "B1", delayed=False)
    b2_tr, _ = phase_param_counts(budget, "B2", delayed=False)
    opt_ratio = optimizer_state_bytes(b1_tr) / optimizer_state_bytes(b2_tr)
    dec_ratio = (budget.dec.stack_total + CACHE_PROJ + budget.lm_head) / (
        budget.emb + budget.lm_head + budget.enc.stack_total + budget.dec.stack_total + CACHE_PROJ
    )
    b0, b1, b2 = FREEZE_BOUNDARIES
    valid_c1 = curriculum_split_table(budget, delayed=False)
    valid_c2 = curriculum_split_table(budget, delayed=True)
    no_b2 = next(r for r in valid_c1 if r[0].startswith("no B2"))
    return [
        Claim(
            "C2 delayed-enc-MoE cheaper than C1 at same split",
            c2 < c1,
            f"C2 {c2 / joint:.0%} < C1 {c1 / joint:.0%}",
            "C2 < C1",
        ),
        Claim(
            "C2 delayed-enc-MoE curriculum ≤80% of joint 50B",
            c2 <= 0.80 * joint,
            f"{c2 / joint:.0%}",
            "≤80%",
        ),
        Claim(
            "dense encoder FFN active < MoE encoder FFN active",
            n_e_dense < n_e_moe,
            f"{b(n_e_dense)} < {b(n_e_moe)}",
            "MiniCPM5 16× dense < 16×8 experts",
        ),
        Claim(
            "B0/B1 detach cache; B2 does not",
            b0.detach_at_cache and b1.detach_at_cache and not b2.detach_at_cache,
            f"B0={b0.detach_at_cache} B1={b1.detach_at_cache} B2={b2.detach_at_cache}",
            "True, True, False",
        ),
        Claim(
            "embed frozen B0/B1; lm_head frozen B0 trained B1",
            b0.tied_emb == "freeze_embed_and_head" and b1.tied_emb == "freeze_embed_train_head",
            f"{b0.tied_emb}/{b1.tied_emb}",
            "freeze_embed_and_head / freeze_embed_train_head",
            note="training embed while encoder frozen leaks X^0 (Theorem E)",
        ),
        Claim(
            "B1 Adam states ≈ decoder/total (~60%)",
            _close(opt_ratio, dec_ratio, rel=0.02),
            f"{opt_ratio:.0%}",
            f"~{dec_ratio:.0%}",
        ),
        Claim(
            "detach drops encoder activations (keep 26/42)",
            abs(activation_keep_frac() - 26 / 42) < 1e-12,
            f"{activation_keep_frac():.0%}",
            "62%",
        ),
        Claim(
            "all valid 50B splits stay ≤85% joint (C1 and C2)",
            all(r[2] <= 0.85 for r in valid_c1 if r[3])
            and all(r[2] <= 0.85 for r in valid_c2 if r[3]),
            f"C1 max {max(r[2] for r in valid_c1 if r[3]):.0%}; "
            f"C2 max {max(r[2] for r in valid_c2 if r[3]):.0%}",
            "≤85%",
        ),
        Claim(
            "B2=0 is cheaper but invalid (no write/read co-adapt)",
            (not no_b2[3]) and no_b2[2] < c1 / joint,
            f"{no_b2[2]:.0%} vs C1 {c1 / joint:.0%}, valid={no_b2[3]}",
            "invalid",
        ),
        Claim(
            "cache proj is a new module (trainable in B0)",
            CACHE_PROJ > 0 and "cache_proj" in b0.trainable,
            m(CACHE_PROJ),
            "W_K, W_V after encoder.detach()",
        ),
        Claim(
            "default B2 ≥10B (encoder expert entropy / co-adapt)",
            DEFAULT_CURRICULUM_SPLIT[2] >= 10e9,
            f"{DEFAULT_CURRICULUM_SPLIT[2] / 1e9:.0f}B",
            "≥10B",
        ),
        Claim(
            "frozen spec is C1 (not delayed encoder MoE)",
            DEFAULT_DELAYED_ENCODER_MOE is False,
            str(DEFAULT_DELAYED_ENCODER_MOE),
            "False (C1 frozen: both stacks MoE, freeze enc)",
        ),
    ]


def fp8_moe_active_frac(budget: ModelBudget) -> float:
    """Share of counted 6NT from MoE expert GEMMs (the considerable low-prec portion)."""
    return (budget.enc.ffn_active + budget.dec.ffn_active) / budget.fwd_active


def fp8_gemm_frac(budget: ModelBudget) -> float:
    """Share of counted 6NT that is tensor-core GEMM if allowed linears go low-prec.

    6NT ignores softmax / LN / router. Embedding stays high precision (Theorem E).
    lm_head is a GEMM and is included (fwd_active − emb).
    """
    return min(1.0, (budget.fwd_active - budget.emb) / budget.fwd_active)


def fp8_speedup_from_gemm_frac(frac: float, gemm_x: float = FP8_SPEEDUP_PEAK) -> float:
    """Amdahl: GEMM gets gemm_x, the rest stays 1×."""
    rest = max(0.0, 1.0 - frac)
    return 1.0 / (frac / gemm_x + rest)


def nvfp4_speedup_from_gemm_frac(
    frac: float, gemm_x: float = NVFP4_SPEEDUP_PEAK
) -> float:
    return fp8_speedup_from_gemm_frac(frac, gemm_x=gemm_x)


def wallclock_h100_h(flops: float, *, speedup: float = 1.0) -> float:
    """Kaplan FLOPs converted to H100-h at the bf16 40% MFU baseline, then scaled."""
    return gpu_hours(flops, H100_BF16_EFF) / speedup


def curriculum_phase_flops(
    budget: ModelBudget,
    tokens: float = 50e9,
    *,
    delayed: bool = False,
    split: tuple[float, float, float] = DEFAULT_CURRICULUM_SPLIT,
) -> tuple[float, float, float]:
    """B0 / B1 / B2 Kaplan FLOPs on a token envelope (sums to ``flops_curriculum``)."""
    t0, t1, t2 = _scale_split(tokens, split)
    return (
        flops_new_modules(budget, t0, encoder_dense=delayed),
        flops_freeze_encoder(budget, t1, encoder_dense=delayed),
        flops_joint(budget, t2),
    )


def encoder_fwd_flops(
    budget: ModelBudget, tokens: float, *, delayed: bool = False
) -> float:
    """Encoder-stack forward only (no embedding): 2 N_enc T."""
    return 2.0 * encoder_active_no_emb(budget, dense=delayed) * tokens


def wallclock_c1_fp8_policy(
    budget: ModelBudget,
    tokens: float = 50e9,
    *,
    speedup: float = FP8_SPEEDUP_CONSERVATIVE,
) -> float:
    """Hopper/Ada fallback wall-clock (C1+FP8). Same mix as NVFP4, S=1.5.

    B0 student stays bf16 (Theorem A neighborhood). Frozen-encoder forward
    GEMM in B0, and all of B1/B2, run at ``speedup``. Does not change 6NT.
    """
    t0, _, _ = _scale_split(tokens)
    f0, f1, f2 = curriculum_phase_flops(budget, tokens)
    enc0 = encoder_fwd_flops(budget, t0)
    return (
        wallclock_h100_h(f0 - enc0)
        + wallclock_h100_h(enc0, speedup=speedup)
        + wallclock_h100_h(f1, speedup=speedup)
        + wallclock_h100_h(f2, speedup=speedup)
    )


def wallclock_c1_nvfp4_policy(
    budget: ModelBudget,
    tokens: float = 50e9,
    *,
    speedup: float = NVFP4_SPEEDUP_CONSERVATIVE,
) -> float:
    """Published Phase B wall-clock (frozen spec C1+NVFP4).

    Same B0-student-bf16 mix as the FP8 fallback, with S=2.0 vs bf16.
    All-phase 2.0× (B0 student also NVFP4) and peak 4× are sensitivity.
    """
    return wallclock_c1_fp8_policy(budget, tokens, speedup=speedup)


@dataclass(frozen=True)
class Fp8PhasePolicy:
    phase: str
    student: str
    frozen_encoder_gemm: str
    note: str


Nvfp4PhasePolicy = Fp8PhasePolicy


# Hopper/Ada fallback. Student = tensors that receive gradients.
FP8_PHASE_POLICY: tuple[Fp8PhasePolicy, ...] = (
    Fp8PhasePolicy(
        "L0",
        "bf16",
        "bf16",
        "tiny correctness; no low prec",
    ),
    Fp8PhasePolicy(
        "B0",
        "bf16",
        "fp8",
        "gate ramp next to Theorem A; frozen encoder is inference GEMM",
    ),
    Fp8PhasePolicy(
        "B1",
        "fp8",
        "fp8",
        "allowed linear GEMMs in FP8; router / LN / embed / softmax stay high prec",
    ),
    Fp8PhasePolicy(
        "B2",
        "fp8",
        "n/a",
        "both stacks; same allowed GEMM set in FP8 after unfreeze",
    ),
    Fp8PhasePolicy(
        "C",
        "bf16",
        "fp8",
        "indexer KL is local and small; keep bf16",
    ),
)

# Frozen NVFP4 policy for C1+NVFP4. Same must-high-prec set as FP8 fallback.
NVFP4_PHASE_POLICY: tuple[Nvfp4PhasePolicy, ...] = (
    Nvfp4PhasePolicy(
        "L0",
        "bf16",
        "bf16",
        "tiny correctness; no NVFP4",
    ),
    Nvfp4PhasePolicy(
        "B0",
        "bf16",
        "nvfp4",
        "gate ramp next to Theorem A; frozen encoder is inference GEMM",
    ),
    Nvfp4PhasePolicy(
        "B1",
        "nvfp4",
        "nvfp4",
        "all allowed linear GEMMs including lm_head and attn QKV/O",
    ),
    Nvfp4PhasePolicy(
        "B2",
        "nvfp4",
        "n/a",
        "both stacks; same allowed GEMM set after unfreeze",
    ),
    Nvfp4PhasePolicy(
        "C",
        "bf16",
        "nvfp4",
        "indexer KL is local and small; keep bf16",
    ),
)


def claims_fp8(budget: ModelBudget) -> list[Claim]:
    """FP8 fallback ledger. Does not change Kaplan 6NT. Not the published wall-clock."""
    joint = flops_joint(budget, 50e9)
    c1 = flops_curriculum(budget, 50e9, delayed=False)
    moe_f = fp8_moe_active_frac(budget)
    moe_x = fp8_speedup_from_gemm_frac(moe_f)
    c1_bf16_h = wallclock_h100_h(c1)
    mixed_h = wallclock_c1_fp8_policy(budget)
    joint_h = wallclock_h100_h(joint)
    f0, f1, f2 = curriculum_phase_flops(budget)
    pol = {p.phase: p for p in FP8_PHASE_POLICY}
    t0, t1, t2 = DEFAULT_CURRICULUM_SPLIT
    later = (t1 + t2) / (t0 + t1 + t2)
    return [
        Claim(
            "FP8 does not change Kaplan 6NT",
            abs(c1 - flops_curriculum(budget, 50e9, delayed=False)) < 1.0,
            "same FLOPs",
            "dtype ≠ operation count",
        ),
        Claim(
            "H100 FP8 peak ≈ 2× bf16",
            _close(FP8_SPEEDUP_PEAK, 2.0, rel=0.01),
            f"{FP8_SPEEDUP_PEAK:.2f}×",
            "~2.0×",
        ),
        Claim(
            "conservative FP8 wall-clock is 1.5× (12B overhead)",
            FP8_SPEEDUP_CONSERVATIVE == 1.5,
            f"{FP8_SPEEDUP_CONSERVATIVE:.2f}×",
            "1.5×",
        ),
        Claim(
            "MoE is ≥70% of 6NT (considerable low-prec portion)",
            moe_f >= 0.70,
            f"{moe_f:.1%}",
            "≥70% of counted 6NT",
        ),
        Claim(
            "MoE-only Amdahl speedup in 1.50–1.75×",
            1.50 <= moe_x <= 1.75,
            f"{moe_x:.2f}× (f={moe_f:.1%})",
            "justifies published 1.5× fallback",
        ),
        Claim(
            "C1+FP8 mixed is the Hopper/Ada fallback (~729)",
            FROZEN_WALLCLOCK == "C1+NVFP4"
            and pol["B0"].student == "bf16"
            and pol["B1"].student == "fp8"
            and pol["B2"].student == "fp8",
            f"{mixed_h:.0f}h fallback",
            "B0 student bf16; B1/B2 fp8; not published wall-clock",
        ),
        Claim(
            "fallback C1+FP8 hours ≈729 (≤60% joint bf16)",
            _close(mixed_h, 729, rel=0.02) and mixed_h <= 0.60 * joint_h,
            f"{mixed_h:.0f} ({mixed_h / joint_h:.0%} of {joint_h:.0f})",
            "~729; ≤60% of joint",
        ),
        Claim(
            "B0 student stays bf16 (Theorem A neighborhood)",
            pol["B0"].student == "bf16",
            pol["B0"].student,
            "bf16",
        ),
        Claim(
            "B1/B2 fallback student GEMM is FP8",
            pol["B1"].student == "fp8" and pol["B2"].student == "fp8",
            f"{pol['B1'].student}/{pol['B2'].student}",
            "fp8",
        ),
        Claim(
            "B1+B2 cover ≥80% of the 50B envelope",
            later >= 0.80,
            f"{later:.0%}",
            "≥80% (27+15 of 50)",
        ),
        Claim(
            "B0 is ≤15% of C1 FLOPs (bf16 student is cheap)",
            f0 / (f0 + f1 + f2) <= 0.15,
            f"{f0 / (f0 + f1 + f2):.1%}",
            "≤15%",
        ),
        Claim(
            "embed / router / LN stay high precision (lm_head is a GEMM)",
            "lm_head" not in KEEP_HIGH_PREC
            and set(KEEP_HIGH_PREC)
            >= {"embed", "router", "rms_norm", "gate", "indexer", "attn_softmax"},
            ",".join(KEEP_HIGH_PREC),
            "embed, router, LN, gate, softmax, indexer (not lm_head)",
        ),
        Claim(
            "C1 bf16 hours unchanged by the FP8 policy",
            _close(c1_bf16_h, 1046, rel=0.02),
            f"{c1_bf16_h:.0f}",
            "~1046",
        ),
    ]


def claims_nvfp4(budget: ModelBudget) -> list[Claim]:
    """NVFP4 wall-clock ledger. Does not change Kaplan 6NT."""
    joint = flops_joint(budget, 50e9)
    c1 = flops_curriculum(budget, 50e9, delayed=False)
    moe_f = fp8_moe_active_frac(budget)
    gemm_f = fp8_gemm_frac(budget)
    mixed_h = wallclock_c1_nvfp4_policy(budget)
    fp8_h = wallclock_c1_fp8_policy(budget)
    joint_h = wallclock_h100_h(joint)
    c1_bf16_h = wallclock_h100_h(c1)
    pol = {p.phase: p for p in NVFP4_PHASE_POLICY}
    vs_fp8 = NVFP4_SPEEDUP_CONSERVATIVE / FP8_SPEEDUP_CONSERVATIVE
    lo, hi = NVFP4_VS_FP8_NVIDIA
    return [
        Claim(
            "NVFP4 does not change Kaplan 6NT",
            abs(c1 - flops_curriculum(budget, 50e9, delayed=False)) < 1.0,
            "same FLOPs",
            "dtype ≠ operation count",
        ),
        Claim(
            "RTX PRO 6000 peaks are 1/2/4 PFLOP",
            RTX_PRO_6000_BF16_PEAK == 1.0e15
            and RTX_PRO_6000_FP8_PEAK == 2.0e15
            and RTX_PRO_6000_FP4_PEAK == 4.0e15,
            "1/2/4",
            "BF16/FP8/FP4",
        ),
        Claim(
            "conservative NVFP4 wall-clock is 2.0× vs bf16 (not 4×)",
            NVFP4_SPEEDUP_CONSERVATIVE == 2.0
            and NVFP4_SPEEDUP_CONSERVATIVE < NVFP4_SPEEDUP_PEAK,
            f"{NVFP4_SPEEDUP_CONSERVATIVE:.2f}×",
            "2.0×; peak 4× is sensitivity",
        ),
        Claim(
            "published NVFP4/FP8 ratio in NVIDIA 1.31–1.73×",
            lo <= vs_fp8 <= hi,
            f"{vs_fp8:.2f}×",
            "2.0/1.5 ≈ 1.33",
        ),
        Claim(
            "frozen wall-clock spec is C1+NVFP4 mixed",
            FROZEN_WALLCLOCK == "C1+NVFP4"
            and pol["B0"].student == "bf16"
            and pol["B1"].student == "nvfp4"
            and pol["B2"].student == "nvfp4",
            FROZEN_WALLCLOCK,
            "B0 student bf16; B1/B2 nvfp4 (not moe-only)",
        ),
        Claim(
            "published C1+NVFP4 hours ≈571 (≤45% joint bf16)",
            _close(mixed_h, 571, rel=0.02) and mixed_h <= 0.45 * joint_h,
            f"{mixed_h:.0f} ({mixed_h / joint_h:.0%} of {joint_h:.0f})",
            "~571; ≤45% of joint",
        ),
        Claim(
            "B0 student stays bf16 (Theorem A neighborhood)",
            pol["B0"].student == "bf16" and pol["B0"].frozen_encoder_gemm == "nvfp4",
            f"{pol['B0'].student}/{pol['B0'].frozen_encoder_gemm}",
            "bf16 / nvfp4",
        ),
        Claim(
            "B1/B2 student is nvfp4 (all allowed GEMMs)",
            pol["B1"].student == "nvfp4" and pol["B2"].student == "nvfp4",
            f"{pol['B1'].student}/{pol['B2'].student}",
            "nvfp4",
        ),
        Claim(
            "lm_head is an NVFP4 GEMM slot (not must-bf16)",
            "lm_head" in NVFP4_GEMM_SLOTS and "lm_head" not in KEEP_HIGH_PREC,
            "lm_head ∈ GEMM, ∉ keep-high",
            "Theorem E is the input table",
        ),
        Claim(
            "attn softmax must stay high prec; QKV is NVFP4",
            "attn_softmax" in KEEP_HIGH_PREC and "attn_qkv" in NVFP4_GEMM_SLOTS,
            "softmax high / QKV nvfp4",
            "score/context ≠ linear projections",
        ),
        Claim(
            "must-high-prec set excludes lm_head",
            "lm_head" not in KEEP_HIGH_PREC
            and set(KEEP_HIGH_PREC)
            >= {"embed", "router", "rms_norm", "gate", "indexer", "attn_softmax", "qk_norm"},
            ",".join(KEEP_HIGH_PREC),
            "embed, LN, router, gate, softmax, indexer",
        ),
        Claim(
            "C1 bf16 hours unchanged by the NVFP4 policy",
            _close(c1_bf16_h, 1046, rel=0.02),
            f"{c1_bf16_h:.0f}",
            "~1046",
        ),
        Claim(
            "C1+FP8 fallback still ≈729",
            _close(fp8_h, 729, rel=0.02) and fp8_h > mixed_h,
            f"{fp8_h:.0f}",
            "~729 Hopper/Ada",
        ),
        Claim(
            "RTX PRO 6000 BF16 peak ≈ H100 (hours comparable)",
            _close(RTX_PRO_6000_BF16_PEAK / H100_BF16_PEAK, 1.0, rel=0.03),
            f"{RTX_PRO_6000_BF16_PEAK / H100_BF16_PEAK:.3f}×",
            "~1.01×",
        ),
        Claim(
            "MoE-only 4× Amdahl stays a sensitivity bound",
            nvfp4_speedup_from_gemm_frac(moe_f) > NVFP4_SPEEDUP_CONSERVATIVE,
            f"{nvfp4_speedup_from_gemm_frac(moe_f):.2f}× (f={moe_f:.1%})",
            ">2.0×; do not publish 4×",
        ),
        Claim(
            "allowed GEMM 6NT share ≥95% (lm_head included)",
            gemm_f >= 0.95,
            f"{gemm_f:.1%}",
            "≥95%",
        ),
    ]


def attn_score_flops(n: int, k: int, layers: int) -> float:
    """QK^T + AV multiply-adds, treating n_heads * head_dim = D.

    2 matmuls × 2 flop/MAC × D × n × k × layers.
    """
    return 4.0 * D * n * k * layers


def keys_csa(n: int, *, m: int = 4, n_win: int = 8192, index_topk: int = 256) -> int:
    compressed = max(n // m, 0)
    selected = min(index_topk, compressed)
    return selected + min(n_win, n)


def keys_hca(n: int, *, m_hca: int = 128, n_win: int = 8192) -> int:
    return n // m_hca + min(n_win, n)


def keys_dense(n: int) -> int:
    return n


@dataclass(frozen=True)
class KvRecipe:
    name: str
    per_token_dim: int  # elements stored per token per layer (K+V or latent)
    dtype_bytes: float
    layers: int
    seq: int
    compress: float = 1.0  # sequence-axis compression of THIS cache

    @property
    def bytes(self) -> float:
        return self.seq / self.compress * self.layers * self.per_token_dim * self.dtype_bytes


def kv_table(seq: int, dtype_bytes: float = 2.0) -> list[KvRecipe]:
    """Reproduce plan §16 plus MiniCPM5-native GQA-2.

    Plan §16's "1M" is 1_000_000 tokens (not 2^20). GQA-2 means 2 KV heads
    at head_dim=128. MLA-576 is a DeepSeek-V3-style latent (kv_lora=512 + rope=64).
    """
    dec_win_layers = LD_DEFAULT
    n_win = 8192
    l0 = LE_DEFAULT + LD_DEFAULT
    return [
        KvRecipe("decoder-only MHA (42L)", MHA_KV_DIM, dtype_bytes, l0, seq),
        KvRecipe("decoder-only GQA-2 (42L)", GQA2_KV_DIM, dtype_bytes, l0, seq),
        KvRecipe("decoder-only MLA-576 (42L)", MLA_LATENT, dtype_bytes, l0, seq),
        KvRecipe("YOCO + MLA-576 (1 global)", MLA_LATENT, dtype_bytes, 1, seq),
        KvRecipe("YOCO + MLA-576 + seq÷8", MLA_LATENT, dtype_bytes, 1, seq, compress=8.0),
        KvRecipe("YOCO + MLA-576 + CSA m=4", MLA_LATENT, dtype_bytes, 1, seq, compress=4.0),
        KvRecipe("YOCO + GQA-2 (1 global)", GQA2_KV_DIM, dtype_bytes, 1, seq),
        KvRecipe(
            "decoder 26×8K window (MLA-576)",
            MLA_LATENT,
            dtype_bytes,
            dec_win_layers,
            min(n_win, seq),
        ),
        KvRecipe(
            "encoder 16×8K window (MLA-576)",
            MLA_LATENT,
            dtype_bytes,
            LE_DEFAULT,
            min(n_win, seq),
        ),
    ]


@dataclass(frozen=True)
class Claim:
    name: str
    ok: bool
    observed: str
    expected: str
    note: str = ""


def _close(x: float, target: float, rel: float = 0.03, abs_tol: float = 0.0) -> bool:
    return abs(x - target) <= max(abs_tol, rel * abs(target))


def claims_middle_placeholder(budget: ModelBudget) -> list[Claim]:
    """Ledger against the published middle-tier spec (plan §1.2 / §3)."""
    tokens = 50e9
    # Plan §3 uses enc_active + dec_active (embed + lm_head counted on each stack).
    flops_plan = training_flops(budget.enc_active + budget.dec_active, tokens)
    h100_plan = gpu_hours(flops_plan, H100_BF16_EFF)
    kv_1m = {r.name: r for r in kv_table(1_000_000)}
    return [
        Claim("total ~12.25B", _close(budget.total, 12.25e9, rel=0.01), b(budget.total), "12.25B"),
        Claim(
            "enc active ~2.03B",
            _close(budget.enc_active, 2.03e9, rel=0.02),
            b(budget.enc_active),
            "2.03B",
        ),
        Claim(
            "dec active ~4.33B",
            _close(budget.dec_active, 4.33e9, rel=0.02),
            b(budget.dec_active),
            "4.33B",
        ),
        Claim(
            "emb ~0.27B",
            _close(budget.emb, 0.267e9, rel=0.02),
            b(budget.emb),
            "0.27B",
        ),
        Claim(
            "enc self-attn ~0.15B",
            _close(budget.enc.attn_total, 0.15e9, rel=0.05),
            b(budget.enc.attn_total),
            "0.15B",
        ),
        Claim(
            "dec self-attn ~0.25B",
            _close(budget.dec.layers * budget.attn.self_attn, 0.25e9, rel=0.05),
            b(budget.dec.layers * budget.attn.self_attn),
            "0.25B",
        ),
        Claim(
            "dec cross-attn ~0.22B",
            _close(budget.dec.layers * budget.attn.cross_attn, 0.22e9, rel=0.05),
            b(budget.dec.layers * budget.attn.cross_attn),
            "0.22B",
        ),
        Claim(
            "sparsity enc 8/21",
            abs(budget.sparsity_enc - 8 / 21) < 1e-12,
            f"{budget.sparsity_enc:.1%}",
            "38.1% (8/21)",
        ),
        Claim(
            "sparsity dec 11/21",
            abs(budget.sparsity_dec - 11 / 21) < 1e-12,
            f"{budget.sparsity_dec:.1%}",
            "52.4% (11/21)",
        ),
        Claim(
            "three tiers share 882 expert-slots",
            budget.expert_slots == 882,
            str(budget.expert_slots),
            "882",
        ),
        Claim(
            "50B tok ≈1325 H100-h (plan convention)",
            _close(h100_plan, 1325, rel=0.03),
            f"{h100_plan:.0f}",
            "~1325",
        ),
        Claim(
            "1M decoder-only MHA KV ≈344 GB",
            _close(kv_1m["decoder-only MHA (42L)"].bytes / 1e9, 343.93, rel=0.01),
            gb(kv_1m["decoder-only MHA (42L)"].bytes),
            "344 GB",
        ),
        Claim(
            "1M decoder-only GQA-2 KV ≈43 GB",
            _close(kv_1m["decoder-only GQA-2 (42L)"].bytes / 1e9, 42.99, rel=0.02),
            gb(kv_1m["decoder-only GQA-2 (42L)"].bytes),
            "43 GB",
        ),
        Claim(
            "1M decoder-only MLA-576 KV ≈48 GB",
            _close(kv_1m["decoder-only MLA-576 (42L)"].bytes / 1e9, 48.38, rel=0.02),
            gb(kv_1m["decoder-only MLA-576 (42L)"].bytes),
            "48 GB",
        ),
        Claim(
            "1M YOCO+MLA KV ≈1.15 GB",
            _close(kv_1m["YOCO + MLA-576 (1 global)"].bytes / 1e9, 1.15, rel=0.02),
            gb(kv_1m["YOCO + MLA-576 (1 global)"].bytes),
            "1.15 GB",
        ),
        Claim(
            "1M YOCO+MLA+÷8 KV ≈0.14 GB",
            _close(kv_1m["YOCO + MLA-576 + seq÷8"].bytes / 1e9, 0.14, rel=0.05),
            gb(kv_1m["YOCO + MLA-576 + seq÷8"].bytes),
            "0.14 GB",
        ),
        Claim(
            "26×8K window (MLA) ≈0.25 GB",
            _close(kv_1m["decoder 26×8K window (MLA-576)"].bytes / 1e9, 0.25, rel=0.15),
            gb(kv_1m["decoder 26×8K window (MLA-576)"].bytes),
            "~0.25 GB",
        ),
        Claim(
            "no MiniCPM μP: logit_scale = 1",
            abs(D / DIM_MODEL_BASE - 1.0) < 1e-12,
            f"{D / DIM_MODEL_BASE:.0f}",
            "1",
        ),
        Claim(
            "residual_scale identity (Llama, not 1.4/√L)",
            SCALE_EMB == 1.0 and SCALE_DEPTH == 1.0,
            f"scale_emb={SCALE_EMB} residual={SCALE_DEPTH}",
            "1 / 1",
            note="MiniCPM5 is Llama; do not keep MiniCPM-2B μP",
        ),
        Claim(
            "freeze-enc train-dec saves ~20% vs joint",
            0.74 <= flops_freeze_encoder(budget, 50e9) / flops_joint(budget, 50e9) <= 0.82,
            f"{flops_freeze_encoder(budget, 50e9) / flops_joint(budget, 50e9):.0%}",
            "~77–80%",
        ),
        Claim(
            "independent 50B+50B+20B stitch is MORE FLOPs",
            next(r.flops for r in staged_recipes(budget, 50e9) if r.name.startswith("independent"))
            > flops_joint(budget, 50e9),
            f"{next(r.flops for r in staged_recipes(budget, 50e9) if r.name.startswith('independent')) / flops_joint(budget, 50e9):.0%}",
            ">100% of joint",
        ),
        Claim(
            "unfreeze curriculum ≤85% of joint 50B",
            next(r.flops for r in staged_recipes(budget, 50e9) if r.name.startswith("unfreeze"))
            <= 0.85 * flops_joint(budget, 50e9),
            f"{next(r.flops for r in staged_recipes(budget, 50e9) if r.name.startswith('unfreeze')) / flops_joint(budget, 50e9):.0%}",
            "≤85%",
        ),
    ]


def verify(budget: ModelBudget | None = None) -> list[Claim]:
    budget = budget or compute_budget(TIERS[DEFAULT_TIER])
    if budget.tier.key != "middle" or budget.first_dense or budget.attn.name != "GQA 16/2":
        raise ValueError("--verify is defined on the published middle-tier GQA budget")
    return (
        claims_middle_placeholder(budget)
        + claims_curriculum(budget)
        + claims_fp8(budget)
        + claims_nvfp4(budget)
    )


def print_budget(budget: ModelBudget) -> None:
    t = budget.tier
    print(f"embedding (untied + lm_head) : {b(budget.emb)} + {b(budget.lm_head)}")
    print(f"expert (single)          : {m(EXPERT)}")
    print(f"dense FFN (MiniCPM5)     : {m(DENSE_FFN)}")
    print(f"attn accounting          : {budget.attn.name}  self={m(budget.attn.self_attn)} cross={m(budget.attn.cross_attn)}")
    print(f"first-layer dense        : {budget.first_dense}")
    print(f"-- Encoder (self-decoder) --")
    print(f"  layers={t.le} experts={t.ns_e}+{t.nr_e} top_k={t.tk_e}  moe_layers={budget.enc.moe_layers}")
    print(f"  ACTIVE / input token   : {b(budget.enc_active)}")
    print(f"  stack total            : {b(budget.enc.stack_total)}")
    print(f"  expert sparsity        : {budget.sparsity_enc:.1%}  ({t.ns_e + t.tk_e}/{t.ns_e + t.nr_e})")
    print(f"-- Decoder (cross-decoder) --")
    print(f"  layers={t.ld} experts={t.ns_d}+{t.nr_d} top_k={t.tk_d}  moe_layers={budget.dec.moe_layers}")
    print(f"  ACTIVE / output token  : {b(budget.dec_active)}")
    print(f"  stack total            : {b(budget.dec.stack_total)}")
    print(f"  expert sparsity        : {budget.sparsity_dec:.1%}  ({t.ns_d + t.tk_d}/{t.ns_d + t.nr_d})")
    print("-- Summary --")
    print(f"  TOTAL parameters       : {b(budget.total)}")
    print(f"  fwd ACTIVE (emb once)  : {b(budget.fwd_active)}")
    print(f"  plan ACTIVE (enc+dec)  : {b(budget.enc_active + budget.dec_active)}")
    print(f"  expert-slots           : {budget.expert_slots}")
    print(f"  attn / moe fraction    : {budget.attn_frac:.1%} / {budget.moe_frac:.1%}")


def print_compute(budget: ModelBudget, tokens: float = 50e9) -> None:
    plan_n = budget.enc_active + budget.dec_active
    rows = [
        ("plan convention (enc_act+dec_act, emb twice)", plan_n),
        ("full forward (emb once)", budget.fwd_active),
        ("encoder-only prefill", budget.enc_active),
    ]
    print(f"-- Training compute @ {tokens/1e9:.0f}B tok, 6NT, 40% MFU --")
    for label, n_act in rows:
        flops = training_flops(n_act, tokens)
        print(
            f"  {label:44s}  N={b(n_act):>7s}  "
            f"{gpu_hours(flops, H100_BF16_EFF):6.0f} H100-h  "
            f"{gpu_hours(flops, A100_BF16_EFF):6.0f} A100-h"
        )


def print_staged(budget: ModelBudget, tokens: float = 50e9) -> None:
    recipes = staged_recipes(budget, tokens)
    joint = recipes[0]
    print(f"-- Staged vs joint @ {tokens/1e9:.0f}B tok envelope (emb-once 6NT) --")
    for r in recipes:
        ratio = r.flops / joint.flops
        if ratio < 0.98:
            delta = "save"
        elif ratio <= 1.02:
            delta = "base"
        else:
            delta = "MORE"
        print(
            f"  {r.name:48s}  {r.h100_h():6.0f} H100-h  "
            f"{ratio:5.0%} vs joint ({delta})  # {r.note}"
        )


def print_curriculum(budget: ModelBudget, tokens: float = 50e9) -> None:
    print("-- Freeze boundary (C1 frozen spec; delayed-enc-MoE is sensitivity) --")
    print(
        f"  {'phase':<4s} {'detach':<7s} {'tied_emb':<16s} {'gate':<8s} "
        f"{'frozen':<42s} {'trainable'}"
    )
    for fb in FREEZE_BOUNDARIES:
        print(
            f"  {fb.phase:<4s} {str(fb.detach_at_cache):<7s} {fb.tied_emb:<16s} "
            f"{fb.gate:<8s} {','.join(fb.frozen) or '—':<42s} {','.join(fb.trainable)}"
        )
        print(f"       # {fb.note}")

    print("-- Stored-param memory (bf16 weights + bf16 grads + Adam m,v; no activations) --")
    print(
        f"  {'phase':<6s} {'variant':<8s} {'train':>8s} {'frozen':>8s} "
        f"{'Adam':>10s} {'footprint':>10s} {'vs B2 Adam'}"
    )
    b2_tr, _ = phase_param_counts(budget, "B2", delayed=False)
    b2_adam = optimizer_state_bytes(b2_tr)
    for delayed, variant in ((False, "C1"), (True, "C2")):
        for phase in ("B0", "B1", "B2"):
            tr, fr = phase_param_counts(budget, phase, delayed=delayed)
            adam = optimizer_state_bytes(tr)
            foot = param_footprint_bytes(tr, fr)
            print(
                f"  {phase:<6s} {variant:<8s} {b(tr):>8s} {b(fr):>8s} "
                f"{gb(adam):>10s} {gb(foot):>10s} {adam / b2_adam:5.0%}"
            )
    print(
        f"  detach drops encoder layer activations: keep "
        f"{activation_keep_frac():.0%} (26/42); save ~38% vs joint"
    )
    print(
        f"  cache W_K/W_V (d_kv={KV_DIM})              : {m(CACHE_PROJ)}  "
        f"(trainable in B0, after encoder.detach())"
    )
    print(
        f"  encoder FFN C1 MoE active / C2 dense   : "
        f"{b(encoder_active_no_emb(budget, dense=False))} / "
        f"{b(encoder_active_no_emb(budget, dense=True))}"
    )

    print(f"-- Token-split sensitivity @ {tokens/1e9:.0f}B envelope --")
    print(f"  {'split':<22s} {'C1 vs joint':>12s} {'C2 vs joint':>12s} {'valid'}")
    c1_rows = {r[0]: r for r in curriculum_split_table(budget, delayed=False)}
    for label, _flops, ratio_c2, valid in curriculum_split_table(budget, delayed=True):
        ratio_c1 = c1_rows[label][2]
        flag = "yes" if valid else "NO (write/read never co-adapt)"
        print(f"  {label:<22s} {ratio_c1:11.0%} {ratio_c2:11.0%}   {flag}")


def print_fp8(budget: ModelBudget, tokens: float = 50e9) -> None:
    joint = flops_joint(budget, tokens)
    c1 = flops_curriculum(budget, tokens, delayed=False)
    moe_f = fp8_moe_active_frac(budget)
    gemm_f = fp8_gemm_frac(budget)
    moe_x = fp8_speedup_from_gemm_frac(moe_f)
    gemm_x = fp8_speedup_from_gemm_frac(gemm_f)
    joint_h = wallclock_h100_h(joint)
    c1_h = wallclock_h100_h(c1)
    mixed_h = wallclock_c1_fp8_policy(budget, tokens)
    print("-- C1+FP8 (Hopper/Ada fallback; does not change 6NT) --")
    print(f"  H100 peak bf16 / FP8              : {H100_BF16_PEAK/1e12:.0f} / {H100_FP8_PEAK/1e12:.0f} TFLOPS")
    print(f"  peak ratio                        : {FP8_SPEEDUP_PEAK:.2f}×")
    attn_f = (budget.enc.attn_total + budget.dec.attn_total) / budget.fwd_active
    print(f"  MoE stored-param / 6NT-active     : {budget.moe_frac:.1%} / {moe_f:.1%}")
    print(f"  attn 6NT-active / GEMM 6NT        : {attn_f:.1%} / {gemm_f:.1%}")
    print(f"  MoE-only Amdahl (2× GEMM)         : {moe_x:.2f}×  (justifies fallback 1.5×)")
    print(f"  MoE+attn+head Amdahl (upper bound): {gemm_x:.2f}×")
    print(f"  conservative 12B wall-clock       : {FP8_SPEEDUP_CONSERVATIVE:.2f}×  (kernel + comm + scale)")
    print(f"  keep high precision               : {', '.join(KEEP_HIGH_PREC)}")
    print(f"  {'phase':<4s} {'student':<12s} {'frozen_enc':<12s} note")
    for p in FP8_PHASE_POLICY:
        print(f"  {p.phase:<4s} {p.student:<12s} {p.frozen_encoder_gemm:<12s} {p.note}")
    print(f"-- Wall-clock @ {tokens/1e9:.0f}B tok (H100-h vs joint bf16) --")
    rows = [
        ("joint bf16", joint_h, 1.0),
        ("C1 bf16", c1_h, c1_h / joint_h),
        (
            "C1+FP8 (Hopper/Ada fallback)",
            mixed_h,
            mixed_h / joint_h,
        ),
        (
            "C1 all-1.5× (sensitivity)",
            wallclock_h100_h(c1, speedup=FP8_SPEEDUP_CONSERVATIVE),
            wallclock_h100_h(c1, speedup=FP8_SPEEDUP_CONSERVATIVE) / joint_h,
        ),
        (
            "C1 MoE-Amdahl all (sensitivity)",
            wallclock_h100_h(c1, speedup=moe_x),
            wallclock_h100_h(c1, speedup=moe_x) / joint_h,
        ),
        (
            "C1 peak 2× (upper bound)",
            wallclock_h100_h(c1, speedup=FP8_SPEEDUP_PEAK),
            wallclock_h100_h(c1, speedup=FP8_SPEEDUP_PEAK) / joint_h,
        ),
    ]
    for name, hours, ratio in rows:
        print(f"  {name:32s}  {hours:6.0f} H100-h  {ratio:5.0%} vs joint bf16")
    n_enc = encoder_stack_total(budget, dense=False)
    print(
        f"  frozen encoder weights bf16 / FP8 : "
        f"{gb(n_enc * WEIGHT_BYTES)} / {gb(n_enc * 1.0)}"
    )
    print("  published wall-clock is C1+NVFP4; see --nvfp4")


def print_nvfp4(budget: ModelBudget, tokens: float = 50e9) -> None:
    joint = flops_joint(budget, tokens)
    c1 = flops_curriculum(budget, tokens, delayed=False)
    moe_f = fp8_moe_active_frac(budget)
    gemm_f = fp8_gemm_frac(budget)
    moe_x = nvfp4_speedup_from_gemm_frac(moe_f)
    gemm_x = nvfp4_speedup_from_gemm_frac(gemm_f)
    joint_h = wallclock_h100_h(joint)
    c1_h = wallclock_h100_h(c1)
    mixed_h = wallclock_c1_nvfp4_policy(budget, tokens)
    fp8_h = wallclock_c1_fp8_policy(budget, tokens)
    print("-- C1+NVFP4 (frozen Phase B wall-clock; does not change 6NT) --")
    print(
        f"  RTX PRO 6000 peak bf16/FP8/FP4   : "
        f"{RTX_PRO_6000_BF16_PEAK/1e15:.0f} / "
        f"{RTX_PRO_6000_FP8_PEAK/1e15:.0f} / "
        f"{RTX_PRO_6000_FP4_PEAK/1e15:.0f} PFLOP"
    )
    print(f"  6000 mem / bandwidth              : {RTX_PRO_6000_MEM_GIB} GiB / {RTX_PRO_6000_BANDWIDTH/1e9:.0f} GB/s")
    print(f"  peak FP4 / BF16                   : {NVFP4_SPEEDUP_PEAK:.2f}×")
    print(f"  NVIDIA NVFP4 vs FP8 (GB200/300)   : {NVFP4_VS_FP8_NVIDIA[0]:.2f}–{NVFP4_VS_FP8_NVIDIA[1]:.2f}×")
    attn_f = (budget.enc.attn_total + budget.dec.attn_total) / budget.fwd_active
    print(f"  MoE stored-param / 6NT-active     : {budget.moe_frac:.1%} / {moe_f:.1%}")
    print(f"  attn 6NT-active / GEMM 6NT        : {attn_f:.1%} / {gemm_f:.1%}")
    print(f"  MoE-only Amdahl (4× GEMM)         : {moe_x:.2f}×  (sensitivity)")
    print(f"  allowed-GEMM Amdahl (upper bound) : {gemm_x:.2f}×")
    print(f"  conservative 12B wall-clock       : {NVFP4_SPEEDUP_CONSERVATIVE:.2f}× vs bf16")
    print(f"  keep high precision               : {', '.join(KEEP_HIGH_PREC)}")
    print(f"  NVFP4 GEMM slots                  : {', '.join(NVFP4_GEMM_SLOTS)}")
    print(f"  {'phase':<4s} {'student':<12s} {'frozen_enc':<12s} note")
    for p in NVFP4_PHASE_POLICY:
        print(f"  {p.phase:<4s} {p.student:<12s} {p.frozen_encoder_gemm:<12s} {p.note}")
    print(f"-- Wall-clock @ {tokens/1e9:.0f}B tok (H100-h ≈ 6000-h vs joint bf16) --")
    rows = [
        ("joint bf16", joint_h, 1.0),
        ("C1 bf16", c1_h, c1_h / joint_h),
        (
            "C1+FP8 (Hopper/Ada fallback)",
            fp8_h,
            fp8_h / joint_h,
        ),
        (
            "C1+NVFP4 (frozen spec)",
            mixed_h,
            mixed_h / joint_h,
        ),
        (
            "C1 all-2.0× (sensitivity)",
            wallclock_h100_h(c1, speedup=NVFP4_SPEEDUP_CONSERVATIVE),
            wallclock_h100_h(c1, speedup=NVFP4_SPEEDUP_CONSERVATIVE) / joint_h,
        ),
        (
            "C1 MoE-Amdahl 4× (sensitivity)",
            wallclock_h100_h(c1, speedup=moe_x),
            wallclock_h100_h(c1, speedup=moe_x) / joint_h,
        ),
        (
            "C1 peak 4× (upper bound)",
            wallclock_h100_h(c1, speedup=NVFP4_SPEEDUP_PEAK),
            wallclock_h100_h(c1, speedup=NVFP4_SPEEDUP_PEAK) / joint_h,
        ),
    ]
    for name, hours, ratio in rows:
        print(f"  {name:32s}  {hours:6.0f} H100-h  {ratio:5.0%} vs joint bf16")
    n_enc = encoder_stack_total(budget, dense=False)
    print(
        f"  frozen encoder weights bf16 / NVFP4 ~ : "
        f"{gb(n_enc * WEIGHT_BYTES)} / {gb(n_enc * 0.5)}"
    )


def print_kv(lengths: Sequence[int] = (8_192, 32_768, 131_072, 262_144, 1_000_000)) -> None:
    print("-- KV cache (bf16, content bytes; GB = 1e9) --")
    header = f"{'recipe':<36s}" + "".join(f"{n:>12d}" for n in lengths)
    print(header)
    recipes_by_name: dict[str, list[KvRecipe]] = {}
    for n in lengths:
        for r in kv_table(n):
            recipes_by_name.setdefault(r.name, []).append(r)
    for name, recs in recipes_by_name.items():
        print(f"{name:<36s}" + "".join(f"{gb(r.bytes):>12s}" for r in recs))


def print_attn_complexity(
    lengths: Sequence[int] = (4_096, 8_192, 32_768, 131_072, 262_144, 1_048_576),
) -> None:
    print("-- Attention keys per query (n_win=8192, index_topk=256, m=4, m'=128) --")
    print(f"{'n':>10s} {'dense':>10s} {'CSA':>10s} {'HCA':>10s} {'CSA/dense':>10s} {'HCA/dense':>10s}")
    for n in lengths:
        d = keys_dense(n)
        c = keys_csa(n)
        h = keys_hca(n)
        print(f"{n:10d} {d:10d} {c:10d} {h:10d} {c/d:10.3f} {h/d:10.3f}")

    print("-- Score+AV FLOPs vs MLP forward (middle-tier fwd_active, one sequence of length n) --")
    mid = compute_budget(TIERS["middle"])
    mlp = 2.0 * mid.fwd_active  # 2 N_active per token; ×n below
    print(
        f"{'n':>10s} {'mlp_fwd':>12s} {'dense42':>12s} {'enc_hyb16':>12s} "
        f"{'enc+dec_win':>12s} {'xattn26':>12s}"
    )
    for n in lengths:
        mlp_n = mlp * n
        dense = attn_score_flops(n, n, 42)
        enc_hyb = 0.0
        for i in range(16):
            k = keys_csa(n) if i % 2 == 0 else keys_hca(n)
            enc_hyb += attn_score_flops(n, k, 1)
        dec_win = attn_score_flops(n, min(8192, n), 26)
        xattn = attn_score_flops(n, n, 26)
        print(
            f"{n:10d} {mlp_n:12.3e} {dense:12.3e} {enc_hyb:12.3e} "
            f"{enc_hyb + dec_win:12.3e} {xattn:12.3e}"
        )


def print_prefill_decode(budget: ModelBudget) -> None:
    print("-- Prefill early-exit vs decode bottleneck --")
    print(
        f"  encoder layer fraction     : {budget.enc.layers}/{budget.enc.layers + budget.dec.layers}"
        f" = {budget.enc.layers / (budget.enc.layers + budget.dec.layers):.0%}"
    )
    print(
        f"  encoder activation share   : {b(budget.enc_active)} / "
        f"{b(budget.enc_active + budget.dec_active)} = "
        f"{budget.enc_active / (budget.enc_active + budget.dec_active):.0%}"
    )
    print("  (YOCO prefill runs encoder on all prompt tokens; decoder only on the last position.)")
    for n in (32_768, 131_072, 262_144, 1_048_576):
        mlp_dec = 2.0 * budget.dec_active
        xattn_full = attn_score_flops(1, n, budget.dec.layers)
        xattn_m4 = attn_score_flops(1, n // 4, budget.dec.layers)
        xattn_topk = attn_score_flops(1, keys_csa(n), budget.dec.layers)
        print(
            f"  decode @ n={n:<8d}  dec-MLP={mlp_dec:.3e}  "
            f"xattn-full={xattn_full:.3e} ({xattn_full / mlp_dec:.2f}×MLP)  "
            f"xattn-m4={xattn_m4:.3e} ({xattn_m4 / mlp_dec:.2f}×)  "
            f"xattn-CSA={xattn_topk:.3e} ({xattn_topk / mlp_dec:.2f}×)"
        )


def print_attn_params() -> None:
    print("-- Self-attn params per layer --")
    print(f"  MiniCPM5 GQA 16/2            : {m(GQA_SELF)}")
    print(f"  MiniCPM5 MHA 4d² (not used)  : {m(MHA)}")
    print(f"  cross-attn Q/O 2d²           : {m(CROSS_QO)}")
    print(f"  CSA MQA-128 (detailed)       : {m(_csa_mqa64_self_attn(True))}")
    print(f"  HCA MQA-128 (detailed)       : {m(_csa_mqa64_self_attn(False))}")
    print(f"  CSA/HCA average              : {m(attn_accounting('csa_mqa64').self_attn)}")
    print("  (8K window dominates long-context score FLOPs; index_topk 256 vs 512 is a ~3% delta.)")


def retune_routed(
    *,
    first_dense: bool,
    attn: AttnAccounting,
    target: float = 12.25e9,
    base: Tier | None = None,
) -> tuple[Tier, ModelBudget]:
    """Brute-force small Nr_e / Nr_d adjustments to hit ``target`` total params."""
    base = base or TIERS["middle"]
    best: tuple[tuple[int, int, float, int], Tier, ModelBudget] | None = None
    for nr_e in range(8, 26):
        for nr_d in range(8, 26):
            tier = Tier(
                name=f"retune Nr={nr_e}/{nr_d}",
                key="retune",
                ns_e=base.ns_e,
                tk_e=base.tk_e,
                nr_e=nr_e,
                ns_d=base.ns_d,
                tk_d=base.tk_d,
                nr_d=nr_d,
                le=base.le,
                ld=base.ld,
            )
            budget = compute_budget(tier, attn=attn, first_dense=first_dense)
            err = abs(budget.total - target)
            drift = abs(nr_e - base.nr_e) + abs(nr_d - base.nr_d)
            balance = abs(nr_e - nr_d)
            close = 0 if err / target <= 0.005 else 1
            score = (close, drift, err, balance)
            if best is None or score < best[0]:
                best = (score, tier, budget)
    assert best is not None
    return best[1], best[2]


def print_retune() -> None:
    print("-- Nr retune to 12.25B (top-k unchanged ⇒ activations almost unchanged) --")
    attn_gqa = attn_accounting("gqa")
    attn_csa = attn_accounting("csa_mqa64")
    for label, first_dense, attn in (
        ("first-dense + GQA attn", True, attn_gqa),
        ("all-MoE + CSA/HCA MQA-64", False, attn_csa),
        ("first-dense + CSA/HCA MQA-64", True, attn_csa),
    ):
        tier, budget = retune_routed(first_dense=first_dense, attn=attn)
        print(
            f"  {label:34s}  Enc 1+{tier.nr_e} top-k{tier.tk_e}  "
            f"Dec 1+{tier.nr_d} top-k{tier.tk_d}  "
            f"total={b(budget.total)}  in={b(budget.enc_active)}  out={b(budget.dec_active)}"
        )


def print_mup() -> None:
    print("-- MiniCPM5 scales (Llama; no MiniCPM-2B μP) --")
    print(f"  scale_emb                 : {SCALE_EMB}")
    print(f"  logits / (d/dim_model_base): {D / DIM_MODEL_BASE:.0f}")
    print(f"  residual_scale            : {SCALE_DEPTH}  (identity; do not use 1.4/√42)")
    print(f"  base_layers               : {BASE_LAYERS}  (16 encoder + 26 decoder)")


def print_claims(cs: Iterable[Claim]) -> int:
    cs = list(cs)
    print("-- Claim ledger (middle tier + C1 + C1+FP8 fallback + C1+NVFP4, GQA 16/2, all-MoE) --")
    failed = 0
    for c in cs:
        mark = "PASS" if c.ok else "FAIL"
        if not c.ok:
            failed += 1
        extra = f"  # {c.note}" if c.note else ""
        print(f"  [{mark}] {c.name:56s}  observed={c.observed:28s}  expected={c.expected}{extra}")
    print(f"  {len(cs) - failed}/{len(cs)} passed")
    return failed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tier",
        choices=["middle", "low", "near_dense", "all"],
        default="middle",
        help="activation tier (default: middle)",
    )
    p.add_argument(
        "--attn",
        choices=["gqa", "placeholder", "csa_mqa64"],
        default="gqa",
        help="self-attn parameter accounting (placeholder is a deprecated alias of gqa)",
    )
    p.add_argument(
        "--first-dense",
        action="store_true",
        help="keep the first layer of each stack dense (plan Phase A)",
    )
    p.add_argument("--tokens", type=float, default=50e9, help="token budget for GPU-hour estimate")
    p.add_argument(
        "--verify",
        action="store_true",
        help="assert published middle-tier + freeze-curriculum + FP8 fallback + NVFP4 claims; exit 1 on failure",
    )
    p.add_argument("--full", action="store_true", help="print KV / attention-complexity / μP sections")
    p.add_argument(
        "--staged",
        action="store_true",
        help="print freeze-curriculum vs independent-merge FLOPs",
    )
    p.add_argument(
        "--curriculum",
        action="store_true",
        help="print freeze boundaries, optimizer memory, token-split sensitivity",
    )
    p.add_argument(
        "--fp8",
        action="store_true",
        help="print C1+FP8 Hopper/Ada fallback wall-clock (does not change 6NT)",
    )
    p.add_argument(
        "--nvfp4",
        action="store_true",
        help="print C1+NVFP4 frozen wall-clock (does not change 6NT)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    attn = attn_accounting(args.attn)

    if args.verify:
        failed = print_claims(verify())
        return 1 if failed else 0

    keys = ["low", "middle", "near_dense"] if args.tier == "all" else [args.tier]
    for i, key in enumerate(keys):
        if i:
            print()
        print(f"===== {TIERS[key].name} =====")
        budget = compute_budget(TIERS[key], attn=attn, first_dense=args.first_dense)
        print_budget(budget)
        print_compute(budget, tokens=args.tokens)
        if (args.staged or args.curriculum or args.fp8 or args.nvfp4) and key == keys[-1]:
            if args.staged:
                print()
                print_staged(budget, tokens=args.tokens)
            if args.curriculum:
                print()
                print_curriculum(budget, tokens=args.tokens)
            if args.fp8:
                print()
                print_fp8(budget, tokens=args.tokens)
            if args.nvfp4:
                print()
                print_nvfp4(budget, tokens=args.tokens)

    if args.full:
        print()
        print_kv()
        print()
        print_attn_params()
        print()
        print_attn_complexity()
        print()
        print_prefill_decode(compute_budget(TIERS["middle"], attn=attn, first_dense=args.first_dense))
        print()
        print_mup()
        print()
        print_retune()
        print()
        print_staged(compute_budget(TIERS["middle"], attn=attn, first_dense=args.first_dense), tokens=args.tokens)
        print()
        print_curriculum(compute_budget(TIERS["middle"], attn=attn, first_dense=args.first_dense), tokens=args.tokens)
        print()
        print_fp8(compute_budget(TIERS["middle"], attn=attn, first_dense=args.first_dense), tokens=args.tokens)
        print()
        print_nvfp4(compute_budget(TIERS["middle"], attn=attn, first_dense=args.first_dense), tokens=args.tokens)
        print()
        if args.tier in ("middle", "all") and args.attn in ("gqa", "placeholder") and not args.first_dense:
            print_claims(verify())
    elif args.curriculum and not args.staged:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
