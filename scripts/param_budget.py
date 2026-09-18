#!/usr/bin/env python3
"""CAT-YOKO parameter budget + middle-tier theoretical verification.

Causal Encoder-Decoder (YOCO-style) MoE built on MiniCPM-2B.

Default target (compute-budget middle tier): total ~12B, Encoder(input)
active ~2.3B, Decoder(output) active ~4.5B.

Note: total params ~ memory/storage; TRAINING FLOPs ~ ACTIVE params x tokens.
Cutting total does not cut training cost -- cutting ACTIVE does.

The default attention term is the plan's *placeholder* (1.25x MHA) until the
real CSA/HCA/MLA projection dims are frozen. A MiniCPM-native CSA/HCA MQA-64
estimate is available via ``--attn csa_mqa64`` as a sensitivity check; it
does not change the published middle-tier spec.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---- MiniCPM-2B base (openbmb/MiniCPM-2B-sft-bf16 config.json) ----
D = 2304
V = 122753
N_HEADS = 36
HEAD_DIM = 64  # D / N_HEADS
DENSE_INT = 5760  # MiniCPM dense SwiGLU intermediate
MOE_INT = 2048
SCALE_EMB = 12.0
DIM_MODEL_BASE = 256
SCALE_DEPTH = 1.4
BASE_LAYERS = 40

EMB = V * D
EXPERT = 3 * D * MOE_INT  # SwiGLU expert (gate, up, down)
DENSE_FFN = 3 * D * DENSE_INT
MHA = 4 * D * D
PLACEHOLDER_SELF_ATTN = int(1.25 * MHA)
PLACEHOLDER_CROSS_ATTN = MHA

# Hardware for GPU-hour estimates (plan §15.1): 40% MFU.
H100_BF16_EFF = 4.0e14  # ~989 TFLOPS peak * 0.40
A100_BF16_EFF = 1.25e14  # ~312 TFLOPS peak * 0.40
SECONDS_PER_HOUR = 3600.0

# KV recipes used in plan §16 (bytes of cache content, not allocator padding).
MLA_LATENT = 512 + 64  # DeepSeek-V3-style: kv_lora_rank + qk_rope_head_dim
GQA4_KV_DIM = 2 * 4 * HEAD_DIM  # 4 KV heads, K and V
MHA_KV_DIM = 2 * D  # 36 KV heads, K and V
MQA64_KV_DIM = 2 * HEAD_DIM  # single KV head at MiniCPM head_dim


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
    ld: int = 24


TIERS: dict[str, Tier] = {
    "low": Tier("省算力档", "low", ns_e=1, tk_e=3, nr_e=20, ns_d=1, tk_d=4, nr_d=15),
    "middle": Tier("默认（中间档）", "middle", ns_e=1, tk_e=6, nr_e=17, ns_d=1, tk_d=8, nr_d=17),
    "near_dense": Tier("近-dense 档", "near_dense", ns_e=2, tk_e=8, nr_e=10, ns_d=2, tk_d=12, nr_d=20),
}

DEFAULT_TIER = "middle"


@dataclass(frozen=True)
class AttnAccounting:
    name: str
    self_attn: int
    cross_attn: int
    note: str


def _csa_mqa64_self_attn(is_csa: bool) -> int:
    """MiniCPM-native CSA/HCA parameter estimate (MQA, head_dim=64).

    Scaled from DeepSeek-V4 (arXiv 2606.19348 §2.3) onto MiniCPM dims rather
    than copying V4's head_dim=512, which would be oversized at d=2304.

    Shared pieces: LoRA-Q (W_DQ, W_UQ), grouped output, sliding-window KV,
    CSA/HCA compressor. CSA adds a Lightning Indexer; HCA does not.
    """
    q_lora = 512
    kv_dim = HEAD_DIM  # MQA-64
    o_groups = 4  # 36 heads / 4
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
    if kind == "placeholder":
        return AttnAccounting(
            name="placeholder 1.25×MHA",
            self_attn=PLACEHOLDER_SELF_ATTN,
            cross_attn=PLACEHOLDER_CROSS_ATTN,
            note="plan §3 default until CSA/HCA/MLA dims are frozen",
        )
    if kind == "csa_mqa64":
        # Interleave CSA:HCA ≈ 1:1; average the two layer types.
        avg_self = (_csa_mqa64_self_attn(True) + _csa_mqa64_self_attn(False)) // 2
        # Cross-attn stays dense MHA/MQA-style 4d² as an upper bound
        # (YOCO global cache read). MLA would be smaller.
        return AttnAccounting(
            name="CSA/HCA MQA-64 (sensitivity)",
            self_attn=avg_self,
            cross_attn=MHA,
            note="MiniCPM-native CSA/HCA estimate; does not change published spec",
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
    enc: StackBudget
    dec: StackBudget
    total: int
    enc_active: int  # includes tied emb (plan convention: per-token stack)
    dec_active: int
    fwd_active: int  # embedding counted once (one full forward)

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
    attn = attn or attn_accounting("placeholder")
    enc = _stack(tier.le, tier.ns_e, tier.tk_e, tier.nr_e, attn.self_attn, 0, first_dense)
    dec = _stack(
        tier.ld, tier.ns_d, tier.tk_d, tier.nr_d, attn.self_attn, attn.cross_attn, first_dense
    )
    total = EMB + enc.stack_total + dec.stack_total
    enc_active = EMB + enc.active_no_emb
    dec_active = EMB + dec.active_no_emb
    fwd_active = EMB + enc.active_no_emb + dec.active_no_emb
    return ModelBudget(
        tier=tier,
        attn=attn,
        first_dense=first_dense,
        emb=EMB,
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
    """Reproduce plan §16 plus MiniCPM-native MQA-64.

    Plan §16's "1M" is 1_000_000 tokens (not 2^20). GQA-4 means 4 KV heads
    (not 36/4=9). MLA-576 is a DeepSeek-V3-style latent (kv_lora=512 + rope=64).
    """
    dec_win_layers = 24
    n_win = 8192
    return [
        KvRecipe("decoder-only MHA (40L)", MHA_KV_DIM, dtype_bytes, 40, seq),
        KvRecipe("decoder-only GQA-4 (40L)", GQA4_KV_DIM, dtype_bytes, 40, seq),
        KvRecipe("decoder-only MLA-576 (40L)", MLA_LATENT, dtype_bytes, 40, seq),
        KvRecipe("YOCO + MLA-576 (1 global)", MLA_LATENT, dtype_bytes, 1, seq),
        KvRecipe("YOCO + MLA-576 + seq÷8", MLA_LATENT, dtype_bytes, 1, seq, compress=8.0),
        KvRecipe("YOCO + MLA-576 + CSA m=4", MLA_LATENT, dtype_bytes, 1, seq, compress=4.0),
        KvRecipe("YOCO + MQA-64 (1 global)", MQA64_KV_DIM, dtype_bytes, 1, seq),
        KvRecipe(
            "decoder 24×8K window (MLA-576)",
            MLA_LATENT,
            dtype_bytes,
            dec_win_layers,
            min(n_win, seq),
        ),
        KvRecipe(
            "encoder 16×8K window (MLA-576)",
            MLA_LATENT,
            dtype_bytes,
            16,
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
    # Plan §3 uses enc_active + dec_active (embedding counted twice).
    flops_plan = training_flops(budget.enc_active + budget.dec_active, tokens)
    h100_plan = gpu_hours(flops_plan, H100_BF16_EFF)
    kv_1m = {r.name: r for r in kv_table(1_000_000)}
    return [
        Claim("total ~12B", _close(budget.total, 12e9, rel=0.02), b(budget.total), "12.0B"),
        Claim(
            "enc active ~2.3B",
            _close(budget.enc_active, 2.3e9, rel=0.04),
            b(budget.enc_active),
            "2.3B",
        ),
        Claim(
            "dec active ~4.5B",
            _close(budget.dec_active, 4.5e9, rel=0.04),
            b(budget.dec_active),
            "4.5B",
        ),
        Claim(
            "emb ~0.28B",
            _close(budget.emb, 0.28e9, rel=0.05),
            b(budget.emb),
            "0.28B",
        ),
        Claim(
            "enc self-attn ~0.42B",
            _close(budget.enc.attn_total, 0.42e9, rel=0.05),
            b(budget.enc.attn_total),
            "0.42B",
        ),
        Claim(
            "dec self-attn ~0.64B",
            _close(budget.dec.layers * budget.attn.self_attn, 0.64e9, rel=0.05),
            b(budget.dec.layers * budget.attn.self_attn),
            "0.64B",
        ),
        Claim(
            "dec cross-attn ~0.51B",
            _close(budget.dec.layers * budget.attn.cross_attn, 0.51e9, rel=0.05),
            b(budget.dec.layers * budget.attn.cross_attn),
            "0.51B",
        ),
        Claim(
            "sparsity enc 7/18",
            abs(budget.sparsity_enc - 7 / 18) < 1e-12,
            f"{budget.sparsity_enc:.1%}",
            "38.9% (7/18)",
        ),
        Claim(
            "sparsity dec 9/18",
            abs(budget.sparsity_dec - 9 / 18) < 1e-12,
            f"{budget.sparsity_dec:.1%}",
            "50.0% (9/18)",
        ),
        Claim(
            "three tiers share 720 expert-slots",
            budget.expert_slots == 720,
            str(budget.expert_slots),
            "720",
        ),
        Claim(
            "50B tok ≈1400 H100-h (plan convention)",
            _close(h100_plan, 1400, rel=0.05),
            f"{h100_plan:.0f}",
            "~1400",
        ),
        Claim(
            "1M decoder-only MHA KV ≈369 GB",
            _close(kv_1m["decoder-only MHA (40L)"].bytes / 1e9, 369, rel=0.01),
            gb(kv_1m["decoder-only MHA (40L)"].bytes),
            "369 GB",
        ),
        Claim(
            "1M decoder-only GQA-4 KV ≈41 GB",
            _close(kv_1m["decoder-only GQA-4 (40L)"].bytes / 1e9, 41, rel=0.02),
            gb(kv_1m["decoder-only GQA-4 (40L)"].bytes),
            "41 GB",
        ),
        Claim(
            "1M decoder-only MLA-576 KV ≈46 GB",
            _close(kv_1m["decoder-only MLA-576 (40L)"].bytes / 1e9, 46, rel=0.02),
            gb(kv_1m["decoder-only MLA-576 (40L)"].bytes),
            "46 GB",
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
            "24×8K window (MLA) ≈0.2 GB",
            _close(kv_1m["decoder 24×8K window (MLA-576)"].bytes / 1e9, 0.2, rel=0.15),
            gb(kv_1m["decoder 24×8K window (MLA-576)"].bytes),
            "~0.2 GB",
        ),
        Claim(
            "μP logits scale d/dim_model_base = 9",
            abs(D / DIM_MODEL_BASE - 9.0) < 1e-12,
            f"{D / DIM_MODEL_BASE:.0f}",
            "9",
        ),
        Claim(
            "keep residual 1.4/√40 after stack split",
            True,
            f"{SCALE_DEPTH / math.sqrt(BASE_LAYERS):.4f}",
            "do not rescale by 16/24",
            note="inherited MiniCPM weights were trained with 1.4/√40",
        ),
    ]


def verify(budget: ModelBudget | None = None) -> list[Claim]:
    budget = budget or compute_budget(TIERS[DEFAULT_TIER])
    if budget.tier.key != "middle" or budget.first_dense or budget.attn.name != "placeholder 1.25×MHA":
        raise ValueError("--verify is defined on the published middle-tier placeholder budget")
    return claims_middle_placeholder(budget)


def print_budget(budget: ModelBudget) -> None:
    t = budget.tier
    print(f"embedding (tied)         : {b(budget.emb)}")
    print(f"expert (single)          : {m(EXPERT)}")
    print(f"dense FFN (MiniCPM)      : {m(DENSE_FFN)}")
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
        f"{'n':>10s} {'mlp_fwd':>12s} {'dense40':>12s} {'enc_hyb16':>12s} "
        f"{'enc+dec_win':>12s} {'xattn24':>12s}"
    )
    for n in lengths:
        mlp_n = mlp * n
        dense = attn_score_flops(n, n, 40)
        enc_hyb = 0.0
        for i in range(16):
            k = keys_csa(n) if i % 2 == 0 else keys_hca(n)
            enc_hyb += attn_score_flops(n, k, 1)
        dec_win = attn_score_flops(n, min(8192, n), 24)
        xattn = attn_score_flops(n, n, 24)
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
    ph = attn_accounting("placeholder")
    print("-- Self-attn params per layer --")
    print(f"  MiniCPM MHA 4d²              : {m(MHA)}")
    print(f"  placeholder 1.25×MHA         : {m(ph.self_attn)}")
    print(f"  CSA MQA-64 (detailed)        : {m(_csa_mqa64_self_attn(True))}")
    print(f"  HCA MQA-64 (detailed)        : {m(_csa_mqa64_self_attn(False))}")
    print(f"  CSA/HCA average              : {m(attn_accounting('csa_mqa64').self_attn)}")
    print("  (8K window dominates long-context score FLOPs; index_topk 256 vs 512 is a ~3% delta.)")


def retune_routed(
    *,
    first_dense: bool,
    attn: AttnAccounting,
    target: float = 12.05e9,
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
    print("-- Nr retune to 12.05B (top-k unchanged ⇒ activations almost unchanged) --")
    attn_ph = attn_accounting("placeholder")
    attn_csa = attn_accounting("csa_mqa64")
    for label, first_dense, attn in (
        ("first-dense + placeholder attn", True, attn_ph),
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
    print("-- MiniCPM μP scales (must keep after the 16/24 split) --")
    print(f"  scale_emb                 : {SCALE_EMB}")
    print(f"  logits / (d/dim_model_base): {D / DIM_MODEL_BASE:.0f}")
    print(f"  residual scale_depth/√40  : {SCALE_DEPTH / math.sqrt(BASE_LAYERS):.4f}  (keep; do not use √16 or √24)")
    print(f"  residual if rescaled √16  : {SCALE_DEPTH / math.sqrt(16):.4f}  (would drift inherited weights)")
    print(f"  residual if rescaled √24  : {SCALE_DEPTH / math.sqrt(24):.4f}")


def print_claims(cs: Iterable[Claim]) -> int:
    cs = list(cs)
    print("-- Claim ledger (published middle tier, placeholder attn, all-MoE) --")
    failed = 0
    for c in cs:
        mark = "PASS" if c.ok else "FAIL"
        if not c.ok:
            failed += 1
        extra = f"  # {c.note}" if c.note else ""
        print(f"  [{mark}] {c.name:42s}  observed={c.observed:16s}  expected={c.expected}{extra}")
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
        choices=["placeholder", "csa_mqa64"],
        default="placeholder",
        help="self-attn parameter accounting",
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
        help="assert published middle-tier claims; exit 1 on failure",
    )
    p.add_argument("--full", action="store_true", help="print KV / attention-complexity / μP sections")
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
        if args.tier in ("middle", "all") and args.attn == "placeholder" and not args.first_dense:
            print_claims(verify())
    return 0


if __name__ == "__main__":
    sys.exit(main())
