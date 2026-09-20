"""Dedicated-dir mini-train proof of the published attention / YOCO / PDSA plan.

Static claims plus Phase **A→E** on a tiny DummyStream graph (A is 0-token
surgery; B0–E are 1–N trainer steps). Writes ``--out/ledger.json``.
Does not download Ultra-FineWeb. Not a CSA CUDA kernel. F/G are registered
but not this mini-train. PDSA Tier 1/3 stay deferred (FROZEN_SPEC: PDSA off).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch import nn

from cat_yoko.attention import CrossAttention, WindowAttention, reset_sdpa_counts
from cat_yoko.blocks import DecoderBlock
from cat_yoko.config import CATYokoConfig, KEEP_HIGH_PREC, encoder_layer_kind
from cat_yoko.data import DummyStream
from cat_yoko.freeze import gate_schedule, set_gate, trainable_names
from cat_yoko.indexer import LightningIndexer, indexer_compressed_keep
from cat_yoko.moe import grouped_mm_available
from cat_yoko.ops import fused_qkv_smoke, probe_sdpa_backends, snapshot_ops
from cat_yoko.optim import unwrap, wsd_lr
from cat_yoko.phases import C_CHAIN, C_CHAIN_KDA, PHASES
from cat_yoko.sparse import (
    compressed_keep_matrix,
    hca_attend,
    hca_slot_keep,
    window_keep_matrix,
)
from cat_yoko.trainer import Trainer, build_model, configure_cuda

# Published pretrain envelope: A (0-token surgery) through E (WSD). F/G are post-train.
PHASES_RUN = (
    "A",
    "B0",
    "B1",
    "B2",
    "C-index",
    "C-topk",
    "C-hca",
    "C-win",
    "D-8k",
    "E",
)

GRAPHS = ("plan", "bf16")


def graph_config(name: str) -> CATYokoConfig:
    """``plan`` is the CPU/CI tiny graph; ``bf16`` is the Ampere Flash-shaped probe."""
    key = str(name).strip().lower().replace("_", "-")
    if key in {"plan", "plan-probe"}:
        return CATYokoConfig.plan_probe()
    if key in {"bf16", "bf16-probe"}:
        return CATYokoConfig.bf16_probe()
    raise ValueError(f"unknown graph {name!r}; expected plan|bf16")


@dataclass
class Claim:
    name: str
    group: str
    ok: bool
    observed: str
    expected: str
    note: str = ""
    deferred: bool = False


def _claim(name: str, group: str, ok: bool, observed, expected, note: str = "", *, deferred: bool = False) -> Claim:
    return Claim(
        name=name,
        group=group,
        ok=bool(ok) or deferred,
        observed=str(observed),
        expected=str(expected),
        note=note,
        deferred=deferred,
    )


def _finite_pos(x: float) -> bool:
    return bool(math.isfinite(x)) and x > 0


def _atol(device: str) -> float:
    return 2e-2 if str(device).startswith("cuda") else 2e-3


def _arch_claims() -> list[Claim]:
    root = Path(__file__).resolve().parents[1]
    scripts = str(root / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import arch_verify as av

    out = []
    for c in av.verify():
        out.append(
            _claim(
                f"arch.{c.name}",
                "arch",
                bool(c.ok),
                c.observed,
                c.expected,
                c.note,
            )
        )
    return out


def _source_has_no_csa_kernel() -> Claim:
    import cat_yoko.attention as attn
    import cat_yoko.indexer as idx
    import cat_yoko.sparse as sparse

    blob = inspect.getsource(attn) + inspect.getsource(sparse) + inspect.getsource(idx)
    ok = "class CSA" not in blob and "LightningIndexer" in blob
    return _claim(
        "attention.no_csa_cuda_kernel",
        "attention",
        ok,
        "no class CSA; LightningIndexer present",
        "PyTorch SDPA / union mask, not a CSA kernel",
    )


def _theorem_b_masks(cfg: CATYokoConfig) -> list[Claim]:
    s, w, m, mh = cfg.seq_len, cfg.n_win, cfg.compress_m, cfg.compress_m_hca
    device = torch.device("cpu")
    win = window_keep_matrix(s, w, device)
    comp = compressed_keep_matrix(s, m, device)
    union = win | comp
    future = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    no_future = not bool((union & future).any().item())
    t, p = s - 1, 0
    beyond = (not bool(win[t, p].item())) and bool(comp[t, p].item()) and bool(union[t, p].item())
    hidden = torch.randn(1, s, cfg.hidden_size)
    indexer = LightningIndexer(cfg)
    sel = indexer_compressed_keep(indexer, hidden, m, cfg.index_topk)
    topk_subset = not bool((sel & ~comp.unsqueeze(0)).any())
    deleted = bool((comp.unsqueeze(0) & ~sel).any())
    n_slots = max(s // mh, 1)
    hca_slots = hca_slot_keep(s, mh, n_slots, device)
    own = (torch.arange(s) // mh).clamp(max=n_slots - 1)
    hca_no_own = not bool(hca_slots[torch.arange(s), own].any())
    hca_src = inspect.getsource(hca_attend).replace(" ", "")
    hca_fallback = "torch.cat([k,k_c],dim=2)" in hca_src
    return [
        _claim("theorem_b.union_causal", "attention", no_future, f"future_hits=0 seq={s}", "W∪S_comp never sees p>t"),
        _claim(
            "theorem_b.window_covers_own_block",
            "attention",
            w >= m and w >= mh,
            f"n_win={w} m={m} m'={mh}",
            "n_win>=m so own-block hole is in the window",
        ),
        _claim(
            "theorem_b.compression_reaches_beyond_window",
            "pdsa",
            beyond,
            f"t={t} p={p} win={bool(win[t, p].item())} comp={bool(comp[t, p].item())}",
            "probe: a past token outside the window stays in S_comp",
        ),
        _claim(
            "theorem_b.indexer_deletes_only",
            "pdsa",
            topk_subset,
            "selected ⊆ S_comp",
            "PDSA/DSA: top-k may delete compressed keys, never add",
        ),
        _claim(
            "theorem_b.indexer_actually_deletes",
            "pdsa",
            deleted,
            f"index_topk={cfg.index_topk} < n_blocks={max(s // m, 1)}",
            "topk is a real subset of S_comp on the probe graph",
        ),
        _claim(
            "theorem_b.hca_excludes_own_slot",
            "pdsa",
            hca_no_own,
            "own compressed slot masked",
            "write-first HCA own-block out; window is the fallback",
        ),
        _claim(
            "pdsa.hca_concat_is_query_time_fallback",
            "pdsa",
            hca_fallback,
            "hca_attend concatenates window KV with pooled slots",
            "PDSA §5: write-first compression keeps a path to raw KV",
        ),
        _claim(
            "pdsa.window_lt_seq_so_compression_matters",
            "pdsa",
            w < s,
            f"n_win={w} seq={s}",
            "probe graph must not hide compression behind a full-seq window",
        ),
    ]


def _live_attention_claims(cfg: CATYokoConfig) -> list[Claim]:
    """Execute window / CSA / HCA / YOCO cross-attn on the probe graph."""
    torch.manual_seed(0)
    s = cfg.seq_len
    attn = WindowAttention(cfg)
    x = torch.randn(1, s, cfg.hidden_size)
    x2 = x.clone()
    x2[:, -1] = torch.randn_like(x2[:, -1])
    y1 = attn(x)
    y2 = attn(x2)
    win_ok = bool(torch.isfinite(y1).all()) and bool(
        torch.allclose(y1[:, :-1], y2[:, :-1], atol=2e-3, rtol=2e-3)
    )
    indexer = LightningIndexer(cfg)
    keep = indexer_compressed_keep(indexer, x, cfg.compress_m, cfg.index_topk)
    from cat_yoko.sparse import csa_attend

    y_csa = csa_attend(attn, x, None, keep[0] if keep.dim() == 3 else keep)
    y_csa2 = csa_attend(attn, x2, None, indexer_compressed_keep(indexer, x2, cfg.compress_m, cfg.index_topk)[0])
    csa_ok = bool(torch.isfinite(y_csa).all()) and bool(
        torch.allclose(y_csa[:, :-1], y_csa2[:, :-1], atol=2e-3, rtol=2e-3)
    )

    captured: list[int] = []
    import cat_yoko.sparse as sparse_mod

    orig = sparse_mod._sdpa

    def _spy(q, k, v, *args, **kwargs):
        captured.append(int(k.shape[-2]))
        return orig(q, k, v, *args, **kwargs)

    sparse_mod._sdpa = _spy  # type: ignore[method-assign]
    try:
        y_hca = hca_attend(attn, x, None, cfg.compress_m_hca)
        y_hca2 = hca_attend(attn, x2, None, cfg.compress_m_hca)
    finally:
        sparse_mod._sdpa = orig  # type: ignore[method-assign]
    n_slots = max(s // cfg.compress_m_hca, 1)
    hca_cat = bool(captured) and captured[0] == s + n_slots
    hca_ok = bool(torch.isfinite(y_hca).all()) and bool(
        torch.allclose(y_hca[:, :-1], y_hca2[:, :-1], atol=2e-3, rtol=2e-3)
    )

    cross = CrossAttention(cfg)
    names = [n for n, _ in cross.named_parameters()]
    no_kv = not any("k_proj" in n or "v_proj" in n for n in names)
    has_qo = any("q_proj" in n for n in names) and any("o_proj" in n for n in names)
    k = torch.randn(1, s, cfg.kv_dim)
    v = torch.randn(1, s, cfg.kv_dim)
    k2 = k.clone()
    v2 = v.clone()
    k2[:, -1] = torch.randn_like(k2[:, -1])
    v2[:, -1] = torch.randn_like(v2[:, -1])
    yc = cross(x, k, v)
    yc2 = cross(x, k2, v2)
    cross_ok = bool(torch.isfinite(yc).all()) and bool(
        torch.allclose(yc[:, :-1], yc2[:, :-1], atol=2e-3, rtol=2e-3)
    )
    thm_a = "self.gate*c" in inspect.getsource(DecoderBlock.forward).replace(" ", "")
    return [
        _claim(
            "attention.window_gqa_causal",
            "attention",
            win_ok,
            "WindowAttention prefix ignores last token",
            "sliding GQA is causal",
        ),
        _claim(
            "attention.csa_union_runs_causal",
            "attention",
            csa_ok,
            "csa_attend finite + prefix-stable",
            "union mask, not a CSA kernel",
        ),
        _claim(
            "pdsa.hca_runtime_concat_slots",
            "pdsa",
            hca_cat,
            f"sdpa_k={captured[:1]} want {s}+{n_slots}",
            "HCA SDPA key length = window tokens + pooled slots",
        ),
        _claim(
            "attention.hca_concat_runs_causal",
            "pdsa",
            hca_ok,
            "hca_attend finite + prefix-stable",
            "own-block out + window fallback",
        ),
        _claim(
            "yoco.cross_attn_q_o_only",
            "yoco",
            no_kv and has_qo,
            names,
            "CrossAttention has W_Q/W_O, no W_K/W_V",
        ),
        _claim(
            "yoco.cross_attn_cache_causal",
            "yoco",
            cross_ok,
            "decoder query t ignores cache p>t",
            "YOCO global cache is prefix-causal",
        ),
        _claim(
            "yoco.theorem_a_gate_multiplies_cross",
            "yoco",
            thm_a,
            "DecoderBlock: x += gate * cross",
            "gate=0 ≡ MiniCPM5 residual cut",
        ),
    ]


def _static_plan_claims(cfg: CATYokoConfig) -> list[Claim]:
    c12 = CATYokoConfig.middle_12b()
    kinds = [encoder_layer_kind(i, cfg.encoder_layers, use_kda=cfg.use_kda) for i in range(cfg.encoder_layers)]
    kinds12 = [encoder_layer_kind(i, 16) for i in range(16)]
    chain = C_CHAIN
    hca_last_sparse = chain.index("C-hca") > chain.index("C-topk") > chain.index("C-index")
    kda_first_if_on = C_CHAIN_KDA[0] == "C-kda"
    keep = set(KEEP_HIGH_PREC)
    early = wsd_lr(0, cfg, "E", tokens_in_phase=0.0, phase_budget=1000)
    late = wsd_lr(0, cfg, "E", tokens_in_phase=1000.0, phase_budget=1000)
    return [
        _claim(
            "yoco.split_16_26_published",
            "yoco",
            (c12.encoder_layers, c12.decoder_layers) == (16, 26),
            f"{c12.encoder_layers}+{c12.decoder_layers}",
            "16 encoder + 26 decoder",
        ),
        _claim(
            "yoco.published_2_7_7",
            "yoco",
            (kinds12.count("sliding"), kinds12.count("csa"), kinds12.count("hca")) == (2, 7, 7),
            f"sliding={kinds12.count('sliding')} csa={kinds12.count('csa')} hca={kinds12.count('hca')}",
            "2+7+7",
        ),
        _claim(
            "yoco.published_n_win_covers_hca",
            "attention",
            c12.n_win >= c12.compress_m_hca and c12.n_win >= c12.compress_m,
            f"n_win={c12.n_win} m={c12.compress_m} m'={c12.compress_m_hca}",
            "8192 ≥ 128 and 8192 ≥ 4",
        ),
        _claim(
            "yoco.probe_has_sliding_csa_hca",
            "yoco",
            kinds == ["sliding", "csa", "hca"],
            kinds,
            ["sliding", "csa", "hca"],
        ),
        _claim("yoco.m2_off_flag", "pdsa", cfg.compress_m >= 1 and not cfg.use_kda, f"use_kda={cfg.use_kda}", "M2/KDA default off"),
        _claim(
            "pdsa.query_aware_before_write_first",
            "pdsa",
            hca_last_sparse,
            "→".join(chain),
            "CSA indexer/topk before HCA (PDSA §14.2 Tier 2)",
        ),
        _claim(
            "plan.kda_lights_first_when_implemented",
            "plan",
            kda_first_if_on,
            "→".join(C_CHAIN_KDA),
            "hybrid-linear: do not train CSA/HCA then swap in KDA",
        ),
        _claim("plan.attn_softmax_high_prec", "attention", "attn_softmax" in keep, ",".join(KEEP_HIGH_PREC), "softmax stays fp32"),
        _claim(
            "plan.e_wsd_decay",
            "plan",
            early > late and abs(late / cfg.lr - cfg.wsd_decay_min_ratio) < 1e-6,
            f"{early:.2e}→{late:.2e}",
            "Phase E WSD to ~1/100 peak",
        ),
        _claim(
            "plan.mini_train_is_a_through_e",
            "plan",
            PHASES_RUN[0] == "A" and PHASES_RUN[-1] == "E" and "F" not in PHASES_RUN,
            "→".join(PHASES_RUN),
            "Phase A surgery through Phase E WSD; F/G are post-train",
        ),
        _claim(
            "plan.fg_registered_after_e",
            "plan",
            PHASES["G"].loss == "grpo" and PHASES["G-dpo"].loss == "dpo" and PHASES["F"].loss == "sft",
            "G=grpo G-dpo=dpo F=sft",
            "F/G exist; this mini-train stops at E",
        ),
        _claim(
            "plan.d8k_published_keeps_hca",
            "plan",
            PHASES["D-8k"].sparse == "hca" and PHASES["C-win"].seq_len == 8192,
            f"D-8k sparse={PHASES['D-8k'].sparse} C-win seq={PHASES['C-win'].seq_len}",
            "long-context keeps C lighting; published C-win is 8K",
        ),
        _claim(
            "pdsa.tier1_calibrated_fallback",
            "pdsa",
            True,
            "not in published graph",
            "FROZEN_SPEC PDSA off; inference-side, not this mini-train",
            "Tier 1 confidence-gated expand/fallback is planned, not C1 B0",
            deferred=True,
        ),
        _claim(
            "pdsa.tier3_editable_memory",
            "pdsa",
            True,
            "YOCO cache is append-only",
            "bounded lifecycle is a later milestone",
            deferred=True,
        ),
        _claim("plan.c1_token_split", "plan", PHASES["B0"].tokens == 8e9 and PHASES["B1"].tokens == 27e9, "8+27+15e9", "C1"),
        _claim(
            "plan.b0_gate_0_to_0.3",
            "yoco",
            gate_schedule("B0", 0.0) == 0.0 and abs(gate_schedule("B0", 1.0) - 0.3) < 1e-9,
            f"{gate_schedule('B0', 0)}→{gate_schedule('B0', 1)}",
            "0→0.3",
        ),
        _claim("plan.b1_gate_to_1", "yoco", abs(gate_schedule("B1", 1.0) - 1.0) < 1e-9, gate_schedule("B1", 1.0), "1.0"),
    ]


def _no_future_leak(model: nn.Module, cfg: CATYokoConfig, device: str) -> bool:
    raw = unwrap(model)
    raw.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
    fut = ids.clone()
    fut[:, -1] = (fut[:, -1] + 1) % cfg.vocab_size
    atol = _atol(device)
    with torch.no_grad():
        a = raw(input_ids=ids)["logits"]
        b = raw(input_ids=fut)["logits"]
    ok = bool(torch.allclose(a[:, :-1], b[:, :-1], atol=atol, rtol=atol))
    raw.train()
    return ok


def _cache_slots(model: nn.Module, cfg: CATYokoConfig, device: str) -> int:
    """YOCO cache sequence dim as seen by decoder CrossAttention (M2 off ⇒ == seq)."""
    raw = unwrap(model)
    raw.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
    captured: list[int] = []

    def _hook(_m, args, _out):
        # CrossAttention.forward(x, k, v, ...)
        if len(args) >= 2 and torch.is_tensor(args[1]) and args[1].dim() >= 2:
            captured.append(int(args[1].shape[1]))

    h = unwrap(raw.decoder[0]).cross_attn.register_forward_hook(_hook)
    with torch.no_grad():
        raw(input_ids=ids)
    h.remove()
    raw.train()
    return captured[0] if captured else -1


def _probe_phase(model: nn.Module, cfg: CATYokoConfig, phase: str, nll: float, device: str) -> list[Claim]:
    raw = unwrap(model)
    names = trainable_names(raw)
    g = float(raw.decoder[0].gate)
    slots = _cache_slots(raw, cfg, device)
    leak_ok = _no_future_leak(raw, cfg, device)
    dec_has_kv = any(hasattr(unwrap(b).cross_attn, "k_proj") for b in raw.decoder)
    idx_enc = [getattr(b, "kind", None) for b in raw.encoder if getattr(b, "indexer", None) is not None]
    idx_dec = any(getattr(b, "indexer", None) is not None for b in raw.decoder)
    modes = [getattr(b, "sparse_mode", "window") for b in raw.encoder]
    kinds = [getattr(b, "kind", "?") for b in raw.encoder]
    claims = [
        _claim(f"train.{phase}.finite_nll", "plan", _finite_pos(nll), nll, ">0 finite"),
        _claim(f"train.{phase}.no_future_leak", "attention", leak_ok, leak_ok, "logits[:, :-1] ignore last token"),
        _claim(
            f"train.{phase}.m2_off_cache_slots",
            "yoco",
            slots == cfg.seq_len,
            f"cache_k seq={slots}",
            f"M2 off ⇒ slots == seq ({cfg.seq_len})",
        ),
        _claim(
            f"train.{phase}.yoco_cache_once",
            "yoco",
            (not dec_has_kv) and hasattr(raw, "cache_k") and hasattr(raw, "cache_v"),
            f"decoder_cross_k_proj={dec_has_kv}",
            "one encoder-top cache; decoder CrossAttention has no W_K/W_V",
        ),
        _claim(
            f"train.{phase}.m1_indexer_not_on_decoder",
            "pdsa",
            not idx_dec,
            f"encoder_indexer_kinds={idx_enc} decoder={idx_dec}",
            "M3 is decoder query; do not reuse M1 LightningIndexer",
        ),
    ]
    if phase == "B0":
        enc_frozen = all(not p.requires_grad for p in raw.encoder.parameters())
        new_ok = any("cross_attn" in n for n in names) and any("cache_k" in n for n in names)
        no_enc = not any(n.startswith("encoder") for n in names)
        claims += [
            _claim("train.B0.encoder_frozen", "plan", enc_frozen and no_enc, no_enc, "C1 B0 new modules only"),
            _claim("train.B0.cross_and_cache_trainable", "yoco", new_ok, names[:8], "cache + cross-attn"),
            _claim("train.B0.gate_in_unit_interval", "yoco", 0.0 <= g <= 0.31, g, "gate ≤ 0.3"),
        ]
    if phase == "B1":
        enc_frozen = all(not p.requires_grad for p in raw.encoder.parameters())
        dec_ok = next(raw.decoder[0].self_attn.parameters()).requires_grad
        claims += [
            _claim("train.B1.encoder_still_frozen", "plan", enc_frozen, enc_frozen, "Theorem E"),
            _claim("train.B1.decoder_trainable", "plan", dec_ok, dec_ok, "decoder + lm_head"),
            _claim("train.B1.gate_opened", "yoco", g >= 0.99, g, "→1"),
        ]
    if phase == "B2":
        claims.append(_claim("train.B2.detach_off", "plan", not bool(raw.detach_cache), raw.detach_cache, False))
        claims.append(
            _claim(
                "train.B2.encoder_unfrozen",
                "plan",
                next(raw.encoder.parameters()).requires_grad,
                True,
                "joint write/read (PDSA: B2 cannot be 0)",
            )
        )
    if phase == "C-index":
        csa_idx = all(getattr(b, "indexer", None) is not None for b in raw.encoder if getattr(b, "kind", None) == "csa")
        non_csa = all(getattr(b, "indexer", None) is None for b in raw.encoder if getattr(b, "kind", None) != "csa")
        claims += [
            _claim("train.C-index.indexer_on_csa_only", "attention", csa_idx and non_csa, idx_enc, ["csa"]),
            _claim(
                "train.C-index.only_indexer_trainable",
                "plan",
                bool(names) and all("indexer" in n for n in names),
                names,
                "freeze backbone",
            ),
        ]
        raw.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
        with torch.no_grad():
            out = raw(input_ids=ids)
        kl = out.get("indexer_kl")
        rec = out.get("indexer_recall")
        raw.train()
        claims.append(
            _claim(
                "train.C-index.kl_finite",
                "attention",
                kl is not None and math.isfinite(float(kl)),
                None if kl is None else float(kl),
                "layer-internal KL vs dense window",
            )
        )
        if rec is not None:
            claims.append(
                _claim("train.C-index.recall_in_unit", "attention", 0.0 <= float(rec) <= 1.0, float(rec), "[0,1]")
            )
    if phase == "C-topk":
        csa_topk = all(
            getattr(b, "sparse_mode", None) == "topk" for b in raw.encoder if getattr(b, "kind", None) == "csa"
        )
        hca_still_win = all(
            getattr(b, "sparse_mode", None) == "window" for b in raw.encoder if getattr(b, "kind", None) == "hca"
        )
        claims += [
            _claim("train.C-topk.csa_lights_topk", "attention", csa_topk, list(zip(kinds, modes)), "CSA → union mask"),
            _claim("train.C-topk.hca_not_yet", "pdsa", hca_still_win, list(zip(kinds, modes)), "write-first HCA stays dark"),
        ]
    if phase in {"C-hca", "C-win", "D-8k", "E", "F"}:
        hca_on = all(
            getattr(b, "sparse_mode", None) == "hca" for b in raw.encoder if getattr(b, "kind", None) == "hca"
        )
        csa_stays = all(
            getattr(b, "sparse_mode", None) == "topk" for b in raw.encoder if getattr(b, "kind", None) == "csa"
        )
        claims += [
            _claim(f"train.{phase}.hca_concat_lit", "pdsa", hca_on, list(zip(kinds, modes)), "window ∥ mean-pool slots"),
            _claim(f"train.{phase}.csa_keeps_topk", "attention", csa_stays, list(zip(kinds, modes)), "monotonic lighting"),
        ]
    if phase.startswith("D"):
        stream = DummyStream(cfg.vocab_size, cfg.seq_len, seed=0, needle=True)
        batch = stream.batch(1, "cpu")
        mid = cfg.seq_len // 2
        planted = int(batch["input_ids"][0, mid].item()) == cfg.vocab_size - 1
        claims.append(
            _claim(
                "train.D.needle_in_dummy",
                "plan",
                planted,
                planted,
                "mid-seq needle; later queries see it via YOCO cache (causal prefix)",
            )
        )
    if phase == "F":
        stream = DummyStream(cfg.vocab_size, cfg.seq_len, seed=0, response_only=True)
        batch = stream.batch(1, "cpu")
        cut = max(int(cfg.seq_len * 0.5), 1)
        masked = bool((batch["labels"][0, :cut] == -100).all())
        claims.append(
            _claim(
                "train.F.sft_masks_prompt",
                "plan",
                masked,
                f"labels[:{cut}]=-100",
                "SFT DummyStream response-only (not in A–E mini-train)",
            )
        )
    return claims


def _run_phase_a(model: nn.Module, cfg: CATYokoConfig, device: str) -> tuple[list[Claim], float]:
    """Offline Phase A: dummy MiniCPM5 upcycle, both stacks MoE, gate=0, window compute."""
    from cat_yoko.moe import MoE
    from cat_yoko.upcycle import dummy_minicpm_state, init_new_modules, upcycle_from_minicpm

    raw = unwrap(model)
    src = dummy_minicpm_state(cfg)
    upcycle_from_minicpm(raw, src, cfg)
    init_new_modules(raw)
    set_gate(raw, 0.0)
    enc_moe = all(isinstance(unwrap(b).mlp, MoE) for b in raw.encoder)
    dec_moe = all(isinstance(unwrap(b).mlp, MoE) for b in raw.decoder)
    hash0 = bool(getattr(unwrap(raw.decoder[0]).mlp, "hash_route", False))
    modes = [getattr(b, "sparse_mode", "window") for b in raw.encoder]
    window_only = all(m == "window" for m in modes)
    g = float(raw.decoder[0].gate)
    raw.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
    atol = _atol(device)
    with torch.no_grad():
        a = raw(input_ids=ids)["logits"]
        delta = torch.randn_like(raw.cache_k.weight)
        raw.cache_k.weight.add_(delta)
        b = raw(input_ids=ids)["logits"]
        raw.cache_k.weight.sub_(delta)
        nll = float(raw(input_ids=ids, labels=ids)["nll"])
    thm_a = bool(torch.allclose(a, b, atol=atol, rtol=atol))
    raw.train()
    claims = [
        _claim("train.A.zero_tokens", "plan", True, "offline dummy upcycle", "Phase A is 0-token surgery"),
        _claim(
            "train.A.both_stacks_moe",
            "plan",
            enc_moe and dec_moe,
            f"enc={enc_moe} dec={dec_moe}",
            "C1: both stacks MoE at token 0",
        ),
        _claim("train.A.no_first_dense", "plan", not cfg.first_dense, cfg.first_dense, False),
        _claim("train.A.gate_zero", "yoco", abs(g) < 1e-6, g, 0.0),
        _claim(
            "train.A.sparse_still_window",
            "attention",
            window_only,
            modes,
            "implement then light; A/B stay window",
        ),
        _claim("train.A.hash_moe_decoder_bootstrap", "plan", hash0, hash0, "decoder first layers hash-route"),
        _claim(
            "train.A.theorem_a_cache_irrelevant",
            "yoco",
            thm_a,
            thm_a,
            "gate=0 ⇒ cache W_K does not change logits",
        ),
        _claim(
            "train.A.no_mup",
            "plan",
            (not cfg.use_mup) and abs(cfg.residual_scale - 1.0) < 1e-9,
            cfg.residual_scale,
            1.0,
        ),
    ]
    claims.extend(_probe_phase(raw, cfg, "A", nll, device))
    return claims, nll


def _ops_source_claims() -> list[Claim]:
    import cat_yoko.attention as attn

    dense = inspect.getsource(attn._cuda_sdpa_kernel)
    masked = inspect.getsource(attn._cuda_masked_sdpa_kernel)
    sdpa = inspect.getsource(attn._sdpa)
    return [
        _claim(
            "ops.dense_sdpa_prefers_flash",
            "attention",
            "FLASH_ATTENTION" in dense and "CUDNN_ATTENTION" in dense,
            "Flash → cuDNN → efficient",
            "dense causal fused kernels",
        ),
        _claim(
            "ops.masked_sdpa_skips_flash",
            "attention",
            "FLASH_ATTENTION" not in masked and "CUDNN_ATTENTION" in masked and "EFFICIENT_ATTENTION" in masked,
            "cuDNN + efficient, no Flash",
            "Flash rejects attn_mask",
        ),
        _claim(
            "ops.masked_tries_bf16_before_fp32",
            "attention",
            "masked_bf16" in sdpa and ".float()" in sdpa,
            "bf16 mask then fp32 math",
            "Ampere CSA/HCA stay bf16 when the fused kernel runs",
        ),
        _claim(
            "ops.masked_isolates_one_backend",
            "attention",
            "MASKED_SDPA_SWITCH_SEQ" in inspect.getsource(attn) and "_masked_backend_order" in sdpa,
            "seq<320 Efficient; else cuDNN",
            "Ampere dispatcher picks the slower kernel if both are enabled",
        ),
        _claim(
            "ops.dense_kernel_list_not_tuple",
            "attention",
            "tuple(" not in dense,
            "sdpa_kernel(list)",
            "PyTorch sdpa_kernel wants a list",
        ),
        _claim(
            "ops.grouped_mm_sm90_gate",
            "attention",
            "major >= 9" in inspect.getsource(grouped_mm_available),
            "SM90+",
            "Ampere skips grouped_mm RuntimeError tax and uses padded bmm",
        ),
        _claim(
            "ops.banded_sliding_window",
            "attention",
            "_banded_window_sdpa" in inspect.getsource(attn)
            and "_window_sdpa" in inspect.getsource(attn.WindowAttention.forward),
            "covering Flash; else fat tiles 256 (seq>=512); CSA union still S×S",
            "do not materialize S×S for a static sliding window",
        ),
    ]


def _ops_graph_claims(cfg: CATYokoConfig) -> list[Claim]:
    claims = [
        _claim(
            "ops.compute_is_bf16",
            "plan",
            (not cfg.use_nvfp4) and (not cfg.use_fp8),
            f"nvfp4={cfg.use_nvfp4} fp8={cfg.use_fp8}",
            "mini-verify is BF16; NVFP4/FP8 stay off",
        ),
        _claim("ops.kda_off", "plan", not cfg.use_kda, cfg.use_kda, False),
        _claim(
            "ops.grouped_mm_recorded",
            "attention",
            True,
            grouped_mm_available(),
            "SM90+ grouped_mm; Ampere records False and uses moe_bmm",
        ),
    ]
    if cfg.name == "bf16-probe":
        claims += [
            _claim(
                "ops.flash_shaped",
                "attention",
                cfg.head_dim >= 32 and cfg.seq_len >= 64,
                f"hd={cfg.head_dim} seq={cfg.seq_len}",
                "hd>=32 seq>=64 so Ampere fused SDPA is in-distribution",
            ),
            _claim(
                "ops.theorem_b_hole_kept",
                "pdsa",
                cfg.n_win < cfg.seq_len and cfg.n_win >= cfg.compress_m,
                f"n_win={cfg.n_win} seq={cfg.seq_len} m={cfg.compress_m}",
                "grow the graph without swallowing compression",
            ),
        ]
    return claims


def _ops_cuda_claims(cfg: CATYokoConfig, device: str, probe: dict) -> list[Claim]:
    dense = str(probe.get("dense_gqa", ""))
    masked = str(probe.get("masked_equal", ""))
    hca = str(probe.get("hca_concat", ""))
    fused = fused_qkv_smoke(device, torch.bfloat16, hidden=cfg.hidden_size, kv=cfg.kv_dim)
    tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    try:
        prec = torch.get_float32_matmul_precision()
    except Exception:
        prec = ""
    return [
        _claim(
            "ops.dense_gqa_fused",
            "attention",
            dense in {"flash", "cudnn", "efficient"},
            dense,
            "YOCO cross / covering window: Flash or cuDNN GQA",
        ),
        _claim(
            "ops.masked_equal_bf16",
            "attention",
            masked in {"cudnn", "efficient"},
            masked,
            "CSA/window mask: cuDNN or mem-efficient bf16",
        ),
        _claim(
            "ops.hca_concat_bf16",
            "attention",
            hca in {"cudnn", "efficient"},
            hca,
            "HCA concat mask stays fused bf16",
        ),
        _claim("ops.fused_qkv_smoke", "attention", fused, fused, "one GEMM for QKV"),
        _claim("ops.tf32", "attention", tf32, f"tf32={tf32} prec={prec}", "TF32 tensor cores"),
    ]


def static_ledger(graph: str = "plan") -> list[Claim]:
    """Architecture + mask + plan claims; no Trainer loop."""
    cfg = graph_config(graph)
    ledger: list[Claim] = []
    ledger.extend(_arch_claims())
    ledger.append(_source_has_no_csa_kernel())
    ledger.extend(_theorem_b_masks(cfg))
    ledger.extend(_live_attention_claims(cfg))
    ledger.extend(_static_plan_claims(cfg))
    ledger.extend(_ops_source_claims())
    ledger.extend(_ops_graph_claims(cfg))
    return ledger


def run(
    *,
    device: str,
    out: Path,
    steps: int = 2,
    graph: str = "plan",
) -> dict:
    cfg = graph_config(graph)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger: list[Claim] = static_ledger(graph)
    cuda = str(device).startswith("cuda")
    if cuda:
        configure_cuda()
    probe = None
    if cuda and torch.cuda.is_available():
        probe = probe_sdpa_backends(
            device,
            torch.bfloat16,
            seq=cfg.seq_len,
            n_heads=cfg.num_heads,
            n_kv=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        )
        ledger.extend(_ops_cuda_claims(cfg, device, probe))

    dtype = "bf16" if cuda else "fp32"
    model = build_model(cfg, device, dtype=dtype)
    t0 = time.perf_counter()
    phases = {}
    ops_phases: dict[str, dict] = {}
    for phase in PHASES_RUN:
        reset_sdpa_counts()
        if phase == "A":
            extra, nll = _run_phase_a(model, cfg, device)
            ledger.extend(extra)
            ops_phases[phase] = snapshot_ops(cfg, device, phase=phase)
            phases[phase] = {
                "nll": float(nll),
                "step": 0,
                "ok": _finite_pos(float(nll)),
                "peak_mib": 0.0,
                "tokens": 0,
                "sdpa": ops_phases[phase].get("sdpa"),
                "sdpa_counts": ops_phases[phase].get("sdpa_counts"),
            }
            continue
        tr = Trainer(
            cfg,
            phase,
            device,
            steps=max(int(steps), 1),
            dtype=dtype,
            seq_len=cfg.seq_len,
            micro_batch=2,
            accum=1,
            save_dir=out / phase,
            save_every=0,
            save_full=False,
            save_trainable=True,
            save_optim=False,
            reuse_model=model,
            log_path=out / phase / "metrics.jsonl",
        )
        row = tr.run()
        finite = _finite_pos(float(row.nll))
        ops_phases[phase] = snapshot_ops(cfg, device, phase=phase)
        phases[phase] = {
            "nll": float(row.nll),
            "step": int(row.step),
            "ok": finite,
            "peak_mib": float(row.peak_mib),
            "sdpa": ops_phases[phase].get("sdpa"),
            "sdpa_counts": ops_phases[phase].get("sdpa_counts"),
        }
        ledger.extend(_probe_phase(model, cfg, phase, float(row.nll), device))
    elapsed = time.perf_counter() - t0
    failed = [c for c in ledger if not c.ok]
    deferred = [c for c in ledger if c.deferred]
    summary = {
        "ok": not failed,
        "n_claims": len(ledger),
        "n_fail": len(failed),
        "n_deferred": len(deferred),
        "elapsed_s": round(elapsed, 3),
        "device": device,
        "graph": graph,
        "name": cfg.name,
        "dtype": dtype,
        "head_dim": cfg.head_dim,
        "seq_len": cfg.seq_len,
        "n_win": cfg.n_win,
        "encoder_layers": cfg.encoder_layers,
        "use_kda": cfg.use_kda,
        "use_nvfp4": cfg.use_nvfp4,
        "use_fp8": cfg.use_fp8,
        "ops": {"probe": probe, "phases": ops_phases},
        "phases": phases,
        "failed": [asdict(c) for c in failed],
        "deferred": [asdict(c) for c in deferred],
        "claims": [asdict(c) for c in ledger],
        "out": str(out),
    }
    (out / "ledger.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAT-YOKO plan mini-train verify (dedicated dir)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--graph", choices=GRAPHS, default="plan")
    p.add_argument("--json", action="store_true")
    p.add_argument("--static-only", action="store_true", help="masks + live attn, no Trainer loop")
    args = p.parse_args(argv)
    if args.static_only:
        ledger = static_ledger(args.graph)
        failed = [c for c in ledger if not c.ok]
        blob = {
            "ok": not failed,
            "n_claims": len(ledger),
            "n_fail": len(failed),
            "n_deferred": sum(1 for c in ledger if c.deferred),
            "graph": args.graph,
            "failed": [asdict(c) for c in failed],
            "deferred": [asdict(c) for c in ledger if c.deferred],
            "claims": [asdict(c) for c in ledger],
        }
        print(json.dumps({k: blob[k] for k in blob if k != "claims"}, indent=2) if args.json else
              f"plan-verify static graph={blob['graph']} ok={blob['ok']} claims={blob['n_claims']} fail={blob['n_fail']}")
        for c in blob["failed"]:
            print(f"FAIL {c['name']}: {c['observed']} (want {c['expected']})", flush=True)
        return 0 if blob["ok"] else 1
    blob = run(device=args.device, out=args.out, steps=max(int(args.steps), 1), graph=args.graph)
    if args.json:
        print(json.dumps({k: blob[k] for k in blob if k != "claims" and k != "ops"}, indent=2))
    else:
        print(
            f"plan-verify graph={blob['graph']} dtype={blob['dtype']} ok={blob['ok']} "
            f"claims={blob['n_claims']} fail={blob['n_fail']} "
            f"deferred={blob['n_deferred']} {blob['elapsed_s']}s",
            flush=True,
        )
        probe = (blob.get("ops") or {}).get("probe") or {}
        if probe:
            print(
                f"  ops dense_gqa={probe.get('dense_gqa')} masked={probe.get('masked_equal')} "
                f"hca={probe.get('hca_concat')}",
                flush=True,
            )
        for phase, st in blob["phases"].items():
            extra = ""
            sdpa = st.get("sdpa") or {}
            counts = st.get("sdpa_counts") or {}
            if sdpa or counts:
                extra = (
                    f" sdpa={sdpa.get('kind')} "
                    f"dense={counts.get('dense', 0)} "
                    f"masked_bf16={counts.get('masked_bf16', 0)} "
                    f"math={counts.get('math_fp32', 0)}"
                )
            print(f"  {phase} nll={st['nll']:.4f} ok={st['ok']}{extra}", flush=True)
        for c in blob["failed"]:
            print(f"FAIL {c['name']}: {c['observed']} (want {c['expected']})", flush=True)
        for c in blob["deferred"]:
            print(f"DEFER {c['name']}: {c['note']}", flush=True)
    return 0 if blob["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
