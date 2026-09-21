# CAT-YOKO Training Plan: Causal Encoder-Decoder (YOCO-style) Hybrid-Attention MoE Upcycled from MiniCPM5-2B

> Goal: using OpenBMB **MiniCPM5-2B (Apache-2.0, Llama GQA)** as the base, train a **Causal Encoder-Decoder (YOCO / "You Only Cache Once" decoder-decoder)** model:
> **≈12.25B total parameters** (already cut from 24B to fit the compute budget); asymmetric activation **default: Encoder processes the input at ≈2.03B active parameters/token, Decoder generates the output at ≈4.33B active parameters/token**
> (cheaper-compute band ≈1.43B/3.02B, near-dense band ≈3.04B/6.29B: see §3);
> attention is **DeepSeek-V4-Flash-style CSA + HCA compressed attention + an 8K large sliding window**; native long context (YOCO single global KV cache).
> The base was switched from MiniCPM-2B-sft-bf16 (GML) to MiniCPM5-2B for **Apache-2.0 license alignment**, not to change the curriculum. CAT-YOKO code and derived weights are likewise **Apache-2.0**.
>
> ⚠️ **Key point**: total parameters mainly affect **VRAM / storage**; **training FLOPs ∝ active parameters × tokens**. To actually cut training cost you must cut **activation** (pick a cheaper-compute band), not only total parameter count.
>
> This document is an executable engineering training plan. **Implementation defaults are frozen** in [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)
> / `cat_yoko.config.CATYokoConfig.middle_12b()`: causal 16/26, C1 full MoE, C1+NVFP4, Phase B sliding window + gated
> cross-attn, AdamW; **KDA / mHC / MTP / Muon / first-layer dense / M2 are not the release defaults**. Remaining bands and ablations in this document are sensitivity studies, not training-code switch defaults.
> **Current training progress** (Hub overlay step, which GPUs have been reclaimed) lives in [`docs/STATUS.md`](STATUS.md). Do not look for step pins in this document.

---

## 0. Requirements clarification and key assumptions

Your original brief had a few points that need an explicit reading. This plan proceeds under the interpretations below; say so if they miss your intent:

| Your wording | This plan's reading | Notes / adjustable |
| --- | --- | --- |
| `based on openbmb minicpm2b` | Base = **MiniCPM5-2B** (Apache-2.0, dense Llama GQA: 42 layers, hidden 2048, FFN 6144, 16 Q / 2 KV, `head_dim=128`, vocab 130560, **untied** embedding) | Native 128K context (`rope_theta=5e6`); no need to switch to MiniCPM-2B-128k |
| `same as deepseekv4.1 flash` | Match the **DeepSeek-V4-Flash** architectural paradigm (CSA/HCA hybrid attention + DeepSeekMoE + mHC + Muon + MTP + Hash-MoE bootstrap) | Official V4-Flash is 284B/13B decoder-only; we recast the same architecture, scaled down to 12B and rebuilt as a YOCO encoder-decoder (asymmetric activation) |
| `Causal-Encoder-Decoder, input activation 3b, output activation 6b`; later **`cut to 12B`** | **YOCO-style decoder-decoder**: **Encoder = self-decoder** processes the input and emits a **single global KV cache**; **Decoder = cross-decoder** generates the output and cross-attends to that global cache. **Total parameters ≈12.25B** (named `CAT-YOKO-12B`); activation default is the **middle band (~2.03B-in / ~4.33B-out)**; cheaper band 1.43/3.02 and near-dense band 3.04/6.29: see §3 | Activation asymmetry comes from **two physically different stacks** (smaller encoder, larger decoder); **the three bands change top-k only, not expert count**. See §2, §3 |
| `large sliding-window attention 8k` | The **uncompressed sliding-window branch** kept at every CSA/HCA layer, `n_win = 8192` | DeepSeek-V4 default `n_win=128`; 8K is a large upscale, more expensive, better local fidelity |
| `CSA HCA` | **Compressed Sparse Attention** + **Heavily Compressed Attention** (DeepSeek-V4's two compressed-attention kinds, interleaved across layers) | See §2 |

> ✅ **This round's locked spec**: Causal Encoder-Decoder (YOCO-style); **12.25B total parameters** (cut from 24B); activation default **≈2.03B-in / ≈4.33B-out**.
> **The release recipe is frozen** ([`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md) / `cat_yoko.config`): causal 16/26, C1 full MoE, C1+NVFP4 wall-clock 571 H100-h (target RTX PRO 6000 / 6000D), Phase B sliding window + gated cross-attn, M2/KDA/mHC/MTP/Muon off.
> The Encoder is **not** bidirectional. The "YOKO" in the repo name **CAT-YOKO** maps to **YOCO**.

> ⚠️ **Important reality check (read this first)**: fully replicating this architecture and pretraining from 32T-scale data is **frontier-lab scale** engineering.
> Upcycling MiniCPM5-2B and continuing training can bring cost down to a few hundred B tokens,
> but it still needs tens to hundreds of H100/H800-class GPUs running continuously. This plan is designed as **upcycling surgery + continued pretraining**,
> which is the only realistic path to a usable model of this architecture on a controllable budget (from-scratch 32T pretraining is not recommended).

---

## 1. Base model and target spec

### 1.1 MiniCPM5-2B (base, from the official config; Apache-2.0)

| Item | Value |
| --- | --- |
| Layers `num_hidden_layers` | 42 |
| Hidden size `hidden_size` | 2048 |
| FFN intermediate `intermediate_size` | 6144 (SwiGLU; `6144/2048=3`) |
| Attention heads | 16 Q / 2 KV **GQA** (Llama; `num_key_value_heads=2`) |
| `head_dim` | 128 (`d / n_q`) |
| `d_kv` | 256 (`n_kv · head_dim`) |
| Vocab `vocab_size` | 130560 |
| Activation | SiLU / SwiGLU |
| Position encoding | RoPE, `rope_theta=5e6` (native 128K context) |
| RMSNorm `rms_eps` | 1e-6 |
| Embedding | **untied** (input table and `lm_head` are not shared) |
| License | **Apache-2.0** (redistributable for a release, unlike MiniCPM-2B-sft-bf16's GML) |
| Special design | **No μP** (Llama identity residuals; logits are not divided by 9); WSD learning-rate schedule follows MiniCPM-family practice |

> MiniCPM5 is Llama. **Do not** import MiniCPM-2B μP constants (`scale_emb`, `scale_depth/√L`, logits `/ (d/dim_model_base)`).
> After the 42 layers are split 16+26, residual multipliers stay identity (do not invent √16 / √26). Cross-check in the theory verification.

### 1.2 Target model `CAT-YOKO-12B` (Causal Encoder-Decoder / YOCO-style)

| Item | Recommended value (default middle band) | Notes |
| --- | --- | --- |
| Architecture | **YOCO decoder-decoder**: Encoder(self-decoder) → global KV cache → Decoder(cross-decoder) | See §2 |
| Total parameters | **≈12.25B** (storage) | See §3 budget |
| **Encoder** active / input token | **≈2.03B** (band options 1.43/2.03/3.04) | 16 layers, MoE, CSA/HCA+8K sliding window, emits the global cache |
| **Decoder** active / output token | **≈4.33B** (band options 3.02/4.33/6.29) | 26 layers, MoE, self-attention + **cross-attn to the global cache** |
| Hidden size | 2048 (keep the base; enc/dec match so the vocab and warm-start can be shared) | |
| FFN | **DeepSeekMoE** fine-grained experts, `moe_intermediate_size=2048` | Enc: 1 shared+20 routed, top-k 7; Dec: 1 shared+20 routed, top-k 10 (default band); **first-layer dense off** |
| Attention | **CSA/HCA hybrid + 8K sliding window**; encoder holds long-range compression; decoder cross-attn reuses the single global cache; **optional 3:1 three-way mix with KDA linear attention** | §2 / §2.5; Phase B is still sliding-window GQA |
| KV cache | **Single global cache (You Only Cache Once)**, `d_kv=256` + compression → O(N)-class VRAM | The main long-context win |
| Residual | **mHC** (optional; first ship ordinary residuals) | Stability extra |
| Training objective | Main CE + **MTP** auxiliary head (optional) | |
| Optimizer | **Muon (2D weights) + AdamW (emb/norm/router/bias)** | Release default is AdamW |
| Context | **Primary delivery target: 128K–256K usable**; staged 4K → 8K → 32K → 128K → 256K; 1M is stretch (inference support + needle) | MiniCPM5 native 128K + 8K sliding window + compressed long-range + YOCO |

---

## 2. Architecture: YOCO-style Causal Encoder-Decoder + CSA/HCA compressed attention

### 2.0 Overall skeleton (decoder-decoder / YOCO)

CAT-YOKO is two causal stacks. Behaviorally it is equivalent to a decoder-only Transformer, except it "caches once":

```
input tokens ─► [Encoder = Self-Decoder, 16 layers, ≈2.03B active/token]
                 │  efficient causal attention (CSA/HCA + 8K sliding window), per-layer long-range compression
                 ▼
          top hidden state  ──►  emit [single global KV cache  K̂, V̂] (You Only Cache Once; d_kv=256)
                 │
                 ▼
        [Decoder = Cross-Decoder, 26 layers, ≈4.33B active/token]
           each layer = efficient causal self-attn (generation sequence, sliding window)  +  Cross-Attn(→ K̂,V̂)  +  MoE-FFN
                 ▼
             RMSNorm ─► LM Head (untied) ─► next token
```

Key properties (from YOCO; formalism and causality proofs are in [`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)):
- **Cache once**: only the one global cache emitted at the encoder top is reused by every cross-decoder layer, so KV-cache VRAM drops from `O(N·L)` to about `O(N)`. Encoder-internal CSA/HCA (**M1**) does **not** automatically reduce this cache's slot count; pooling along the sequence (**M2**) is what cuts slots from \(N\) to \(N/m\).
- **Prefill can exit early (inference only)**: the prompt only needs to finish the encoder to write the global cache; the decoder runs once on the last position to emit the first token. Training is still a full-length forward through both stacks. That is the motive for "**lighter on the input side (≈2.03B)**". Middle-band encoder activation share is about 32% (see `docs/THEORY_VERIFICATION.md` §6).
- **Asymmetric activation comes from two physical stacks**: the encoder is smaller (16 layers, fewer experts/active params → ≈2.03B), the decoder is larger (26 layers, includes cross-attn, more experts/active params → ≈4.33B). **The three bands change top-k only** (low 4/6, middle 7/10, near 12/16), not the 1+20 expert instances.
- **Global attention is retained**: globality comes from the decoder **cross-attn reading the cache (M3)**, not from stacking encoder sliding-window receptive fields (16×8K=131072, which does not cover 256K and does not need to).

> Design tradeoff: the original YOCO paper used sliding-window or gated retention in the self-decoder. We replace the self-decoder's
> intra-layer attention with **CSA/HCA + 8K sliding window (M1)**, which cuts the encoder's quadratic term when \(n\gg 8K\), and optionally mixes long-range into the write representation.
> **A tighter global cache is M2 (optional pooling). Query-aware retrieval at generation time is M3 (decoder-side indexer / calibrated fallback)** — not the same thing as M1.
> Decoder **self-attention also sees the full-length sequence at train time**, so it must use a sliding window / KDA; "generation sequences are usually short" describes inference only. All long-range traffic goes through cross-attn.

### 2.1 CSA / HCA / 8K sliding window (attention details)

DeepSeek-V4 replaces V3's full MLA attention with **two compressed-attention kinds interleaved layer by layer**. The point is to cut long-context attention cost from
`O(L²)` down to roughly `O(L·k)`, and to compress the KV cache heavily. This compressed attention is used **inside the Encoder (self-decoder)**, and also for its self-attention. Three layer kinds:

### 2.2 Three layer types (`layer_types`)

1. **Sliding-window (bootstrap layers)**: local sliding-window causal attention only, window = `sliding_window`, no long-range branch. The Encoder uses this on the **first 2 layers** (frozen; 16 layers is what still fits 7+7 CSA/HCA).
2. **CSA (Compressed Sparse Attention)**
   - Compress every `m=4` tokens of KV into 1 slot (learnable compression weights `Z` and a position bias, overlapping window).
   - Use a **Lightning Indexer** to score the query against compressed entries and keep **top-`index_topk`** (default 512) for attention (that is **DSA** on the compressed sequence).
   - Concatenate an extra **uncompressed sliding-window K/V branch** (size `n_win`) to keep local detail.
3. **HCA (Heavily Compressed Attention)**
   - Compress every `m'=128` tokens into 1 slot (non-overlapping), **no indexer**, **dense attention** over all compressed entries.
   - Likewise concatenate an uncompressed sliding-window branch.

> Implementation note (matches the official reference): both CSA and HCA **concat** **raw sliding-window K/V** with **compressed K/V** along the sequence axis,
> build a combined mask, and run **one** standard masked attention; CSA's mask is filtered by `top_k`, HCA's mask is fully visible.
> The only differences are "compression rate" and "whether top-k is used".

### 2.3 Attention config for this project (recommended)

| Parameter | Recommended value | Matching DeepSeek-V4 name |
| --- | --- | --- |
| `sliding_window` (`n_win`) | **8192** | The requested "8K large sliding window"; must also be \(\ge m'=128\) so HCA can fill holes in its own block |
| Encoder `layer_types` | **2× sliding bootstrap, then CSA:HCA=1:1** → 16 layers are **2 sliding + 7 CSA + 7 HCA** | V4-Flash starts with 2 sliding layers; 16 layers cannot do "3 bootstrap + 1:1" |
| Decoder self-attn | **all sliding** (optional later KDA mix); **CSA is not laid down by default** | Global mix already lives in cross-attn (M3) |
| CSA compression rate `m` | 4 | `compress_rate_csa` |
| HCA compression rate `m'` | 128 | `compress_rate_hca` |
| `index_topk` (CSA, **M1**) | 256～512 | Encoder intra-layer Lightning Indexer; **do not** reuse on the decoder query |
| Decoder-side selection (**M3**) | Needed from 128K: dense→top-k or calibrated fallback | The real generation-time retrieval; see architecture theory §3 |
| Attention base | Phase B = MiniCPM5 **GQA 16 Q / 2 KV** (`d_kv=256`); CSA/HCA can compress further to MQA-128. **Do not** use 1.25×MHA placeholder accounting | V4 implements DSA in MLA's MQA mode; the release ledger is GQA |

> **Cost of the 8K window**: the sliding-window branch is uncompressed `O(L·n_win)` cost. `n_win=8192` is 64× the default 128,
> so local-attention overhead rises sharply. If long-context throughput is tight, roll `n_win` back to 2K–4K once long-range capacity is enough,
> or use an 8K window on only some layers. Ablate `n_win ∈ {2K,4K,8K}` in §7.

### 2.4 Initialization: MiniCPM5 GQA → CSA/HCA + Encoder/Decoder

MiniCPM5 is a 42-layer decoder-only **Llama GQA** (16 Q / 2 KV, `head_dim=128`). It has no MLA latent, no compressor/indexer, and no cross-attn. Migration:

- **Split into two stacks**: slice MiniCPM5's 42 layer weights onto Encoder(16) + Decoder(26). Keep the Encoder at 16 layers so **2+7+7 CSA/HCA** still fits; give both extra layers to the Decoder (see §4 Phase A).
- **Attention backbone**: inherit **GQA** weights for the `q/k/v/o` projections (K/V is `(d_kv, d)=(256, 2048)`, not full MHA). If you later cut over to MLA, SVD-factor the K/V projections into low-rank `W^{DKV}·W^{UK/UV}` for init; same idea for `q_lora_rank`.
- **Decoder cross-attn**: `q`/`o` projections may be initialized from the matching self-attention `q`/`o` (**cross counts only Q/O = \(2d^2\)**; K/V come from the YOCO cache). **Bypass cross-attn first (gate≈0) and open it gradually** to keep training stable. At gate=0, the 16/26 split is equivalent to the original 42-layer residual stream (architecture theory Theorem A).
- **New modules** (compressors `W^{aKV}/W^{bKV}/W^{aZ}/W^{bZ}`, position bias `B`, Lightning Indexer): small-scale random init, **dense alignment first, then sparsify** (see §4 Phase C).

### 2.5 Optional extra: add KDA linear attention (three-way mix)

**Motive, and a misconception that must be cleared up**: it is tempting to think "adding KDA linear attention will keep CSA/HCA from dropping information in the **middle** of the context".
The literature actually says the opposite — **linear attention (including KDA) is the weakest link for exact mid-context recall**: its fixed-size RNN state collides,
and needles that fall outside the local window are easy to lose (arXiv 2507.06457, LoLA). Hybrid models recover recall by **keeping
full / sparse attention layers** as the retrieval path, not by relying on the linear layers. **Lost-in-the-middle is also a positional bias** (softmax models have it too),
and is mainly relieved by RoPE/NoPE calibration; adding linear attention does not fix it directly. **So mid-context exact retrieval is carried by CSA top-k and a few full anchors, not by KDA.**

**Why still recommend KDA?** Two complementary gains:
1. **Efficiency leverage (primary)**: Kimi Linear's **3:1 KDA:MLA** mix cuts 1M-context KV cache by ~75% and raises decode by ~6×, with quality that holds or improves.
   Replacing most layers with KDA and keeping a few CSA/HCA layers can cut long-context cost substantially.
2. **"Fallback coverage" when CSA misses (secondary)**: CSA's risk is that the Lightning Indexer's top-k **drops** a relevant mid-context block.
   KDA is a **no-top-k, order-sensitive path that writes every token into state**, giving gist-level full-sequence coverage as a safety net on misses
   (coarse coverage; it does not replace exact retrieval).

**Recommended config (optional; settle with ablations)**:
- **Encoder (self-decoder) three-way mix**: about **3:1 KDA : (CSA/HCA)**, e.g. every 4 layers `[KDA, KDA, KDA, CSA]`, insert 1 HCA every few groups,
  and keep **1–2 high-`index_topk` CSA layers or true full attention as "recall anchors"** (hybrid-linear: gated-delta families reach Transformer-level recall at 3:1~6:1).
- **Position encoding**: KDA uses learned decay for position / recency; full/CSA anchor layers may use **NoPE** (Kimi Linear's practice).
- **Decoder (cross-decoder)**: self-attention uses sliding window / KDA (train sequences can be long; do not revert to full attention); cross-segment retrieval is cross-attn → global cache (M3).

**Cost / caveats**:
- KDA needs an extra **DPLR chunked kernel** + **per-layer recurrent state** (orthogonal to YOCO "cache once": YOCO saves KV cache; KDA state is a small per-layer state).
- Known mix pitfall: **the model can learn to ignore the linear path** (if you first train strong CSA/HCA retrieval, then swap most layers to KDA). So **implement the 3:1 graph first (B, still sliding window), then light up KDA→CSA→HCA (C)**. Do not change layer types only at C. HCA layers are fewest and write-first, so light them last. Release default remains `use_kda=False`.
- Parameter-wise, KDA layers are usually cheaper than GQA/CSA (no large KV projections). Replacing some CSA/HCA layers with KDA slightly cuts per-stack parameters; recompute in the §3 script with the actual KDA dims and top the 12.25B back up with routed expert count.

> Bottom line: **worth adding, but the role is "efficiency + fallback coverage", not "protect exact mid-context retrieval"**. Whether to ship it, and the exact KDA:CSA:HCA ratio, is a §7 ablation.

---

## 3. Parameter-budget derivation (`CAT-YOKO-12B`, Encoder-Decoder split)

Base dims `d=2048`, `vocab=130560`, **untied** embedding (one input table + one `lm_head`). Encoder 16 layers, Decoder 26 layers (42 total, MiniCPM5 depth; the extra 2 layers go to the decoder).
`moe_intermediate_size=2048` (DeepSeek-V4 same order of magnitude, hardware-friendly; dense FFN 6144 is divisible by 2048), single-expert SwiGLU ≈ `3·d·2048 ≈ 12.58M`.
Attention is booked as **GQA 16/2** (`2d² + 2d·d_kv`), **not** 1.25×MHA. YOCO cache `d_kv=256`. Decoder cross counts only Q/O = \(2d^2\).

### 3.1 Budget table (12.25B total params; the three activation bands change top-k only)

Single-expert SwiGLU ≈ `3·d·2048 ≈ 12.58M`; Embedding (untied) ≈0.27B + lm_head ≈0.27B; all three bands have **12.25B** total parameters and differ only in activation / sparsity.

| Band | Enc experts (shared+routed, top-k) | Dec experts (shared+routed, top-k) | Enc active / input | Dec active / output | Sparsity enc/dec | Train compute (×50B tok) |
| --- | --- | --- | --- | --- | --- | --- |
| Cheaper-compute | 1+20, top-k **4** | 1+20, top-k **6** | 1.43B | 3.02B | 23.8% / 33.3% | 926 H100-h |
| **Default (middle)** | **1+20, top-k 7** | **1+20, top-k 10** | **2.03B** | **4.33B** | **38.1% / 52.4%** | **1,325 H100-h** |
| Near-dense | 1+20, top-k **12** | 1+20, top-k **16** | 3.04B | 6.29B | 61.9% / 81.0% | 1,943 H100-h |

Fixed pieces (same across bands): Enc self-attention ≈0.15B, Dec self-attention ≈0.25B, Dec cross-attn ≈0.22B (Q/O), emb ≈0.27B, lm_head ≈0.27B.
All three bands have **882 expert instances** (16×21 + 26×21; only top-k changes), so total params stay 12.25B and training cost moves only with activation. Under untied accounting, the plan's \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) and one forward (emb + lm_head each counted once) are both **6.36B → 1,325 H100-h**.

> 🔑 **Total parameters ≈ VRAM / storage; training FLOPs ∝ activation × tokens.** All three bands are 12.25B total params, but training cost moves with **activation** (926 → 1,325 → 1,943 H100-h @ 50B tok).
> Default is the middle band (enc 2.03B / dec 4.33B) — a compromise between "1.4 is too little, 3.0 is too much". Layer split (16/26), `moe_intermediate_size`,
> and per-stack shared/routed stay fixed at 1+20; **the only band knob is top-k**. Attention is booked as MiniCPM5 GQA, not 1.25×MHA.
> Side note: at 12.25B, GQA attention is **~5.0%** of total params (MoE ~90.6%); **adding KDA still does not change the total-param budget** (see §2.5). Line-item recomputes, KV/FLOPs, and the claim ledger are in [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md).

### 3.2 Recompute script (`scripts/param_budget.py`, default middle band)

```python
d, V, moe_int = 2048, 130560, 2048
kv_dim = 256                   # n_kv * head_dim = 2 * 128
emb    = V*d                   # untied input table
lm_h   = V*d                   # untied lm_head
expert = 3*d*moe_int           # SwiGLU single expert
attn   = 2*d*d + 2*d*kv_dim    # GQA 16Q/2KV, not 1.25×MHA
cross  = 2*d*d                 # per-layer cross Q/O; K/V from YOCO cache (d_kv=256)

# default middle band (12.25B / ~2.03B-in / ~4.33B-out); all three bands change top-k only
Le, ns_e, tk_e, Nr_e = 16, 1, 7, 20   # Encoder = self-decoder
Ld, ns_d, tk_d, Nr_d = 26, 1, 10, 20  # Decoder = cross-decoder
# cheaper-compute band: top-k 4/6    near-dense band: top-k 12/16

enc_act = emb + attn*Le          + Le*(ns_e+tk_e)*expert
dec_act = lm_h + (attn+cross)*Ld + Ld*(ns_d+tk_d)*expert
total   = emb + lm_h + attn*Le + Le*(ns_e+Nr_e)*expert \
              + (attn+cross)*Ld + Ld*(ns_d+Nr_d)*expert
print(f"enc_active(input)={enc_act/1e9:.2f}B "
      f"dec_active(output)={dec_act/1e9:.2f}B total={total/1e9:.2f}B")
```

The full recompute (KV / FLOPs / three-band comparison / spec asserts) is `scripts/param_budget.py`; derivation is in [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md):

```bash
python3 scripts/param_budget.py --full
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
python3 scripts/param_budget.py --nvfp4
```

---

## 4. Staged training recipe (core)

Overall idea: **upcycling + surgical conversion + staged continued training**. Use MiniCPM5-2B's existing skill as a warm start,
introduce one large change at a time, and let the model recover, so a single giant edit does not collapse it. Token budgets are order-of-magnitude guidance, not calendar time.

```
Phase A  Architecture surgery and initialization        —— 0 token (offline weight transform)
Phase B  Upcycling recovery continued pretraining       —— 50–150B token @ seq 4K (dense/sliding-window attention, no sparsity yet)
Phase C  Attention sparsification alignment             —— 20–50B token (default Indexer→top-k→HCA; with `use_kda`, C-kda first, then CSA/HCA)
Phase D  Long-context extension                         —— 20–60B token (8K→32K→128K, stepwise RoPE scaling)
Phase E  WSD anneal / high-quality data                 —— 20–50B token (LR decay stage, pile on math/code/long documents)
Phase F  SFT                                            —— 1–10B token (instruction + long context + tools)
Phase G  RL (GRPO / optional DPO)                       —— batched by domain
(optional)  MTP head joint training                     —— attach an MTP head from Phase B, weight 0.1–0.3
```

### 4.0 Train stacks / layers separately then merge? Feasible, but do not default to two independent LMs

YOCO is not seq2seq: at train time **the same sequence runs through Encoder then Decoder**, and the loss sits on the Decoder top. Encoder and Decoder representations were already jointly trained in MiniCPM5's 42 layers; Theorem A ([`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)) says a gate=0 16/26 split **is** that residual stream. Treating the two stacks as two independent LMs, training them apart, and splicing them throws that alignment away. **No franken-merge.**

Full freeze boundaries, gradient cutoff, untied embedding, optimizer / activation memory, and split sensitivity are in [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md). NVFP4 module policy and wall-clock are in [`docs/NVFP4_THEORY.md`](NVFP4_THEORY.md); Hopper/Ada FP8 fallback is in [`docs/FP8_THEORY.md`](FP8_THEORY.md). Numbers: `python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4`.

**The Phase B recipe is locked as C1+NVFP4**: C1 freeze boundary (Phase A MoE on both stacks → B0/B1 freeze the Encoder → B2 short joint) × mixed NVFP4. Delayed Encoder MoE, full-phase 2.0×, and peak 4× are sensitivity only; they are not in the recipe. Joint bf16 is the 100% control. C1+FP8 is the fallback when there is no Blackwell.

Middle band, 50B tokens, untied, emb and lm_head each counted once (same as one forward):

| Approach | H100-h | vs joint 50B |
| --- | ---: | ---: |
| Train both stacks together (joint bf16 control) | 1,325 | 100% |
| Encoder frozen, train only Decoder + cross-attn (entire run; no co-adaptation) | 987 | 75% |
| Train only new modules (cross-attn + \(W_K/W_V\); backbone frozen) | 720 | 54% |
| C1 unfreeze curriculum 8+27+15B (bf16 operand ledger) | 1,046 | 79% |
| **C1+FP8 (Hopper/Ada fallback)** | 729 | 55% |
| **C1+NVFP4 (locked)** | **571** | **43%** |
| Encoder as an independent LM 50B + frozen-Encoder Decoder 50B + 20B splice recovery | 1,940 | **146% (more expensive)** |
| Encoder as an LM 25B then joint 25B | 874 | 66% (**quality gamble**: is half the joint tokens enough to recover?) |

**Conclusions:**

1. **"Train two models apart and weld them" does not save compute.** The same 50B per stack plus splice recovery is 1.5× joint training. Caching Encoder hidden states for the Decoder is also unrealistic (50B tokens × \(d\) × 2 bytes ≈ 205 TB).
2. **"On the same sliced weights, unfreeze trainable subsets by stage" is what saves.** What you save is Encoder backward (and, in the new-module stage, Decoder-backbone weight grads), not Encoder forward — YOCO's CE is on the Decoder, so Encoder forward cannot be skipped.
3. Freezing the Encoder saves about **25% FLOPs**, but the Encoder no longer rewrites memory for the Decoder's queries (write/read do not co-adapt). PDSA's "no write-time signal" result also warns: train only the reader and freeze the writer forever, and retrieval hits a ceiling. So freeze the Encoder only as **the middle of Phase B**; the end must have a short joint (B2 ≥ 10B, default 15B).
4. Training only cross-attn / indexer (Phase A warmup, Phase C step 1) is cheapest (~54%). That is already written into Phase C; it is not a new invention.
5. Greedy layer-by-layer add (2 layers → freeze → add 2 more) has no stable compute-saving evidence on LLMs, and you still need a final joint finetune. **Do not do it.**
6. DeepSeek-style "per-domain experts each SFT+RL then distill" applies only to **Phase F/G post-training**, not to this 12B pretrain skeleton.
7. **The C1 freeze boundary is locked; wall-clock is further locked as C1+NVFP4.** Phase A virtual-group MoE on both stacks, B0/B1 freeze the Encoder: one offline surgery, token 0 is already the 12.25B middle band, B2 only unfreezes. B1/B2 all non-must-bf16 linear GEMMs (MoE, attn QKV/O, lm_head) + frozen Encoder forward run NVFP4; the B0 student stays bf16. Release wall-clock **571 H100-h** (**43%** of joint bf16 **1,325**; target RTX PRO 6000 / 6000D). Do not delay Encoder upcycling until B2, and do not put 2.0× on the B0 student (that is a sensitivity control, not in the recipe).

**Freeze rules (Theorems D/E, C1 locked; MiniCPM5 is untied; write these into the trainer, they are not a verbal convention):**

| Item | B0 | B1 | B2 |
| --- | --- | --- | --- |
| Global cache | `X^{16}.detach()` then multiply \(W_K,W_V\) | same as left | **drop** detach |
| \(W_K,W_V\) | new modules, trainable (\(2\,d\,d_{\mathrm{kv}}=1.05\mathrm{M}\), \(d_{\mathrm{kv}}=256\)) | trainable | trainable |
| Input embedding \(E\) | **frozen** | **frozen** | unfrozen |
| `lm_head` | **frozen** | **trainable** (untied; Theorem E does not govern the head) | unfrozen |
| Encoder weights | frozen (already virtual-group MoE) | frozen | unfrozen; experts specialize from here |
| Decoder | freeze backbone, train only cross-attn | unfreeze self-attn+MoE | unfreeze |

Forbidden: training input embedding \(E\) while the Encoder is frozen (input distribution drift, Theorem E). **B1 may train `lm_head`** — MiniCPM5 is untied; the head is not shared.

**Locked recipe (when the budget is tight, replace "Phase B 50B fully joint bf16" with C1+NVFP4):**

| Sub-stage | tokens | Trainable | Encoder FFN | gate |
| --- | ---: | --- | --- | --- |
| B0 | 8B (5–10B) | new modules (cross-attn, \(W_K/W_V\), gate, new LN); backbone + embed + `lm_head` frozen | frozen virtual-group MoE | 0 → 0.3 |
| B1 | 27B (20–40B) | unfreeze Decoder + **`lm_head`**; **Encoder + input \(E\) still frozen**; cache still detached | same as above | → 1 |
| B2 | 15B (10–20B) | both stacks unfrozen (including embed and `lm_head`), smaller LR | unfrozen; experts start specializing | 1 |

Total tokens still about 50B. C1 operands are about **79% of joint**; release wall-clock is **C1+NVFP4 = 571 H100-h (43% of joint bf16 1,325)**. C1 bf16 **1,046 H100-h**. C1+FP8 **729** is the Hopper/Ada fallback. Adam state at B1 is only **62%** of joint; detach drops Encoder activations of about **38%** (keeps 26/42). If quality is unstable, lengthen B2; do not go back to two independent LMs, and do not revert to full-run joint bf16. Phase C's "freeze the trunk, train only the indexer" still stacks **after** this curriculum (intra-layer KL, not CE through the cache; indexer stays bf16).

### Phase A — Architecture surgery and initialization (offline)

0. **Split into Encoder / Decoder stacks (YOCO-ize)**: map MiniCPM5-2B's 42 dense layers onto **Encoder 16 layers + Decoder 26 layers**.
   Keep the Encoder at 16 layers so **2+7+7 CSA/HCA** still fits; give both extra layers to the Decoder.
   Recommended: Encoder takes the base's **first 16 layers**, Decoder takes the **last 26 layers** (keep depth semantics); copy the **untied** embed and `lm_head` separately.
   Each Decoder layer **adds a cross-attn sublayer** (initial gate≈0 bypass, see §2.4; Q/O = \(2d^2\)), so the initial forward ≈ original decoder-only behavior and recovery is easier.
1. **MoE upcycling (dense FFN → fine-grained MoE)**, run **separately** on Encoder and Decoder, using Megatron-LM `upcycling_utils.py` (C1 locked: both stacks finish in this phase; B0/B1 then freeze the Encoder, see §4.0):
   - Slice the dense FFN intermediate into G groups and copy each group into several experts (**virtual-group init**: at the conversion instant, top-k exactly selects one copy of each shard, equivalent to the original dense function).
   - MiniCPM5 dense SwiGLU **6144 is divisible by 2048 (G=3)**. Upcycling is still **copy-and-scale the first `moe_int` rows** (each expert copies the dense first 2048 rows, then scales by \((E G^2 / T)^{1/3}\)); it is not exact identity at the surgical instant. Recovery is Phase B.
   - **Weight scaling**: SwiGLU expert projections scale on the order of `(E·G²/T)^(1/3)` (paper reports about 1.5% loss drop).
   - **Routing**: `softmax-then-topK` (better than topK-then-softmax); affinity scores use **Sqrt(Softplus(·))** (V4 practice).
   - Each stack is **full MoE by default** (C1: both stacks are MoE at token 0). DeepSeek-style "first-layer dense" is **off** and not in the release recipe; use `--first-dense` for sensitivity if needed.
   - **No μP**: do not invent residual multipliers for 16/26, and do not import MiniCPM-2B's `scale_depth/√40`.
   - **Hash-MoE bootstrap**: the Decoder's first few MoE layers use a frozen `token_id → expert_id` hash route (V4 practice, early stability). The Encoder is already frozen in B0/B1, so hash routing on the Encoder is redundant; apply it when B2 unfreezes.
2. **Attention conversion**: inherit GQA `q/k/v/o` (K/V shape `(256, 2048)`); new CSA/HCA compressors, position bias, and Lightning Indexer get small-scale random init. In this phase treat every attention layer as **dense/sliding window** (no top-k, no HCA compression), approximately the original attention. **This repository implements the Phase C indexer intra-layer KL and the C/D/E/F/G training entry points; it does not implement a CSA CUDA kernel.**
3. **Do not introduce MiniCPM-2B μP scale constants** (emb multiplier, `scale_depth`, logits scale). MiniCPM5 residuals are identity.
4. **Optional mHC**: first ship ordinary residuals through Phase B/C; switch to mHC after that is stable (constrain residual maps to the Birkhoff polytope / doubly stochastic matrices, spectral norm ≤1).

> Encoder/Decoder boundary: the Encoder top output goes through a (learnable) projection \(W_K,W_V\) to produce global `K̂,V̂` reused by every cross-decoder layer (YOCO cache-once). \(W_K,W_V\) are new modules. B0/B1 must `X^{16}.detach()` **before** multiplying by the projection (Theorem D); B2 drops detach.

### Phase B — Upcycling recovery continued pretraining

- **Goal**: after MoE conversion + attention conversion + encoder-decoder conversion, restore language-modeling skill (including gradually opening cross-attn).
- Sequence length 4K, attention still dense/sliding window (not yet sparse), data is a general pretrain mix (see §5).
- **Open cross-attn gradually**: Decoder cross-attn gate ramps linearly from 0 to 1. Under the unfreeze curriculum: B0 (~8B) only goes to 0.3, B1 goes to 1, so you do not slam \(g\) to 1 on a frozen backbone.
- **Distillation to speed recovery**: use **MiniCPM5-2B-Base (dense, teacher)** for logit KD (KL(teacher‖student), temperature 1–2, weight 0.5→0 linear decay), which shortens recovery a lot.
- **MoE load balance**: aux-loss-free bias method (`e_score_correction_bias`, update bias from per-expert load, rate e.g. 1e-3) + **light sequence-wise balance loss** (weight ~1e-3) against extreme per-sequence imbalance. Encoder experts update only from B2, so Encoder load monitoring starts at B2.
- Learning rate: **WSD** (Warmup-Stable-Decay) — short warmup (0.5–1B tokens), then the stable segment (LR ≈ 30–50% of MiniCPM pretrain peak, because this is continued training). This phase stays stable with no decay. Drop LR another notch when B2 unfreezes the Encoder.
- **Untied embedding**: B0/B1 **freeze input \(E\)**; B0 also freezes `lm_head`, B1 **trains `lm_head`**; forbidden to train input \(E\) while the Encoder is frozen (Theorem E).
- **When the budget is tight, do not rewrite this as two independent LMs**: use the already locked **C1+NVFP4** in §4.0 (B0 new modules → B1 freeze Encoder → B2 short joint; B1/B2 allowed linear GEMMs run NVFP4), same ~50B tokens, wall-clock **571 H100-h (43% of joint bf16 1,325)**.

### Phase C — Attention sparsification alignment (critical, easy to blow up)

**Flow: implement first, then light up.** Code and layer labels enter the graph at B; C only changes `sparse_mode`. Do not weld the modules on only at C.

| Step | When | What |
| --- | --- | --- |
| **Implement** | Phase A surgery + **B graph build** | Default layer labels 2 sliding+7 CSA+7 HCA; `--use-kda` then 2+11 KDA+2 CSA+1 HCA, `KDAGates` enter the graph. **Compute is still sliding-window GQA** (`sparse_mode=window`). KDA parameters are frozen at B and do not enter Adam. Release B0 default `use_kda=False` (132-tensor overlay). |
| **Light up** | Phase C | Without KDA: indexer→topk→hca→win. With KDA (already implemented at B): **C-kda → indexer → topk → hca (last) → win**. Do not patch KDA modules onto a `use_kda=False` B overlay. |

Follow DeepSeek-V3.2 "dense warm-start, then sparse" to introduce DSA/compression. **Release default** (2 sliding + 7 CSA + 7 HCA, `use_kda=False`):

1. **Indexer dense alignment**: freeze the trunk, train only the Lightning Indexer on Encoder CSA so its score distribution **matches dense attention weights** (intra-layer KL). This step does not change the main output; it only teaches the indexer "whom to pick". Supervision stacks **after** B2. Indexer top-k may only delete from the compressed-block set \(S_{\mathrm{comp}}\), never add (Theorem B).
2. **Turn on CSA top-k**: CSA layers switch from dense sliding window to **sliding window ∪ indexer-selected compressed blocks** (own block excluded, window fills holes), then take small steps so the trunk adapts to sparsity. This is not "keep only top-k tokens".
3. **Turn on HCA compression**: HCA layers **concat sliding-window KV with mean-pooled slots** (\(m'=128\), own block excluded).
4. **Joint-train the 8K sliding window**: `C-win` opens CSA top-k + HCA concat together at seq=8192, and confirms window-branch vs compressed-branch masks are correct.
- Monitor loss spikes at every step; on instability, roll that step back, lengthen alignment, or lower LR.

**If you turn on 3:1 KDA** (12B: 2 sliding + 11 KDA + 2 CSA + 1 HCA), the light-up order becomes:

0. **`C-kda`**: switch only KDA-kind layers to gated-delta; CSA/HCA-kind **still run sliding window**. Most paths learn to be used first, avoiding hybrid-linear "strong retrieval first, then swap in linear layers and get ignored".
1. **Indexer**: KDA stays lit; intra-layer KL only on the remaining CSA anchors (12B has only 2 such layers; tokens cut from 10e9 to 5e9; C still totals 25e9).
2. **CSA top-k**, then **HCA last** (1 layer, write-first), then `C-win`.

CSA/HCA **implementation is not deferred or deleted**; what is deferred is **lighting up**. Same for KDA: B `--use-kda` implements, C lights up. B0 default remains `use_kda=False`.

### Phase D — Long-context extension

- Raise train sequence length in steps: **8K → 32K → 128K** (continue if you need longer).
- RoPE: frequency-scale to the target length (NTK/YaRN class) or continue training on long sequences directly; MiniCPM5-2B is native **128K context / `rope_theta=5e6`**, so use it as the long-context base and do not consult MiniCPM-2B-128k `rope_scaling`.
- CSA/HCA keep long-range attention cost under control; the 8K uncompressed window keeps local fidelity.
- Switch data to long documents / concatenated long samples; monitor with needle & RULER.
- **Implementation**: `python3 -m cat_yoko.d --stage 8k|32k|128k` or `--chain`. Phase B 4K packed `.bin` is **re-windowed to the target seq** (flat int32 concatenated rows). `--try` DummyStream writes a needle at mid-sequence. D/E/F **default `sparse=hca`** (after C lights up, do not go back to window); `--use-kda` is inherited from resume extra, and only KDA-kind layers keep running gated-delta. prepare `--mix phase-d`: en 45% / zh 20% / math 10% / StarCoder 25%.

### Phase E — WSD anneal (high-quality data)

- Enter WSD **Decay**: LR decays quickly (exponential / 1-sqrt) to ~1/100 of peak.
- Mix shifts toward **high quality + math + code + long context + instruction-ized** (MiniCPM experience: the anneal stage benefits most from high-quality data).
- **Implementation**: `python3 -m cat_yoko.e`. `wsd_lr(..., lr_mode=decay)`. prepare `--mix phase-e`: en 30% / zh 15% / math 25% / StarCoder 15% / UltraChat body 15%. Do not download in CI.

### Phase F — SFT

- Instruction / multi-turn / long context / tool use / code / math; pack to the target length; loss only on the response.
- You can follow DeepSeek-V4 "**per-domain experts each SFT+RL, then on-policy distill into one model**", but at this project's scale (12B) start with a single mixed SFT.
- **Implementation**: `python3 -m cat_yoko.f`. seq=8192. prepare `--mix phase-f` writes jsonl (`tokens`+`labels=-100` on user; multi-turn packed to target length). Parse UltraChat `data` lists, Chat `messages`, alpaca `instruction`/`output`, encode `role: text`. Trainer `FileStream` also accepts already-tokenized `prompt_ids`/`response_ids` or `messages[].ids` and concatenates the same way. `--try` still uses DummyStream and masks the prompt prefix. sparse stays `hca`.

### Phase G — RL

- **GRPO** (DeepSeek family) as primary; reward covers verifiable math, executable code, instruction following; DPO can be added as light preference alignment.
- **Long-context usability RL (critical, and cheap in compute)**: run RLVR on **verifiable long-context tasks** so "actually read and use the long input" is rewarded directly — much cheaper than feeding another ocean of long tokens, and aimed at lost-in-the-middle / long-instruction non-compliance / multi-hop misses:
  - **Reward signal**: long-doc QA (RULER-style, needle variants, multi-hop HotpotQA extensions) with exact-match/F1; **grounding / citation reward** (the answer must cite the right passage/line, matching §14 PDSA evidence selection); long-instruction following (checkable constraints).
  - **Curriculum**: RL in 128K→256K steps, deliberately place key information in the **middle / long-range**, reinforcing mid-context recall (complements §12 IN2/FILM training).
  - **Save compute**: long-trace RL is expensive per rollout → use **PS-PPO (prefix-sampling PPO)** to backprop only the sampled prefix with unbiased truncation, which cuts long-sequence RL compute/VRAM; or select evidence with PDSA on ultra-long context before RL (shorten effective rollout length).
- Watch MoE routing and sparse-attention stability on long rollouts during RL.

---

## 5. Data

| Stage | Main data | Magnitude (tokens) |
| --- | --- | --- |
| B recovery (think) | Ultra-FineWeb en 55% / zh 30% + UltraData-Math L2 10% + StarCoder **5%** | 50B envelope (prepare slices by `--max-tokens`; do not pull 50B) |
| C sparsify | Same distribution as B, biased to long documents | 20–50B |
| D long context | Long documents, books, repo-scale code concat, synthetic long-dependency tasks | 20–60B |
| E anneal | High-quality curated + math + code + instruction-ized SFT precursor | 20–50B |
| F SFT | **UltraChat** and similar instruction / multi-turn (not Phase B) | 1–10B |
| G RL | Verifiable-task prompt sets (math/code/agent) | prompt-level |

Key points:

- **Tokenizer must be MiniCPM5-2B** (`openbmb/MiniCPM5-2B`, `V=130560`). Do not use MiniCPM3 / MiniCPM4 tokenizers, and do not feed MiniCPM-2B-sft-bf16 into a MiniCPM5 upcycled graph. Ultra-FineWeb is a MiniCPM4-era web filter set and **must be re-tokenized**. **This repository does not download Ultra-FineWeb onto a small VM / CI.**
- Default mix `phase-b` (think / B0–B2 / C) is OpenBMB web+math **plus 5% StarCoder**. The model is meant to do code work, so the think stage is not 0% code; 5% is "a little, not none", not Phase D's 25%. Optional `phase-b-code` raises code to 10% (Ultra-FineWeb paper eval mix). StarCoder is not OpenBMB; **do not download it in CI / on a small VM**. Current 3090 B0 still uses DummyStream: the same 5% is hashed from in-repo short snippets into rows, not StarCoder itself.
- UltraChat / instruction dialogue is reserved for Phase F/G, not Phase B.
- Implementation: `python3 -m cat_yoko.prepare --mix phase-b --tokenizer openbmb/MiniCPM5-2B --out data/phaseb.bin --max-tokens 1e8` → int32 packed mmap; `cat_yoko.train --data data/phaseb.bin --upcycle-hf openbmb/MiniCPM5-2B-Base`. Sidecar `*.bin.meta.json` carries `eos_id`. The repo **does not check in corpora**.
- Long-context samples use document concat + synthetic "needle / multi-hop"; strict dedup and eval-set decontamination. Ultra-FineWeb is marked Apache 2.0; source-page copyright still follows each site's terms.

---

## 6. Optimizer / hparams / stability

| Item | Recommended |
| --- | --- |
| Optimizer | **Release default AdamW** (\(\beta=(0.9,0.95)\), wd=0.1). Muon switch exists, **default off** (see [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)) |
| Muon | Newton-Schulz orthogonalization of momentum; **hybrid ZeRO** implementation (V4 practice); lr needs its own sweep (usually larger than Adam) |
| LR schedule | **WSD**: warmup(0.5–1B) → stable → decay; continued-training peak is 0.3–0.5× the base pretrain peak |
| Batch | Global batch grows by stage (e.g. 4M→16M token/step); long-context stages use seq packing |
| Precision | **Mixed precision locked** ([`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md)): all linear GEMMs that are not must-bf16 are NVFP4 (MoE experts, attn QKV/O, lm_head, frozen Encoder forward). B0 student, L0, Phase C indexer, **embed**, RMSNorm, router, gate, attn softmax stay high precision. Do not change 6NT; wall-clock is booked at 2.0× vs bf16. C1+FP8 is the Hopper/Ada fallback. V4-style FP4 expert storage is a later option, not in this recipe |
| Regularization / stability | zero-centered & weight-decayed RMSNorm, router z-loss (light), grad clip 1.0 |
| MoE balance | aux-loss-free bias updates + light seq-balance loss; monitor expert utilization / drop rate |
| MTP | Auxiliary-head weight 0.1–0.3; may be used only on B–E; at inference it can be dropped or used for speculative decoding |
| μP | **None**. MiniCPM5 is Llama; do not import MiniCPM-2B emb/residual/logits scale constants |

Muon's Newton-Schulz orthogonalization stays fp32, orthogonal to network NVFP4 GEMMs. Do not turn on NVFP4 at L0.

---

## 7. Eval and ablations

**Capability eval**: MMLU / CMMLU / C-Eval (knowledge), GSM8K / MATH (math), HumanEval / MBPP (code),
BBH (reasoning), IFEval (instruction following).
**Long context**: **RULER**, **Needle-in-a-Haystack**, LongBench; check retrieval fidelity of 8K sliding window + compressed long-range + YOCO global cache at 32K/128K/1M (YOCO reports near-perfect needle at 1M).
**Efficiency**: per-token inference FLOPs, **KV cache size (YOCO caches once, should be well below a decoder-only baseline)**, prefill latency (encoder early-exit gain), decode throughput.
**Required ablations**:
1. **Encoder/Decoder layer split** (e.g. 16/26 vs 20/22 vs 12/30) on 2.03B/4.33B activation and quality; release default remains **16 encoder** (2+7+7 CSA/HCA still fits);
2. `n_win ∈ {2K, 4K, 8K}` quality/throughput tradeoff;
3. CSA:HCA layer ratio (1:1 vs 2:1 vs 3:1);
4. `index_topk ∈ {128,256,512}`;
5. MoE granularity / expert count (`moe_intermediate_size`, `n_routed`, `top_k`) vs hitting the 12B total / per-band activation targets;
6. **YOCO decoder-decoder vs equal-param decoder-only** (confirm KV cache / prefill gains without dropping quality);
7. **Whether to introduce KDA and the KDA:CSA:HCA ratio** (e.g. pure CSA/HCA vs 3:1 KDA mix vs 6:1) — watch whether RULER / multi-hop mid-context recall **drops because KDA was added** (linear layers are expected to slightly cut exact recall; full/CSA anchors must compensate) and long-context throughput / KV-cache gains;
8. Upcycling vs continued training from the dense base (confirm the upcycling gain);
9. Muon vs AdamW; mHC vs ordinary residual; cross-attn gate ramp vs open immediately; full-anchor NoPE vs RoPE.
10. **Unfreeze curriculum**: C1 (locked) vs full joint vs **illegal** B2=0 (curriculum doc §9; look at recovery PPL and RULER, not FLOPs alone). Delayed Encoder MoE is sensitivity only, not in the recipe.
11. **NVFP4**: B1 `nvfp4` vs full-run bf16 (look at recovery PPL / overflow, not wall-clock alone); B0 student wrongly on NVFP4 as a negative control. The must-high-precision set must not enter 4-bit. If QKV/O diverges, fall back to high precision (MaxText reading); do not change C1.

---

## 8. Infrastructure

| Component | Suggestion |
| --- | --- |
| Training framework | This repo's reference implementation is **PyTorch** (`cat_yoko.train --backend torch`). When 12B does not fit one GPU, use **DeepSpeed ZeRO** (`--backend deepspeed`, [`docs/DEEPSPEED_ZERO.md`](DEEPSPEED_ZERO.md); `--dump-deepspeed` writes JSON, CI does not hard-require the install). Scale-out EP/TP is reserved for **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** / Megatron-Core. Mapping: `cat_yoko.megatron.mapping.megatron_blueprint`; `--dump-megatron` writes JSON. YOCO is **not** `GPTModel`. |
| Parallelism | `ParallelPlan`: TP/PP/EP/CP/SP. 12B: TP ∈ {1,2,4,8,16} (divides 16 heads and \(d=2048\)); **EP ∈ {1,2,4,5,10,20}** (divides 20 routed). When PP>1, encoder|decoder cuts at layer 16 (`pipeline_split_rank`). Long context uses Context/Sequence Parallel. YOCO is **not** Megatron `GPTModel`. |
| Attention kernel | **FlashMLA** sparse prefill/decode kernel (backs DSA, FP8 KV); **NSA** Triton kernels are a reference for the compress+select+window three-branch implementation |
| MoE kernel | Fused MoE dispatch/combine kernel (compute/comm/memory overlap) |
| Precision | bf16 master + locked NVFP4 GEMM (§6 / [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md)); Hopper/Ada fallback FP8; deterministic/reproducible kernels (optional) |
| VRAM | Tensor-level recompute (`--grad-ckpt`), B0/B1 frozen-Encoder CPU offload, B2 per-layer offload, Adam momentum CPU offload (`--optim-cpu`); **DeepSpeed ZeRO-3 + CPU offload** to shard frozen weights (48GiB Ampere). This is not a Megatron EP/TP loop |
| Inference | vLLM / SGLang (already integrated DSA/FlashMLA sparse kernels) for eval and RL rollout |

> If you cannot write a CSA/HCA kernel in-house, **start from HuggingFace `transformers`' `DeepseekV4` reference implementation**
> (`layer_types`, `compress_rates`, `sliding_window`, `index_topk`, etc. are already exposed) to prove correctness and small-scale training,
> then migrate to high-performance kernels for scale-out.

---

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Sparse-attention training unstable / quality drop | Strictly follow Phase C "dense align → gradual sparse"; align the indexer alone first; on failure, roll back one step |
| MoE load collapse / idle experts | aux-loss-free bias + seq-balance loss + Hash-MoE bootstrap + utilization monitoring |
| Capability regression after upcycling | virtual-group init + weight scaling + teacher distillation + LR reset to a relatively high stable segment |
| 8K sliding window too expensive | Ablate `n_win` downward; 8K on only some layers; long-range goes to CSA/HCA |
| MiniCPM-2B μP constants transplanted, causing numeric drift | MiniCPM5 is Llama: `scale_emb=1`, identity residuals, logits not divided by 9; unit-test forward scale after surgery; do not import MiniCPM-2B μP |
| Muon does not converge / unfamiliar hparams | Ship an AdamW baseline first, then switch to Muon and sweep lr separately; keep a fallback switch |
| Missing kernels | Prove correctness on the HF reference first, then bring up high-performance kernels |
| NVFP4 overflow / loss spikes | B0 student stays bf16; when B1 switches to `nvfp4`, watch NaNs and expert utilization; first fallback is QKV/O back to high precision, then FP8/bf16; do not change the C1 freeze boundary |
| Poor long-context extrapolation | Staged RoPE scaling + long-sample curriculum + RULER process monitoring |

---

## 10. Milestones (by capability / budget, not by calendar)

1. **M1 surgery ready**: offline `CAT-YOKO-12B` (Encoder 16L / Decoder 26L) initial weights, forward numeric-scale self-check passes, short-run loss does not diverge with cross-attn bypassed.
2. **M2 recovery met**: after Phase B (cross-attn fully open), general benchmarks recover to ~95%+ of MiniCPM5-2B.
3. **M3 sparsification met**: after Phase C, CSA top-k + HCA + 8K sliding window are on; short-context quality roughly matches M2; efficiency is clearly better.
4. **M4 long context**: **128K–256K RULER/Needle pass and quality is usable (primary target)**; 1M is stretch (inference runs + needle can pass; do not chase quality); per-token FLOPs and **KV cache (YOCO single cache)** well below a decoder-only control; prefill early-exit gain is realized.
5. **M5 post-training**: after SFT + GRPO, instruction/math/code land in the target band; produce a releasable checkpoint.

---

## 11. Immediate next steps

The release spec is frozen; see [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md). Next is to run this repo's **12B training code** (tiny unit tests → meta 12B graph → `--dump-megatron` → FSDP / Megatron once you have GPUs).

1. `python3 -m unittest tests.test_param_budget tests.test_arch_verify tests.test_train tests.test_trainer tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload`
2. `python3 -m cat_yoko.prepare --mix local --local texts.jsonl --tokenizer dummy --config tiny --out /tmp/t.bin --max-tokens 256` then `python3 -m cat_yoko.train --config tiny --phase B0 --steps 3 --accum 1 --data /tmp/t.bin`
3. With a GPU: `python3 -m cat_yoko.gpu_smoke` (tiny); `python3 -m cat_yoko.gpu_smoke --middle` (one 12B B0 step, ≥28GiB, bf16 graph built directly); `--middle --phase B1` (Encoder offload + CPU Adam); `--c1` (same 12B graph B0→B1→B2)
4. `python3 -m cat_yoko.train --config 12b --meta` (count parameters, do not allocate 24GB)
5. `python3 -m cat_yoko.train --config 12b --dump-megatron` (dual-stack TransformerConfig JSON, does not run Megatron)
5b. `python3 -m cat_yoko.train --config 12b --dump-deepspeed --zero 3 --zero-offload-param` (ZeRO JSON, does not install DeepSpeed)
6. With network + GPU: `pip install 'cat-yoko[data]'`, `prepare --mix phase-b --tokenizer openbmb/MiniCPM5-2B --out data/phaseb.bin --max-tokens 1e8`, then `--config 12b --phase B0 --upcycle-hf openbmb/MiniCPM5-2B-Base --data data/phaseb.bin --save-dir runs/b0 --dtype bf16 --grad-ckpt --device cuda --steps N` to start training under C1+NVFP4. B1/B2 use `--resume` against `latest.pt` or the save directory (weights + packed cursor + RNG; do not restore the previous phase's Adam / step). `latest.pt` hardlinks when `step_{last}.pt` already exists; do not write a second 23GiB copy for 12B; put ckpts on a large disk (`/root/autodl-tmp`), not `/tmp`. You can also `--c1 --save-dir runs/c1 --steps N` to run the three stages on the same graph, writing `runs/c1/{B0,B1,B2}/latest.pt`. 12B does not save Adam by default. Single 32GB GPU + ~62GiB host cgroup: B0 one step directly; B1 offloads the frozen Encoder + CPU Adam (one-step smoke uses ephemeral momentum); B2 per-layer offload, clip+Adam as soon as a layer's backward finishes (`--accum 1`). Scale-out later with `--backend megatron`. Do not download Ultra-FineWeb or 12B weights on a small VM / in CI. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO.

Do not reopen 16/26, C1, C1+NVFP4, causal Encoder, or the M2 default. If quality is the problem, lengthen B2 or fall back dtype; do not change the freeze boundary.

---

## 12. Further advanced techniques (layered by value/risk, avoid piling on)

> Principle: the more novel components, the harder training is. Ordered "low risk first, high risk later / optional"; each one should be independently switchable and revertible.

### Tier 1 — Low risk, high reward (recommended as defaults to add)

- **QK-Norm** (RMSNorm on query/key) + **z-loss** (router-z against routing-logit explosion + output-logit z-loss) + **dual RMSNorm (pre+post, OLMo2/Gemma2 style)**: the cheap stabilizer for a deep + MoE + sparse-attention novel stack.
- **Document-aware attention mask**: packed long sequences **must not attend across documents**, so the long-context training signal is not polluted.
- **FIM (Fill-in-the-Middle)**: fill-middle training on code data, better completion/edit skill.
- **IN2 / information-intensive long-context training (FILM-class)**: synthesize samples whose **key information sits in the middle** of a long document — **this is the actual fix for lost-in-the-middle**, more direct than adding KDA (see the §2.5 clarification).

### Tier 2 — Medium risk, high reward (add after the base is stable)

- **MTP → speculative decoding**: reuse the already-attached MTP head for EAGLE-style self-speculation; inference speedup; almost zero extra train cost.
- **NVFP4 training** ([`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) locked): linear GEMMs that are not must-bf16; B0 student / indexer / must-high-precision set stay high precision. Release wall-clock 2.0× vs bf16; do not publish 4×. **V4-style FP4 expert storage** is a later option, not in this recipe. Hopper/Ada fallback: [`docs/FP8_THEORY.md`](FP8_THEORY.md).
- **attention logit soft-cap / QK-clip** (Gemma2 / Kimi): suppress extreme logits, further stabilize training.
- **RoPE/NoPE calibration and frequency scaling (YaRN)**: long-context extrapolation + less positional bias.

### Tier 3 — Cautious / last (off by default)

- **mHC** (already in §2, marked optional), **Muon** (keep an AdamW fallback), **3-way KDA mix** (§2.5).
- **Zero-compute / elastic top-k experts**: adaptive activation by token difficulty (potentially a fit for "2.03B/4.33B asymmetric", but complex and easy to destablize; research item).
- **Shared attention blocks (Zamba-style) reused across layers**: further save params/cache, but the coupling is strong.

---

## 13. Training-difficulty management: staged de-risking (important)

**Core: do not light up every new component at once.** The risk of combined novelty (YOCO + CSA/HCA + optional KDA + DeepSeekMoE + Muon + mHC + MTP) multiplies.
Gradual introduction + per-step rollback + metric monitoring is the only stable path.

**De-risk ladder:**
- **L0 (tiny correctness)**: small config (`hidden 256, enc2L/dec2L, sliding=8, m=4, m'=8, index_topk=2`) verifies: YOCO dataflow (encoder→global cache→cross-decoder), CSA/HCA/KDA masks and kernels, MoE routing/balance. Only "is it correct, does it NaN". **Precision bf16, no NVFP4.**
- **L1 (half-scale de-risk prototype, ≈3–6B)**: the **full novel architecture stack** but **half the experts**, run tens of B tokens for stability, upcycling recovery curves, sparsification alignment, and cross-attn ramp. A cheap architecture proving ground.
- **L2 (scale to the 12B target)**: **MoE expert count is the safest scale axis** — once the architecture is proven at L1, half-scale→12B is mostly adding routed experts (+ a little continued training so new experts specialize), much lower risk than changing the architecture.
- **Each new component is its own step**: first **implement the module into the graph** (B still sliding window), then **light it up**. Default light-up is indexer→CSA→HCA; **a 3:1 KDA graph is C-kda → sparsify**. **NVFP4 does not enter L0** (tiny stays bf16); allowed linear GEMMs start at B1, with a separate rollback switch from Muon orthogonalization. Watch loss spikes / expert utilization / recall metrics; on failure, roll that step back.

**Difficulty–benefit cheat sheet:**

| Want less hassle / a faster result | Want extreme long-context efficiency |
| --- | --- |
| Start decoder-only + CSA/HCA (no YOCO/KDA/mHC/Muon), then add gradually once that runs | Full-stack YOCO + 3:1 KDA + CSA/HCA + NVFP4, but strictly L0→L1→L2 |

---

## 14. Fold in PDSA memory management (trainable lifecycle + calibrated fallback)

> Source: **Memory-Managed Long-Context Attention** (Zou & Donz, arXiv `2606.28876`, this team's work, hereafter PDSA).
> That paper is a **memory system around a frozen LLM at inference/eval** (not a trainable architecture). Core: query-independent writer +
> hard-boundary lifecycle (overwrite/protection/eviction, ≤32 slots) + query-aware reader + **calibrated sparse fallback** + frozen LLM generation from raw evidence.
> It is **orthogonal and complementary** to CSA/HCA/KDA: those compress token-level state; PDSA is managed memory at the semantic-unit level.
> PDSA's stated "next step" is to **put the lifecycle inside the model, trainable** — CAT-YOKO is a natural instance of that.

### 14.1 Transferable conclusions

- **"No write-time signal" bound (PDSA §5, measured)**: on static text, query-blind writability is ≈ random (AUC 0.63–0.66 vs query-aware 0.89–0.97); pure bounded memory recalls only ~0.56 of gold evidence. **Implication**: any **write-first compression** (HCA, KDA, even CSA's compress step) systematically drops information that had no signal at write time and is only later hit by a query → **you must keep a query-time fallback to less-compressed / raw KV**.
- **Bounded selection beats reading the full document on long text (PDSA §4)**: at 8.2k words, reading the full document actually scores worse (lost-in-the-middle); ≤10% evidence already reaches 102–116% of full-document F1. Supports CAT-YOKO's "compress + select" direction.

### 14.2 Layered integration (fits §13 difficulty management)

- **Tier 1 (low risk, recommended) — calibrated confidence-gated sparse fallback**:
  Give CSA's Lightning Indexer a **confidence signal** (e.g. mean/entropy of top-k scores); below threshold, **widen top-k or fall back to less-compressed / raw KV retrieval**.
  Calibrate the threshold **on the deployment length regime** (PDSA recorded negative result: calibrating on short context makes fallback never fire and long-doc coverage collapse).
  This is a principled treatment of the "mid-context recall" worry, using **this team's own measurements**, and it is an inference-time increment that can be added later.
- **Tier 2 (medium risk) — query-aware over write-first**: in layer schedule **use more CSA (query-aware selection)** and be conservative about HCA/KDA (write-first) for critical retrieval; position HCA/KDA as "cheap gist coverage + safety net", and give exact retrieval to CSA + full anchors + fallback.
- **Tier 3 (research-grade, after the base is stable) — trainable bounded editable memory lifecycle**:
  Upgrade YOCO's "append-only global cache" to **bounded, editable, lifecycle-bearing** memory: a learned **writer** (write/overwrite/protect/evict, by key/salience) manages a capacity-limited memory that decoder cross-attn reads.
  Gain: a truly bounded KV cache + **versioned / protected semantics** (agents, long-horizon task differentiation); risk: switched-process stability (PDSA Appendix H), unstable writes — unsolved research, its own milestone, keep a symbolic / frozen fallback.

### 14.3 Matching ablations (add to §7)

- With/without **calibrated sparse fallback** on RULER / multi-hop mid-context recall and long-doc F1; sensitivity of fallback threshold **calibrated across length regimes**.
- **More CSA (query-aware) vs more HCA/KDA (write-first)** on recall of "information only later hit by a query".
- (Research item) trainable editable memory vs append-only global cache: KV-cache upper bound, versioned-task correctness, stability.

> Positioning: PDSA's contribution is **memory management**; it does not replace CSA/HCA/KDA **state compression**. Stacking both is the full design.
> Do not migrate its frozen-reader eval harness or the specific 32-slot number (those are methodological evidence, not architecture).

---

## 15. Compute-budget estimate and low-budget routes (important real constraint)

> Premise correction: this plan originally defaulted to "tens to hundreds of H100s". **If compute is limited, 24B from-scratch / heavy continued pretraining is not feasible**, and priorities must be reordered:
> **prove the architecture at small scale first, scale up when there is budget**. What must be shown is "architectural novelty (YOCO×CSA/HCA + KDA + PDSA trainable lifecycle)",
> not "large-model scale" — nearby hybrid-linear analyses finished at 340M/1.3B, and this team's PDSA core is only ~2.74M parameters + a frozen backbone.

### 15.1 Training compute estimate

`training FLOPs ≈ 6 × N_active × tokens`. The bf16 rows below convert operands to hours at 40% MFU (H100 effective ~4.0e14, A100 ~1.25e14 FLOPS); **default Phase B wall-clock is C1+NVFP4**, not joint bf16. RTX PRO 6000 Server BF16 peak ≈ H100, so hours are comparable.

| Scheme | H100-h | A100-h | 8×H100 days |
| --- | ---: | ---: | ---: |
| **C1+NVFP4 (Phase B locked)** | **571** | **1,827** | **3.0** |
| C1+FP8 (Hopper/Ada fallback) | 729 | 2,333 | 3.8 |
| C1 unfreeze curriculum 8+27+15B (bf16 operand ledger) | 1,046 | 3,347 | 5.4 |
| 12.25B middle band × 50B tok (joint bf16 control; same as the plan ledger under untied) | 1,325 | 4,239 | 6.9 |
| 12.25B middle band × 200B tok | 5,299 | 16,956 | 27.6 |
| 24B × 200B tok (later; using the then 3B+6B activation) | 7,583 | 24,038 | 39.5 |
| 24B × 50B tok (later minimum recovery) | 1,896 | 6,010 | 9.9 |
| ~6B × 60B tok | 885 | 2,804 | 4.6 |
| ~3B × 50B tok | 316 | 1,002 | 1.6 |
| ~1B × 20B tok | 51 | 160 | 0.3 |
| ~0.5B × 10B tok (architecture verification) | 13 | 40 | 0.1 |

> **C1+NVFP4 = 571 is the Phase B release wall-clock (43% of joint bf16 1,325).** 1,325 / 1,046 are bf16 operand controls; 729 is the C1+FP8 Hopper/Ada fallback. NVFP4 does **not** change 6NT ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). Release speedup is **2.0× vs bf16** (relative to FP8 1.5× then ×1.33, on the low end of NVIDIA 1.31–1.73× vs FP8); **4× is only an RTX PRO 6000 peak upper bound, not written into the recipe**. Distillation / upcycling cut required tokens a lot; the long-context stage is a small share and is counted separately.

### 15.2 Three low-budget routes (pick by actual GPU count)

- **Route A — architecture verification (cheapest, ≤ a few GPUs, ~10–50 H100-h, rentable)**: upcycle a small MiniCPM5 (fewer experts) → **0.5–1.5B small MoE**, install YOCO+CSA/HCA(+ optional KDA), continue train 10–20B tok. Goal: prove this attention/encoder-decoder runs, does not drop quality, and saves KV on long context. **Recommended default starting point.** The in-repo probe is [`docs/PLAN_VERIFY.md`](PLAN_VERIFY.md): `plan-probe` graph, short DummyStream train **Phase A→E** (A is offline surgery; do not go to F/G).
- **Route B — scale to the 12B target (~8×A100/H100 or one RTX PRO 6000; Phase B locked C1+NVFP4 ≈ 571 H100-h)**: MiniCPM5-2B → **12.25B** upcycle (optionally via a 3–6B milestone), continue train 50–60B tok + a short long-context stage, obtain the target model. Joint bf16 1,325 is control only.
- **Route C — PDSA extension (almost no training compute, fits existing work)**: freeze the backbone, train only small components (writer / reranker / threshold) + land "calibrated fallback / trainable editable memory" (§14). **Best zero-budget option**, directly produces PDSA's trainable-lifecycle follow-on.
- **Route D — 24B (later, not a current target)**: consider scaling only after a real cluster / compute grant.

### 15.3 Compute-saving levers (priority high to low)

1. **upcycling** (reuse MiniCPM5 weights, never from-scratch); 2. **distillation** (teacher=`openbmb/MiniCPM5-2B-Base`, fewer tokens);
3. **high-sparsity MoE** (fewer active params = fewer FLOPs); 4. **4K context is most of training**, long context is only a short stage;
5. **unfreeze curriculum** (§4.0 / [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md): **C1 locked** — both stacks MoE first, B0/B1 freeze Encoder and input embed, B0 freeze `lm_head` / B1 train `lm_head`, about 21% Phase B FLOPs saved, B1 Adam state 62%, Encoder activations ~38%; the end must be a short joint B2≥10B);
6. **NVFP4** ([`NVFP4_THEORY.md`](NVFP4_THEORY.md): **locked with C1 as C1+NVFP4** — linear GEMMs that are not must-bf16; B0 student / L0 / indexer / must-high-precision set stay high precision; release 2.0× vs bf16. Phase B wall-clock **571 H100-h, 43% of joint bf16 1,325**; 4× is peak upper bound only. Hopper/Ada fallback C1+FP8 = 729); 7. **Muon** (fewer steps; Newton-Schulz still fp32, orthogonal to NVFP4 GEMM); 8. **full train on new modules + LoRA on the rest** (less optimizer VRAM, fits smaller / fewer GPUs);
9. **rent spot GPUs for critical short runs** (no need to buy); 10. seq packing + activation recompute (fit fewer GPUs).

### 15.4 Revised default path

**L0 tiny correctness → plan-probe A→E ledger ([`PLAN_VERIFY.md`](PLAN_VERIFY.md)) → Route A (0.5–1.5B architecture verification, architecture paper) → Route B when there is budget (scale to the 12B target) → later (optional) Route D (24B).**
§1.2's **12B is the target spec**; when the budget is tight, **the current default is to run Route A first**. The §3 budget script can shrink `Nr_e/Nr_d` directly to a 0.5–1.5B band for verification.

---

## 16. Long-context feasibility (primary target 128K–256K; 1M is stretch)

> Set the tone first: **the primary delivery target is "the first 128K–256K usable"**, which is realistic at 12B; the 1M discussion below is stretch analysis of "can we even support it", **not** a bid to match 2T-frontier quality at 1M.

Three layers of conclusion; do not mix them:

### 16.1 Inference side: very promising (this architecture is built for 1M)

The bottleneck is KV cache. **YOCO caches once (single global cache) + CSA/HCA sequence compression** squeeze 1M KV from "does not fit" to "pocket change":

| 12B model KV cache at 1M tokens | Size |
| --- | --- |
| decoder-only + MHA (all 42 layers cached) | ≈344 GB (does not fit) |
| decoder-only + GQA-2 / MLA-576 | ≈43 / 48 GB |
| **YOCO + GQA-2 (single global cache, \(d_{\mathrm{kv}}=256\))** | **≈1.02 GB** |
| **YOCO + MLA (single global cache)** | **≈1.15 GB** |
| **YOCO + MLA + CSA \(m=4\) (global cache ÷4 along the sequence)** | **≈0.29 GB** |
| YOCO + MLA + extra sequence÷8 (stretch, not the CSA default) | ≈0.14 GB |
| (plus 26-layer 8K sliding-window branch, independent of N) | +≈0.25 GB |

On top of that, **encoder (self-decoder) prefill early-exit** — ultra-long input only needs to finish the encoder to emit the global cache, not run every layer — so 1M **prefill is cheap too**. That is the point of "light on the input side (≈2.03B)". Middle-band prefill activation share is about 32%. **Decode side**: if the global cache is not compressed, 26-layer cross-attn at 128K/256K is about 3.2× / 6.5× the decoder MLP; you must use CSA \(m=4\) or top-\(k\) selection, or the encoder's saved compute is paid back in cross-attn (see theory verification §6). The original YOCO paper reports near-perfect needle retrieval at 1M. **So 1M inference is feasible on mid-range hardware.**

### 16.2 Training a model that "supports 1M (needle/RULER pass)": realistic, but it takes care

Do not pretrain at 1M; main training is 4–8K, then a **progressive long-context extension stage** at the end (8K→32K→128K→256K→1M). Key points:
1. **RoPE/YaRN scaling** to the target length; 2. **long data**: books / repo-scale concat + synthetic long dependency + IN2 mid-context samples; 3. **progressive length curriculum**;
4. **The train-time VRAM bottleneck is 1M-sequence activations** (not KV) → use **context/sequence parallelism**; YOCO early-exit + CSA/HCA compression cut activations a lot; **the asymmetric design is a natural fit** (encoder handles long input, decoder generates short → long prefill is cheap);
5. **Verify**: RULER-1M / needle-in-haystack. This stage is not many tokens (a few B to low tens of B), cheap relative to main training; short rented-GPU runs are fine.

### 16.3 Matching frontier 1M quality: not reachable on a small budget

DeepSeek/Kimi 1M is "full use" fed by 32T-scale data + large compute. A small budget can get **"supports 1M + decent needle/RULER"**, but "frontier multi-hop deep reasoning at 1M" is not realistic — say so plainly.

### 16.4 Low-budget practical advice

- **Stage it**: first stabilize **128K–256K** (cheap, enough), then separately push **1M capability** and verify with needle/RULER; do not start at 1M.
- **The PDSA route is a cheaper substitute for "effective 1M"**: bounded editable memory + calibrated sparse fallback + retrieval (§14), **without training native 1M attention**, still gets long-range recall — this team's own work, and §5 measurements show bounded selection already beats reading the full document at 8.2k. For extreme long context, that may be a better deal than hard-training native 1M attention.
- On milestones, put 1M under **M4** (long context), as a capability target not a quality target.
- **Use RL later to raise "usability"**: pretrain/extension only solves "can swallow 128K–256K"; **whether it is actually usable** depends a lot on later **long-context RLVR** (see §4 Phase G) — verifiable long-doc tasks + grounding reward to optimize mid-context recall / long-instruction following directly, and PS-PPO / PDSA evidence selection to keep long-rollout cost down. That is the key lever for another usability step on a small budget.

> **Positioning (important, to avoid misunderstanding)**: this project is **not** using 12B to match 2T-class frontier quality — that is impossible.
> **The primary delivery target is "the first 128K–256K of long context usable"** (at 12B scale this is plausible in both theory and engineering);
> 1M is only a bonus of "the architecture can support it + needle/RULER can pass", not a quality target.
> For more extreme long context on a budget, prefer PDSA memory + retrieval fallback (§14) over hard-training native 1M attention.

> One line: **the main battlefield = 128K–256K usable; 1M inference/needle is stretch; 2T-class frontier quality is not in scope.**

---

## References (architectural basis for this plan)

- **DeepSeek-V4** (CSA/HCA, mHC, Muon, MTP, Hash-MoE bootstrap; V4-Flash 284B/13B, 1M ctx, 32T tokens): arXiv `2606.19348`; HuggingFace `transformers` `deepseek_v4` model docs (`layer_types`, `compress_rates`, `sliding_window`, `index_topk`, `mlp_layer_types`, and related config).
- **DeepSeek Sparse Attention (DSA)** and **FlashMLA** sparse kernels (Lightning Indexer + top-k + FlashMLA): DeepSeek-V3.2 report; `deepseek-ai/FlashMLA`.
- **Native Sparse Attention (NSA)** (compress + select + sliding-window three-branch, hardware-aligned, natively trainable): arXiv `2502.11089`.
- **Upcycling LLMs into MoE** (virtual-group init, weight scaling, softmax-then-topK; Megatron `upcycling_utils.py`): arXiv `2410.07524`.
- **DeepSeekMoE** (fine-grained experts + shared experts): arXiv `2401.06066`.
- **MiniCPM5-2B** (Apache-2.0 Llama GQA: d=2048, 42 layers, 16 Q / 2 KV, `head_dim=128`, V=130560, untied, no μP, native 128K / `rope_theta=5e6`; WSD schedule follows MiniCPM-family practice): `openbmb/MiniCPM5-2B` (tokenizer) / `openbmb/MiniCPM5-2B-Base` (upcycle / teacher). Do not use MiniCPM-2B-sft-bf16 (GML) or MiniCPM3/4 tokenizers.
- **Gemma 2 / Qwen3-Next** (local sliding window × global attention interleaving, hybrid attention layer ratios): reference for layer schedule and window design.
- **YOCO — You Only Cache Once** (decoder-decoder: self-decoder emits a single global KV cache, cross-decoder reuses it; prefill early-exit; near-perfect needle at 1M ctx): arXiv `2405.05254`; `microsoft/unilm` YOCO.
- **Kimi Linear / KDA** (Kimi Delta Attention: fine-grained gated Gated-DeltaNet + DPLR chunk kernel; 3:1 KDA:MLA mix, MLA uses NoPE; 1M KV cache ↓~75%, decode ↑~6×): arXiv `2510.26692`; `MoonshotAI/Kimi-Linear`.
- **Hybrid Linear Attention systematic analysis** (linear attention is weak at recall and needs full layers to compensate; gated-delta reaches Transformer-level recall at 3:1~6:1): arXiv `2507.06457`.
- **Lost in the Middle** (mid-context positional bias, present in softmax too, relieved by position-encoding calibration): arXiv `2307.03172`.
- **FILM / IN2 training** (information-intensive long-context training; synthesize "key information in the middle" samples to fix lost-in-the-middle): `Make Your LLM Fully Utilize the Context`, arXiv `2404.16811`.
- **OLMo 2 / Gemma 2** (QK-Norm, dual RMSNorm, logit soft-capping, z-loss, and other stability tricks): arXiv `2501.00656` / `2408.00118`.
- **EAGLE / speculative decoding** (reuse an MTP head for self-speculation speedup): arXiv `2401.15077`.
- **YaRN** (RoPE long-context extrapolation scaling): arXiv `2309.00071`.
- **PS-PPO — Prefix-Sampling PPO** (critic-free RLHF backprops only the sampled prefix with unbiased truncation; cuts long-trace RL compute/VRAM): arXiv `2606.29758`.
- **PDSA / Memory-Managed Long-Context Attention** (bounded editable memory + hard lifecycle overwrite/protection/eviction + query-independent writer + query-aware read + calibrated sparse fallback; measured "no write-time signal" bound, bounded selection beats reading the full document on long text): Zou & Donz, arXiv `2606.28876` (this team's work; its "next step" is a trainable lifecycle, which this plan takes up in §14).
- **MSA — Memory Sparse Attention** (static document sparse memory, PDSA's nearest neighbor): arXiv `2603.23516`.
- **Gated DeltaNet / Gated DeltaNet-2** (KDA's predecessor; decoupled erase and write): arXiv `2412.06464` / `2605.22791`.
