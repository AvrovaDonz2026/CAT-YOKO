# CAT-YOKO Training Plan: Causal Encoder-Decoder (YOCO-style) Hybrid-Attention MoE Upcycled from MiniCPM5-2B

> Goal: using OpenBMB **MiniCPM5-2B (Apache-2.0, Llama GQA)** as the base, train a **Causal Encoder-Decoder (YOCO / "You Only Cache Once" decoder-decoder)** model:
> **≈12.25B total parameters** (already reduced from 24B to match the compute budget); asymmetric activation **default: Encoder processes input ≈2.03B active/token, Decoder generates output ≈4.33B active/token**
> (more compute-efficient tier ≈1.43B/3.02B, near-dense tier ≈3.04B/6.29B: see §3);
> attention uses **DeepSeek-V4-Flash-style CSA + HCA compressed attention + an 8K large sliding window**; native long context (YOCO single global KV cache).
> Switching the base from MiniCPM-2B-sft-bf16 (GML) to MiniCPM5-2B is for **Apache-2.0 license alignment**, not a change of curriculum. CAT-YOKO code and derived weights are likewise **Apache-2.0**.
>
> ⚠️ **Key**: total parameters mainly affect **VRAM/storage**; **training compute ∝ active parameters × tokens**. To actually cut training cost you must cut **activation** (pick a more compute-efficient tier), not just cut total parameters.
>
> This document is an executable engineering training plan. **Implementation defaults are frozen** in [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)
> / `cat_yoko.config.CATYokoConfig.middle_12b()`: causal 16/26, C1 full MoE, C1+NVFP4, Phase B sliding window + gated
> cross-attn, AdamW; **KDA / mHC / MTP / Muon / first-layer dense / M2 are not published defaults**. Remaining tiers and ablations in this document are sensitivity analyses, not training-code switch defaults.
> **Current training progress** (Hub overlay step, which GPUs have been reclaimed) lives in [`docs/STATUS.md`](STATUS.md); do not look for step pins in this document.

---

## 0. Requirements and key assumptions

Your original description has several points that need explicit confirmation. This plan proceeds under the interpretations below; please flag anything that diverges from your intent:

| Your wording | This plan's interpretation | Notes / adjustable items |
| --- | --- | --- |
| `based on openbmb minicpm2b` | Base = **MiniCPM5-2B** (Apache-2.0, dense Llama GQA: 42 layers, hidden 2048, FFN 6144, 16 Q / 2 KV, `head_dim=128`, vocab 130560, **untied** embedding) | Native 128K context (`rope_theta=5e6`); no need to switch to MiniCPM-2B-128k |
| `same as deepseekv4.1 flash` | Match the architectural paradigm of **DeepSeek-V4-Flash** (CSA/HCA hybrid attention + DeepSeekMoE + mHC + Muon + MTP + Hash-MoE bootstrap) | Official V4-Flash is 284B/13B decoder-only; we are doing a **same-architecture replica scaled down to 12B and converted into a YOCO encoder-decoder (asymmetric activation)** |
| `Causal-Encoder-Decoder, input activation 3b, output activation 6b`; later **`cut to 12B`** | **YOCO-style decoder-decoder**: **Encoder=self-decoder** processes input and produces a **single global KV cache**; **Decoder=cross-decoder** generates output and cross-attends to that global cache. **Total parameters ≈12.25B** (named `CAT-YOKO-12B`); activation default is the **middle tier (~2.03B-in/~4.33B-out)**; more efficient 1.43/3.02 and near-dense 3.04/6.29: see §3 | Activation asymmetry comes from **two physically distinct stacks** (smaller encoder, larger decoder); **the three tiers only change top-k, not expert count**. See §2, §3 |
| `large sliding-window attention 8k` | The **uncompressed sliding-window branch** retained in each CSA/HCA layer, `n_win = 8192` | DeepSeek-V4 default is `n_win=128`; 8K is a clear increase, higher cost but better local fidelity |
| `CSA HCA` | **Compressed Sparse Attention** + **Heavily Compressed Attention** (DeepSeek-V4's two compressed-attention types, interleaved across layers) | See §2 |

> ✅ **Spec confirmed for this round**: Causal Encoder-Decoder (YOCO-style); **total parameters 12.25B** (reduced from 24B); activation default **≈2.03B-in / ≈4.33B-out**.
> **Published recipe is frozen** ([`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md) / `cat_yoko.config`): causal 16/26, C1 full MoE, C1+NVFP4 wall-clock 571 H100-h (target RTX PRO 6000 / 6000D), Phase B sliding window + gated cross-attn, M2/KDA/mHC/MTP/Muon off.
> The encoder is **not** bidirectional. "YOKO" in the repo name **CAT-YOKO** corresponds to **YOCO**.

> ⚠️ **Important reality check (read first)**: Fully replicating this architecture and pretraining from 32T-scale data is **frontier-lab-scale** engineering.
> Using MiniCPM5-2B as the base for **upcycling + continued training** can bring cost down to the hundreds-of-B-tokens scale,
> but still requires sustained compute of tens to over a hundred H100/H800-class GPUs. This plan is designed around the "**upcycling conversion + continued pretraining**" route,
> which is the only realistic path to a usable model of this architecture within a controllable budget (from-scratch 32T pretraining is not in the recommended range).

---

## 1. Base model and target spec

### 1.1 MiniCPM5-2B (base, from official config; Apache-2.0)

| Item | Value |
| --- | --- |
| Layers `num_hidden_layers` | 42 |
| Hidden dim `hidden_size` | 2048 |
| FFN intermediate dim `intermediate_size` | 6144 (SwiGLU; `6144/2048=3`) |
| Attention heads | 16 Q / 2 KV **GQA** (Llama; `num_key_value_heads=2`) |
| `head_dim` | 128 (`d / n_q`) |
| `d_kv` | 256 (`n_kv · head_dim`) |
| Vocab `vocab_size` | 130560 |
| Activation | SiLU / SwiGLU |
| Position encoding | RoPE, `rope_theta=5e6` (native 128K context) |
| RMSNorm `rms_eps` | 1e-6 |
| Embedding | **untied** (input table and `lm_head` are not shared) |
| License | **Apache-2.0** (vs MiniCPM-2B-sft-bf16's GML; publishable and redistributable) |
| Special design | **no μP** (Llama identity residual; logits not divided by 9); WSD LR schedule follows MiniCPM-family practice |

> MiniCPM5 is Llama; **do not** transplant MiniCPM-2B μP constants (`scale_emb`, `scale_depth/√L`, logits `/ (d/dim_model_base)`).
> After splitting 42 layers into 16+26 the residual multipliers remain identity (do not invent √16 / √26). See the theory verification for the check.

### 1.2 Target model `CAT-YOKO-12B` (Causal Encoder-Decoder / YOCO-style)

| Item | Recommended value (default middle tier) | Notes |
| --- | --- | --- |
| Architecture | **YOCO decoder-decoder**: Encoder(self-decoder) → global KV cache → Decoder(cross-decoder) | See §2 |
| Total parameters | **≈12.25B** (storage) | See §3 budget |
| **Encoder** active/input token | **≈2.03B** (tiers 1.43/2.03/3.04) | 16 layers, MoE, CSA/HCA+8K sliding window, produces the global cache |
| **Decoder** active/output token | **≈4.33B** (tiers 3.02/4.33/6.29) | 26 layers, MoE, self-attention + **cross-attn to the global cache** |
| Hidden dim | 2048 (keep the base; enc/dec match so the vocab and warm-start can be shared) | |
| FFN | **DeepSeekMoE** fine-grained experts, `moe_intermediate_size=2048` | Enc: 1 shared+20 routed, top-k 7; Dec: 1 shared+20 routed, top-k 10 (default tier); **first-layer dense off** |
| Attention | **CSA/HCA hybrid + 8K sliding window**; enc includes long-range compression; dec cross-attn reuses the single global cache; **optional overlay of KDA linear attention as a 3:1 three-way mix** | §2 / §2.5; Phase B is still sliding-window GQA |
| KV cache | **single global cache (You Only Cache Once)**, `d_kv=256` + compression → O(N)-scale memory | the key long-context gain |
| Residual | **mHC** (optional; first run ordinary residual to completion) | stability enhancement |
| Training objective | main CE + **MTP** auxiliary head (optional) | |
| Optimizer | **Muon (2D weights) + AdamW (emb/norm/router/bias)** | published default AdamW |
| Context | **primary delivery target: 128K–256K usable**; staged 4K → 8K → 32K → 128K → 256K; 1M is stretch (inference support + needle) | MiniCPM5 native 128K + 8K sliding window + compressed long-range + YOCO |

---

## 2. Architecture: YOCO-style Causal Encoder-Decoder + CSA/HCA compressed attention

### 2.0 Overall skeleton (decoder-decoder / YOCO)

CAT-YOKO consists of two causal stacks. Behaviorally it is equivalent to a decoder-only Transformer, but it "caches only once":

```
input tokens ─► [Encoder = Self-Decoder, 16 layers, ≈2.03B active/token]
                 │  efficient causal attention (CSA/HCA + 8K sliding window), compressing long-range per layer
                 ▼
          top hidden states  ──►  produce [single global KV cache  K̂, V̂] (You Only Cache Once; d_kv=256)
                 │
                 ▼
        [Decoder = Cross-Decoder, 26 layers, ≈4.33B active/token]
           each layer = efficient causal self-attn (generation sequence, sliding window)  +  Cross-Attn(→ K̂,V̂)  +  MoE-FFN
                 ▼
             RMSNorm ─► LM Head (untied) ─► next token
```

Key properties (from YOCO; formalization and causality proofs in [`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)):
- **Cache only once**: only the one global cache produced at the encoder top is reused by all cross-decoder layers; KV-cache memory drops from `O(N·L)` to about `O(N)`. CSA/HCA inside encoder layers (**M1**) does **not** automatically reduce this cache's slot count; pooling along the sequence (**M2**) is what drops slots from \(N\) to \(N/m\).
- **Prefill can early-exit (inference only)**: the prompt only needs to finish the encoder to write the global cache; the decoder runs once on the last position to emit the first token. Training is still a full-length forward of both stacks. This is exactly the motivation for "**a lighter input side (≈2.03B)**". Middle-tier encoder activation share is about 32% (see `docs/THEORY_VERIFICATION.md` §6).
- **Asymmetric activation comes from two physical stacks**: the encoder is smaller (16 layers, fewer experts/activation → ≈2.03B), the decoder is larger (26 layers, includes cross-attn, more experts/activation → ≈4.33B). **The three tiers only change top-k** (low 4/6, middle 7/10, near 12/16), not the 1+20 expert instances.
- **Global attention is retained**: globality comes from the decoder **cross-attn reading the cache (M3)**, not from stacking encoder sliding-window receptive fields (16×8K=131072, which does not cover 256K and does not need to).

> Design tradeoff: the original YOCO paper uses sliding-window or gated retention in the self-decoder. We replace the self-decoder's
> intra-layer attention with **CSA/HCA + 8K sliding window (M1)**, lowering the encoder's quadratic term when \(n\gg 8K\), and optionally mixing the write representation over long range.
> **A more compact global cache is M2 (optional pooling); query-aware retrieval at generation time is M3 (decoder-side indexer / calibrated fallback)** — not the same thing as M1.
> Decoder **self-attention also sees the full-length sequence at training time**, so it must use sliding window / KDA; "generation sequences are usually not long" describes inference only. Long range always goes through cross-attn.

### 2.1 CSA / HCA / 8K sliding window (attention details)

DeepSeek-V4 replaces V3's full MLA attention with **two compressed-attention types interleaved layer by layer**. The core goal is to cut long-context attention cost from
`O(L²)` to approximately `O(L·k)`, and to heavily compress the KV cache. This compressed attention is used **inside the Encoder (self-decoder)**, and also for its self-attention. Three layer kinds:

### 2.2 Three layer types (`layer_types`)

1. **Sliding-window (bootstrap layers)**: local sliding-window causal attention only; window = `sliding_window`; no long-range branch. The encoder uses this for the **first 2 layers** (frozen; 16 layers is just enough to fit 7+7 CSA/HCA).
2. **CSA (Compressed Sparse Attention)**
   - Compress every `m=4` tokens of KV into 1 entry (with learnable compression weights `Z` and a position bias; overlapping window).
   - Use a **Lightning Indexer** to score the query against compressed entries and take **top-`index_topk`** (default 512) entries into attention (i.e. **DSA** on the compressed sequence).
   - Additionally concat an **uncompressed sliding-window K/V branch** (size `n_win`) to keep local detail.
3. **HCA (Heavily Compressed Attention)**
   - Compress every `m'=128` tokens into 1 entry (non-overlapping); **no indexer**; **dense attention** over all compressed entries.
   - Likewise concat an uncompressed sliding-window branch.

> Implementation notes (aligned with the official reference): both CSA and HCA **concat raw sliding-window K/V** with **compressed K/V** along the sequence axis,
> build a combined mask, then run **one** standard masked attention; CSA's mask is filtered by `top_k`, HCA's mask is fully visible.
> The only differences are "compression rate" and "whether top-k".

### 2.3 Attention configuration for this project (recommended)

| Parameter | Recommended value | Matching DeepSeek-V4 name |
| --- | --- | --- |
| `sliding_window` (`n_win`) | **8192** | the requested "8K large sliding window"; must be \(\ge m'=128\) so it can fill holes in HCA's own blocks |
| Encoder `layer_types` | **2× sliding bootstrap, then CSA:HCA=1:1** → 16 layers = **2 sliding + 7 CSA + 7 HCA** | V4-Flash starts with 2 sliding layers; 16 layers cannot do "3 bootstrap + 1:1" |
| Decoder self-attn | **all sliding** (optional later KDA mix); **CSA is not laid out by default** | global mixing already lives in cross-attn (M3) |
| CSA compression rate `m` | 4 | `compress_rate_csa` |
| HCA compression rate `m'` | 128 | `compress_rate_hca` |
| `index_topk` (CSA, **M1**) | 256～512 | Lightning Indexer inside encoder layers; **do not** reuse on decoder queries |
| Decoder-side selection (**M3**) | needed from 128K up: dense→top-k or calibrated fallback | the real generation-time retrieval; see architecture theory §3 |
| Attention base | Phase B = MiniCPM5 **GQA 16 Q / 2 KV** (`d_kv=256`); CSA/HCA can further compress to MQA-128. **Do not** use a 1.25×MHA placeholder ledger | V4 implements DSA in MLA's MQA mode; the published ledger is GQA |

> **Cost of the 8K sliding window**: the window branch is uncompressed `O(L·n_win)` cost; `n_win=8192` is 64× the default 128,
> so local-attention overhead rises substantially. If long-context throughput is tight, once long-range capability is sufficient you can roll `n_win` back to 2K–4K,
> or use 8K windows on only some layers. Recommend an `n_win ∈ {2K,4K,8K}` ablation in §7.

### 2.4 Initialization: migrating MiniCPM5 GQA to CSA/HCA + Encoder/Decoder

MiniCPM5 is a 42-layer decoder-only **Llama GQA** (16 Q / 2 KV, `head_dim=128`); it has no MLA latent, no compressor/indexer, and no cross-attn. Migration approach:

- **Split into two stacks**: cut MiniCPM5's 42 layer weights into Encoder(16) + Decoder(26). Keep the encoder at 16 layers so **2+7+7 CSA/HCA** still fits; the extra 2 layers all go to the decoder (see §4 Phase A).
- **Attention backbone**: `q/k/v/o` projections inherit **GQA** weights (K/V is `(d_kv, d)=(256, 2048)`, not full MHA). If cutting to MLA, SVD-factor the K/V projections into low-rank `W^{DKV}·W^{UK/UV}` for init; same idea for `q_lora_rank`.
- **Decoder cross-attn**: `q`/`o` projections can be initialized from the matching self-attention `q`/`o` (**cross counts only Q/O = \(2d^2\)**; K/V come from the YOCO cache). **Bypass cross-attn first (gate≈0) then open it gradually** to stabilize training. At gate=0 the 16/26 split is equivalent to the original 42-layer residual stream (architecture theory Theorem A).
- **New modules** (compressors `W^{aKV}/W^{bKV}/W^{aZ}/W^{bZ}`, position bias `B`, Lightning Indexer): small-scale random init; **dense-align first, then sparsify** (see §4 Phase C).

### 2.5 Optional enhancement: adding KDA linear attention (three-way mix)

**Motivation and a misconception that must be cleared up**: intuition says "adding KDA linear attention will keep CSA/HCA from dropping information in the **middle** of the context".
The literature actually concludes the opposite — **linear attention (including KDA) is precisely the weakest link for "precise mid-context recall"**: its fixed-size RNN state collides in memory,
and needles that fall outside the local window are easily lost (arXiv 2507.06457, LoLA). Hybrid models recover recall by **keeping full/sparse attention layers**
to carry the retrieval path, not by relying on the linear layers. Moreover **lost-in-the-middle is essentially a position bias** (softmax models have it too),
mainly mitigated by RoPE/NoPE calibration; adding linear attention does not directly fix it. **Therefore what preserves precise mid-context retrieval is CSA's top-k plus a few full anchors, not KDA.**

**Then why still recommend introducing KDA?** Two complementary gains:
1. **Efficiency leverage (primary)**: Kimi Linear uses a **3:1 KDA:MLA** mix; 1M-context KV cache ↓~75%, decode ↑~6×, and quality does not drop — it rises.
   Replacing most layers with KDA and keeping a few CSA/HCA layers can substantially cut the model's long-context cost.
2. **"Safety-net coverage" for CSA's selective misses (secondary)**: CSA's risk is that the Lightning Indexer's top-k **misses** a relevant mid-context block.
   KDA is a **no-top-k, order-sensitive path that writes every token into state**, providing gist-level full-sequence coverage as a safety net when selection misses
   (note: coarse coverage; it does not replace precise retrieval).

**Recommended configuration (optional; decide via ablation)**:
- **Encoder (self-decoder) three-way mix**: about **3:1 KDA : (CSA/HCA)**, e.g. every 4 layers `[KDA, KDA, KDA, CSA]`, insert 1 HCA every few groups,
  and keep **1–2 high-`index_topk` CSA layers or true full-attention layers as "recall anchors"** (hybrid-linear: gated-delta class reaches Transformer-level recall at 3:1~6:1).
- **Position encoding**: KDA uses learned decay for position / recency; full/CSA anchor layers may consider **NoPE** (Kimi Linear practice).
- **Decoder (cross-decoder)**: self-attention uses sliding window / KDA (training sequences can be long; do not revert to full attention); cross-segment retrieval is delegated to cross-attn → global cache (M3).

**Cost / caveats**:
- KDA needs an extra **DPLR chunked kernel** + **independent recurrent-state** management (orthogonal to YOCO "cache only once": YOCO saves KV cache; KDA state is a small per-layer state).
- Known pitfall of hybrid conversion: **the model may learn to ignore the linear path** (if you first train strong CSA/HCA retrieval, then replace most layers with KDA). So **implement the 3:1 graph first (B, still sliding window), then light KDA→CSA→HCA (C)**. Do not change layer types only at C. HCA has the fewest layers and is write-first; light it last. Published default remains `use_kda=False`.
- In parameters, KDA layers are usually cheaper than GQA/CSA (no large KV projections); replacing some CSA/HCA layers with KDA slightly drops per-stack parameters, so §3 scripts must be recomputed at the actual KDA dims and routed expert count must be topped back up to 12.25B.

> Conclusion: **worth adding, but the role is "efficiency + safety-net coverage", not "preserve precise mid-context retrieval"**. Whether to enable it, and the exact KDA:CSA:HCA ratio, is decided by §7 ablations.

---

## 3. Parameter budget derivation (`CAT-YOKO-12B`, Encoder-Decoder split)

Base dims `d=2048`, `vocab=130560`, **untied** embedding (one copy each of the input table + `lm_head`). Encoder 16 layers, Decoder 26 layers (42 total, MiniCPM5 depth kept; the extra 2 layers go to the decoder).
`moe_intermediate_size=2048` (DeepSeek-V4 same order of magnitude, hardware-friendly; dense FFN 6144 is divisible by 2048), single-expert SwiGLU ≈ `3·d·2048 ≈ 12.58M`.
Attention is ledgered as **GQA 16/2** (`2d² + 2d·d_kv`), **not** 1.25×MHA. YOCO cache `d_kv=256`. Decoder cross counts only Q/O = \(2d^2\).

### 3.1 Budget table (12.25B total params; the three activation tiers only change top-k)

Single-expert SwiGLU ≈ `3·d·2048 ≈ 12.58M`; Embedding (untied) ≈0.27B + lm_head ≈0.27B; all three tiers have **12.25B** total parameters, differing only in activation / sparsity.

| Tier | Enc experts (shared+routed, top-k) | Dec experts (shared+routed, top-k) | Enc active/input | Dec active/output | Sparsity enc/dec | Training compute (×50B tok) |
| --- | --- | --- | --- | --- | --- | --- |
| Low-compute | 1+20, top-k **4** | 1+20, top-k **6** | 1.43B | 3.02B | 23.8% / 33.3% | 926 H100-h |
| **Default (middle)** | **1+20, top-k 7** | **1+20, top-k 10** | **2.03B** | **4.33B** | **38.1% / 52.4%** | **1,325 H100-h** |
| Near-dense | 1+20, top-k **12** | 1+20, top-k **16** | 3.04B | 6.29B | 61.9% / 81.0% | 1,943 H100-h |

Fixed parts (same across the three tiers): Enc self-attention ≈0.15B, Dec self-attention ≈0.25B, Dec cross-attn ≈0.22B (Q/O), emb ≈0.27B, lm_head ≈0.27B.
All three tiers have **882 expert instances** (16×21 + 26×21; only top-k changes), so total params are the same 12.25B and training cost varies only with activation. Under untied, the plan's \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) and one forward (emb + lm_head each counted once) are both **6.36B → 1,325 H100-h**.

> 🔑 **Total parameters ≈ VRAM/storage; training compute ∝ activation × tokens.** All three tiers are 12.25B total, but training cost follows **activation** (926 → 1,325 → 1,943 H100-h @ 50B tok).
> Default is the middle tier (enc 2.03B / dec 4.33B) — a compromise of "1.4 is too little, 3.0 is too much". Layer split (16/26), `moe_intermediate_size`,
> and per-stack shared/routed stay fixed at 1+20; **the only tier knob is top-k**. Attention is ledgered as MiniCPM5 GQA, not 1.25×MHA.
> Also: at 12.25B, GQA attention is **~5.0%** of total params (MoE ~90.6%); **adding KDA still does not change the total-param budget** (see §2.5). Itemized recalc, KV/FLOPs, and the claim ledger: [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md).

### 3.2 Recalc script (`scripts/param_budget.py`, default middle tier)

```python
d, V, moe_int = 2048, 130560, 2048
kv_dim = 256                   # n_kv * head_dim = 2 * 128
emb    = V*d                   # untied input table
lm_h   = V*d                   # untied lm_head
expert = 3*d*moe_int           # SwiGLU single expert
attn   = 2*d*d + 2*d*kv_dim    # GQA 16Q/2KV, not 1.25×MHA
cross  = 2*d*d                 # per-layer cross Q/O; K/V from YOCO cache (d_kv=256)

# default middle tier (12.25B / ~2.03B-in / ~4.33B-out); the three tiers only change top-k
Le, ns_e, tk_e, Nr_e = 16, 1, 7, 20   # Encoder = self-decoder
Ld, ns_d, tk_d, Nr_d = 26, 1, 10, 20  # Decoder = cross-decoder
# low-compute tier: top-k 4/6    near-dense tier: top-k 12/16

enc_act = emb + attn*Le          + Le*(ns_e+tk_e)*expert
dec_act = lm_h + (attn+cross)*Ld + Ld*(ns_d+tk_d)*expert
total   = emb + lm_h + attn*Le + Le*(ns_e+Nr_e)*expert \
              + (attn+cross)*Ld + Ld*(ns_d+Nr_d)*expert
print(f"enc_active(input)={enc_act/1e9:.2f}B "
      f"dec_active(output)={dec_act/1e9:.2f}B total={total/1e9:.2f}B")
```

Full recalc (KV / FLOPs / three-tier comparison / spec asserts) is authoritative in `scripts/param_budget.py`; derivation in [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md):

```bash
python3 scripts/param_budget.py --full
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
python3 scripts/param_budget.py --nvfp4
```

---

## 4. Staged training recipe (core)

Overall approach: **upcycling + surgical conversion + staged continued training**, using MiniCPM5-2B's existing capability as a warm start.
Introduce only one large change at a time and let the model recover, avoiding collapse from too many simultaneous edits. Token budgets are order-of-magnitude suggestions, not calendar time.

```
Phase A  Architecture surgery and initialization        —— 0 token (offline weight transform)
Phase B  Upcycling recovery continued pretraining       —— 50–150B token @ seq 4K (dense/sliding-window attention, no sparsity yet)
Phase C  Attention sparsification alignment             —— 20–50B token (default Indexer→top-k→HCA; with `use_kda` do C-kda first, then CSA/HCA)
Phase D  Long-context extension                         —— 20–60B token (8K→32K→128K, stepwise RoPE scaling)
Phase E  WSD anneal / high-quality data                 —— 20–50B token (LR decay segment; stack math/code/long-form)
Phase F  SFT                                            —— 1–10B token (instruction + long context + tools)
Phase G  RL (GRPO / optional DPO)                       —— batched by domain
(optional)  MTP-head joint training                     —— hang one MTP head from Phase B, weight 0.1–0.3
```

### 4.0 Train stacks / layers separately then merge? — Feasible, but do not by default split into two independent LMs

YOCO is not seq2seq: at training time **the same sequence passes through Encoder then Decoder**, and the loss sits at the Decoder top. Encoder and Decoder representations were already jointly trained inside MiniCPM5's 42 layers; Theorem A ([`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)) says a gate=0 16/26 split **is** that residual stream. Treating the two stacks as two independent LMs, training them separately, then stitching them together, throws that alignment away. **No franken-merge.**

Full freeze boundaries, gradient truncation, untied embedding, optimizer/activation memory, and split sensitivity: [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md). NVFP4 module policy and wall-clock: [`docs/NVFP4_THEORY.md`](NVFP4_THEORY.md); Hopper/Ada FP8 fallback: [`docs/FP8_THEORY.md`](FP8_THEORY.md). Numbers: `python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4`.

**The Phase B recipe is frozen as C1+NVFP4**: C1 freeze boundary (Phase A both stacks MoE → B0/B1 freeze Encoder → B2 short joint) × mixed NVFP4. Delaying Encoder MoE, whole-phase 2.0×, and peak 4× are sensitivity only and do not enter the recipe. Joint bf16 is the 100% control only. C1+FP8 is the fallback when there is no Blackwell.

Middle tier 50B tokens, untied, emb and lm_head each counted once (same as one forward):

| Approach | H100-h | vs joint 50B |
| --- | ---: | ---: |
| Train both stacks together (joint bf16 control) | 1,325 | 100% |
| Freeze Encoder; train only Decoder + cross-attn (entire run; no co-adaptation) | 987 | 75% |
| Train only new modules (cross-attn + \(W_K/W_V\); backbone frozen) | 720 | 54% |
| C1 unfreeze curriculum 8+27+15B (bf16 operation-count ledger) | 1,046 | 79% |
| **C1+FP8 (Hopper/Ada fallback)** | 729 | 55% |
| **C1+NVFP4 (frozen)** | **571** | **43%** |
| Encoder as independent LM 50B + frozen-Encoder train Decoder 50B + 20B stitch recovery | 1,940 | **146% (more expensive)** |
| Encoder first as LM 25B then joint 25B | 874 | 66% (**quality gamble**: is half the joint tokens enough to recover) |

**Conclusions:**

1. **"Train two models separately then weld them together" does not save compute.** The same 50B/stack plus stitch recovery is 1.5× joint training. Caching Encoder hidden states for the Decoder is also unrealistic (50B tokens × \(d\) × 2 bytes ≈ 205 TB).
2. **"On the same cut weights, unfreeze trainable subsets in layers" is what saves.** What you save is the Encoder backward (and Decoder-backbone weight gradients in the new-module stage), not skipping the Encoder forward — YOCO's CE sits on the Decoder, so the Encoder forward cannot be skipped.
3. Freezing the Encoder saves about **25% FLOPs**, but the Encoder no longer rewrites memory for the Decoder's queries (write/read do not co-adapt). PDSA's "no write-time signal" also suggests: train only the reader and freeze the writer forever, and the retrieval ceiling sticks. So freezing the Encoder can only be the **middle of Phase B**; the end must have a short joint (B2 ≥ 10B, default 15B).
4. Training only cross-attn / indexer (Phase A warmup, Phase C step 1) is cheapest (~54%); this is already written into Phase C, not a new invention.
5. Greedy layer-by-layer growth (2 layers → freeze → add 2 more) has no stable compute-saving evidence on LLMs, and still needs a final joint finetune; **do not**.
6. DeepSeek-style "per-domain experts each SFT+RL then distill" applies only to **Phase F/G post-training**, not to this 12B pretraining skeleton.
7. **The C1 freeze boundary is frozen; wall-clock is further nailed as C1+NVFP4.** Phase A applies virtual-group MoE to both stacks; B0/B1 freeze the Encoder: one offline surgery, token 0 is already the 12.25B middle tier, B2 only unfreezes. On B1/B2, all linear GEMMs that need not be bf16 (MoE, attn QKV/O, lm_head) + frozen Encoder forward run NVFP4; B0 student stays bf16. Published wall-clock **571 H100-h** (**43%** of joint bf16 **1,325**; target RTX PRO 6000 / 6000D). Do not delay Encoder upcycling until B2, and do not apply 2.0× to the B0 student (that is a sensitivity control, not in the recipe).

**Freeze rules (Theorems D/E, C1 frozen; MiniCPM5 is untied; write these into the trainer at implementation time, not as a verbal convention):**

| Item | B0 | B1 | B2 |
| --- | --- | --- | --- |
| Global cache | `X^{16}.detach()` then multiply \(W_K,W_V\) | same as left | **remove** detach |
| \(W_K,W_V\) | new modules, trainable (\(2\,d\,d_{\mathrm{kv}}=1.05\mathrm{M}\), \(d_{\mathrm{kv}}=256\)) | trainable | trainable |
| Input embedding \(E\) | **frozen** | **frozen** | unfreeze |
| `lm_head` | **frozen** | **trainable** (untied; Theorem E does not govern the head) | unfreeze |
| Encoder weights | frozen (already virtual-group MoE) | frozen | unfreeze; experts specialize from here |
| Decoder | freeze backbone, train only cross-attn | unfreeze self-attn+MoE | unfreeze |

Forbidden: still training the input embedding \(E\) while the Encoder is frozen (input distribution drift, Theorem E). **B1 may train `lm_head`** — MiniCPM5 is neither tied nor shared.

**Frozen recipe (when budget is tight, replace "Phase B 50B full-run joint bf16" with C1+NVFP4):**

| Sub-phase | tokens | Trainable | Encoder FFN | gate |
| --- | ---: | --- | --- | --- |
| B0 | 8B (5–10B) | new modules (cross-attn, \(W_K/W_V\), gate, new LN); backbone + embed + `lm_head` frozen | frozen virtual-group MoE | 0 → 0.3 |
| B1 | 27B (20–40B) | unfreeze Decoder + **`lm_head`**; **Encoder + input \(E\) still frozen**; cache still detach | same as above | → 1 |
| B2 | 15B (10–20B) | both stacks unfrozen (including embed and `lm_head`), smaller LR | unfreeze; experts start specializing | 1 |

Total tokens still ~50B. C1 operation count is about **79% of joint**; published wall-clock is **C1+NVFP4 = 571 H100-h (43% of joint bf16 1,325)**. C1 bf16 **1,046 H100-h**. C1+FP8 **729** is the Hopper/Ada fallback. Adam state in B1 is only **62%** of joint; detach drops Encoder activations by about **38%** (keep 26/42). If quality is unstable, lengthen B2 rather than going back to two independent LMs, and rather than reverting to full-run joint bf16. Phase C's "freeze backbone, train only indexer" still stacks **after** this curriculum (intra-layer KL, not CE through the cache; indexer stays bf16).

### Phase A — Architecture surgery and initialization (offline)

0. **Split into Encoder / Decoder stacks (YOCOize)**: map MiniCPM5-2B's 42 dense layers onto **Encoder 16 layers + Decoder 26 layers**.
   Keep the encoder at 16 layers so **2+7+7 CSA/HCA** still fits; the extra 2 layers all go to the Decoder.
   Recommended scheme: Encoder takes the base's **first 16 layer** weights, Decoder takes the **last 26 layer** weights (keep depth semantics); copy the **untied** embed and `lm_head` separately.
   Each Decoder layer **adds a cross-attn sublayer** (initial gate≈0 bypass, see §2.4; Q/O = \(2d^2\)), so the initial forward ≈ original decoder-only behavior, which eases recovery.
1. **MoE upcycling (dense FFN → fine-grained MoE)**, run **separately** on Encoder and Decoder, using Megatron-LM `upcycling_utils.py` (C1 frozen: both stacks finish in this phase; B0/B1 then freeze Encoder, see §4.0):
   - Slice the dense FFN intermediate dim into G segments and replicate each segment into multiple experts (**virtual-group init**: at the conversion instant, top-k exactly selects one copy of each shard, equivalent to the original dense function).
   - MiniCPM5 dense SwiGLU **6144 is divisible by 2048 (G=3)**. Upcycling is still **copy-and-scale the first `moe_int` rows** (each expert copies the dense first 2048 rows, then scales by \((E G^2 / T)^{1/3}\)); it is not instant exact identity at surgery time; recovery is Phase B's job.
   - **Weight scaling**: SwiGLU expert projections scale at the `(E·G²/T)^(1/3)` magnitude (paper-validated ~1.5% loss drop).
   - **Routing**: `softmax-then-topK` (better than topK-then-softmax); affinity scores use **Sqrt(Softplus(·))** (V4 practice).
   - Each stack is **full MoE by default** (C1: both stacks are MoE at token 0). DeepSeek-style "first-layer dense" is **off** and does not enter the published recipe; use `--first-dense` sensitivity when needed.
   - **No μP**: do not invent residual multipliers for 16/26, and do not transplant MiniCPM-2B's `scale_depth/√40`.
   - **Hash-MoE bootstrap**: the Decoder's first few MoE layers use a frozen `token_id → expert_id` hash route (V4 practice; stabilizes the early stage). Encoder is already frozen in B0/B1, so hash routing on the Encoder is redundant; apply it when unfreezing at B2.
2. **Attention conversion**: inherit GQA `q/k/v/o` (K/V shape `(256, 2048)`); newly added CSA/HCA compressors, position bias, and Lightning Indexer get small-scale random init. At this stage treat every attention layer as **dense/sliding-window** (no top-k, no HCA compression), approximately equivalent to the original attention. **This repo implements Phase C indexer intra-layer KL and C/D/E/F/G training entry points; it does not implement a CSA CUDA kernel.**
3. **Do not introduce MiniCPM-2B μP scale constants** (emb multiplier, `scale_depth`, logits scaling). MiniCPM5 residual is identity.
4. **Optional mHC**: first run ordinary residual through Phase B/C; switch to mHC after stability (constrain residual maps onto the Birkhoff polytope / doubly stochastic matrices, spectral norm ≤1).

> Encoder/Decoder boundary: the Encoder top output is projected by a (learnable) \(W_K,W_V\) into global `K̂,V̂` reused by all cross-decoder layers (YOCO single cache). \(W_K,W_V\) are new modules. B0/B1 must `X^{16}.detach()` **then** multiply by the projection (Theorem D); B2 removes detach.

### Phase B — Upcycling recovery continued pretraining

- **Purpose**: after MoE conversion + attention conversion + encoder-decoder conversion, restore language-modeling capability (including gradually opening cross-attn).
- Sequence length 4K; attention is still dense/sliding-window (not yet sparse); data is a general pretraining mix (see §5).
- **Open cross-attn gradually**: Decoder cross-attn gate rises linearly from 0 to 1. Under the unfreeze curriculum: B0 (~8B) only rises to 0.3, B1 rises to 1, so \(g\) is not pulled to full on a frozen backbone.
- **Distillation acceleration**: use **MiniCPM5-2B-Base (dense, teacher)** for logit KD (KL(teacher‖student), temperature 1–2, weight 0.5→0 linear decay) to substantially shorten recovery.
- **MoE load balancing**: aux-loss-free bias method (`e_score_correction_bias`, update bias from each expert's load, update rate e.g. 1e-3) + **light sequence-wise balance loss** (weight ~1e-3) against extreme per-sequence imbalance. Encoder experts start updating only at B2; Encoder load monitoring starts at B2.
- Learning rate: **WSD** (Warmup-Stable-Decay) — short warmup (0.5–1B tokens), then the stable segment (LR ≈ 30–50% of MiniCPM pretraining peak, because this is continued training). This phase stays stable, no decay. When B2 unfreezes the Encoder, drop LR another notch.
- **Untied embedding**: B0/B1 **freeze input \(E\)**; B0 also freezes `lm_head`, B1 **trains `lm_head`**; forbidden to train input \(E\) while the Encoder is frozen (Theorem E).
- **When budget is tight, do not switch to two independent LMs**: use the already-frozen **C1+NVFP4** of §4.0 (B0 new modules → B1 freeze Encoder → B2 short joint; B1/B2 allowed linear GEMMs run NVFP4), same ~50B tokens, wall-clock **571 H100-h (43% of joint bf16 1,325)**.

### Phase C — Attention sparsification alignment (critical, easy to fail)

**Flow: implement first, light later.** Code and layer labels enter the graph at B; C only changes `sparse_mode`; do not weld modules on only at C.

| Step | When | What to do |
| --- | --- | --- |
| **Implement** | Phase A surgery + **B graph build** | default layer labels 2 sliding+7 CSA+7 HCA; `--use-kda` then 2+11 KDA+2 CSA+1 HCA, `KDAGates` enter the graph. **Compute is still sliding-window GQA** (`sparse_mode=window`). KDA params are frozen at B, not in Adam. Published B0 default `use_kda=False` (132-tensor overlay). |
| **Light** | Phase C | without KDA: indexer→topk→hca→win. with KDA (already implemented at B): **C-kda → indexer → topk → hca (last) → win**. Forbidden to patch KDA modules onto a `use_kda=False` B overlay. |

Follow DeepSeek-V3.2's "dense warm-start, then sparse" idea to introduce DSA/compression. **Published default** (2 sliding + 7 CSA + 7 HCA, `use_kda=False`):

1. **Indexer dense alignment**: freeze the backbone; train only the Lightning Indexer on Encoder CSA so its score distribution **aligns with dense attention weights** (intra-layer KL). This step does not change the main output; it only teaches the indexer "whom to pick". Supervision stacks **after** B2. Indexer top-k may only delete from the compressed-block set \(S_{\mathrm{comp}}\), never add (Theorem B).
2. **Open CSA top-k**: CSA layers cut from dense sliding window to **sliding window ∪ indexer-selected compressed blocks** (own block excluded; window fills holes); continue training in small steps so the backbone adapts to sparsity. This is not a replacement with "keep only top-k tokens".
3. **Open HCA compression**: HCA layers **sliding-window KV concat mean-pooled slots** (\(m'=128\), own block excluded).
4. **Open 8K sliding-window joint training**: `C-win` at seq=8192 opens CSA top-k + HCA concat together, confirming window-branch and compressed-branch masks are correct.
- Monitor loss spikes at every step; if unstable, roll that step back, lengthen alignment, or lower LR.

**If opening 3:1 KDA** (12B: 2 sliding + 11 KDA + 2 CSA + 1 HCA), lighting order becomes:

0. **`C-kda`**: switch only KDA-kind layers to gated-delta; CSA/HCA-kind **still run sliding window**. Most paths learn to be used first, avoiding hybrid-linear "strong retrieval first, then linear layers get ignored".
1. **Indexer**: KDA stays lit; intra-layer KL only on remaining CSA anchors (12B has only 2 layers; tokens cut from 10e9 to 5e9; C total still 25e9).
2. **CSA top-k**, then **HCA last** (1 layer, write-first), then `C-win`.

CSA/HCA **implementation is not deferred or deleted**; what is deferred is **lighting**. Same for KDA: B `--use-kda` implements, C lights. B0 default remains `use_kda=False`.

### Phase D — Long-context extension

- Raise training sequence length stepwise: **8K → 32K → 128K** (continue if longer is needed).
- RoPE: frequency-scale to the target length (NTK/YaRN class) or continue training directly on long sequences; MiniCPM5-2B has native **128K context / `rope_theta=5e6`** as the long-context base; no need to consult MiniCPM-2B-128k's `rope_scaling`.
- CSA/HCA keeps long-range attention cost controllable; the 8K uncompressed window guarantees local fidelity.
- Switch data to long documents / concatenated long samples; use needle & RULER for in-process monitoring.
- **Implementation**: `python3 -m cat_yoko.d --stage 8k|32k|128k` or `--chain`. Phase B 4K packed `.bin` is **re-windowed to the target seq** (flat int32 packed rows). `--try` DummyStream writes a needle at mid-sequence. D/E/F **default `sparse=hca`** (after C lighting, do not go back to window); `--use-kda` is inherited from resume extra, and only KDA-kind continues gated-delta. prepare `--mix phase-d`: en 45% / zh 20% / math 10% / StarCoder 25%.

### Phase E — WSD annealing (high-quality data)

- Enter WSD **Decay**: LR decays quickly (exponential / 1-sqrt) to ~1/100 of peak.
- Mix shifts toward **high quality + math + code + long context + instruction-ized** (MiniCPM experience: the anneal segment benefits most from high-quality data).
- **Implementation**: `python3 -m cat_yoko.e`. `wsd_lr(..., lr_mode=decay)`. prepare `--mix phase-e`: en 30% / zh 15% / math 25% / StarCoder 15% / UltraChat body 15%. Do not download in CI.

### Phase F — SFT

- Instruction / multi-turn dialogue / long context / tool use / code / math; pack to target length; loss only on the response.
- Can follow DeepSeek-V4's "**per-domain experts first each SFT+RL, then on-policy distill into a unified model**", but at this project's scale (12B) start with a single mixed SFT.
- **Implementation**: `python3 -m cat_yoko.f`. seq=8192. prepare `--mix phase-f` writes jsonl (`tokens`+`labels=-100` on user; multi-turn conversations packed to target length). Parse UltraChat `data` lists, Chat `messages`, alpaca `instruction`/`output`, encode `role: text`. Trainer `FileStream` also eats already-tokenized `prompt_ids`/`response_ids` or `messages[].ids` and packs rows the same way. `--try` still uses DummyStream to mask the prompt prefix. sparse stays `hca`.

### Phase G — RL

- **GRPO** (DeepSeek family) as primary; reward covers verifiable math, executable code, instruction following; DPO may be added as light preference alignment.
- **Long-context usability RL (critical, and compute-saving)**: use **verifiable long-context tasks** for RLVR to directly reward "truly reading and using long input" — much cheaper than feeding another ocean of long tokens, and specifically treats lost-in-the-middle / long-instruction non-compliance / multi-hop misses:
  - **Reward signals**: long-doc QA (RULER-style, needle variants, multi-hop HotpotQA extensions) with exact-match/F1 verifiable; **grounding/citation reward** (the answer must cite the correct passage/line, matching §14 PDSA evidence selection); long-instruction following (constraints checkable).
  - **Curriculum**: do RL stepwise at 128K→256K, deliberately place key information in the **middle / long-distance**, strengthening mid-context recall (complements §12 IN2/FILM training).
  - **Save compute**: long-trace RL is expensive per rollout → use **PS-PPO (prefix-sampling PPO)** to backprop only the sampled prefix with unbiased truncation, substantially cutting long-sequence RL compute/VRAM; or for ultra-long context, PDSA-select evidence then RL (shorten effective rollout length).
- In the RL stage, watch MoE routing and sparse-attention stability under long rollouts.

---

## 5. Data

| Stage | Primary data | Scale (tokens) |
| --- | --- | --- |
| B recovery | **OpenBMB**: Ultra-FineWeb en 60% / zh 30% + UltraData-Math L2 10% | 50B envelope (prepare slices by `--max-tokens`) |
| C sparsify | same distribution as B, biased to long documents | 20–50B |
| D long context | long documents, books, repo-scale code concat, synthetic long-dependency tasks | 20–60B |
| E anneal | high-quality curated + math + code + instruction-ized SFT precursor | 20–50B |
| F SFT | **UltraChat** and similar instruction/multi-turn (not Phase B) | 1–10B |
| G RL | verifiable-task prompt sets (math/code/agent) | prompt-level |

Key points:

- **Tokenizer must be MiniCPM5-2B** (`openbmb/MiniCPM5-2B`, `V=130560`). Do not use MiniCPM3 / MiniCPM4 tokenizer, and do not feed MiniCPM-2B-sft-bf16 into a MiniCPM5 upcycling graph. Ultra-FineWeb is a MiniCPM4-era web-filter set and **must be re-tokenized**. **This repo does not download Ultra-FineWeb onto small VMs / CI.**
- Default mix `phase-b` is all OpenBMB. Optional `phase-b-code` replaces 10% with StarCoder (not OpenBMB; the Ultra-FineWeb paper eval mix used 10% code).
- UltraChat / instruction dialogue is reserved for Phase F/G; it does not enter Phase B.
- Implementation: `python3 -m cat_yoko.prepare --mix phase-b --tokenizer openbmb/MiniCPM5-2B --out data/phaseb.bin --max-tokens 1e8` → int32 packed mmap; `cat_yoko.train --data data/phaseb.bin --upcycle-hf openbmb/MiniCPM5-2B-Base`. Sidecar `*.bin.meta.json` carries `eos_id`. The repo **does not check in corpora**.
- Long-context samples use document concat + synthetic "needle-in-a-haystack / multi-hop"; strict dedup and eval-set decontamination. Ultra-FineWeb is marked Apache 2.0; source-page copyright still follows each site's terms.

---

## 6. Optimizer / hyperparameters / stability

| Item | Recommended |
| --- | --- |
| Optimizer | **published default AdamW** (\(\beta=(0.9,0.95)\), wd=0.1). Muon switch is kept, **off by default** (see [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)) |
| Muon | Newton-Schulz orthogonalization of momentum; pair with **hybrid ZeRO** implementation (V4 practice); lr needs a separate sweep (usually larger than Adam) |
| LR schedule | **WSD**: warmup(0.5–1B) → stable → decay; continued-training peak is 0.3–0.5× the base pretraining peak |
| Batch | global batch grows with stage (e.g. 4M→16M token/step); long-context stages use seq packing |
| Precision | **mixed precision frozen** ([`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md)): all linear GEMMs that need not be bf16 are NVFP4 (MoE experts, attn QKV/O, lm_head, frozen Encoder forward). B0 student, L0, Phase C indexer, **embed**, RMSNorm, router, gate, attn softmax stay high precision. 6NT is not changed; wall-clock is counted at 2.0× vs bf16. C1+FP8 is the Hopper/Ada fallback. V4-style FP4 expert storage is a later option and does not enter this recipe |
| Regularization / stability | zero-centered & weight-decayed RMSNorm, router z-loss (light), grad clip 1.0 |
| MoE balance | aux-loss-free bias updates + light seq-balance loss; monitor expert utilization / drop rate |
| MTP | auxiliary-head weight 0.1–0.3; may be used only in B–E; at inference can be dropped or used for speculative decoding |
| μP | **none**. MiniCPM5 is Llama; do not transplant MiniCPM-2B emb/residual/logits scale constants |

Muon's Newton-Schulz orthogonalization stays fp32, orthogonal to network NVFP4 GEMMs. Do not turn on NVFP4 at L0.

---

## 7. Evaluation and ablations

**Capability eval**: MMLU / CMMLU / C-Eval (knowledge), GSM8K / MATH (math), HumanEval / MBPP (code),
BBH (reasoning), IFEval (instruction following).
**Long context**: **RULER**, **Needle-in-a-Haystack**, LongBench; check retrieval fidelity of 8K sliding window + compressed long-range + YOCO global cache at 32K/128K/1M (YOCO reports near-perfect needle at 1M).
**Efficiency**: per-token inference FLOPs, **KV cache size (YOCO caches only once; should be substantially below a decoder-only baseline)**, prefill latency (encoder early-exit gain), decode throughput.
**Required ablations**:
1. **Encoder/Decoder layer split** (e.g. 16/26 vs 20/22 vs 12/30) vs 2.03B/4.33B activation and quality; published default remains **16 encoder** (2+7+7 CSA/HCA still fits);
2. `n_win ∈ {2K, 4K, 8K}` quality/throughput tradeoff;
3. CSA:HCA layer ratio (1:1 vs 2:1 vs 3:1);
4. `index_topk ∈ {128,256,512}`;
5. MoE granularity / expert count (`moe_intermediate_size`, `n_routed`, `top_k`) vs hitting the 12B total / per-tier activation targets;
6. **YOCO decoder-decoder vs equal-parameter decoder-only** (verify KV cache / prefill gains without dropping points);
7. **Whether to introduce KDA and the KDA:CSA:HCA ratio** (e.g. pure CSA/HCA vs 3:1 KDA mix vs 6:1) — focus on whether RULER/multi-hop mid-context recall **drops because KDA was added** (linear layers are expected to slightly drop precise recall; full/CSA anchors must compensate) and on long-context throughput / KV-cache gains;
8. Upcycling vs continued training directly from the dense base (verify the upcycling gain);
9. Muon vs AdamW; mHC vs ordinary residual; cross-attn gate gradual-open vs immediate-open; full-anchor-layer NoPE vs RoPE.
10. **Unfreeze curriculum**: C1 (frozen) vs full-run joint vs **illegal** B2=0 (curriculum chapter §9; look at recovery PPL and RULER, not FLOPs alone). Delayed Encoder MoE is sensitivity only and does not enter the recipe.
11. **NVFP4**: B1 `nvfp4` vs full-run bf16 (look at recovery PPL / overflow, not wall-clock alone); B0 student mistakenly opening NVFP4 as a negative control. The must-stay-high-precision set cannot go 4-bit. If QKV/O diverges, fall back to high precision (MaxText convention); do not change C1.

---

## 8. Infrastructure

| Component | Suggestion |
| --- | --- |
| Training framework | This repo's reference implementation is **PyTorch** (`cat_yoko.train --backend torch`). When 12B does not fit on one GPU, use **DeepSpeed ZeRO** (`--backend deepspeed`, [`docs/DEEPSPEED_ZERO.md`](DEEPSPEED_ZERO.md); `--dump-deepspeed` dumps JSON; CI does not force-install). Scale-out EP/TP is reserved for **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** / Megatron-Core. Mapping: `cat_yoko.megatron.mapping.megatron_blueprint`; `--dump-megatron` dumps JSON. YOCO is **not** `GPTModel`. |
| Parallelism | `ParallelPlan`: TP/PP/EP/CP/SP. 12B: TP ∈ {1,2,4,8,16} (divides 16 heads and \(d=2048\)); **EP ∈ {1,2,4,5,10,20}** (divides 20 routed). When PP>1, encoder\|decoder splits at layer 16 (`pipeline_split_rank`). Long context uses Context/Sequence Parallel. YOCO is **not** Megatron `GPTModel`. |
| Attention kernel | **FlashMLA** sparse prefill/decode kernel (supports DSA, FP8 KV); **NSA** Triton kernels can inform the compress+select+sliding-window three-branch implementation |
| MoE kernel | fused MoE dispatch/combine kernel (compute/comm/memory overlap) |
| Precision | bf16 master + frozen NVFP4 GEMM (§6 / [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md)); Hopper/Ada fallback FP8; deterministic/reproducible kernels (optional) |
| Memory | tensor-level recompute (`--grad-ckpt`), B0/B1 frozen Encoder CPU offload, B2 layer-wise offload, Adam momentum CPU offload (`--optim-cpu`); **DeepSpeed ZeRO-3 + CPU offload** for frozen weights (48GiB Ampere). Not Megatron EP/TP |
| Inference | vLLM / SGLang (already integrated DSA/FlashMLA sparse kernels) for eval and RL rollout |

> If you cannot author CSA/HCA kernels yourself, **start from HuggingFace `transformers`' `DeepseekV4` reference implementation**
> (`layer_types`, `compress_rates`, `sliding_window`, `index_topk`, etc. already exposed) to get correctness and small-scale training working,
> then migrate to high-performance kernels for scale-out.

---

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Sparse-attention training unstable / drops points | Strictly follow Phase C "dense align → gradual sparsify"; align the indexer alone first; on trouble, roll back a single step |
| MoE load collapse / idle experts | aux-loss-free bias + seq-balance loss + Hash-MoE bootstrap + monitor utilization |
| Capability regression after upcycling | virtual-group init + weight scaling + teacher distillation + reset LR to a higher stable segment |
| 8K sliding-window cost too high | ablation-roll back `n_win`; 8K on only some layers; long range delegated to CSA/HCA |
| MiniCPM-2B μP constants mistakenly transplanted, causing numerical drift | MiniCPM5 is Llama: `scale_emb=1`, identity residual, logits not divided by 9; unit-test forward scale after surgery; do not transplant MiniCPM-2B μP |
| Muon does not converge / unfamiliar hyperparameters | first run an AdamW baseline, then switch to Muon and sweep lr separately; keep a fallback switch |
| Missing kernels | first verify correctness with the HF reference, then bring up high-performance kernels |
| NVFP4 overflow / loss spikes | B0 student stays bf16; when B1 switches `nvfp4` watch NaN and expert utilization; first fallback is QKV/O back to high precision, then FP8/bf16; do not change the C1 freeze boundary |
| Poor long-context extrapolation | staged RoPE scaling + long-sample curriculum + RULER in-process monitoring |

---

## 10. Milestones (by capability/budget, not calendar)

1. **M1 surgery ready**: offline `CAT-YOKO-12B` (Encoder 16L / Decoder 26L) initial weights; forward numerical-scale self-check passes; short-run loss does not diverge with cross-attn bypassed.
2. **M2 recovery met**: after Phase B (cross-attn fully open), general benchmarks recover to MiniCPM5-2B's ~95%+.
3. **M3 sparsification met**: after Phase C, CSA top-k + HCA + 8K sliding window are on; short-context quality roughly matches M2; efficiency clearly improves.
4. **M4 long context**: **128K–256K RULER/Needle pass and quality is usable (primary target)**; 1M as stretch (inference runs + needle can pass is enough; do not chase quality); per-token FLOPs and **KV cache (YOCO single cache)** substantially below a decoder-only control; prefill early-exit gain is realized.
5. **M5 post-training**: after SFT + GRPO, instruction/math/code reach the target band; produce a releasable checkpoint.

---

## 11. Immediate next steps

Published spec is frozen; see [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md). Next is to run the repo's **12B training code** (tiny unit tests → meta 12B graph → `--dump-megatron` → FSDP / Megatron when GPUs are available).

1. `python3 -m unittest tests.test_param_budget tests.test_arch_verify tests.test_train tests.test_trainer tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload`
2. `python3 -m cat_yoko.prepare --mix local --local texts.jsonl --tokenizer dummy --config tiny --out /tmp/t.bin --max-tokens 256` then `python3 -m cat_yoko.train --config tiny --phase B0 --steps 3 --accum 1 --data /tmp/t.bin`
3. With GPU: `python3 -m cat_yoko.gpu_smoke` (tiny); `python3 -m cat_yoko.gpu_smoke --middle` (12B B0 one step, ≥28GiB, bf16 builds the graph directly); `--middle --phase B1` (Encoder offload + CPU Adam); `--c1` (same 12B graph B0→B1→B2)
4. `python3 -m cat_yoko.train --config 12b --meta` (count parameters; do not allocate 24GB)
5. `python3 -m cat_yoko.train --config 12b --dump-megatron` (dual-stack TransformerConfig JSON; does not run Megatron)
5b. `python3 -m cat_yoko.train --config 12b --dump-deepspeed --zero 3 --zero-offload-param` (ZeRO JSON; does not install DeepSpeed)
6. With network + GPU: `pip install 'cat-yoko[data]'`, `prepare --mix phase-b --tokenizer openbmb/MiniCPM5-2B --out data/phaseb.bin --max-tokens 1e8`, then `--config 12b --phase B0 --upcycle-hf openbmb/MiniCPM5-2B-Base --data data/phaseb.bin --save-dir runs/b0 --dtype bf16 --grad-ckpt --device cuda --steps N` to start training under C1+NVFP4. B1/B2 `--resume` from `latest.pt` or the save dir (weights + packed cursor + RNG; does not restore previous-phase Adam / step). `latest.pt` hardlinks when `step_{last}.pt` already exists; do not write a second 23GiB copy for 12B; put ckpts on a large disk (`/root/autodl-tmp`), not `/tmp`. You can also `--c1 --save-dir runs/c1 --steps N` to run the three phases on the same graph, writing `runs/c1/{B0,B1,B2}/latest.pt`. 12B does not save Adam by default. Single 32GB GPU + ~62GiB host cgroup: B0 one step directly; B1 offloads frozen Encoder + CPU Adam (one-step smoke uses ephemeral momentum); B2 layer-wise offload, clip+Adam as soon as a layer's backward finishes (`--accum 1`). Scale-out later with `--backend megatron`. Do not download Ultra-FineWeb or 12B weights on a small VM / CI. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO.

Do not change 16/26, C1, C1+NVFP4, causal Encoder, or M2 default. For quality issues, lengthen B2 or fall back dtype; do not change the freeze boundary.

---

## 12. Additional advanced techniques (tiered by value/risk; avoid stacking)

> Principle: the more novel components, the harder training. Ordered "low-risk first, high-risk later / optional"; each should be independently switchable and roll-back-able.

### Tier 1 — Low risk, high reward (recommended as defaults)

- **QK-Norm** (RMSNorm on query/key) + **z-loss** (router-z to suppress routing-logit explosion + output-logit z-loss) + **dual RMSNorm (pre+post, OLMo2/Gemma2 style)**: key stabilizers for a deep + MoE + sparse-attention novel stack, extremely cheap.
- **Document-aware attention mask**: packed long sequences **do not attend across documents**, avoiding contamination of the long-context training signal.
- **FIM (Fill-in-the-Middle)**: fill-in training on code data, improving completion/edit.
- **IN2 / information-intensive long-context training (FILM class)**: synthesize training samples where "key information sits in the **middle** of a long document" — **this is the correct fix for lost-in-the-middle**, more directly effective than adding KDA (see the clarification in §2.5).

### Tier 2 — Medium risk, high reward (add after the base is stable)

- **MTP → speculative decoding**: reuse the already-hung MTP head for EAGLE-style self-speculation; inference speedup; almost zero extra training cost.
- **NVFP4 training** ([`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) frozen): linear GEMMs that need not be bf16; B0 student / indexer / must-stay-high-precision set stay high precision. Published wall-clock 2.0× vs bf16; do not publish 4×. **V4-style FP4 expert storage** is a later option and does not enter this recipe. Hopper/Ada fallback: [`docs/FP8_THEORY.md`](FP8_THEORY.md).
- **attention logit soft-cap / QK-clip** (Gemma2 / Kimi): suppress extreme logits, further stabilize training.
- **RoPE/NoPE calibration and frequency scaling (YaRN)**: long-context extrapolation + mitigate position bias.

### Tier 3 — Cautious / last to add (off by default)

- **mHC** (already in §2, marked optional), **Muon** (keep AdamW fallback), **3-way KDA mix** (§2.5).
- **Zero-compute / elastic top-k experts**: adaptive activation by token difficulty (potentially fits the "2.03B/4.33B asymmetric" story, but complex and easy to destabilize; research item).
- **Shared attention blocks (Zamba-style) reused across layers**: further save params/cache, but coupling is strong.

---

## 13. Training-difficulty management: staged de-risking (important)

**Core: do not light every new component at once.** Combined innovation (YOCO + CSA/HCA + optional KDA + DeepSeekMoE + Muon + mHC + MTP) has multiplicative risk;
stepwise introduction + per-step rollback + metric monitoring is the only safe path.

**De-risking ladder:**
- **L0 (tiny correctness)**: small config (`hidden 256, enc2L/dec2L, sliding=8, m=4, m'=8, index_topk=2`) verifies: YOCO dataflow (encoder→global cache→cross-decoder), CSA/HCA/KDA masks and kernels, MoE routing/balance. Only ask "does it match, does it NaN". **Precision bf16, no NVFP4.**
- **L1 (half-scale de-risk prototype, ≈3–6B)**: **full novel architecture stack** but **half the experts**, run tens of B tokens through stability, upcycling recovery curves, sparsification alignment, gradual cross-attn open. A cheap architecture validation bench.
- **L2 (scale to the 12B target)**: **MoE expert count is the safest scaling axis** — after the architecture is validated at L1, half-scale→12B is mainly adding routed experts (+ a little continued training so new experts differentiate); much lower risk than changing the architecture.
- **Each new component is its own step**: first **implement the module into the graph** (B still sliding window), then **light** it. Default lighting indexer→CSA→HCA; **a 3:1 KDA graph then C-kda → sparsify**. **NVFP4 does not enter L0** (tiny stays bf16); from B1, open allowed linear GEMMs, keep a rollback switch separate from Muon orthogonalization. Watch loss spikes / expert utilization / recall metrics; on failure, roll that step back.

**Difficulty–benefit tradeoff cheat sheet:**

| Want less hassle / faster results | Want extreme long-context efficiency |
| --- | --- |
| First decoder-only + CSA/HCA (no YOCO/KDA/mHC/Muon); get it working, then add stepwise | Full-stack YOCO + 3:1 KDA + CSA/HCA + NVFP4, but strictly L0→L1→L2 |

---

## 14. Integrating PDSA memory management (trainable lifecycle + calibrated fallback)

> Source: **Memory-Managed Long-Context Attention** (Zou & Donz, arXiv `2606.28876`, this team's work, hereafter PDSA).
> That paper is a **memory system around a frozen LLM at inference/eval** (not a trainable architecture). Core: query-independent writer +
> hard-boundary lifecycle (overwrite/protection/eviction, ≤32 slots) + query-aware reader + **calibrated sparse fallback** + frozen LLM generating from original evidence.
> It is **orthogonal and complementary** to CSA/HCA/KDA: the latter is token-level state compression; PDSA is managed memory at the semantic-unit level.
> PDSA's explicit "next step" is to **put the lifecycle inside the model, trainable** — CAT-YOKO is a natural instance of that.

### 14.1 Key transferable conclusions

- **"No write-time signal" bound (PDSA §5 measured)**: on static text, query-blind writability judgments ≈ random (AUC 0.63–0.66 vs query-aware 0.89–0.97); pure bounded memory recalls only ~0.56 gold evidence. **Implication**: any **write-first compression** (HCA, KDA, even CSA's compression step) will systematically drop information that "had no signal at write time and is only later hit by a query" → **must keep a query-time fallback to less-compressed / raw KV**.
- **Bounded selection beats reading the full text on long documents (PDSA §4)**: at 8.2k words, reading the full text actually drops (lost-in-the-middle); ≤10% evidence already reaches 102–116% of full-text F1. Supports that CAT-YOKO's "compress + select" direction is correct.

### 14.2 Layered integration (aligned with §13 difficulty management)

- **Tier 1 (low risk, recommended) — calibrated confidence-gated sparse fallback**:
  Give CSA's Lightning Indexer a **confidence signal** (e.g. mean/entropy of top-k scores); below threshold, **widen top-k or fall back to less-compressed / raw KV retrieval**.
  Thresholds are **calibrated on the deployment length regime** (PDSA recorded a negative result: calibrating on short context makes fallback never fire and long-doc coverage collapses).
  This is a principled treatment of the earlier "mid-context recall" concern **using this team's own measurements**, and it is an inference-time increment that can be added later.
- **Tier 2 (medium risk) — query-aware preferred over write-first**: in layer scheduling **use more CSA (query-aware selection)** and use HCA/KDA (write-first) cautiously for critical retrieval; position HCA/KDA as "cheap gist coverage + safety net"; precise retrieval goes to CSA + full anchors + fallback.
- **Tier 3 (research-grade, after the base is stable) — trainable bounded editable memory lifecycle**:
  Upgrade YOCO's "append-only global cache" to **bounded, editable, lifecycle-bearing** memory: a learned **writer** (write/overwrite/protect/evict, by key/salience) manages a capacity-limited memory that decoder cross-attn reads.
  Gain: truly bounded KV cache + **versioned/protected semantics** (agents, long-horizon task differentiation); risk: switched-process stability (PDSA Appendix H), write instability — caution; this is unsolved research, advance as a separate milestone, keep a symbolic/frozen fallback.

### 14.3 Matching ablations (add to §7)

- With/without **calibrated sparse fallback** on RULER/multi-hop mid-context recall and long-doc F1; sensitivity of fallback thresholds **calibrated across length regimes**.
- **CSA ratio ↑ (query-aware) vs HCA/KDA ratio ↑ (write-first)** on recall of "information only later hit".
- (Research item) trainable editable memory vs append-only global cache: KV-cache upper bound, versioned-task correctness, stability.

> Positioning reminder: PDSA's contribution is **memory management**, not a replacement for CSA/HCA/KDA **state compression**; stacking both is the complete scheme.
> Do not migrate its frozen-reader eval harness or the specific 32-slot number (that is methodological evidence, not architecture).

---

## 15. Compute-budget estimates and low-budget routes (important real-world constraints)

> Premise correction: this plan originally defaulted to "tens to over a hundred H100s". **If compute is limited, 24B from-scratch / heavy continued pretraining is not feasible**, so re-prioritize:
> **validate the architecture at small scale first, scale when there is budget**. What we need to prove is "architectural innovation (YOCO×CSA/HCA + KDA + PDSA trainable lifecycle)",
> not "large-model scale" — neighboring hybrid-linear analyses finished at 340M/1.3B, and this team's PDSA core components are only ~2.74M parameters + a frozen backbone.

### 15.1 Training compute estimates

`training FLOPs ≈ 6 × N_active × tokens`. bf16 rows below convert operation count to hours at 40% MFU (H100 effective ~4.0e14, A100 ~1.25e14 FLOPS); **the default Phase B wall-clock is C1+NVFP4**, not joint bf16. RTX PRO 6000 Server BF16 peak ≈ H100, so hours are comparable.

| Plan | H100-h | A100-h | 8×H100 days |
| --- | ---: | ---: | ---: |
| **C1+NVFP4 (Phase B frozen)** | **571** | **1,827** | **3.0** |
| C1+FP8 (Hopper/Ada fallback) | 729 | 2,333 | 3.8 |
| C1 unfreeze curriculum 8+27+15B (bf16 operation-count ledger) | 1,046 | 3,347 | 5.4 |
| 12.25B middle × 50B tok (joint bf16 control; same as the plan ledger under untied) | 1,325 | 4,239 | 6.9 |
| 12.25B middle × 200B tok | 5,299 | 16,956 | 27.6 |
| 24B × 200B tok (future; at the then 3B+6B activation) | 7,583 | 24,038 | 39.5 |
| 24B × 50B tok (future minimum recovery) | 1,896 | 6,010 | 9.9 |
| ~6B × 60B tok | 885 | 2,804 | 4.6 |
| ~3B × 50B tok | 316 | 1,002 | 1.6 |
| ~1B × 20B tok | 51 | 160 | 0.3 |
| ~0.5B × 10B tok (architecture validation) | 13 | 40 | 0.1 |

> **C1+NVFP4 = 571 is the Phase B published wall-clock (43% of joint bf16 1,325).** 1,325 / 1,046 are bf16 operation-count controls; 729 is the C1+FP8 Hopper/Ada fallback. NVFP4 does **not** change 6NT ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). Published speedup **2.0× vs bf16** (relative to FP8 1.5× then ×1.33, landing on the low end of NVIDIA 1.31–1.73× vs FP8); **4× is only the RTX PRO 6000 peak upper bound and is not written into the recipe**. Distillation/upcycling substantially reduce required tokens; the long-context stage is a small share and is counted separately.

### 15.2 Three low-budget routes (pick by actual GPU count)

- **Route A — architecture validation (cheapest, ≤ a few GPUs, ~10–50 H100-h, rentable)**: upcycle a small MiniCPM5 (fewer experts) → **0.5–1.5B small MoE**, install YOCO+CSA/HCA(+optional KDA), continue training 10–20B tok. Goal: prove this attention/encoder-decoder can run, does not drop points, and saves KV at long context. **Recommended as the default starting point.** The in-repo probe is [`docs/PLAN_VERIFY.md`](PLAN_VERIFY.md): the `plan-probe` graph short-trains **Phase A→E** on DummyStream (A is offline surgery; does not go to F/G).
- **Route B — scale to the 12B target (~8×A100/H100 or one RTX PRO 6000; Phase B frozen C1+NVFP4 ≈ 571 H100-h)**: MiniCPM5-2B → **12.25B** upcycle (optionally via a 3–6B milestone), continue training 50–60B tok + a short long-context stage, obtain the target model. Joint bf16 1,325 is control only.
- **Route C — PDSA extension (almost no training compute, fits existing work)**: freeze the backbone, train only small components (writer/reranker/threshold) + land "calibrated fallback / trainable editable memory" (§14). **Best zero-budget option**, directly producing PDSA's trainable-lifecycle follow-on.
- **Route D — 24B (future; not a current target)**: consider scaling only after a real cluster / compute grant.

### 15.3 Compute-saving levers (priority high to low)

1. **upcycling** (reuse MiniCPM5 weights; never from-scratch); 2. **distillation** (teacher=`openbmb/MiniCPM5-2B-Base`, fewer tokens);
3. **high-sparsity MoE** (fewer active params = fewer FLOPs); 4. **4K context is the bulk of training**, long context only a brief segment;
5. **unfreeze curriculum** (§4.0 / [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md): **C1 frozen** — both stacks MoE first, B0/B1 freeze Encoder and input embed, B0 freeze `lm_head` / B1 train `lm_head`, ~21% Phase B FLOPs saved, B1 Adam state 62%, Encoder activations ~38%; the end must have short joint B2≥10B);
6. **NVFP4** ([`NVFP4_THEORY.md`](NVFP4_THEORY.md): **nailed with C1 as C1+NVFP4** — linear GEMMs that need not be bf16; B0 student / L0 / indexer / must-stay-high-precision set stay high precision; published 2.0× vs bf16. Phase B wall-clock **571 H100-h, 43% of joint bf16 1,325**; 4× is peak upper bound only. Hopper/Ada fallback C1+FP8 = 729); 7. **Muon** (fewer steps; Newton-Schulz still fp32, orthogonal to NVFP4 GEMM); 8. **full-train new modules + LoRA on the rest** (less optimizer memory; can use smaller/fewer GPUs);
9. **rent spot GPUs for critical short runs** (no need to buy); 10. seq packing + activation recompute (fit onto fewer GPUs).

### 15.4 Revised default path

**L0 tiny correctness → plan-probe A→E ledger ([`PLAN_VERIFY.md`](PLAN_VERIFY.md)) → Route A (0.5–1.5B architecture validation, produce an architecture paper) → with budget, Route B (scale to the 12B target) → future (optional) Route D (24B).**
§1.2's **12B is the target spec**; when budget is tight the **current default is to execute Route A first**; §3's budget script can directly shrink `Nr_e/Nr_d` to a 0.5–1.5B tier for validation.

---

## 16. Long-context feasibility (primary target 128K–256K; 1M is stretch)

> First, the tone: **the primary delivery target is "the first 128K–256K usable"**; that is realistic at 12B. The 1M discussion below is a stretch analysis of "can we even support it", **not** an attempt to match 2T-frontier quality at 1M.

The conclusion has three layers; do not mix them:

### 16.1 Inference: very promising (this architecture is designed for 1M)

The key is KV cache. **YOCO caches only once (single global cache) + CSA/HCA sequence compression** squeeze 1M KV from "won't fit" to "a rounding error":

| 12B model KV cache at 1M tokens | Size |
| --- | --- |
| decoder-only + MHA (all 42 layers cached) | ≈344 GB (won't fit) |
| decoder-only + GQA-2 / MLA-576 | ≈43 / 48 GB |
| **YOCO + GQA-2 (single global cache, \(d_{\mathrm{kv}}=256\))** | **≈1.02 GB** |
| **YOCO + MLA (single global cache)** | **≈1.15 GB** |
| **YOCO + MLA + CSA \(m=4\) (global cache ÷4 along sequence)** | **≈0.29 GB** |
| YOCO + MLA + extra sequence÷8 (stretch, not CSA default) | ≈0.14 GB |
| (plus 26-layer 8K sliding-window branch, independent of N) | +≈0.25 GB |

Stack **encoder (self-decoder) prefill early-exit** on top — ultra-long input only needs to finish the encoder to produce the global cache, not run every layer — so 1M **prefill is also cheap**. This is exactly what "a light input side (≈2.03B)" means. Middle-tier prefill activation share is about 32%. **Decode side**: if the global cache is not compressed, 26-layer cross-attn at 128K/256K is about 3.2× / 6.5× the decoder MLP; you must use CSA \(m=4\) or top-\(k\) selection, otherwise compute saved on the encoder is spit back out in cross-attn (see theory verification §6). The original YOCO paper reports near-perfect needle retrieval at 1M. **So 1M inference is feasible even on mid-range hardware.**

### 16.2 Training a model that "supports 1M (needle/RULER pass)": realistic, but it takes care

Do not pretrain at 1M; main training is at 4–8K, with a **progressive long-context extension stage** at the end (8K→32K→128K→256K→1M). Keys:
1. **RoPE/YaRN scaling** to the target length; 2. **long data**: books / repo-scale code concat + synthetic long-dependency + IN2 mid-context samples; 3. **progressive length curriculum**;
4. **the training-time memory bottleneck is 1M-sequence activations** (not KV) → use **context/sequence parallelism**; YOCO early-exit + CSA/HCA compression substantially cut activations; **the asymmetric design is a natural fit** (encoder handles long input, decoder generates short → long prefill is cheap);
5. **Validation**: RULER-1M / needle-in-haystack. This stage is not many tokens (a few B to tens of B); cost is small relative to main training; can rent GPUs for a short run.

### 16.3 Matching frontier 1M quality: not reachable on a small budget

DeepSeek/Kimi 1M is "full utilization" fed by 32T-scale data + large compute. A small budget can get "**supports 1M + decent needle/RULER**", but "multi-hop deep reasoning at 1M matching the frontier" is not realistic — say so honestly.

### 16.4 Low-budget practical recommendations

- **Staged**: first stabilize **128K–256K** (cheap, enough), then separately push **1M capability** and verify with needle/RULER; do not start at 1M.
- **The PDSA route is a cheaper substitute for "effective 1M"**: bounded editable memory + calibrated sparse fallback + retrieval (§14), **without training native 1M attention**, still gets long-range recall — your own work, and §5 measurements show bounded selection already beats reading the full text at 8.2k. For extreme long context this may be more cost-effective than hard-training 1M attention.
- On milestones, put 1M under **M4** (long context) as a capability target, not a quality target.
- **Later use RL to raise "usability"**: pretraining/extension only solve "can swallow 128K–256K"; **whether it can actually be used well** depends largely on later **long-context RLVR** (see §4 Phase G) — verifiable long-doc tasks + grounding reward to directly optimize mid-context recall / long-instruction following, and PS-PPO/PDSA evidence selection to keep long-rollout cost down. This is the key lever, on a small budget, for lifting "usability" another notch.

> **Positioning (important, to avoid misunderstanding)**: this project is **not** using 12B to match 2T-class frontier model quality — that is impossible.
> **The primary delivery target is "the first 128K–256K long context usable"** (at 12B scale, this is promising in both theory and engineering);
> 1M is only a bonus of "the architecture can support it + needle/RULER can pass", not a quality target.
> For more extreme long context on a budget, prefer PDSA memory + retrieval fallback (§14) over hard-training native 1M attention.

> One sentence: **main battlefield = 128K–256K usable; 1M inference/needle is stretch; 2T-class frontier quality is not in scope.**

---

## References (architectural sources for this plan)

- **DeepSeek-V4** (CSA/HCA, mHC, Muon, MTP, Hash-MoE bootstrap; V4-Flash 284B/13B, 1M ctx, 32T tokens): arXiv `2606.19348`; HuggingFace `transformers` `deepseek_v4` model docs (`layer_types`, `compress_rates`, `sliding_window`, `index_topk`, `mlp_layer_types`, and other config).
- **DeepSeek Sparse Attention (DSA)** and **FlashMLA** sparse kernels (Lightning Indexer + top-k + FlashMLA): DeepSeek-V3.2 report; `deepseek-ai/FlashMLA`.
- **Native Sparse Attention (NSA)** (compress + select + sliding-window three-branch, hardware-aligned, natively trainable): arXiv `2502.11089`.
- **Upcycling LLMs into MoE** (virtual-group init, weight scaling, softmax-then-topK; Megatron `upcycling_utils.py`): arXiv `2410.07524`.
- **DeepSeekMoE** (fine-grained experts + shared experts): arXiv `2401.06066`.
- **MiniCPM5-2B** (Apache-2.0 Llama GQA: d=2048, 42 layers, 16 Q / 2 KV, `head_dim=128`, V=130560, untied, no μP, native 128K / `rope_theta=5e6`; WSD schedule follows MiniCPM-family practice): `openbmb/MiniCPM5-2B` (tokenizer) / `openbmb/MiniCPM5-2B-Base` (upcycle / teacher). Do not use MiniCPM-2B-sft-bf16 (GML) or MiniCPM3/4 tokenizer.
- **Gemma 2 / Qwen3-Next** (local sliding window × global attention interleaving, hybrid-attention layer ratios): reference for layer scheduling and sliding-window design.
- **YOCO — You Only Cache Once** (decoder-decoder: self-decoder produces a single global KV cache, cross-decoder reuses it; prefill early-exit; 1M ctx near-perfect needle): arXiv `2405.05254`; `microsoft/unilm` YOCO.
- **Kimi Linear / KDA** (Kimi Delta Attention: fine-grained gated Gated-DeltaNet + DPLR chunk kernel; 3:1 KDA:MLA mix, MLA uses NoPE; 1M KV cache ↓~75%, decode ↑~6×): arXiv `2510.26692`; `MoonshotAI/Kimi-Linear`.
- **Hybrid Linear Attention systematic analysis** (linear attention is weak at recall and needs full layers to compensate; gated-delta reaches Transformer-level recall at 3:1~6:1): arXiv `2507.06457`.
- **Lost in the Middle** (mid-context position bias, present in softmax too, mitigated by position-encoding calibration): arXiv `2307.03172`.
- **FILM / IN2 training** (information-intensive long-context training; synthesize "key information in the middle" samples to fix lost-in-the-middle): `Make Your LLM Fully Utilize the Context`, arXiv `2404.16811`.
- **OLMo 2 / Gemma 2** (QK-Norm, dual RMSNorm, logit soft-capping, z-loss, and other stability tricks): arXiv `2501.00656` / `2408.00118`.
- **EAGLE / speculative decoding** (reuse MTP heads for self-speculation speedup): arXiv `2401.15077`.
- **YaRN** (RoPE long-context extrapolation scaling): arXiv `2309.00071`.
- **PS-PPO — Prefix-Sampling PPO** (critic-free RLHF backprops only the sampled prefix with unbiased truncation; cuts long-trace RL compute/VRAM): arXiv `2606.29758`.
- **PDSA / Memory-Managed Long-Context Attention** (bounded editable memory + hard lifecycle overwrite/protection/eviction + query-independent writer + query-aware read + calibrated sparse fallback; measured "no write-time signal" bound; bounded selection beats reading the full text on long documents): Zou & Donz, arXiv `2606.28876` (this team's work; its "next step" is a trainable lifecycle, taken up in this plan's §14).
- **MSA — Memory Sparse Attention** (static document sparse memory, PDSA's nearest neighbor): arXiv `2603.23516`.
- **Gated DeltaNet / Gated DeltaNet-2** (KDA's predecessor; decoupled erase and write): arXiv `2412.06464` / `2605.22791`.
