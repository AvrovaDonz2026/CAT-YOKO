# CAT-YOKO Middle-Tier Theory Verification

> Spec is frozen to the training-plan default tier: `CAT-YOKO-12B`, Encoder active ≈2.03B / input token, Decoder active ≈4.33B / output token.
> Base is Apache-2.0 **MiniCPM5-2B** (Llama GQA), not MiniCPM-2B.
> This note only does **recomputable theoretical checks** (parameters, FLOPs, KV, complexity, no μP); it introduces no new architecture.
> Number source: `python3 scripts/param_budget.py --full`; assertions: `python3 scripts/param_budget.py --verify` and `python3 -m unittest tests.test_param_budget`.
> Architecture (causality, split equivalence, M1/M2/M3): [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md).
> Unfreeze curriculum (C1 published recipe: freeze boundaries, Theorems D/E): [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md).
> NVFP4: Phase B wall-clock published as **C1+NVFP4 = 571 H100-h** (Theorem G; does not change 6NT): [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Hopper/Ada fallback C1+FP8 = 729: [`FP8_THEORY.md`](FP8_THEORY.md).

---

## 0. Conclusions (read this first)

The middle tier is self-consistent under **the accounting convention the plan adopts**:

| Quantity | Plan claim | Exact value | Verdict |
| --- | ---: | ---: | --- |
| Total parameters | ≈12B | **12.25B** | pass |
| Encoder active / input token | ≈2.03B | **2.03B** | pass |
| Decoder active / output token | ≈4.33B | **4.33B** | pass |
| 50B tok training | ~1,325 H100-h | **1,325 H100-h** | pass |
| 1M KV (decoder-only MHA / GQA-2 / MLA / YOCO+MLA / ÷8) | 344 / 43 / 48 / 1.15 / 0.14 GB | **344.06 / 43.01 / 48.38 / 1.15 / 0.14 GB** | pass |

Places where the plan text must change relative to the old MiniCPM-2B ledger, without changing the spec:

1. **Expert sparsity** is **38.1% (8/21) / 52.4% (11/21)**, not 7/18, 9/18, and not 37%/53%.
2. **Attention is ~5.0% of total parameters** (GQA 16/2); MoE is still **90.6%**. Do not port MiniCPM-2B placeholder attention's 13%, or the ~7% from the 24B era. KDA remains parameter-neutral.
3. Diagrams should be **≈2.03B / ≈4.33B**, layer counts **16+26=42**, not 16/24 and 2.3/4.5.
4. §16's "YOCO+MLA+÷8 → 0.14 GB" treats **global-cache further compressed 8× along the sequence** as already implemented; CSA's \(m=4\) only gives **÷4 → 0.29 GB**. ÷8 is an extra design choice, not a necessary corollary of CSA.

Two theory-level conclusions that change training/inference priority:

- **The uncompressed 8K sliding window dominates long-context attention FLOPs.** Changing `index_topk` from 256 to 512 moves only about 3% of score+AV volume; CSA/HCA savings vs dense appear only when \(n \gg 8\mathrm{K}\). Phase B keeping dense/sliding at 4K is correct on complexity — at short context CSA concat is even slightly more expensive than dense.
- **Uncompressed decoder cross-attn overtakes decoder MLP from 128K** (about 3.2× at 128K, about 6.5× at 256K). For the middle tier to deliver "128K–256K usable", the global cache must go through compression or query-aware selection; encoder-top raw KV cannot be handed as-is to 26 layers of cross-attn.

---

## 1. Notation and base

Base is `openbmb/MiniCPM5-2B` `config.json` (Llama GQA, Apache-2.0):

\[
d=2048,\quad V=130560,\quad n_h=16,\quad n_{\mathrm{kv}}=2,\quad d_h=128,\quad L_0=42,\quad d_{\mathrm{ff}}^{\mathrm{dense}}=6144.
\]

**No MiniCPM-2B μP.** `scale_emb=1`, `logit_scale=d/\mathrm{dim\_model\_base}=2048/2048=1`, residual multiplier \(1\) (not \(1.4/\sqrt{L}\)).

YOCO split: Encoder (self-decoder) \(L_e=16\), Decoder (cross-decoder) \(L_d=26\). DeepSeekMoE single expert

\[
E = 3\,d\,d_{\mathrm{moe}} = 3\cdot 2048\cdot 2048 = 12.58\mathrm{M}.
\]

Untied embedding + LM head

\[
\mathrm{Emb} = Vd = 130560\cdot 2048 = 0.2674\mathrm{B},\qquad
\mathrm{LM_{head}} = Vd = 0.2674\mathrm{B}.
\]

**Published self-attention** is MiniCPM5 GQA (16 Q / 2 KV), not MiniCPM-2B's 1.25×MHA placeholder:

\[
A_{\mathrm{self}} = 2d^{2} + 2d\cdot(n_{\mathrm{kv}}d_h) = 9.437\mathrm{M}.
\]

**Cross-attn** counts only Q/O (K/V come from the YOCO cache): \(A_{\times}=2d^{2}=8.389\mathrm{M}\). Cache projections \(W_K,W_V\): \(2\cdot d\cdot 256=1.05\mathrm{M}\), not included in the published 12.25B stack total.

Per-token, per-stack activation (plan accounting: input table counted in Encoder, untied head counted in Decoder):

\[
\begin{aligned}
N_{\mathrm{enc}}^{\mathrm{act}} &= \mathrm{Emb} + L_e A_{\mathrm{self}} + L_e(n_s^{e}+k^{e})E,\\
N_{\mathrm{dec}}^{\mathrm{act}} &= \mathrm{LM_{head}} + L_d(A_{\mathrm{self}}+A_{\times}) + L_d(n_s^{d}+k^{d})E.
\end{aligned}
\]

One full forward (input table and head each counted once):

\[
N_{\mathrm{fwd}}^{\mathrm{act}} = \mathrm{Emb} + \mathrm{LM_{head}} + (N_{\mathrm{enc}}^{\mathrm{act}}-\mathrm{Emb}) + (N_{\mathrm{dec}}^{\mathrm{act}}-\mathrm{LM_{head}}).
\]

After untying, plan-accounting \(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}}\) **equals** \(N_{\mathrm{fwd}}^{\mathrm{act}}\) (each table counted once; the tied-era emb×2 no longer appears). Below they are still listed separately; the numbers are the same.

---

## 2. 882 expert-slot conservation: three tiers, same total params, same expert count, different activation

All three tiers allocate **882 expert instances** across 16+26 layers, **expert layout fixed at 1+20**, changing only top-\(k\):

| Tier | Enc \(n_s+N_r\), top-\(k\) | Dec \(n_s+N_r\), top-\(k\) | Expert instances | Total params |
| --- | --- | --- | ---: | ---: |
| Compute-saving | 1+20, \(k=4\) | 1+20, \(k=6\) | \(16\cdot21+26\cdot21=882\) | 12.25B |
| **Middle (default)** | **1+20, \(k=7\)** | **1+20, \(k=10\)** | **882** | **12.25B** |
| Near-dense | 1+20, \(k=12\) | 1+20, \(k=16\) | 882 | 12.25B |

This is the algebraic reason the middle tier can "tune training cost without changing total params": total params are determined by routed count, training FLOPs by top-\(k\). Cutting total params will not cut training cost. The three tiers have the same expert count, so equal total params is not a coincidence — only top-\(k\) is spun.

Exact activation and sparsity (expert sparsity \(=(n_s+k)/(n_s+N_r)\)):

| Tier | Enc active | Dec active | \(N_{\mathrm{fwd}}^{\mathrm{act}}\) | Plan-accounting \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) | Sparsity enc/dec | 50B tok H100-h (plan accounting) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Compute-saving | 1.43B | 3.02B | 4.45B | 4.45B | 23.8% / 33.3% | 926 |
| **Middle** | **2.03B** | **4.33B** | **6.36B** | **6.36B** | **38.1% / 52.4%** | **1,325** |
| Near-dense | 3.04B | 6.29B | 9.33B | 9.33B | 61.9% / 81.0% | 1,943 |

The plan table's ~930 / ~1,325 / ~1,940 are rounding of 1.43+3.02, 2.03+4.33, 3.04+6.29; exact values as above. Middle-tier sparsity should be written **38% / 52%** (8/21, 11/21).

Fixed portion (same across three tiers, GQA attention):

| Block | Parameters |
| --- | ---: |
| Embedding (untied input table) | 0.267B |
| LM head (untied) | 0.267B |
| Enc self-attention \(16\times 9.437\mathrm{M}\) | 0.151B |
| Dec self-attention \(26\times 9.437\mathrm{M}\) | 0.245B |
| Dec cross-attn \(26\times 8.389\mathrm{M}\) | 0.218B |
| MoE experts \(882\times 12.58\mathrm{M}\) | 11.10B |
| **Total** | **12.25B** |

Attention total \(0.151+0.245+0.218=0.614\mathrm{B}\), **5.0%** of 12.25B; MoE **90.6%**. GQA compresses attention from MiniCPM-2B placeholder ~13%; replacing some CSA/HCA layers with KDA still only moves the attention block (order \(10^{-1}\mathrm{B}\)), **no need to change the 12B total-param target in order to add KDA**.

---

## 3. Middle-tier itemized expansion

Encoder (16 layers, 1 shared + 20 routed, top-\(k=7\)):

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 16\cdot 8\cdot 12.58\mathrm{M} = 1.611\mathrm{B},\\
N_{\mathrm{enc}}^{\mathrm{act}} &= 0.267 + 0.151 + 1.611 = 2.029\mathrm{B},\\
\mathrm{stack_{enc}} &= 0.151 + 16\cdot 21\cdot 12.58\mathrm{M} = 4.379\mathrm{B}.
\end{aligned}
\]

Decoder (26 layers, 1 shared + 20 routed, top-\(k=10\), including cross-attn Q/O):

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 26\cdot 11\cdot 12.58\mathrm{M} = 3.599\mathrm{B},\\
N_{\mathrm{dec}}^{\mathrm{act}} &= 0.267 + 0.245 + 0.218 + 3.599 = 4.330\mathrm{B},\\
\mathrm{stack_{dec}} &= 0.245 + 0.218 + 26\cdot 21\cdot 12.58\mathrm{M} = 7.334\mathrm{B}.
\end{aligned}
\]

\[
N_{\mathrm{total}} = 0.267 + 0.267 + 4.379 + 7.334 = 12.247\mathrm{B},\quad
N_{\mathrm{fwd}}^{\mathrm{act}} = 6.359\mathrm{B}.
\]

Consistent with the plan's "≈12B / ≈2.03B-in / ≈4.33B-out".

---

## 4. Training compute (6NT) and Chinchilla position

Kaplan / Hoffmann accounting: training FLOPs \(\approx 6 N_{\mathrm{act}} T\) (forward 2NT + backward 4NT, ignoring the attention quadratic). H100 effective compute is plan §15.1's \(4.0\times 10^{14}\) FLOPS (about 40% MFU).

Middle tier, \(T=50\mathrm{B}\):

| \(N_{\mathrm{act}}\) accounting | \(N\) | FLOPs | H100-h | A100-h |
| --- | ---: | ---: | ---: | ---: |
| Plan (enc+dec) | 6.36B | \(1.91\times 10^{21}\) | **1,325** | 4,239 |
| Full forward (same as plan, untied) | 6.36B | \(1.91\times 10^{21}\) | 1,325 | 4,239 |
| Encoder only (prefill / early-exit) | 2.03B | \(6.09\times 10^{20}\) | 423 | 1,353 |

1,325 matches the plan's ~1,325. That is the **joint bf16** baseline, not Phase B published wall-clock. The published value is **C1+NVFP4 = 571 H100-h** (43% of joint bf16). C1+FP8 = 729 is the Hopper/Ada fallback. NVFP4 does not change this 6NT table. See [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

**This is not Chinchilla pretraining.** Hoffmann-optimal is about \(20 N\) tokens (dense). 50B / 6.36B ≈ **7.9 tokens / active parameter**, which is upcycle recovery + continued training, not training 12B from scratch. Writing 50–150B as the Phase B recovery budget is correct; treating it as "12B is already fully trained" overfills.

The quadratic attention term is small relative to 6NT on 4K main training (§6); it is not negligible on Phase D's 32K–128K — this is exactly CSA/HCA's train-time value, not only inference-time value.

---

## 5. Attention complexity: the 8K sliding window is the bulk

Number of KV entries that actually participate in core attention per CSA / HCA query (concat = compression branch + uncompressed sliding window, causal):

\[
\begin{aligned}
k_{\mathrm{CSA}}(n) &= \min(k_{\mathrm{index}}, \lfloor n/m\rfloor) + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{HCA}}(n) &= \lfloor n/m'\rfloor + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{dense}}(n) &= n,
\end{aligned}
\]

where \(m=4\), \(m'=128\), \(n_{\mathrm{win}}=8192\), \(k_{\mathrm{index}}=256\) (lower end of the plan's 256–512).

| \(n\) | dense | CSA | HCA | CSA/dense |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,096 | 4,352 | 4,128 | **1.06** |
| 8,192 | 8,192 | 8,448 | 8,256 | **1.03** |
| 32,768 | 32,768 | 8,448 | 8,448 | 0.26 |
| 131,072 | 131,072 | 8,448 | 9,216 | 0.064 |
| 262,144 | 262,144 | 8,448 | 10,240 | 0.032 |
| 1,048,576 | 1,048,576 | 8,448 | 16,384 | 0.008 |

Key points:

1. **When \(n \le 8\mathrm{K}\), CSA/HCA does not save FLOPs** (concat is even slightly more than dense). Phase B first running dense/sliding at 4K, then sparsifying in Phase C, is consistent with the complexity curve, not merely a "training stability" preference.
2. **Past 8K, the 8192-entry sliding window almost pins CSA's \(k\).** `index_topk=256` vs 512: \(k=8448\) vs \(8704\), about **3%**. Long-context ablations should preferentially sweep `n_win ∈ {2K,4K,8K}`, rather than treating `index_topk` as the main cost knob.
3. At 1M, HCA has \(k=\lfloor 10^6/128\rfloor+8192=16384\), still far below dense, but already about 2× CSA; HCA layers should not be the majority at 1M.

Score+AV uses \(4\,d\,n\,k\,L\) (two matmuls × 2 flop/MAC). For one middle-tier forward of length \(n\), MLP term \(2 N_{\mathrm{fwd}}^{\mathrm{act}} n\) vs the attention quadratic:

| \(n\) | MLP forward | 42L dense | Encoder 16L CSA/HCA interleaved | Enc + Dec 26L sliding | Dec 26L **uncompressed** cross-attn |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | \(5.2\times10^{13}\) | \(5.8\times10^{12}\) | \(2.3\times10^{12}\) | \(5.9\times10^{12}\) | \(3.6\times10^{12}\) |
| 32K | \(4.2\times10^{14}\) | \(3.7\times10^{14}\) | \(3.6\times10^{13}\) | \(9.3\times10^{13}\) | \(2.3\times10^{14}\) |
| 128K | \(1.7\times10^{15}\) | \(5.9\times10^{15}\) | \(1.5\times10^{14}\) | \(3.8\times10^{14}\) | \(3.7\times10^{15}\) |
| 256K | \(3.3\times10^{15}\) | \(2.4\times10^{16}\) | \(3.2\times10^{14}\) | \(7.8\times10^{14}\) | \(1.5\times10^{16}\) |

Reading the table:

- 4K: attention quadratic ≪ MLP; 6NT is a good approximation.
- 32K: 42L dense is already the same order as MLP; after encoder compression it can still be an order of magnitude below MLP.
- **From 128K, uncompressed cross-attn alone exceeds the whole-network MLP.** If YOCO stores the global cache as raw top-layer KV, the decoder spends in cross-attn what CSA/HCA saved on the encoder side.

---

## 6. Prefill early-exit and the decode bottleneck

YOCO (arXiv 2405.05254): prefill only needs to finish the self-decoder to write global \(\hat K,\hat V\); the cross-decoder enters only when generating the first token. Middle tier:

| | Value |
| --- | --- |
| Layer-count ratio \(L_e/(L_e+L_d)\) | 16/42 = **38%** |
| Activation ratio \(N_{\mathrm{enc}}^{\mathrm{act}}/(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}})\) | 2.03/6.36 = **32%** |

So "lighter on the input side" is not just halving layers: middle-tier asymmetric MoE further compresses prefill compute to about one-third of a full-model forward. This is consistent with "primary delivery 128K–256K; ultra-long input is the encoder's job".

One decode step (decoder MLP forward \(2 N_{\mathrm{dec}}^{\mathrm{act}}\) vs cross-attn over a global cache of length \(n\)):

| \(n\) | dec-MLP | xattn full | full / MLP | xattn CSA \(m=4\) | xattn CSA top-\(k\)+8K window |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | \(8.66\times10^9\) | \(6.98\times10^9\) | 0.81× | 0.20× | 0.21× |
| **128K** | \(8.66\times10^9\) | \(2.79\times10^{10}\) | **3.22×** | 0.81× | **0.21×** |
| **256K** | \(8.66\times10^9\) | \(5.58\times10^{10}\) | **6.45×** | 1.61× | **0.21×** |
| 1M | \(8.66\times10^9\) | \(2.23\times10^{11}\) | 25.8× | 6.45× | 0.21× |

**For the middle tier to be usable at 128K–256K, the global cache cannot be uncompressed raw KV.** Minimum: \(m=4\) compression along the sequence (at 256K, cross-attn is still about 1.6× MLP). To make decode MLP-dominated again, apply CSA-style top-\(k\) (or PDSA calibrated fallback) to the global cache, pinning per-query \(k\) at \(n_{\mathrm{win}}+k_{\mathrm{index}}\approx 8.4\mathrm{K}\). This is in the same direction as plan §14 "query-aware before write-first, calibrated sparse fallback", and it is a compute constraint, not only a recall constraint.

---

## 7. KV cache: §16's numbers are right; the assumptions need to be written clearly

bf16, content bytes (excluding allocator padding). "1M" = \(10^6\) tokens (not \(2^{20}\)). GB = \(10^9\) B.

MHA: \(2d\) elements per layer per token (K and V). **GQA-2**: 2 KV heads (\(n_{\mathrm{kv}}=2\), \(d_h=128\), \(2\times 256\) elements per token). MLA-576: DeepSeek-V3-style latent \(k_{\mathrm{lora}}+d_{\mathrm{rope}}=512+64\).

\[
\mathrm{bytes} = n \cdot L_{\mathrm{cache}} \cdot d_{\mathrm{kv}} \cdot 2.
\]

| Recipe | 8K | 32K | 128K | 256K | 1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| decoder-only MHA 42L | 2.82 | 11.27 | 45.10 | 90.19 | **344.06** |
| decoder-only GQA-2 42L | 0.35 | 1.41 | 5.64 | 11.27 | **43.01** |
| decoder-only MLA-576 42L | 0.40 | 1.59 | 6.34 | 12.68 | **48.38** |
| **YOCO + MLA-576 (1-layer global)** | 0.01 | 0.04 | 0.15 | 0.30 | **1.15** |
| YOCO + MLA + sequence ÷8 | — | — | 0.02 | 0.04 | **0.14** |
| YOCO + MLA + CSA \(m=4\) | — | — | 0.04 | 0.08 | **0.29** |
| YOCO + MiniCPM5 GQA-2 (1-layer global) | 0.01 | 0.03 | 0.13 | 0.27 | 1.02 |
| Decoder 26×8K window (MLA) | 0.25 | 0.25 | 0.25 | 0.25 | **0.25** |
| Encoder 16×8K window (MLA) | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |

8K/32K/128K/256K columns are \(8192/32768/131072/262144\); 1M column is \(10^6\) (same as plan §16).

Checking plan §16:

- 344 / 43 / 48 / 1.15 / 0.14 GB: **pass** (decimal GB, \(n=10^6\), GQA=2 heads, MLA=576, 42 layers).
- "+≈0.2 GB of 8K sliding window": Decoder 26 layers is exactly **0.25 GB**, order of magnitude passes. Encoder itself still has about 0.15 GB of 8K window (original YOCO also acknowledges the self-decoder has a constant cache). At 128K–256K the global cache (0.15–0.30 GB) and the 8K windows (0.25+0.15 GB) are the same order — another reminder that **the 8K window is not free**.
- **÷8 is not a corollary of CSA \(m=4\).** \(m=4\) gives 0.29 GB @ 1M. 0.14 GB needs extra sequence compression (heavier HCA-style global pooling, or another compression layer on the global cache). Mark it in the plan as "optional stretch", so impl does not default to CSA and think it already matches 0.14 GB.

Primary target 128K–256K: YOCO+MLA global cache **0.15–0.30 GB**, even plus both-side 8K windows still far below decoder-only MHA's 45–90 GB. 1M inference is feasible on mid-range hardware; this claim holds in theory.

---

## 8. Published GQA vs MiniCPM5-native CSA/HCA (sensitivity, spec unchanged)

The published ledger is MiniCPM5 GQA 16/2, **no longer using the 1.25×MHA placeholder**. DeepSeek-V4-Flash uses `head_dim=512`, `q_lora_rank=1024` (\(d=4096\)). Those dimensions cannot be ported to MiniCPM5's \(d=2048\). The sensitivity model uses MiniCPM5-native MQA-128 (\(d_h=128\), single KV head) + LoRA-Q 512 + grouped output, counting CSA/HCA projections per V4 paper §2.3:

| Per-layer self-attention | Parameters |
| --- | ---: |
| MiniCPM5 MHA \(4d^2\) (not adopted) | 16.78M |
| **Published GQA 16/2** | **9.437M** |
| CSA MQA-128 (including indexer) | 10.00M |
| HCA MQA-128 | 8.41M |
| CSA/HCA average | 9.20M |

The detailed model is **almost the same size** as published GQA (9.20M vs 9.44M). If later frozen to MQA-128 CSA/HCA:

- Total params land at about **12.24B**, active about **2.03B / 4.32B** (attention is also in the activation).
- Aligns with 12.25B / 2.03 / 4.33 to two decimal places; no need to change routed for CSA.
- If one still wants to fine-tune activation, add top-\(k\) rather than add routed.

**Until projection dimensions are frozen, keep using GQA 16/2 as the middle-tier spec.** The detailed model only shows: switching to CSA/HCA will not suddenly blow past 12B.

---

## 9. First-layer dense (Phase A) and "no μP"

Plan Phase A: "keep the first layer of each stack dense". The current §3 table accounts **all MoE**. Replacing layer 1 of each stack with MiniCPM5 dense SwiGLU (\(3\cdot d\cdot 6144=37.75\mathrm{M}\)):

| | All MoE (spec table) | First-layer dense, Nr unchanged |
| --- | ---: | ---: |
| Total params | 12.25B | **11.79B** |
| Enc active | 2.03B | 1.97B |
| Dec active | 4.33B | 4.23B |

Activation is still inside the ≈2.0 / ≈4.3 rounding. Total-param gap is 0.45B. To restore ~12.25B without changing top-\(k\): Encoder/Decoder routed **20→21** (everything else unchanged) → total params **12.30B**, activation still 1.97 / 4.23. Recommend adopting "Enc 1+21 / Dec 1+21, first-layer dense" when landing Phase A, still calling it middle-tier externally.

**MiniCPM5 is Llama; there is no μP scaling.**

- `scale_emb=1`, logits not divided by 9, residual multiplier is identity \(1\).
- **Do not** port MiniCPM-2B's \(1.4/\sqrt{40}\) or \(1.4/\sqrt{42}\). After splitting into 16+26, do not recompute residuals from the new stack depths either.
- What is split is layers, not scale.

---

## 10. Information-theoretic constraint (PDSA) and the middle tier

PDSA (Zou & Donz, arXiv 2606.28876) gives a constraint orthogonal to the parameter budget but in the same direction as §6's decode bottleneck:

- **No-signal-at-write-time**: query-independent writability on static text is near-random (AUC 0.63–0.66). HCA / KDA / CSA compression steps are all write-first and will systematically drop blocks that have no signal at write time and are only hit at query time.
- Therefore **a query-time fallback to low-compression / raw KV must be kept**. §6 shows this fallback is also the compute path to take at 128K–256K: only by shrinking cross-attn \(k\) from \(n\) to \(n_{\mathrm{win}}+k_{\mathrm{index}}\) does decode return to MLP-dominated.
- The middle tier gives the precise-retrieval budget to Decoder cross-attn (4.33B active, 26 layers) and the long-input compression budget to the Encoder (2.03B, 16 layers). This is consistent with "query-aware read happens at decode, write-first compression happens at prefill". Do not, to "save a bit more prefill", turn the encoder into pure HCA/KDA with no CSA anchor + fallback.

Linear attention cannot fix lost-in-the-middle (already clarified in plan §2.5); the middle tier also does not rely on KDA to protect mid-span recall. IN2/FILM + calibrated fallback is the mid-span path.

---

## 11. Claim ledger

Mechanized checks of the plan's published numbers (`scripts/param_budget.py --verify`, GQA 16/2, all MoE, middle tier):

| Claim | Result |
| --- | --- |
| Total params ≈12B | PASS (12.25B) |
| Enc active ≈2.03B | PASS (2.03B) |
| Dec active ≈4.33B | PASS (4.33B) |
| Emb / Enc attn / Dec attn / cross ≈ 0.27 / 0.15 / 0.25 / 0.22B | PASS |
| Expert sparsity 8/21, 11/21 | PASS (38.1%/52.4%) |
| Three tiers share 882 expert slots | PASS |
| 50B tok ≈1325 H100-h | PASS (1325) |
| 1M KV 344 / 43 / 48 / 1.15 / 0.14 GB | PASS |
| 26×8K window ≈0.25 GB | PASS (0.25 GB) |
| No μP: logit_scale = 1; residual identity | PASS |
| freeze-enc ≈75% joint; independent splice 146%; C1 curriculum 79% | PASS (see curriculum note) |

Unfreeze curriculum another 12 claims (published C1, detach, untied \(E\), Adam 62%, legal split ≤85%, etc.) plus middle-tier 22, FP8 fallback 13, NVFP4 16 total **`--verify` 63/63**, see [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md), [`FP8_THEORY.md`](FP8_THEORY.md), [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

Text layer (not in `--verify`, already expanded above):

| Text | Treatment |
| --- | --- |
| MiniCPM-2B / \(d=2304\) / \(V=122753\) / 16/24 | already changed to MiniCPM5-2B / \(d=2048\) / \(V=130560\) / 16+26 |
| 720 expert slots, 1+17 layout | already changed to 882 slots, both stacks 1+20, only top-\(k\) changes |
| Sparsity 37%/53% or 7/18, 9/18 | already changed to 38.1%/52.4% (8/21, 11/21) |
| Attention ~7% or ~13% | already changed to ~5.0% (GQA) |
| Diagrams 3B/6B or 2.3/4.5 | already changed to 2.03B/4.33B |
| Global cache ÷8 | already marked optional; CSA \(m=4\) corresponds to ÷4 → 0.29 GB |
| μP logits /9, residual \(1.4/\sqrt{40}\) | already changed to identity 1 |
| First-layer dense not in spec table | total params land at 11.79B; restore 12.30B with Enc/Dec \(N_r=21\) |

---

## 12. Constraints on implementation (read out of the theory, not new features)

1. **Spec stays locked to middle-tier**: Enc 16L, 1+20, top-\(k=7\); Dec 26L, 1+20, top-\(k=10\); published attention GQA 16/2. If Phase A first-layer dense, both stacks' routed go to 21.
2. **Global cache must be compressed or selectable.** Otherwise 128K–256K decode is dominated by 26 layers of cross-attn, and YOCO+CSA encoder gains are spent back.
3. **`n_win=8K` is the first cost knob**; `index_topk` is not. Ablate {2K,4K,8K} per plan §7.
4. **Do not introduce MiniCPM-2B μP**, and do not recompute residuals from the new stack depths.
5. **Do not expect CSA to save compute on 4K main training**; sparsify in Phase C, long context in Phase D, consistent with the FLOPs curve.
6. After projection dimensions freeze, re-run `python3 scripts/param_budget.py --attn csa_mqa64 --full`; CSA/HCA is almost the same params as GQA, do not change the layer split.
7. **Phase B follows published C1** (both stacks MoE first, freeze Encoder, detach cache, freeze input table, B1 may train lm_head, B2≥10B). Do not treat the two stacks as independent LMs then splice.
8. **Phase B wall-clock follows published C1+NVFP4** (571 H100-h; linear GEMMs that need not be bf16; B0 student / L0 / indexer / embed / LN / router / softmax high precision). Published 2.0× vs bf16, does not change 6NT, do not publish 4×. Joint bf16 is baseline only. C1+FP8 729 is Hopper/Ada fallback.

Recompute commands:

```bash
python3 scripts/param_budget.py              # middle-tier summary
python3 scripts/param_budget.py --tier all   # three-tier comparison
python3 scripts/param_budget.py --full       # KV / complexity / no μP / Nr restore search
python3 scripts/param_budget.py --verify     # spec + unfreeze curriculum + FP8 fallback + NVFP4 assertions
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
