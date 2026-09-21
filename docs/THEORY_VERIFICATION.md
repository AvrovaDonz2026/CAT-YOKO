# CAT-YOKO Middle-Tier Theory Verification

> Spec frozen as the training-plan default: `CAT-YOKO-12B`, encoder activation ≈2.03B / input token, decoder activation ≈4.33B / output token.
> The base is Apache-2.0 **MiniCPM5-2B** (Llama GQA), not MiniCPM-2B.
> This document is a **recomputable theoretical check** only (parameters, FLOPs, KV, complexity, no μP). It does not introduce a new architecture.
> Number source: `python3 scripts/param_budget.py --full`; assertions: `python3 scripts/param_budget.py --verify` and `python3 -m unittest tests.test_param_budget`.
> Architecture (causality, split equivalence, M1/M2/M3) is in [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md).
> Unfreeze curriculum (C1 locked-in recipe: freeze boundaries, Theorems D/E) is in [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md).
> NVFP4: Phase B wall-clock is locked in as **C1+NVFP4 = 571 H100-h** (Theorem G; does not change 6NT) in [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Hopper/Ada fallback C1+FP8 = 729 is in [`FP8_THEORY.md`](FP8_THEORY.md).

---

## 0. Conclusions (read this first)

Under **the plan's chosen accounting convention**, the middle tier is internally consistent:

| Quantity | Plan claim | Exact value | Verdict |
| --- | ---: | ---: | --- |
| Total parameters | ≈12B | **12.25B** | PASS |
| Encoder activation / input token | ≈2.03B | **2.03B** | PASS |
| Decoder activation / output token | ≈4.33B | **4.33B** | PASS |
| 50B tok training | ~1,325 H100-h | **1,325 H100-h** | PASS |
| 1M KV (decoder-only MHA / GQA-2 / MLA / YOCO+MLA / ÷8) | 344 / 43 / 48 / 1.15 / 0.14 GB | **344.06 / 43.01 / 48.38 / 1.15 / 0.14 GB** | PASS |

Places where the plan text must be updated relative to the old MiniCPM-2B ledger, without changing the spec:

1. **Expert sparsity** is **38.1% (8/21) / 52.4% (11/21)**, not 7/18, 9/18, and not 37%/53%.
2. **Attention is ~5.0% of total parameters** (GQA 16/2); MoE is still **90.6%**. Do not carry over MiniCPM-2B placeholder attention at 13%, or the ~7% from the 24B era. KDA remains parameter-neutral.
3. Diagrams should show **≈2.03B / ≈4.33B**, layer counts **16+26=42**, not 16/24 and 2.3/4.5.
4. The §16 line “YOCO+MLA+÷8 → 0.14 GB” treats **an extra 8× sequence-wise compression of the global cache** as already implemented. CSA’s \(m=4\) only gives **÷4 → 0.29 GB**. ÷8 is an extra design choice, not a necessary corollary of CSA.

Two theoretical conclusions that change training/inference priorities:

- **The uncompressed 8K sliding window dominates long-context attention FLOPs.** Changing `index_topk` from 256 to 512 moves only about 3% of the score+AV volume. CSA/HCA savings versus dense appear only at \(n \gg 8\mathrm{K}\). Phase B staying dense/windowed at 4K is correct on complexity grounds — at short context, CSA concat is even slightly more expensive than dense.
- **Uncompressed decoder cross-attn overtakes decoder MLP starting at 128K** (about 3.2× at 128K, about 6.5× at 256K). For the middle tier to be usable at 128K–256K, the global cache must be compressed or query-aware selected. The encoder’s top-layer raw KV cannot be handed unchanged to 26 layers of cross-attn.

---

## 1. Notation and base

The base is `openbmb/MiniCPM5-2B` `config.json` (Llama GQA, Apache-2.0):

\[
d=2048,\quad V=130560,\quad n_h=16,\quad n_{\mathrm{kv}}=2,\quad d_h=128,\quad L_0=42,\quad d_{\mathrm{ff}}^{\mathrm{dense}}=6144.
\]

**There is no MiniCPM-2B μP.** `scale_emb=1`, `logit_scale=d/\mathrm{dim\_model\_base}=2048/2048=1`, residual multiplier \(1\) (not \(1.4/\sqrt{L}\)).

YOCO split: Encoder (self-decoder) \(L_e=16\), Decoder (cross-decoder) \(L_d=26\). DeepSeekMoE single expert

\[
E = 3\,d\,d_{\mathrm{moe}} = 3\cdot 2048\cdot 2048 = 12.58\mathrm{M}.
\]

Untied embedding + LM head

\[
\mathrm{Emb} = Vd = 130560\cdot 2048 = 0.2674\mathrm{B},\qquad
\mathrm{LM_{head}} = Vd = 0.2674\mathrm{B}.
\]

**Published self-attention** is MiniCPM5 GQA (16 Q / 2 KV), not MiniCPM-2B’s 1.25×MHA placeholder:

\[
A_{\mathrm{self}} = 2d^{2} + 2d\cdot(n_{\mathrm{kv}}d_h) = 9.437\mathrm{M}.
\]

**Cross-attn** counts only Q/O (K/V come from the YOCO cache): \(A_{\times}=2d^{2}=8.389\mathrm{M}\). Cache projections \(W_K,W_V\): \(2\cdot d\cdot 256=1.05\mathrm{M}\), not included in the published 12.25B stack total.

Per-token, per-stack activation (plan accounting: input table counted on the Encoder, untied head counted on the Decoder):

\[
\begin{aligned}
N_{\mathrm{enc}}^{\mathrm{act}} &= \mathrm{Emb} + L_e A_{\mathrm{self}} + L_e(n_s^{e}+k^{e})E,\\
N_{\mathrm{dec}}^{\mathrm{act}} &= \mathrm{LM_{head}} + L_d(A_{\mathrm{self}}+A_{\times}) + L_d(n_s^{d}+k^{d})E.
\end{aligned}
\]

One full forward pass (input table and head each counted once):

\[
N_{\mathrm{fwd}}^{\mathrm{act}} = \mathrm{Emb} + \mathrm{LM_{head}} + (N_{\mathrm{enc}}^{\mathrm{act}}-\mathrm{Emb}) + (N_{\mathrm{dec}}^{\mathrm{act}}-\mathrm{LM_{head}}).
\]

After untying, the plan accounting \(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}}\) **equals** \(N_{\mathrm{fwd}}^{\mathrm{act}}\) (each table counted once; the tied-era emb×2 no longer appears). The sections below still list the two stacks separately; the numbers are the same.

---

## 2. 882 expert-slot conservation: three tiers, same total params, same expert count, different activation

All three tiers assign **882 expert instances** to 16+26 layers. **Expert layout is fixed at 1+20**; only top-\(k\) changes:

| Tier | Enc \(n_s+N_r\), top-\(k\) | Dec \(n_s+N_r\), top-\(k\) | Expert instances | Total params |
| --- | --- | --- | ---: | ---: |
| Low-compute | 1+20, \(k=4\) | 1+20, \(k=6\) | \(16\cdot21+26\cdot21=882\) | 12.25B |
| **Middle (default)** | **1+20, \(k=7\)** | **1+20, \(k=10\)** | **882** | **12.25B** |
| Near-dense | 1+20, \(k=12\) | 1+20, \(k=16\) | 882 | 12.25B |

This is the algebraic reason the middle tier can change training cost without changing total parameters: total params are set by the routed count; training FLOPs are set by top-\(k\). Cutting total params does not cut training cost. The three tiers have the same expert count, so matching total params is not a coincidence — only top-\(k\) is rotated.

Exact activation and sparsity (expert sparsity \(=(n_s+k)/(n_s+N_r)\)):

| Tier | Enc act. | Dec act. | \(N_{\mathrm{fwd}}^{\mathrm{act}}\) | Plan accounting \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) | Sparsity enc/dec | 50B tok H100-h (plan accounting) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Low-compute | 1.43B | 3.02B | 4.45B | 4.45B | 23.8% / 33.3% | 926 |
| **Middle** | **2.03B** | **4.33B** | **6.36B** | **6.36B** | **38.1% / 52.4%** | **1,325** |
| Near-dense | 3.04B | 6.29B | 9.33B | 9.33B | 61.9% / 81.0% | 1,943 |

The plan table’s ~930 / ~1,325 / ~1,940 are roundings of 1.43+3.02, 2.03+4.33, 3.04+6.29; exact values are above. Middle-tier sparsity should be written **38% / 52%** (8/21, 11/21).

Fixed part (identical across the three tiers, GQA attention):

| Block | Params |
| --- | ---: |
| Embedding (untied input table) | 0.267B |
| LM head (untied) | 0.267B |
| Enc self-attention \(16\times 9.437\mathrm{M}\) | 0.151B |
| Dec self-attention \(26\times 9.437\mathrm{M}\) | 0.245B |
| Dec cross-attn \(26\times 8.389\mathrm{M}\) | 0.218B |
| MoE experts \(882\times 12.58\mathrm{M}\) | 11.10B |
| **Total** | **12.25B** |

Attention total \(0.151+0.245+0.218=0.614\mathrm{B}\), **5.0%** of 12.25B; MoE **90.6%**. GQA compresses attention from MiniCPM-2B’s placeholder ~13%. Replacing some CSA/HCA layers with KDA still only touches the attention block (order \(10^{-1}\mathrm{B}\)), so **the 12B total-parameter target does not need to change to add KDA**.

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

This matches the plan’s “≈12B / ≈2.03B-in / ≈4.33B-out”.

---

## 4. Training compute (6NT) and the Chinchilla position

Kaplan / Hoffmann accounting: training FLOPs \(\approx 6 N_{\mathrm{act}} T\) (forward 2NT + backward 4NT, ignoring the quadratic attention term). Effective H100 compute is the plan §15.1 figure \(4.0\times 10^{14}\) FLOPS (about 40% MFU).

Middle tier, \(T=50\mathrm{B}\):

| \(N_{\mathrm{act}}\) convention | \(N\) | FLOPs | H100-h | A100-h |
| --- | ---: | ---: | ---: | ---: |
| Plan (enc+dec) | 6.36B | \(1.91\times 10^{21}\) | **1,325** | 4,239 |
| Full forward (same as plan, untied) | 6.36B | \(1.91\times 10^{21}\) | 1,325 | 4,239 |
| Encoder only (prefill / early-exit) | 2.03B | \(6.09\times 10^{20}\) | 423 | 1,353 |

1,325 matches the plan’s ~1,325. That is the **joint bf16** control, not the Phase B published wall-clock. The published value is **C1+NVFP4 = 571 H100-h** (43% of joint bf16). C1+FP8 = 729 is the Hopper/Ada fallback. NVFP4 does not change this 6NT table. See [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

**This is not Chinchilla pretraining.** Hoffmann-optimal is about \(20 N\) tokens (dense). 50B / 6.36B ≈ **7.9 tokens / active parameter**, which is upsampling recovery plus continued training, not training a 12B from scratch. Writing 50–150B as a Phase B recovery budget is correct; reading it as “the 12B is already fully trained” overstates how much training it has received.

The quadratic attention term is small relative to 6NT on the 4K main training run (§6). On Phase D’s 32K–128K it is not negligible — that is the training-time value of CSA/HCA, not only the inference-time value.

---

## 5. Attention complexity: the 8K window is the bulk

Number of KV entries that actually participate in core attention per query for CSA / HCA (concat = compressed branch + uncompressed sliding window, causal):

\[
\begin{aligned}
k_{\mathrm{CSA}}(n) &= \min(k_{\mathrm{index}}, \lfloor n/m\rfloor) + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{HCA}}(n) &= \lfloor n/m'\rfloor + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{dense}}(n) &= n,
\end{aligned}
\]

where \(m=4\), \(m'=128\), \(n_{\mathrm{win}}=8192\), \(k_{\mathrm{index}}=256\) (the lower end of the plan’s 256–512 range).

| \(n\) | dense | CSA | HCA | CSA/dense |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,096 | 4,352 | 4,128 | **1.06** |
| 8,192 | 8,192 | 8,448 | 8,256 | **1.03** |
| 32,768 | 32,768 | 8,448 | 8,448 | 0.26 |
| 131,072 | 131,072 | 8,448 | 9,216 | 0.064 |
| 262,144 | 262,144 | 8,448 | 10,240 | 0.032 |
| 1,048,576 | 1,048,576 | 8,448 | 16,384 | 0.008 |

Takeaways:

1. **At \(n \le 8\mathrm{K}\), CSA/HCA do not save FLOPs** (concat is even slightly larger than dense). Running dense/windowed first in Phase B at 4K, then sparsifying in Phase C, matches the complexity curve. It is not merely a “training stability” preference.
2. **Past 8K, the 8192-entry window almost pins CSA’s \(k\).** `index_topk=256` versus 512: \(k=8448\) vs \(8704\), about **3%**. Long-context ablations should sweep `n_win ∈ {2K,4K,8K}` first, rather than treating `index_topk` as the main cost knob.
3. At 1M, HCA has \(k=\lfloor 10^6/128\rfloor+8192=16384\), still far below dense, but already about 2× CSA. HCA layers should not be the majority of layers at 1M.

Score+AV uses \(4\,d\,n\,k\,L\) (two matmuls × 2 flop/MAC). For one middle-tier forward of length \(n\), MLP term \(2 N_{\mathrm{fwd}}^{\mathrm{act}} n\) versus the quadratic attention term:

| \(n\) | MLP forward | 42L dense | Encoder 16L CSA/HCA interleaved | Enc + Dec 26L window | Dec 26L **uncompressed** cross-attn |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | \(5.2\times10^{13}\) | \(5.8\times10^{12}\) | \(2.3\times10^{12}\) | \(5.9\times10^{12}\) | \(3.6\times10^{12}\) |
| 32K | \(4.2\times10^{14}\) | \(3.7\times10^{14}\) | \(3.6\times10^{13}\) | \(9.3\times10^{13}\) | \(2.3\times10^{14}\) |
| 128K | \(1.7\times10^{15}\) | \(5.9\times10^{15}\) | \(1.5\times10^{14}\) | \(3.8\times10^{14}\) | \(3.7\times10^{15}\) |
| 256K | \(3.3\times10^{15}\) | \(2.4\times10^{16}\) | \(3.2\times10^{14}\) | \(7.8\times10^{14}\) | \(1.5\times10^{16}\) |

Reading the table:

- 4K: quadratic attention ≪ MLP; 6NT is a good approximation.
- 32K: 42L dense is already the same order as MLP; after encoder compression it can still be an order of magnitude below MLP.
- **From 128K on, uncompressed cross-attn alone exceeds the whole-network MLP.** If YOCO stores the global cache as raw top-layer KV, the decoder spends in cross-attn what CSA/HCA saved on the encoder side.

---

## 6. Prefill early-exit and the decode bottleneck

YOCO (arXiv 2405.05254): prefill only needs to finish the self-decoder to write the global \(\hat K,\hat V\); the cross-decoder enters when the first generated token is produced. Middle tier:

| | Value |
| --- | --- |
| Layer ratio \(L_e/(L_e+L_d)\) | 16/42 = **38%** |
| Activation ratio \(N_{\mathrm{enc}}^{\mathrm{act}}/(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}})\) | 2.03/6.36 = **32%** |

So “lighter on the input side” is more than halving the layer count: the middle tier’s asymmetric MoE further compresses prefill compute to about one third of a full-model forward. That is consistent with “primary delivery 128K–256K; ultra-long input is the encoder’s job”.

One decode step (decoder MLP forward \(2 N_{\mathrm{dec}}^{\mathrm{act}}\) vs cross-attn against a global cache of length \(n\)):

| \(n\) | dec-MLP | xattn full | full / MLP | xattn CSA \(m=4\) | xattn CSA top-\(k\)+8K window |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | \(8.66\times10^9\) | \(6.98\times10^9\) | 0.81× | 0.20× | 0.21× |
| **128K** | \(8.66\times10^9\) | \(2.79\times10^{10}\) | **3.22×** | 0.81× | **0.21×** |
| **256K** | \(8.66\times10^9\) | \(5.58\times10^{10}\) | **6.45×** | 1.61× | **0.21×** |
| 1M | \(8.66\times10^9\) | \(2.23\times10^{11}\) | 25.8× | 6.45× | 0.21× |

**For the middle tier to be usable at 128K–256K, the global cache cannot be uncompressed raw KV.** Minimum: \(m=4\) compression along the sequence (at 256K, cross-attn is still about 1.6× MLP). To make decode MLP-dominated again, the global cache needs CSA-style top-\(k\) (or a PDSA calibrated fallback) that pins per-query \(k\) at \(n_{\mathrm{win}}+k_{\mathrm{index}}\approx 8.4\mathrm{K}\). That is the same direction as plan §14 “query-aware before write-first, calibrated sparse fallback”, and it is a compute constraint, not only a recall constraint.

---

## 7. KV cache: the §16 numbers are correct; the assumptions need to be written down

bf16, content bytes (no allocator padding). “1M” = \(10^6\) tokens (not \(2^{20}\)). GB = \(10^9\) B.

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

The 8K/32K/128K/256K columns are \(8192/32768/131072/262144\); the 1M column is \(10^6\) (same as plan §16).

Checking plan §16:

- 344 / 43 / 48 / 1.15 / 0.14 GB: **PASS** (GB to two decimals, \(n=10^6\), GQA=2 heads, MLA=576, 42 layers).
- “+≈0.2 GB of 8K sliding window”: Decoder 26 layers is exactly **0.25 GB**, order of magnitude PASS. The encoder itself still has about 0.15 GB of 8K window (the original YOCO paper also acknowledges a constant cache on the self-decoder). At 128K–256K the global cache (0.15–0.30 GB) and the 8K windows (0.25+0.15 GB) are the same order — another reminder that **the 8K window is not free**.
- **÷8 is not a corollary of CSA \(m=4\).** \(m=4\) gives 0.29 GB @ 1M. 0.14 GB needs extra sequence compression (heavier HCA-style global pooling, or another compression layer on the global cache). Mark it in the plan as “optional stretch”, so an implementation that ships CSA defaults is not assumed to have already hit 0.14 GB.

Primary target 128K–256K: YOCO+MLA global cache **0.15–0.30 GB**, and even with both-side 8K windows it is far below decoder-only MHA’s 45–90 GB. 1M inference on mid-range hardware is feasible; that claim holds theoretically.

---

## 8. Published GQA vs MiniCPM5-native CSA/HCA (sensitivity; spec unchanged)

The published ledger is MiniCPM5 GQA 16/2; **the 1.25×MHA placeholder is no longer used**. DeepSeek-V4-Flash uses `head_dim=512`, `q_lora_rank=1024` (\(d=4096\)). Those dimensions cannot be copied onto MiniCPM5’s \(d=2048\). The sensitivity model uses MiniCPM5-native MQA-128 (\(d_h=128\), single KV head) + LoRA-Q 512 + grouped output, counting CSA/HCA projections per V4 paper §2.3:

| Per-layer self-attention | Params |
| --- | ---: |
| MiniCPM5 MHA \(4d^2\) (not adopted) | 16.78M |
| **Published GQA 16/2** | **9.437M** |
| CSA MQA-128 (including indexer) | 10.00M |
| HCA MQA-128 | 8.41M |
| CSA/HCA average | 9.20M |

The detailed model is **almost the same size** as published GQA (9.20M vs 9.44M). If later frozen to MQA-128 CSA/HCA:

- Total params land at about **12.24B**, activation about **2.03B / 4.32B** (attention is in the activation too).
- That lines up with 12.25B / 2.03 / 4.33 to two decimal places; routed count does not need to change for CSA.
- If activation still needs a tweak, add top-\(k\), not routed experts.

**Until projection dimensions are frozen, keep GQA 16/2 as the middle-tier spec.** The detailed model only shows that switching to CSA/HCA will not suddenly blow past 12B.

---

## 9. First-layer dense (Phase A) and “no μP”

Plan Phase A: “keep the first layer of each stack dense”. The current §3 table accounts **all-MoE**. Replacing layer 1 of each stack with MiniCPM5 dense SwiGLU (\(3\cdot d\cdot 6144=37.75\mathrm{M}\)):

| | All MoE (spec table) | First-layer dense, Nr unchanged |
| --- | ---: | ---: |
| Total params | 12.25B | **11.79B** |
| Enc activation | 2.03B | 1.97B |
| Dec activation | 4.33B | 4.23B |

Activation is still inside the ≈2.0 / ≈4.3 rounding. Total params differ by 0.45B. To restore ~12.25B without changing top-\(k\): Encoder/Decoder routed **20→21** (everything else unchanged) → total params **12.30B**, activation still 1.97 / 4.23. Recommend adopting “Enc 1+21 / Dec 1+21, first-layer dense” when Phase A lands, still calling it the middle tier externally.

**MiniCPM5 is Llama; there is no μP scaling.**

- `scale_emb=1`, logits are not divided by 9, residual multiplier is identity \(1\).
- **Do not** copy MiniCPM-2B’s \(1.4/\sqrt{40}\) or \(1.4/\sqrt{42}\). After the 16+26 split, do not recompute residuals from the new stack depths either.
- The cut is layers, not scale.

---

## 10. Information-theoretic constraint (PDSA) and the middle tier

PDSA (Zou & Donz, arXiv 2606.28876) gives a constraint orthogonal to the parameter budget, but in the same direction as the §6 decode bottleneck:

- **No write-time signal**: query-independent writability on static text is near chance (AUC 0.63–0.66). The compression steps of HCA / KDA / CSA are all write-first, and systematically drop blocks that have no signal at write time and are only hit at query time.
- Therefore **a query-time fallback to low-compression / raw KV must be kept**. §6 shows that this fallback is also the right path on compute at 128K–256K: shrinking cross-attn \(k\) from \(n\) to \(n_{\mathrm{win}}+k_{\mathrm{index}}\) is what returns decode to MLP dominance.
- The middle tier gives the precise-retrieval budget to Decoder cross-attn (4.33B activation, 26 layers) and the long-input compression budget to the Encoder (2.03B, 16 layers). That matches “query-aware read happens at decode; write-first compression happens at prefill”. Do not turn the encoder into pure HCA/KDA with no CSA anchor + fallback just to “save a bit more prefill”.

Linear attention cannot fix lost-in-the-middle (already clarified in plan §2.5); the middle tier also does not rely on KDA to preserve mid-context recall. IN2/FILM + calibrated fallback is the mid-context path.

---

## 11. Claim ledger

Mechanical check of numbers already published in the plan (`scripts/param_budget.py --verify`, GQA 16/2, all-MoE, middle tier):

| Claim | Result |
| --- | --- |
| Total params ≈12B | PASS (12.25B) |
| Enc activation ≈2.03B | PASS (2.03B) |
| Dec activation ≈4.33B | PASS (4.33B) |
| Emb / Enc attn / Dec attn / cross ≈ 0.27 / 0.15 / 0.25 / 0.22B | PASS |
| Expert sparsity 8/21, 11/21 | PASS (38.1%/52.4%) |
| Three tiers share 882 expert slots | PASS |
| 50B tok ≈1325 H100-h | PASS (1325) |
| 1M KV 344 / 43 / 48 / 1.15 / 0.14 GB | PASS |
| 26×8K window ≈0.25 GB | PASS (0.25 GB) |
| No μP: logit_scale = 1; residual identity | PASS |
| freeze-enc ≈75% joint; independent stitch 146%; C1 curriculum 79% | PASS (see curriculum doc) |

The unfreeze curriculum’s extra 12 claims (locked-in C1, detach, untied \(E\), Adam 62%, legal split ≤85%, etc.) plus the middle tier’s 22, the FP8 fallback’s 13, and NVFP4’s 16 total **`--verify` 63/63**; see [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md), [`FP8_THEORY.md`](FP8_THEORY.md), [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

Text-layer items (not in `--verify`; expanded above):

| Text | Treatment |
| --- | --- |
| MiniCPM-2B / \(d=2304\) / \(V=122753\) / 16/24 | Changed to MiniCPM5-2B / \(d=2048\) / \(V=130560\) / 16+26 |
| 720 expert slots, 1+17 layout | Changed to 882 slots, 1+20 on both stacks, only top-\(k\) changes |
| Sparsity 37%/53% or 7/18, 9/18 | Changed to 38.1%/52.4% (8/21, 11/21) |
| Attention ~7% or ~13% | Changed to ~5.0% (GQA) |
| Diagrams 3B/6B or 2.3/4.5 | Changed to 2.03B/4.33B |
| Global cache ÷8 | Marked optional; CSA \(m=4\) corresponds to ÷4 → 0.29 GB |
| μP logits /9, residual \(1.4/\sqrt{40}\) | Changed to identity 1 |
| First-layer dense not in the spec table | Total params drop to 11.79B; Enc/Dec \(N_r=21\) restores 12.30B |

---

## 12. Constraints on the implementation (read out of the theory, not new features)

1. **Keep the spec locked on the middle tier**: Enc 16L, 1+20, top-\(k=7\); Dec 26L, 1+20, top-\(k=10\); published attention GQA 16/2. If Phase A uses first-layer dense, raise routed on both stacks to 21.
2. **The global cache must be compressed or selectable.** Otherwise 128K–256K decode is dominated by 26 layers of cross-attn, and the encoder gains from YOCO+CSA are paid back.
3. **`n_win=8K` is the first cost knob**; `index_topk` is not. Ablate {2K,4K,8K} per plan §7.
4. **Do not introduce MiniCPM-2B μP**, and do not recompute residuals from the new stack depths.
5. **Do not expect CSA to save compute on 4K main training**; sparsify in Phase C and put long context in Phase D, matching the FLOPs curve.
6. After projection dimensions freeze, rerun `python3 scripts/param_budget.py --attn csa_mqa64 --full`; CSA/HCA and GQA are almost the same parameter count — do not change the layer split.
7. **Phase B follows the locked-in C1 recipe** (MoE both stacks first, freeze Encoder, detach cache, freeze input table, B1 may train lm_head, B2≥10B). Do not treat the two stacks as independent LMs and then stitch them.
8. **Phase B wall-clock follows locked-in C1+NVFP4** (571 H100-h; linear GEMMs that need not be bf16; B0 student / L0 / indexer / embed / LN / router / softmax stay high precision). Publish 2.0× vs bf16, do not change 6NT, do not publish 4×. Joint bf16 is control only. C1+FP8 729 is the Hopper/Ada fallback.

Recompute commands:

```bash
python3 scripts/param_budget.py              # middle-tier summary
python3 scripts/param_budget.py --tier all   # three-tier comparison
python3 scripts/param_budget.py --full       # KV / complexity / no μP / Nr search
python3 scripts/param_budget.py --verify     # spec + unfreeze curriculum + FP8 fallback + NVFP4 assertions
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
