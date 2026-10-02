# CAT-YOKO Architecture Theory Verification

> Division of labor with [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md): that note checks **middle-tier parameters / FLOPs / KV**; this note checks **dataflow, causality, receptive field, and the global-cache interface**. Spec remains middle-tier: Encoder 16L / Decoder 26L, CSA+HCA+8K sliding window, primary target 128K–256K. Base is MiniCPM5-2B (Llama GQA, untied).
> Executable assertions: `python3 scripts/arch_verify.py --verify` and `python3 -m unittest tests.test_arch_verify`.
> Theory can prove **self-consistency, causality, complexity, and information flow**; it cannot prove quality after 12B upcycling. Quality still goes through L0→L1 experiments.

---

## 0. Conclusions (read this first)

The architecture is self-consistent on causality and residual split, but the plan text collapsed **three distinct mechanisms** into one sentence: "CSA/HCA compresses the global cache". They must be separated, or implementation will run top-k in the wrong place.

| Mechanism | Where it happens | What is the query | Role |
| --- | --- | --- | --- |
| **M1** Encoder CSA/HCA | every self-decoder layer | **current input token** | cheaper write of per-token representations |
| **M2** Global-cache re-pooling | Encoder top → \(\hat K,\hat V\) | none (write-first) | compress \(N\) slots to \(N/m\); **optional** |
| **M3** Decoder-side indexer / calibrated fallback | cross-attn | **generation query** | the actual query-aware retrieval |

YOCO's globality comes from **M3 reading all (or selected) cache slots**, not from the encoder sliding-window receptive field. 16 layers × 8K window stacked RF is only **131072**, which does not cover 256K; this **does not block** 256K retrieval, because token 0's cache slot is still there and the decoder can read it directly.

Must also nail down:

1. **When gate≈0, the 16/26 split ≡ original 42-layer residual stream** (minus a not-yet-opened cross-attn). This is why MiniCPM5 warm-start is valid.
2. **The CSA/HCA compression branch must exclude the query's own block**, otherwise future tokens inside the block leak; **the sliding window patches this hole**. Sufficient condition: \(n_{\mathrm{win}}\ge m'\). 8K ≥ 128, holds.
3. **Early-exit belongs only to inference prefill**. Training runs both stacks over all tokens; do not treat the 32% activation share as training FLOPs.
4. **Decoder self-attention also sees the full-length sequence at train time.** "Generation sequences are usually not long" describes inference only. In long-context training, decoder self-attn must be sliding-window/KDA; the global mix is left to cross-attn.
5. Encoder 16 layers cannot do "3-layer bootstrap + CSA:HCA=1:1". Frozen as **2× sliding + 7 CSA + 7 HCA** (aligns with V4-Flash's 2-layer sliding-window bootstrap). Adding decoder to 26 layers **does not change** this Encoder schedule.

Claim ledger: 14/14 pass.

---

## 1. Formal dataflow

A sequence of length \(n\), hidden state \(X^{0}\in\mathbb{R}^{n\times d}\) is the embedding (MiniCPM5 `scale_emb=1`, no μP).

\[
\begin{aligned}
X^{\ell} &= \mathrm{SelfDec}^{\ell}(X^{\ell-1}), && \ell=1,\ldots,L_e=16,\\
(\hat K,\hat V) &= \big(X^{L_e}W_K,\; X^{L_e}W_V\big), && \hat K,\hat V\in\mathbb{R}^{n\times d_{\mathrm{kv}}},\\
X^{\ell} &= \mathrm{CrossDec}^{\ell}(X^{\ell-1},\hat K,\hat V), && \ell=L_e+1,\ldots,L_e+L_d.
\end{aligned}
\]

\(d_{\mathrm{kv}}=n_{\mathrm{kv}}d_h=256\) (GQA-2). Each CrossDec block:

\[
\begin{aligned}
U &= X + \mathrm{SelfAttn}_{\le n_{\mathrm{win}}}(X),\\
Z &= U + g\cdot \mathrm{CrossAttn}(Q=UW_Q,\; K=\hat K,\; V=\hat V),\\
X' &= Z + \mathrm{MoE}(Z).
\end{aligned}
\]

\(g\in[0,1]\) is the Phase A/B cross-attn gate. Causal mask: both self-attn and cross-attn query \(t\) must not see positions \(>t\).

External behavior is a causal LM: logits come from the **untied** head \(W_{\mathrm{head}}\) on \(X^{L_e+L_d}\). This is what YOCO means by "looks like decoder-only, cache only once".

---

## 2. Theorem A — at gate=0 the split equals the original residual stream

**Theorem A.** If (i) the self-attn+FFN of \(\mathrm{SelfDec}\) and \(\mathrm{CrossDec}\) are exactly original MiniCPM5 layers \(1..16\) and \(17..42\), (ii) \(g=0\), (iii) not yet MoE-ified and attention not yet replaced by CSA, then for any input, CAT-YOKO's \(X^{42}\) equals original MiniCPM5's \(X^{42}\).

**Proof.** When \(g=0\), CrossDec degenerates to SelfAttn+FFN. Residual input is \(X^{16}\), which is exactly MiniCPM5 layer 17's input. Induction over layers yields the claim. □

Corollaries:

- Phase A's "cross-attn bypass" is not a heuristic; it is a sufficient condition that the **split point is valid**.
- MoE upcycling and CSA replacement each break the equivalence and must be staged (Phase A split → B recovery → C sparsification), consistent with the §13 de-risking ladder.
- Decoder layer 1 (global layer 17) already has residual input \(X^{16}\); after opening \(g\), cross-attn also reads the projection of \(X^{16}\). So raising \(g\) from 0 to 1 **adds a "mix-prefix-by-position" pathway on the same memory**, rather than suddenly attaching a foreign encoder.

Capability that appears after opening \(g\): if self-attn is sliding-window, position \(t\)'s residual contains only local context; cross-attn lets \(t\) mix \(\hat K_{1:t}\), i.e. the **causal prefix of \(X^{16}\)**. This is exactly YOCO's mechanism of trading one layer of memory for a global receptive field.

---

## 3. Three mechanisms: do not treat M1 as M3

Plan §2.0 writes "self-decoder CSA/HCA produces a more compact global cache". Read literally, this would make one think that after Encoder CSA, the cache slot count is already \(N/m\). **It is not.**

### M1 — Intra-layer Encoder CSA/HCA

Each layer compresses that layer's KV along the sequence, then lets **the current layer's query (input token)** select. Output is still \(n\) hidden states. Top-layer \(X^{L_e}\) is still \(n\times d\).
For the decoder this is **write-time contextualization**: the query used when writing the cache is "this input itself", not the user's later question. PDSA's "no-signal-at-write-time" hits exactly here — Encoder CSA top-k cannot save a needle that is only asked about after the fact.

### M2 — Global-cache re-pooling (optional)

\[
\hat K' = \mathrm{Pool}_m(\hat K)\in\mathbb{R}^{(n/m)\times d_{\mathrm{kv}}}.
\]

This is what actually reduces the slot count of **YOCO's single** cache. The budget note §7 figures 0.29 GB (\(m=4\)) and 0.14 GB (÷8) belong to M2, not M1. M2 is write-first; there is no decoder query.

### M3 — Decoder-side selection (the actual retrieval)

Generation query \(q_t\) does dense / top-k / calibrated fallback over \(\hat K_{1:t}\) (or the slots after M2). This is the query-aware step. Uncompressed cross-attn overtakes decoder MLP at 128K–256K (budget note §6), so **at least one of M3 or M2 is required**; recommend M3 (with fallback), M2 as a memory-saving add-on.

**Architecture constraint.** Implementation: make cross-attn causal dense by default first (valid; the continuous relaxation of Theorem A), then add M3 in Phase C. Do not reuse Encoder Lightning Indexer top-k weights on decoder queries — the two query distributions differ.

---

## 4. Theorem B — CSA/HCA causality + sliding window fills the hole

Write compression ratio \(m\), query position \(t\) (0-index). Own block index \(b=\lfloor t/m\rfloor\). Block \(s\) covers tokens \([sm,(s+1)m)\).

**Compression-branch visibility (V4 §2.3.1)**

\[
\mathcal{S}_{\mathrm{comp}}(t)=\{s:s < \lfloor t/m\rfloor\}.
\]

**Lemma B1 (compression branch does not leak the future).** If \(s\in\mathcal{S}_{\mathrm{comp}}(t)\), then the largest index inside block \(s\) is \((s+1)m-1 \le m\lfloor t/m\rfloor-1 \le t-1\). CSA's overlapping branch \(C^b\) uses the previous block; the newest token is no later. Hence the compression branch is causal.

**Lemma B2 (own block has a hole).** Tokens \(\le t\) inside the own block (including itself) are not in \(\mathcal{S}_{\mathrm{comp}}\). If tokens \(>t\) inside the own block were included in the compressed keys, the future would leak — that is why the entire block must be excluded.

**Sliding window**

\[
\mathcal{W}(t)=\{p:\max(0,t-n_{\mathrm{win}}+1)\le p\le t\}.
\]

**Theorem B (hole filling).** If \(n_{\mathrm{win}}\ge m\), then all tokens \(\le t\) inside the own block are in \(\mathcal{W}(t)\). Therefore

\[
\mathcal{V}(t)=\mathcal{W}(t)\;\cup\;\bigcup_{s\in\mathcal{S}_{\mathrm{comp}}(t)}[sm,(s+1)m)
\]

is causal, and the own-block past is visible to the query.

CAT-YOKO: \(n_{\mathrm{win}}=8192\), \(m=4\), \(m'=128\), \(8192\ge 128\). The same applies to HCA. If anyone drops the window below \(<128\) while keeping HCA \(m'=128\), hole filling fails — this is why the ablation `n_win∈{2K,4K,8K}` has a **lower bound that is not 0, at least 128**. 2K/4K/8K are all safe.

**Indexer.** top-\(k\) may only delete from \(\mathcal{S}_{\mathrm{comp}}(t)\), never add. Dropping points under sparsification is a recall problem, not a causality problem; so Phase C must do indexer alignment first.

Exhaustive check on finite \(n=64\) is in `scripts/arch_verify.py` (CSA/HCA/YOCO zero leak; top-k is a subset).

---

## 5. Theorem C — YOCO cross-attn causality and early-exit

**Causality.** Query \(t\) may read cache slots \(\{0,\ldots,t\}\). The Encoder is causal; \(\hat K_j\) depends only on tokens \(\le j\), so when \(j\le t\), \(\hat K_j\) contains no future.

**Inference prefill early-exit.** The prompt \(x_{0:n-1}\)'s global cache is closed once \(X^{L_e}_{0:n-1}\) is computed. To emit the first generated token, it is enough to **run the decoder once at position \(n-1\)** (reading the already-written cache); there is no need to run \(L_d\) layers on all \(n\) prompt positions. This is YOCO Table 1's early-exit: what it saves is \(O(L_d n)\) decoder prefill, not the decoder's last position.

**Training has no early-exit.** Every position has CE, so the decoder must forward on all \(t=0..n-1\). Training FLOPs use \(N_{\mathrm{fwd}}^{\mathrm{act}}\) (budget note); do not estimate Phase B from the 32% encoder share.

---

## 6. Receptive field: where globality comes from

| Path | Receptive field |
| --- | --- |
| Encoder pure sliding-window stacking | \(L_e\cdot n_{\mathrm{win}}=16\cdot 8192=131072\) |
| Encoder CSA/HCA | at write time can already see compressed long-range (lossy) |
| Decoder self-attention | \(n_{\mathrm{win}}=8192\) (**cannot** add across the YOCO split point with the encoder window) |
| Decoder cross-attn | causal prefix length \(t+1\) (or M3's \(k\)) |

In the original YOCO, the self-decoder is already sliding-window or retention: the encoder **need not** mix globally. Token 0's slot is still there; the decoder at \(t=256\mathrm{K}\) can still read it. The 256K primary target **does not depend** on encoder RF ≥ 256K.

The real roles of CSA/HCA in the encoder are: (a) reducing the encoder attention quadratic term at train/prefill time (only happens when \(n\gg 8K\), budget note §5); (b) optionally giving written representations a bit of long-range context. It is not a sufficient condition for 256K retrieval — the sufficient condition is **M3's global read**.

The decoder window cannot stack RF with the encoder window: after the split point, decoder self-attn only sees the local decoder residual stream; long-range must go through \(\hat K\).

---

## 7. Information bottleneck of the shared global cache

Decoder-only: layer \(\ell\)'s keys come from that layer's hidden state, \(L_d\) mutually distinct memories.
YOCO: \(L_d\) layers read **the same** \(\hat K(X^{L_e})\); only \(W_Q^{\ell}\) differs per layer.

The memory tensor drops from \(O(L_d n d_{\mathrm{kv}})\) to \(O(n d_{\mathrm{kv}})\), a factor of \(L_d=26\). This is a memory theorem and also an expressivity bet: layers can no longer "rewrite" keys, only change the query. YOCO scores near-perfect on 1M needle, showing this memory is enough for retrieval-type tasks; it does **not** prove it is enough for multi-layer reasoning/rewrite (agent state, versioning) — that is the motivation for plan §14 Tier 3 trainable lifecycle, not the motivation for CSA.

M2 further pools along the sequence, dropping the bottleneck from \(n\) to \(n/m\). PDSA has already measured that write-first selection drops needles with "no-signal-at-write-time", so by default M2 should not be harsher than \(m=4\); ÷8 is stretch only.

---

## 8. 16/26 asymmetry

Original YOCO is \(L/2+L/2\). CAT-YOKO takes 16/26 (MiniCPM5's \(L_0=42\)), corresponding to "lighter writer, heavier reader":

- Prefill / long input binds the writer (2.03B, 32% plan-accounting active).
- Generation leaves compute for the reader (4.33B) to do more layers of \(W_Q\) queries over the memory.
- MiniCPM5's first 16 layers as writer, last 26 as reader, consistent with Theorem A's split point (low-level features in front, high-level processing in back).

Theory does **not uniquely determine** 16/26. 12/30 would be cheaper prefill and weaker memory; 21/21 is closer to original YOCO. This is a §7 ablation item, not an error. 16/26 just needs to be consistent with the middle-tier narrative of "light input, heavy output". Encoder is still 16 layers; CSA schedule 2/7/7 is unchanged.

---

## 9. Encoder layer schedule: why 2/7/7 rather than "first 3 layers bootstrap"

16 layers must simultaneously satisfy: (i) the first few layers stay close to MiniCPM5 dense attention for warm-start, (ii) CSA:HCA=1:1.

- 2-layer sliding bootstrap → 14 remaining → **7 CSA + 7 HCA**.
- 3-layer bootstrap → 13 remaining → **cannot 1:1**.

V4-Flash `compress_ratios` starts with two `0`s (sliding); V4-Pro text is 2× HCA bootstrap. For MiniCPM5 upcycling, **sliding bootstrap is closer to original GQA** (just adding a window), better than jumping straight into HCA \(m'=128\).

**Frozen:** Encoder `layer_types` =

```
sliding, sliding, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca
```

Decoder self-attn: all sliding (or later KDA mix); **do not** by default lay CSA on decoder self-attn as well — globality is already in cross-attn. CSA on the decoder is extra complexity and duplicates YOCO's "decoder self uses efficient local attention".

The plan's original side-by-side wording "first 3 layers sliding/HCA" and "2× HCA bootstrap" is frozen in this section to the single schedule above. Decoder 26 layers does not change this Encoder 16-layer schedule.

---

## 10. MoE, Hash-MoE, no μP (architecture side)

- Asymmetric activation comes from **layer counts and top-k of the two stacks**, not from changing k per token on the same layer. Compatible with Theorem A: replacing FFN with MoE breaks the equivalence, so virtual-group upcycling must be recovered separately (Phase B).
- First layer of each stack dense: routing is unstable at layer 1 (DeepSeekMoE convention), isomorphic to attention bootstrap — both are "don't apply the riskiest inductive bias first". Budget note: after first-layer dense, Enc/Dec routed 20→21 restores 12.30B.
- Hash-MoE: freeze `token_id→expert_id`, no learned routing, does not enter attention causality. Encoder Hash-MoE is placed after B2 unfreeze (curriculum note §5).
- **No μP**: `scale_emb=1`, residual identity 1, logits not divided by 9. Do not port MiniCPM-2B's \(1.4/\sqrt{40}\). Orthogonal to the YOCO split: what is split is layers, not scale.

mHC / Muon / MTP do not enter this note's causality checks. mHC is a residual spectral constraint; turning it off does not affect YOCO/CSA validity.

---

## 11. Where to put KDA and PDSA

KDA is **a fixed-size recurrent state per layer**, orthogonal to YOCO KV slots. It cannot provide M3: the state is the same gist for all future queries, exactly the write-first PDSA describes. Plan §2.5's placement (efficiency + coarse coverage, not mid-span precise retrieval) is consistent with this note.

If one does 3:1 KDA:CSA on the Encoder, that is swapping linear layers for CSA layers **inside M1**, reducing the encoder quadratic term and per-layer KV; it does **not** reduce YOCO global-cache slot count (that is M2), and does **not** give decoder-side query-awareness (that is M3). Mix ratio still uses the §7 ablation; this note only forbids "shipping KDA equals protecting the middle".

PDSA calibrated fallback is M3's gate, not a fourth attention. The threshold must be calibrated at the target length (plan §14); calibrating on short context will make fallback never fire — the same length scale as the budget note's "cross-attn only becomes the bottleneck from 128K".

---

## 12. Failure modes implied by theory (avoid before implementing)

| Failure | Source | Avoid |
| --- | --- | --- |
| Intra-block future leak | compression branch sees own block | Theorem B: exclude own block + \(n_{\mathrm{win}}\ge m'\) |
| 256K retrieval is 0 | mistakenly thinking encoder RF is required | ensure M3 can read token 0's slot |
| Decode killed by cross-attn | only M1, no M2/M3 | from 128K open M3 or \(m=4\) M2 |
| Training FLOPs undercounted 3× | applying early-exit to training | train both stacks full length |
| Decoder long-train \(O(n^2)\) | thinking "generation is short" so decoder is full attention | decoder self is also windowed at train time |
| Warm-start collapse | \(g=1\) or CSA top-k from the start | Theorem A: \(g=0\); sparsify only in Phase C |
| 16 layers cannot fit 1:1 | 3-layer bootstrap | freeze 2/7/7 |
| Indexer reused on the wrong query | wiring M1's indexer into M3 | two query sets, two (or later-added) indexers |

What theory **cannot** rule out: MoE load collapse, indexer misalignment, lost-in-the-middle (position bias, IN2/FILM + RoPE/NoPE), 12B not recovering MiniCPM5 95%. Those are experiments.

---

## 13. Claim ledger

`python3 scripts/arch_verify.py --verify`:

| Claim | Result |
| --- | --- |
| 16+26=42 | PASS |
| \(n_{\mathrm{win}}\ge m,m'\) | PASS (8192≥128) |
| Encoder schedule 2 sliding + 7 CSA + 7 HCA | PASS |
| Window-only encoder RF = 131072 < 256K | PASS (and ≥256K is not required) |
| Shared cache = 1 writer × 26 readers | PASS |
| Early-exit inference only | PASS (text) |
| n=64 CSA/HCA/YOCO causal, own block not in compression, window fills hole, top-k subset | PASS |
| M1 ≠ M2 ≠ M3 | PASS (text; tests check slot counts and query sets differ) |

---

## 14. Plan revisions (this PR)

1. §2.0: CSA/HCA does not automatically compress YOCO slot count; globality comes from cross-attn.
2. §2.3: Encoder `layer_types` frozen as 2/7/7 sliding→CSA/HCA; Decoder self defaults to all sliding.
3. §2.0 "generation sequences are usually not long": mark as inference only; train-time decoder self is still windowed.
4. Tie-in with the budget note: 128K–256K must have M2 or M3; default push M3. Split is **16/26**, not 16/24.

---

## 15. Train separately then splice (compute, not causality)

Training the two stacks **independently as LMs then splicing** breaks Theorem A's representation alignment, and 50B+50B+splice is **more expensive** than joint 50B (about 1.46×). What can be saved is an **unfreeze curriculum** on the same split weights.

Theorem D: B0/B1 `.detach()` on \(X^{16}\); Encoder has no weight/activation gradients; \(W_K,W_V\) hang after detach and are new modules.
Theorem E: input \(E_{\mathrm{in}}\) must freeze with the Encoder; B0 freezes lm_head, B1 **may train** the untied head. Forbidden to train the input table while the Encoder is frozen, otherwise the frozen Encoder's input distribution drifts.

Published recipe **C1**: Phase A both stacks MoE, B0/B1 freeze Encoder (virtual-group freeze ⇒ Encoder ≈ MiniCPM5-16; only B2 lets Encoder experts specialize).

Numbers: C1 is about **79%** of joint 50B; independent splice **146%**. B1 Adam state is about **62%** of joint. Freeze boundaries: [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) and training plan §4.0. **Published wall-clock C1+NVFP4 = 571 H100-h** ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). `python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4`.

Recompute:

```bash
python3 scripts/arch_verify.py --verify
python3 -m unittest tests.test_arch_verify
python3 scripts/param_budget.py --verify                 # middle-tier + unfreeze curriculum + FP8 fallback + NVFP4 ledger
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
```
