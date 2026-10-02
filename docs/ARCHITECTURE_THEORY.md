# CAT-YOKO Architecture Theory Verification

> Division of labor with [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md): that document checks **middle-tier parameters / FLOPs / KV**; this one checks **data flow, causality, receptive field, and the global-cache interface**. Spec remains the middle tier: Encoder 16L / Decoder 26L, CSA+HCA+8K sliding window, primary target 128K–256K. Base MiniCPM5-2B (Llama GQA, untied).
> Executable assertions: `python3 scripts/arch_verify.py --verify` and `python3 -m unittest tests.test_arch_verify`.
> Theory can prove **consistency, causality, complexity, and information flow**; it cannot prove quality after 12B upsampling. Quality still goes through L0→L1 experiments.

---

## 0. Conclusions (read this first)

The architecture is consistent on causality and residual split, but the plan text collapsed **three different mechanisms** into one sentence, “CSA/HCA compress the global cache”. They must be separated, or the implementation will do top-k in the wrong place.

| Mechanism | Where it happens | Whose query | Role |
| --- | --- | --- | --- |
| **M1** Encoder CSA/HCA | every self-decoder layer | **the current input token** | cheaper per-token representations |
| **M2** global-cache re-pooling | Encoder top → \(\hat K,\hat V\) | none (write-first) | compress \(N\) slots to \(N/m\); **optional** |
| **M3** Decoder-side indexer / calibrated fallback | cross-attn | **the generation query** | the actual query-aware retrieval |

YOCO globality comes from **M3 reading all (or selected) cache slots**, not from the encoder sliding-window receptive field. 16 layers × 8K window stacked RF is only **131072**, which does not cover 256K; that **does not block** 256K retrieval, because token 0’s cache slot is still there and the decoder can read it directly.

These must also be nailed down:

1. **At gate≈0, the 16/26 split ≡ the original 42-layer residual stream** (difference: a cross-attn that is not yet open). That is why MiniCPM5 warm-start is legitimate.
2. **The CSA/HCA compression branch must exclude the query’s own block**, or future tokens inside the block leak; **the sliding window patches that hole**. Sufficient condition: \(n_{\mathrm{win}}\ge m'\). 8K ≥ 128, holds.
3. **Early-exit belongs only to inference prefill.** Training runs every token through both stacks; do not mistake the 32% activation share for training FLOPs.
4. **Decoder self-attention also sees the full-length sequence at training time.** “Generated sequences are usually not long” describes inference only. In long-context training, decoder self-attn must be windowed/KDA; global mixing is left to cross-attn.
5. Encoder 16 layers cannot do “3-layer bootstrap + CSA:HCA=1:1”. Lock it as **2× sliding + 7 CSA + 7 HCA** (aligned with V4-Flash’s 2-layer sliding bootstrap). Adding Decoder layers to 26 **does not change** this Encoder schedule.

Claim ledger: 14/14 PASS.

---

## 1. Formal data flow

Sequence of length \(n\), hidden state \(X^{0}\in\mathbb{R}^{n\times d}\) is the embedding (MiniCPM5 `scale_emb=1`, no μP).

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

\(g\in[0,1]\) is the Phase A/B cross-attn gate. Causal mask: neither self-attn nor cross-attn query \(t\) may see positions \(>t\).

External behavior is a causal LM: logits come from the **untied** head \(W_{\mathrm{head}}\) on \(X^{L_e+L_d}\). This is YOCO’s “looks like decoder-only, cache once”.

---

## 2. Theorem A — at gate=0 the split equals the original residual stream

**Theorem A.** If (i) \(\mathrm{SelfDec}\) and \(\mathrm{CrossDec}\) self-attn+FFN are the original MiniCPM5 layers \(1..16\) and \(17..42\), (ii) \(g=0\), (iii) not yet MoE-ized and attention not yet swapped to CSA, then for any input, CAT-YOKO’s \(X^{42}\) equals original MiniCPM5’s \(X^{42}\).

**Proof.** At \(g=0\), CrossDec degenerates to SelfAttn+FFN. Residual input is \(X^{16}\), which is exactly MiniCPM5 layer 17’s input. Layer-wise induction gives the result. □

Corollaries:

- Phase A’s “cross-attn bypass” is not a heuristic; it is a sufficient condition for a **legitimate cut**.
- MoE upsampling and CSA replacement each break the equivalence and must be staged (Phase A split → B recovery → C sparsify), matching the §13 derisking ladder.
- Decoder layer 1 (global layer 17) already has residual \(X^{16}\); after opening \(g\), cross-attn also reads a projection of \(X^{16}\). So raising \(g\) from 0 to 1 **adds a “mix the prefix by position” path on the same memory**, rather than suddenly attaching a foreign encoder.

Capability added after opening \(g\): if self-attn is windowed, position \(t\)’s residual is only local; cross-attn lets \(t\) mix \(\hat K_{1:t}\), i.e. **the causal prefix of \(X^{16}\)**. That is YOCO’s mechanism for trading one layer of memory for a global receptive field.

---

## 3. Three mechanisms: do not treat M1 as M3

Plan §2.0 writes “self-decoder CSA/HCA produce a more compact global cache”. Read literally, that would mean Encoder CSA already leaves cache slots at \(N/m\). **It does not.**

### M1 — Encoder in-layer CSA/HCA

Each layer compresses that layer’s KV along the sequence, then lets **the current layer’s query (input token)** select. Output is still \(n\) hidden states. Top-layer \(X^{L_e}\) is still \(n\times d\).
For the decoder this is **write-time contextualization**: the query used when writing the cache is “this input itself”, not the user’s later question. PDSA’s “no write-time signal” lands directly here — Encoder CSA top-k cannot save a needle that is only asked about after the fact.

### M2 — global-cache re-pooling (optional)

\[
\hat K' = \mathrm{Pool}_m(\hat K)\in\mathbb{R}^{(n/m)\times d_{\mathrm{kv}}}.
\]

This is what actually reduces slot count of **YOCO’s single** cache. The budget doc §7 figures 0.29 GB (\(m=4\)) and 0.14 GB (÷8) belong to M2, not M1. M2 is write-first; there is no decoder query.

### M3 — Decoder-side selection (the actual retrieval)

Generation query \(q_t\) does dense / top-k / calibrated fallback over \(\hat K_{1:t}\) (or the slots after M2). This is the query-aware step. At 128K–256K, uncompressed cross-attn overtakes decoder MLP (budget doc §6), so **at least one of M3 or M2 is required**; recommend M3 (with fallback), M2 as a memory-saving add-on.

**Architecture constraint.** Implement cross-attn as causal dense first by default (legal; a continuous relaxation of Theorem A), then add M3 in Phase C. Do not reuse Encoder Lightning Indexer top-k weights on the decoder query — the two query distributions differ.

---

## 4. Theorem B — CSA/HCA causality + window patches the hole

Write compression ratio \(m\), query position \(t\) (0-index). Own-block index \(b=\lfloor t/m\rfloor\). Block \(s\) covers tokens \([sm,(s+1)m)\).

**Compression-branch visibility (V4 §2.3.1)**

\[
\mathcal{S}_{\mathrm{comp}}(t)=\{s:s < \lfloor t/m\rfloor\}.
\]

**Lemma B1 (compression branch does not leak the future).** If \(s\in\mathcal{S}_{\mathrm{comp}}(t)\), then the largest index inside block \(s\) is \((s+1)m-1 \le m\lfloor t/m\rfloor-1 \le t-1\). CSA’s overlapping branch \(C^b\) uses the previous block; the newest token is no later. Hence the compression branch is causal.

**Lemma B2 (own block has a hole).** Tokens \(\le t\) inside the own block (including self) are not in \(\mathcal{S}_{\mathrm{comp}}\). Tokens \(>t\) inside the own block, if folded into the compressed key, would leak the future — that is why the whole block must be excluded.

**Sliding window**

\[
\mathcal{W}(t)=\{p:\max(0,t-n_{\mathrm{win}}+1)\le p\le t\}.
\]

**Theorem B (hole patch).** If \(n_{\mathrm{win}}\ge m\), then every token \(\le t\) inside the own block is in \(\mathcal{W}(t)\). Therefore

\[
\mathcal{V}(t)=\mathcal{W}(t)\;\cup\;\bigcup_{s\in\mathcal{S}_{\mathrm{comp}}(t)}[sm,(s+1)m)
\]

is causal, and the own-block past is visible to the query.

CAT-YOKO: \(n_{\mathrm{win}}=8192\), \(m=4\), \(m'=128\), \(8192\ge 128\). The same applies to HCA. If someone drops the window below \(<128\) while keeping HCA \(m'=128\), the hole patch fails — that is why the ablation `n_win∈{2K,4K,8K}` has a **floor that is not 0, at least 128**. 2K/4K/8K are all safe.

**Indexer.** top-\(k\) may only delete from \(\mathcal{S}_{\mathrm{comp}}(t)\), never add. Dropped points under sparsification are a recall issue, not a causality issue; that is why Phase C does indexer alignment first.

Exhaustive check on finite \(n=64\) is in `scripts/arch_verify.py` (CSA/HCA/YOCO zero leak; top-k is a subset).

---

## 5. Theorem C — YOCO cross-attn causality and early-exit

**Causality.** Query \(t\) may read cache slots \(\{0,\ldots,t\}\). The Encoder is causal, \(\hat K_j\) depends only on tokens \(\le j\), so for \(j\le t\), \(\hat K_j\) contains no future.

**Inference prefill early-exit.** The prompt \(x_{0:n-1}\) global cache closes once \(X^{L_e}_{0:n-1}\) is computed. To emit the first generated token, **the decoder only needs to run once at position \(n-1\)** (reading the already-written cache); it need not run \(L_d\) layers at all \(n\) prompt positions. This is YOCO Table 1 early-exit: what is saved is \(O(L_d n)\) decoder prefill, not the decoder’s last position.

**Training has no early-exit.** Every position has CE, so the decoder must forward at all \(t=0..n-1\). Training FLOPs use \(N_{\mathrm{fwd}}^{\mathrm{act}}\) (budget doc); do not estimate Phase B from the 32% encoder share.

---

## 6. Receptive field: where globality comes from

| Path | Receptive field |
| --- | --- |
| Encoder pure sliding-window stack | \(L_e\cdot n_{\mathrm{win}}=16\cdot 8192=131072\) |
| Encoder CSA/HCA | can see compressed long range at write time (lossy) |
| Decoder self-attention | \(n_{\mathrm{win}}=8192\) (**cannot** add to encoder windows across the YOCO cut) |
| Decoder cross-attn | causal prefix length \(t+1\) (or M3’s \(k\)) |

Original YOCO self-decoder is windowed or retention: the encoder **need not** mix globally. Token 0’s slot is still there; the decoder at \(t=256\mathrm{K}\) can still read it. The 256K primary target **does not depend** on encoder RF ≥ 256K.

What CSA/HCA actually do in the encoder: (a) cut the encoder’s quadratic attention term at train/prefill (only when \(n\gg 8K\), budget doc §5); (b) optionally give written representations a little long-range context. It is not a sufficient condition for 256K retrieval — the sufficient condition is **M3’s global read**.

Decoder windows cannot stack RF with encoder windows: after the cut, decoder self-attn only sees a local piece of the decoder residual stream; long range must go through \(\hat K\).

---

## 7. Information bottleneck of the shared global cache

Decoder-only: layer \(\ell\)’s keys come from that layer’s hidden state, \(L_d\) mutually different memories.
YOCO: \(L_d\) layers read **the same** \(\hat K(X^{L_e})\); only \(W_Q^{\ell}\) differs per layer.

The memory tensor drops from \(O(L_d n d_{\mathrm{kv}})\) to \(O(n d_{\mathrm{kv}})\), factor \(L_d=26\). That is a memory theorem and an expressivity bet: later layers can no longer “rewrite” keys, only change the query. YOCO scoring near-full on 1M needle shows this memory is enough for retrieval-style tasks; it does **not** prove it is enough for multi-layer reasoning/rewrite (agent state, versioning) — that is the motivation for plan §14 Tier 3 trainable lifecycle, not for CSA.

M2 pools further along the sequence, shrinking the bottleneck from \(n\) to \(n/m\). PDSA already measured that write-first selection drops needles with no write-time signal, so M2 should not default more aggressive than \(m=4\); ÷8 is stretch only.

---

## 8. 16/26 asymmetry

Original YOCO is \(L/2+L/2\). CAT-YOKO takes 16/26 (MiniCPM5 \(L_0=42\)), matching “lighter writer, heavier reader”:

- Prefill / long input bound to the writer (2.03B, 32% plan-accounting activation).
- Generation leaves compute for the reader (4.33B) to do more layers of \(W_Q\) queries on the memory.
- MiniCPM5 first 16 layers as writer, last 26 as reader, matching Theorem A’s cut (lower-layer features first, higher-layer processing after).

Theory **does not uniquely determine** 16/26. 12/30 would be cheaper prefill and weaker memory; 21/21 is closer to original YOCO. That is a §7 ablation item, not an error. 16/26 only needs to match the middle-tier story of “light input, heavy output”. The Encoder is still 16 layers; CSA schedule 2/7/7 is unchanged.

---

## 9. Encoder layer schedule: why 2/7/7, not “first 3 layers bootstrap”

16 layers must satisfy both: (i) the first few layers stay close to MiniCPM5 dense attention for warm-start, (ii) CSA:HCA=1:1.

- 2-layer sliding bootstrap → 14 remaining → **7 CSA + 7 HCA**.
- 3-layer bootstrap → 13 remaining → **cannot 1:1**.

V4-Flash `compress_ratios` starts with two `0`s (sliding); V4-Pro text is 2× HCA bootstrap. For MiniCPM5 upsampling, **sliding bootstrap is closer to original GQA** (just add a window) than starting with HCA \(m'=128\).

**Locked:** Encoder `layer_types` =

```
sliding, sliding, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca
```

Decoder self-attn: all sliding (or later KDA mix). **Do not** default to laying CSA on decoder self-attn — globality is already in cross-attn. CSA on the Decoder is extra complexity and duplicates YOCO’s “decoder self uses efficient local attention”.

The plan originally listed “first 3 layers sliding/HCA” alongside “2× HCA bootstrap”; this section locks that to the schedule above. Decoder 26 layers do not change this Encoder 16-layer schedule.

---

## 10. MoE, Hash-MoE, no μP (architecture side)

- Asymmetric activation comes from **layer counts and top-k on the two stacks**, not from changing k per token inside one layer. Compatible with Theorem A: replacing FFN with MoE breaks equivalence, so virtual-group upsampling is recovered separately (Phase B).
- First layer of each stack dense: routing is unstable at layer 1 (DeepSeekMoE convention), isomorphic to the attention bootstrap — both are “do not apply the riskiest inductive bias first”. Budget doc: after first-layer dense, Enc/Dec routed 20→21 restores 12.30B.
- Hash-MoE: frozen `token_id→expert_id`, no learned routing, does not enter attention causality. Encoder Hash-MoE is placed after the B2 unfreeze (curriculum doc §5).
- **No μP**: `scale_emb=1`, residual identity 1, logits not divided by 9. Do not copy MiniCPM-2B’s \(1.4/\sqrt{40}\). Orthogonal to the YOCO split: the cut is layers, not scale.

mHC / Muon / MTP do not enter this document’s causality check. mHC is a residual spectral constraint; turning it off does not affect YOCO/CSA legality.

---

## 11. Where KDA and PDSA sit

KDA is **one fixed-size recurrent state per layer**, orthogonal to YOCO KV slots. It cannot provide M3: the state is the same gist for all future queries, exactly PDSA’s write-first. Plan §2.5’s placement (efficiency + coarse coverage, not precise mid-context retrieval) matches this document.

A 3:1 KDA:CSA mix on the Encoder swaps linear layers for CSA layers **inside M1**, cutting encoder quadratic terms and per-layer KV; it does **not** reduce YOCO global-cache slot count (that is M2), and does **not** give decoder-side query-awareness (that is M3). Mix ratios still use the §7 ablation; this document only forbids “turning on KDA equals preserving the mid-context”.

PDSA calibrated fallback is M3’s gate, not a fourth attention. The threshold must be calibrated at the target length (plan §14); calibrating on short context makes the fallback never fire — the same length scale as the budget doc’s “cross-attn becomes the bottleneck from 128K”.

---

## 12. Failure modes implied by the theory (avoid these before implementing)

| Failure | Source | Avoid |
| --- | --- | --- |
| In-block future leak | Compression branch sees own block | Theorem B: exclude own block + \(n_{\mathrm{win}}\ge m'\) |
| 256K retrieval is 0 | Mistakenly require encoder RF | Guarantee M3 can read token 0’s slot |
| Decode killed by cross-attn | Only M1, no M2/M3 | From 128K, turn on M3 or M2 with \(m=4\) |
| Training FLOPs undercounted 3× | Apply early-exit to training | Train both stacks full length |
| Decoder long-train \(O(n^2)\) | Think “generation is short” so decoder is full attention | Decoder self is windowed at train time too |
| Warm-start collapse | \(g=1\) or CSA top-k from the start | Theorem A: \(g=0\); sparsify in Phase C |
| 16 layers cannot schedule 1:1 | 3-layer bootstrap | Lock 2/7/7 |
| Indexer reused on the wrong query | Wire M1’s indexer into M3 | Two queries, two (or later-added) indexers |

Theory **cannot** rule out: MoE load collapse, indexer misalignment, lost-in-the-middle (position bias, IN2/FILM + RoPE/NoPE), 12B recovery missing MiniCPM5 95%. Those are experiments.

---

## 13. Claim ledger

`python3 scripts/arch_verify.py --verify`:

| Claim | Result |
| --- | --- |
| 16+26=42 | PASS |
| \(n_{\mathrm{win}}\ge m,m'\) | PASS (8192≥128) |
| Encoder schedule 2 sliding + 7 CSA + 7 HCA | PASS |
| Window-only encoder RF = 131072 < 256K | PASS (and need not be ≥256K) |
| Shared cache = 1 writer × 26 readers | PASS |
| Early-exit inference only | PASS (prose) |
| n=64 CSA/HCA/YOCO causal, own block not compressed, window patches hole, top-k subset | PASS |
| M1 ≠ M2 ≠ M3 | PASS (prose; tests check slot count and query sets differ) |

---

## 14. Plan revisions (this PR)

1. §2.0: CSA/HCA do not automatically compress YOCO slot count; globality comes from cross-attn.
2. §2.3: Encoder `layer_types` locked to 2/7/7 sliding→CSA/HCA; Decoder self defaults to all windowed.
3. §2.0 “generated sequences are usually not long”: mark as inference-only; train-time decoder self is still windowed.
4. Tie-in with the budget doc: 128K–256K must have M2 or M3; default push M3. Split **16/26**, not 16/24.

---

## 15. Train separately then merge (compute, not causality)

Training the two stacks **independently as LMs and then stitching** breaks Theorem A’s representation alignment, and 50B+50B+stitch is **more expensive** than joint 50B (about 1.46×). What can be saved is an **unfreeze curriculum** on the same split weights.

Theorem D: B0/B1 `.detach()` at \(X^{16}\); Encoder has no weight/activation gradients; \(W_K,W_V\) attach after detach and are new modules.
Theorem E: input \(E_{\mathrm{in}}\) must freeze with the Encoder; B0 freezes lm_head, B1 **may train** the untied head. Training the input table while the Encoder is frozen is forbidden, or the frozen Encoder’s input distribution drifts.

Locked-in **C1**: Phase A MoE both stacks, B0/B1 freeze Encoder (virtual-group freeze ⇒ Encoder ≈ MiniCPM5-16; Encoder experts specialize only at B2).

Numbers: C1 about **79%** of joint 50B; independent stitch **146%**. B1 Adam state about **62%** of joint. Freeze boundaries in [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) and training plan §4.0. **Published wall-clock C1+NVFP4 = 571 H100-h** ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). `python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4`.

Recompute:

```bash
python3 scripts/arch_verify.py --verify
python3 -m unittest tests.test_arch_verify
python3 scripts/param_budget.py --verify                 # middle-tier + unfreeze curriculum + FP8 fallback + NVFP4 ledger
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
```
