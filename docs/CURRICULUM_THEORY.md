# CAT-YOKO Unfreeze-Curriculum Theory Verification

> Division of labor with the previous two notes: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks middle-tier parameters / FLOPs / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; **this note checks Phase B's trainable subset** — freeze boundaries, gradient truncation at the cache, untied embedding, **C1 published recipe**, optimizer/activation memory, token split. Delayed Encoder MoE is sensitivity comparison only.
> Spec remains middle-tier: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. Base MiniCPM5-2B (Llama GQA, untied). Introduces no new architecture; does not split the two stacks into two independent LMs.
> Executable assertions: `python3 scripts/param_budget.py --verify` (including curriculum claims), `--staged`, `--curriculum`, `--fp8`, `--nvfp4`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **FLOPs / memory / gradient flow / compatibility with Theorem A**; it cannot prove quality after 12B upcycling. Quality still goes through L0→L1 experiments; if B2 is not enough, lengthen B2.
> NVFP4 wall-clock (does not change 6NT; Phase B published **C1+NVFP4 = 571 H100-h**): [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Hopper/Ada fallback C1+FP8 = 729: [`FP8_THEORY.md`](FP8_THEORY.md).

---

## 0. Conclusions (read this first)

"Train two models separately then weld them together" is **more expensive and breaks Theorem A**. What can be saved is a **B0 → B1 → B2 unfreeze curriculum** on the same split weights. Freeze boundaries follow published **C1**; Phase B **wall-clock follows published C1+NVFP4 (571 H100-h)**.

| Approach | 50B tok H100-h | vs joint | Role |
| --- | ---: | ---: | --- |
| Train both stacks together (joint bf16) | 1,325 | 100% | baseline |
| C1: both stacks MoE first, B0/B1 freeze Encoder | 1,046 | 79% | operation-count ledger |
| **C1+FP8 (Hopper/Ada fallback)** | 729 | 55% | no Blackwell |
| **C1+NVFP4** | **571** | **43%** | **published wall-clock** |
| Independent 50B+50B+20B splice | 1,940 | **146% (more expensive)** | do not do this |

Must also nail down:

1. **Gradients `.detach()` at the global cache.** Encoder has no weight gradients and no activation gradients; the forward cannot be skipped (YOCO's CE is at the Decoder top). This is Theorem D.
2. **Input embedding must freeze with the Encoder; never train the input table while the Encoder is frozen.** MiniCPM5 is **untied**: \(E_{\mathrm{in}}\) and lm_head are two matrices. B0 freezes the input table **and** the head; B1 **may train lm_head** (input table still frozen). This is Theorem E. `train_embed_while_frozen` is forbidden.
3. **\(W_K,W_V\) are new modules**, hanging **after** `encoder.detach()`, trainable from B0. They are not in the published 12.25B stack total (\(2\cdot d\cdot 256=1.05\mathrm{M}\)).
4. **The recipe is C1.** Phase A does virtual-group MoE on both stacks; B0/B1 freeze Encoder: one offline surgery, from token 0 it is already the published 12.25B middle tier, B2 only unfreezes and does not change structure. While frozen, Encoder ≈ MiniCPM5 1–16 (Proposition F). B0/B1 pay MoE Encoder forward (1.76B vs dense 0.75B); expert copies only specialize at B2 — hence B2 defaults to 15B. Delayed Encoder upcycling (original C2) is not in the recipe, only in the `--curriculum` sensitivity table.
5. **C1 saves about 21% FLOPs vs joint; Adam state at B1 is only 62% of joint; detach drops Encoder activations about 38% (leaving 26/42 ≈ 62%).** When packing onto fewer GPUs, the memory lever may be more useful than the FLOPs lever.
6. **B2 cannot be 0** (write/read never co-adapt; PDSA already warns). Default B2 = 15B ≥ 10B. If quality is unstable, lengthen B2; do not revert to two independent LMs.
7. Phase C "freeze backbone, train indexer only" stacks **after** B2; indexer alignment is intra-layer KL, not LM backprop through the cache.
8. **Wall-clock is nailed as C1+NVFP4.** Linear GEMMs that need not be bf16 go NVFP4; B0 student / L0 / indexer stay bf16. Publish **571 H100-h (43% of joint bf16)**. See [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

Claim ledger: middle-tier 22 + curriculum 12 + FP8 fallback 13 + NVFP4 16, `--verify` **63/63** pass.

---

## 1. Formalization: one compute graph, three trainable subsets

Causal LM of length \(n\), notation same as architecture note §1:

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E_{\mathrm{in}},\quad
X^{\ell}=\mathrm{SelfDec}^{\ell}(X^{\ell-1}),\ \ell=1..L_e,
\]

\[
(\hat K,\hat V)=\big(\mathrm{sg}(X^{L_e})W_K,\;\mathrm{sg}(X^{L_e})W_V\big)
\quad\text{(B0/B1; }\mathrm{sg}=\texttt{.detach()}\text{)},
\]

\[
X^{\ell}=\mathrm{CrossDec}^{\ell}(X^{\ell-1},\hat K,\hat V),\ \ell=L_e+1..L_e+L_d,
\quad
z=X^{L_e+L_d}W_{\mathrm{head}}^{\top}/\alpha,\quad \alpha=1.
\]

B2 removes \(\mathrm{sg}\); both \(E_{\mathrm{in}}\) and \(W_{\mathrm{head}}\) unfreeze. MiniCPM5 has no μP, \(s_{\mathrm{emb}}=1\), \(\alpha=1\).

Kaplan accounting (same as the budget note: forward \(2NT\) + activation backward \(2NT\) + weight backward \(2NT\)):

| Subset | Forward | Activation backward | Weight backward |
| --- | --- | --- | --- |
| Frozen, and \(\mathrm{sg}\) truncated | \(2N\) | 0 | 0 |
| Frozen, but still on the Decoder residual chain (B0 Decoder backbone) | \(2N\) | \(2N\) | 0 |
| Trainable | \(2N\) | \(2N\) | \(2N\) |

Hence:

\[
\begin{aligned}
F_{\mathrm{joint}}&=6\,N_{\mathrm{fwd}}T,\\
F_{\mathrm{freeze\text{-}enc}}&=\big(2(N_{\mathrm{emb}}+N_{\mathrm{enc}})+6N_{\mathrm{dec}}\big)T,\\
F_{\mathrm{new\text{-}mod}}&=\big(2N_{\mathrm{fwd}}+2N_{\mathrm{new}}+2N_{\mathrm{dec}}\big)T.
\end{aligned}
\]

\(N_{\mathrm{new}}=L_d A_{\times}+2d\cdot 256\) (26-layer cross-attn Q/O + cache projections \(1.05\mathrm{M}\)). Published C1's \(N_{\mathrm{enc}}\) is MoE active \(1.76\mathrm{B}\) (excluding emb).

Curriculum

\[
F_{\mathrm{C}}=F_{\mathrm{new\text{-}mod}}(T_0)+F_{\mathrm{freeze\text{-}enc}}(T_1)+F_{\mathrm{joint}}(T_2),\quad T_0+T_1+T_2=50\mathrm{B}.
\]

C1 uses MoE Encoder throughout (frozen in B0/B1, unfrozen in B2).

---

## 2. Theorem D — gradients stop at the detached cache

**Theorem D.** Let \((\hat K,\hat V)\) be obtained by right-multiplying \(\mathrm{sg}(X^{L_e})\) by \(W_K,W_V\), and let the loss \(L\) depend on the input only through the Decoder and \((\hat K,\hat V)\). Then

\[
\frac{\partial L}{\partial\theta_{\mathrm{enc}}}=0,\qquad
\frac{\partial L}{\partial X^{\ell}}=0\quad(\ell\le L_e).
\]

Hence Encoder activations need not be retained for backprop.

**Proof.** \(\mathrm{sg}\) treats \(X^{L_e}\) as a constant. The chain rule breaks at the cache. If \(W_K,W_V\) are **after** \(\mathrm{sg}\), \(\partial L/\partial W_K\) is still obtained. □

**Implementation constraint (freeze boundary, not a heuristic):**

```
X_Le = encoder(embed(x))
X_Le = X_Le.detach()          # B0/B1
K, V = X_Le @ W_K, X_Le @ W_V # new modules, trainable
```

If one projects first then detaches, \(W_K,W_V\) are frozen into the Encoder, and B0 cannot train the YOCO interface. If one neither detaches nor sets Encoder `requires_grad=False`, activation memory is not saved, and a mis-tagged Encoder parameter may still receive gradient.

B2 must **remove** detach, otherwise write/read cannot co-adapt.

---

## 3. Theorem E — input-table leak while Encoder is frozen (untied MiniCPM5)

MiniCPM5 does **not share** the input table and output head: \(E_{\mathrm{in}},W_{\mathrm{head}}\in\mathbb{R}^{V\times d}\) are two matrices.

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E_{\mathrm{in}},\qquad
z=X^{42}W_{\mathrm{head}}^{\top}/\alpha,\quad \alpha=1,\ s_{\mathrm{emb}}=1.
\]

**Theorem E.** If \(\theta_{\mathrm{enc}}\) is frozen while \(E_{\mathrm{in}}\) is trainable, then \(\partial L/\partial E_{\mathrm{in}}\) is nonzero through \(X^{0}\) (as long as Decoder CE can still depend on the prefix through the cache, or some non-detached path exists before B2), so \(X^{0}\) drifts. Even if gradient only comes back through the **head**: in YOCO the head does not directly update \(E_{\mathrm{in}}\) (untied), but **any** update to \(E_{\mathrm{in}}\) while the Encoder is frozen feeds the frozen stack an \(X^{0}\) that is no longer the MiniCPM5 input distribution. Theorem A's "same residual stream" no longer holds for the input distribution.

**Corollary E1 (freeze boundary).** The input table **must be frozen** in B0/B1. The head is unbound from the input table, so:

| Policy | What it does | Use |
| --- | --- | --- |
| **freeze_embed_and_head (B0 default)** | both \(E_{\mathrm{in}}\) and \(W_{\mathrm{head}}\) frozen | new-module warmup; logits still go through MiniCPM5 head |
| **freeze_embed_train_head (B1 default)** | \(E_{\mathrm{in}}\) frozen, \(W_{\mathrm{head}}\) **trainable** | LP-FT: adapt the head to frozen features first, then move the backbone |
| train_embed_while_frozen (forbidden) | Encoder frozen, still train \(E_{\mathrm{in}}\) | Theorem E leak |

B0 freezes the head: new modules are random, gate is still 0→0.3; do not also change the classification surface. When B1 unfreezes the Decoder it **may** train lm_head — the head is no longer bound to the input table, so it will not drag \(X^{0}\). Only B2 unfreezes \(E_{\mathrm{in}}\). **Never train the input embedding while the Encoder is frozen.**

ULMFiT (Howard & Ruder 2018) is "unfreeze top-down": B0 new modules → B1 unfreeze Decoder + head → B2 unfreeze Encoder+\(E_{\mathrm{in}}\). LP-FT is "first train the head (or new interface) until it can use frozen features, then move the backbone". YOCO's new interface is cross-attn + \(W_K,W_V\), not a randomly initialized classification head — gate rises from 0, and at the start of B0 the forward is still Theorem A. Both literatures support **move the new interface first, then the reader (including head), finally the writer + input table**, and do not support training \(E_{\mathrm{in}}\) in B0/B1.

---

## 4. Freeze boundaries (B0 / B1 / B2)

`scripts/param_budget.py`'s `FREEZE_BOUNDARIES` (C1 published):

| Sub-phase | tokens (default) | Frozen | Trainable | detach | embedding | gate |
| --- | ---: | --- | --- | --- | --- | --- |
| **B0** | 8B | Encoder, Decoder backbone, \(E_{\mathrm{in}}\), lm_head | cross-attn, \(W_K/W_V\), new LN, gate | yes | freeze_embed_and_head | 0→0.3 |
| **B1** | 27B | Encoder, \(E_{\mathrm{in}}\) | Decoder self-attn + MoE, cross-attn, \(W_K/W_V\), **lm_head** | yes | freeze_embed_train_head | →1 |
| **B2** | 15B | — | all (including \(E_{\mathrm{in}}\) and head) | **no** | train_untied | 1 |

B0 tokens align with the existing 5–10B gate ramp (here 8B; gate only rises to 0.3, to avoid pulling \(g\) to 1 on a frozen backbone and overfitting cross-attn to frozen features — the symmetric failure of LP-FT "random head forces the backbone to change features": here the head is good, the new branch is random).

B1 unfreezes the reader **and** lm_head. Writer and input table are a **frozen virtual-group MoE Encoder** (≈ MiniCPM5-16, Proposition F).

B2 ≥ 10B: Encoder experts only start specializing from this moment and need load entropy; write/read need to co-adapt. If quality is unstable, lengthen B2, not B0.

**Do not:** greedy layer-by-layer growth (2 layers → freeze → add 2 more). There is no stable compute-saving evidence on LLMs; the end still needs joint training, and it also breaks the 16/26 split point.

### C1 timeline (published)

```
Phase A  offline: 16/26 split, gate=0, Encoder+Decoder both virtual-group MoE,
         add cross-attn and W_K/W_V. Thereafter total params are 12.25B.
B0  8B   Encoder / Decoder backbone / E_in / lm_head all frozen. Train new modules only.
         cache = X^16.detach() @ W_{K,V}. g: 0→0.3.
         Encoder forward = frozen MoE copies ≈ MiniCPM5 1–16 (Proposition F).
B1  27B  Unfreeze Decoder + lm_head. Encoder + E_in still frozen, still detach. g→1.
         Reader learns to use frozen memory; writer and input table do not change.
B2  15B  Remove detach, unfreeze Encoder + E_in, smaller LR.
         Encoder experts only start specializing here; write/read co-adapt.
```

C1's MoE initialization is compatible with freezing: virtual-group guarantees equality to dense at the conversion instant; freeze routing and experts, and that equality is maintained until B2. Hash-MoE on the Encoder is redundant in B0/B1 (routing is already frozen dead); use it when B2 unfreezes.

---

## 5. Virtual-group freeze ⇒ approximate Theorem A

**Proposition F.** After Phase A virtual-group upcycling of a stack, at the conversion instant \(f_{\mathrm{MoE}}=f_{\mathrm{dense}}\) (one copy per slice is hit by top-\(k\)). If that stack's experts and routing are subsequently frozen, the equality holds numerically. Therefore C1's Encoder in B0/B1 ≈ MiniCPM5 layers 1–16.

Decoder is likewise upcycled in Phase A: B0 freezes Decoder backbone ⇒ Decoder also keeps the virtual-group equality until B1 unfreezes. B1 is the MoE recovery segment; only B2 lets the writer participate.

**Hash-MoE.** Encoder is already frozen, so hash routing on the Encoder is redundant. Place it in the first few steps of B2 unfreeze. Decoder Hash-MoE is still useful in B1.

---

## 6. Sensitivity: delayed Encoder upcycling (not published)

The script still computes FLOPs for "B0/B1 Encoder stays MiniCPM5 dense" as a comparison; **not in the recipe**.

Per-layer dense SwiGLU \(37.75\mathrm{M} < 8\times 12.58\mathrm{M}=100.7\mathrm{M}\) active experts, so 16 layers \(0.75\mathrm{B}<1.76\mathrm{B}\). Under the same 8+27+15B split this comparison is 997 H100-h (75%), about 4pp cheaper than C1, but requires a second Encoder virtual-group at B2. Not adopted as published.

---

## 7. FLOPs ledger (aligned with `--staged`)

Middle tier, emb and head each counted once, 50B token envelope, 40% MFU:

| Approach | H100-h | vs joint |
| --- | ---: | ---: |
| Train both stacks together | 1,325 | 100% |
| Freeze Encoder, train Decoder (full run; write/read do not co-adapt) | 987 | 75% |
| Train new modules only | 720 | 54% |
| **C1 unfreeze curriculum 8+27+15B (published)** | **1,046** | **79%** |
| Independent 50B+50B+20B splice | 1,940 | 146% |
| Encoder as LM 25B then joint 25B | 874 | 66% (quality gamble) |

Why independent splice is more expensive is unchanged: each stack pays 50B of 6NT, plus splice recovery; Theorem A's representation alignment is thrown away. 50B tokens × \(d\) × 2 bytes of Encoder hidden-state cache ≈ 205 TB; one cannot skip the forward by "storing Encoder outputs".

---

## 8. Optimizer and activation memory (may be worth more than FLOPs)

Accounting: bf16 weights 2 B/param; trainable additionally bf16 gradients 2 B + Adam \(m,v\) fp32 totaling 8 B ⇒ trainable 12 B/param, frozen 2 B/param. Excludes activations, excludes fp32 master. Cache projections counted in storage (12.248B rather than the published 12.25B stack total). Numbers are **C1**.

| Stage | Trainable | Frozen | Adam \(m,v\) | weights+grads+Adam | vs B2 Adam |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.22B | 12.03B | 1.75 GB | 26.69 GB | 2% |
| B1 | 7.60B | 4.65B | 60.82 GB | 100.52 GB | **62%** |
| B2 | 12.25B | 0 | 97.99 GB | 146.98 GB | 100% |

B1 Adam state is 62% of joint (Decoder total params + lm_head + cache projections / overall total params).

**Activations:** after detach, Encoder 16 layers need not be retained for backprop. Layer-count model: leave \(26/42\approx 62\%\), save about 38% activation memory. Not the old ledger's \(24/40=60\%\).

Muon stores one momentum only on 2D matrices; the B1/B2 optimizer gap will be slightly smaller than Adam's 8 B/param, direction unchanged.

This is why §15.3 ranks the unfreeze curriculum ahead of NVFP4: it simultaneously cuts backward FLOPs, Adam state, and activations. The published product is **C1+NVFP4 = 571** (C1 bf16 1,046 is operation-count ledger only; C1+FP8 729 is Hopper/Ada fallback); it does not replace freezing.

---

## 9. Token-split sensitivity

Envelope fixed at 50B. Script `--curriculum` (C1 column is published; dense-enc column is sensitivity):

| split | C1 vs joint | Legal |
| --- | ---: | --- |
| **Published 8+27+15** | **79%** | yes |
| Short B2 5+35+10 | 78% | yes (B2 against the 10B floor) |
| Long B2 10+20+20 | 81% | yes (quality first) |
| Short B0 5+25+20 | 83% | yes |
| Skip B0 0+35+15 | 82% | yes (misses new-module warmup) |
| **B2=0 (8+42+0)** | 71% | **no** |

All legal splits under C1 are ≤85% of joint (max 83%). B2=0 is cheapest, but the Encoder never rewrites memory for Decoder queries — PDSA "no-signal-at-write-time" hits this path directly. Published recipe does not adopt B2 shorter than 10B.

---

## 10. Composition with Phase C / D

Phase C step 1 (published default, no KDA): freeze backbone, train Lightning Indexer only; the target is **intra-layer** indexer distribution aligned to dense attention (KL/MSE), not CE through the YOCO cache.

If `use_kda`: B already implements the 3:1 graph (still sliding); C step 0 is `C-kda` (light majority KDA layers only; CSA/HCA still sliding), indexer/CSA/HCA later; C total still 25e9. Do not weld KDA into a `use_kda=False` overlay only at C.

- Encoder indexer supervision is intra-Encoder-layer; it does not need CE backprop into the Encoder, and **need not** drop B0/B1 detach in order to train the indexer; C happens **after** B2, when detach has already been turned off for one round.
- Decoder indexer likewise; supervision is intra-cross-attn / CSA layer.
- C's FLOPs shape is the same as B0 (new modules + Decoder activation backprop + Encoder forward only), but \(N_{\mathrm{new}}\) is the indexer rather than 26-layer cross-attn, smaller.
- **Do not** make C's "freeze backbone" into forever-freeze Encoder: M1's Encoder CSA may still need light adaptation at long context (Phase D). After C alignment, a short joint fine-tune.

Phase D long context: default both stacks unfrozen, small LR. If memory is insufficient, Encoder can be frozen again for a stretch (C1-style freeze-enc), but from 128K cross-attn is already the bottleneck (budget note §6); freezing the writer equals giving up write-side adaptation, stopgap only.

Phase F/G domain-expert distillation is orthogonal to this pretraining curriculum.

---

## 11. Failure modes (avoid before implementing)

| Failure | Source | Avoid |
| --- | --- | --- |
| Independent splice more expensive, split-point drift | two stacks as two LMs | B0/B1/B2 only; Theorem A |
| Encoder backprop sneaks back | no detach, or \(W_K\) before detach | Theorem D code order |
| Encoder frozen but \(X^0\) drifting | train input \(E_{\mathrm{in}}\) | Theorem E: B0/B1 freeze input table; B1 may train head only |
| Encoder experts occupy slots without specializing | Phase A both stacks MoE then freeze Encoder | **accept**: B2≥15B is when Encoder experts specialize; do not revert to independent LMs because of this |
| write/read ceiling stall | B2=0 or B2≪10B | default 15B; if unstable lengthen B2 |
| B0 \(g\to1\) overfits frozen features | new branch random, backbone frozen | B0 only to \(g=0.3\) |
| Counting early-exit into curriculum FLOPs | training has no early-exit | architecture note Theorem C; curriculum uses full-length 6NT |
| Encoder Hash-MoE placed in B0 | Encoder already frozen | Hash-MoE follows Encoder unfreeze (B2) |
| Thinking Encoder forward is skipped | CE is at Decoder top | forward must run; what is saved is backward and memory |

What theory **cannot** rule out: B2 too short causing Encoder load collapse, loss spike at unfreeze, 12B not recovering MiniCPM5 95%. Those are experiments; on a spike roll back that sub-phase and drop LR, do not revert to independent LMs.

---

## 12. Claim ledger

`python3 scripts/param_budget.py --verify` adds, beyond the middle-tier 22:

| Claim | Result |
| --- | --- |
| **Published recipe is C1** | PASS |
| C1 curriculum ≤85% joint | PASS (79%; middle-tier ledger) |
| B0/B1 detach, B2 no detach | PASS |
| B0/B1 freeze input table; B0 freeze head, B1 train head | PASS |
| B1 Adam ≈ Decoder/total params ~62% | PASS |
| detach retains 26/42 layer activations | PASS (62%) |
| All legal 50B splits ≤85% joint | PASS (C1 max 83%) |
| B2=0 cheaper but illegal | PASS |
| cache projections are B0 new modules | PASS (1.05M) |
| Default B2 ≥10B | PASS (15B) |
| Encoder dense FFN < MoE FFN (sensitivity, not published) | PASS (0.75B < 1.76B) |
| delayed-enc comparison cheaper (sensitivity, not published) | PASS (75% < 79%) |

C1 ≤85% is still in the middle-tier ledger (79%). FP8 fallback 13 claims: [`FP8_THEORY.md`](FP8_THEORY.md); NVFP4 16 claims: [`NVFP4_THEORY.md`](NVFP4_THEORY.md); total `--verify` **63/63**. Published wall-clock is **C1+NVFP4 = 571**.

---

## 13. Plan revisions (this PR)

1. §4.0: **C1 published** (both stacks MoE first, B0/B1 freeze Encoder; detach, freeze \(E_{\mathrm{in}}\), B1 may train lm_head, \(W_K/W_V\)). Delayed Encoder upcycling not in the recipe.
2. Phase A: virtual-group on Encoder and Decoder **each**. Split **16/26**.
3. §15.3 lever #5: C1 saves about 21% Phase B FLOPs, B1 Adam 62%, activations ~38%.
4. Do not write a training-code skeleton (per user request, close the theory loop first).
5. **Wall-clock nailed as C1+NVFP4**: 571 H100-h ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). Joint bf16, C1+FP8, and full-stage 2.0× not published.

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
