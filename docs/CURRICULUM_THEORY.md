# CAT-YOKO Unfreeze Curriculum Theory Verification

> Division of labor with the previous two documents: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks middle-tier parameters / FLOPs / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; **this document checks Phase B’s trainable subset** — freeze boundaries, gradient cutoff at the cache, untied embedding, **the locked-in C1 recipe**, optimizer/activation memory, and the token split. Delayed Encoder MoE is a sensitivity control only.
> Spec remains the middle tier: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. Base MiniCPM5-2B (Llama GQA, untied). No new architecture is introduced, and the two stacks are not split into two independent LMs.
> Executable assertions: `python3 scripts/param_budget.py --verify` (including curriculum claims), `--staged`, `--curriculum`, `--fp8`, `--nvfp4`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **FLOPs / memory / gradient flow / compatibility with Theorem A**; it cannot prove quality after 12B upsampling. Quality still goes through L0→L1 experiments. If B2 is not enough, lengthen B2.
> NVFP4 wall-clock (does not change 6NT; Phase B is locked in as **C1+NVFP4 = 571 H100-h**) is in [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Hopper/Ada fallback C1+FP8 = 729 is in [`FP8_THEORY.md`](FP8_THEORY.md).

---

## 0. Conclusions (read this first)

“Train two models separately and then weld them together” is **more expensive and breaks Theorem A**. What can be saved is a **B0 → B1 → B2 unfreeze curriculum** on the same split weights. Freeze boundaries follow the locked-in **C1** recipe; Phase B **wall-clock is locked in as C1+NVFP4 (571 H100-h)**.

| Approach | 50B tok H100-h | vs joint | Role |
| --- | ---: | ---: | --- |
| Train both stacks together (joint bf16) | 1,325 | 100% | Control |
| C1: MoE both stacks first, freeze Encoder in B0/B1 | 1,046 | 79% | Operand ledger |
| **C1+FP8 (Hopper/Ada fallback)** | 729 | 55% | No Blackwell |
| **C1+NVFP4** | **571** | **43%** | **Locked-in wall-clock** |
| Independent 50B+50B+20B stitch | 1,940 | **146% (more expensive)** | Do not do this |

These must also be nailed down:

1. **Gradients `.detach()` at the global cache.** The Encoder has no weight gradients and no activation gradients; the forward pass cannot be skipped (YOCO’s CE is at the Decoder top). This is Theorem D.
2. **The input embedding must freeze with the Encoder; never train the input table while the Encoder is frozen.** MiniCPM5 is **untied**: \(E_{\mathrm{in}}\) and lm_head are two matrices. B0 freezes the input table **and** the head; B1 **may train lm_head** (input table still frozen). This is Theorem E. `train_embed_while_frozen` is forbidden.
3. **\(W_K,W_V\) are new modules**, attached **after** `encoder.detach()`, and trainable from B0. They are not in the published 12.25B stack total (\(2\cdot d\cdot 256=1.05\mathrm{M}\)).
4. **The recipe is C1.** Phase A does virtual-group MoE on both stacks; B0/B1 freeze the Encoder: one offline surgery, the published 12.25B middle tier from token 0, B2 only unfreezes and does not change structure. While frozen, the Encoder ≈ MiniCPM5 1–16 (Proposition F). B0/B1 pay the MoE Encoder forward (1.76B vs dense 0.75B); expert copies specialize only at B2 — that is why B2 defaults to 15B. Delayed Encoder upsampling (the old C2) is not in the recipe; it stays in the `--curriculum` sensitivity table.
5. **C1 saves about 21% FLOPs versus joint; Adam state in B1 is only 62% of joint; detach drops Encoder activations, about 38% (leaving 26/42 ≈ 62%).** When packing onto fewer GPUs, the memory lever may be more useful than the FLOPs lever.
6. **B2 cannot be 0** (write/read never co-adapt; PDSA already warns). Default B2 = 15B ≥ 10B. If quality is unstable, lengthen B2; do not revert to two independent LMs.
7. Phase C “freeze the trunk, train only the indexer” stacks **after** B2; indexer alignment is in-layer KL, not LM backprop through the cache.
8. **Wall-clock is locked as C1+NVFP4.** Linear GEMMs that need not be bf16 go NVFP4; B0 student / L0 / indexer stay bf16. Publish **571 H100-h (43% of joint bf16)**. See [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

Claim ledger: middle-tier 22 + curriculum 12 + FP8 fallback 13 + NVFP4 16, `--verify` **63/63** PASS.

---

## 1. Formalization: one compute graph, three trainable subsets

Causal LM of length \(n\), notation the same as architecture doc §1:

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

B2 drops \(\mathrm{sg}\); both \(E_{\mathrm{in}}\) and \(W_{\mathrm{head}}\) unfreeze. MiniCPM5 has no μP, \(s_{\mathrm{emb}}=1\), \(\alpha=1\).

Kaplan accounting (same as the budget doc: forward \(2NT\) + backward activations \(2NT\) + backward weights \(2NT\)):

| Subset | Forward | Backward activations | Backward weights |
| --- | --- | --- | --- |
| Frozen, and \(\mathrm{sg}\) cut | \(2N\) | 0 | 0 |
| Frozen, but still on the Decoder residual chain (B0 Decoder backbone) | \(2N\) | \(2N\) | 0 |
| Trainable | \(2N\) | \(2N\) | \(2N\) |

Therefore:

\[
\begin{aligned}
F_{\mathrm{joint}}&=6\,N_{\mathrm{fwd}}T,\\
F_{\mathrm{freeze\text{-}enc}}&=\big(2(N_{\mathrm{emb}}+N_{\mathrm{enc}})+6N_{\mathrm{dec}}\big)T,\\
F_{\mathrm{new\text{-}mod}}&=\big(2N_{\mathrm{fwd}}+2N_{\mathrm{new}}+2N_{\mathrm{dec}}\big)T.
\end{aligned}
\]

\(N_{\mathrm{new}}=L_d A_{\times}+2d\cdot 256\) (26 layers of cross-attn Q/O + cache projections \(1.05\mathrm{M}\)). Locked-in C1’s \(N_{\mathrm{enc}}\) is MoE activation \(1.76\mathrm{B}\) (excluding emb).

Curriculum

\[
F_{\mathrm{C}}=F_{\mathrm{new\text{-}mod}}(T_0)+F_{\mathrm{freeze\text{-}enc}}(T_1)+F_{\mathrm{joint}}(T_2),\quad T_0+T_1+T_2=50\mathrm{B}.
\]

C1 uses a MoE Encoder throughout (frozen in B0/B1, unfrozen in B2).

---

## 2. Theorem D — gradients stop at the detached cache

**Theorem D.** Let \((\hat K,\hat V)\) be obtained by right-multiplying \(\mathrm{sg}(X^{L_e})\) by \(W_K,W_V\), and let the loss \(L\) depend on the input only through the Decoder and \((\hat K,\hat V)\). Then

\[
\frac{\partial L}{\partial\theta_{\mathrm{enc}}}=0,\qquad
\frac{\partial L}{\partial X^{\ell}}=0\quad(\ell\le L_e).
\]

Hence Encoder activations need not be kept for backprop.

**Proof.** \(\mathrm{sg}\) treats \(X^{L_e}\) as a constant. The chain rule breaks at the cache. If \(W_K,W_V\) sit **after** \(\mathrm{sg}\), \(\partial L/\partial W_K\) is still obtained. □

**Implementation constraint (freeze boundary, not a heuristic):**

```
X_Le = encoder(embed(x))
X_Le = X_Le.detach()          # B0/B1
K, V = X_Le @ W_K, X_Le @ W_V # new modules, trainable
```

If you project first and then detach, \(W_K,W_V\) are frozen into the Encoder and B0 cannot train the YOCO interface. If you neither detach nor set Encoder `requires_grad=False`, activation memory is not saved, and a mis-tagged Encoder parameter may still receive a gradient.

B2 **must drop** detach, or write/read cannot co-adapt.

---

## 3. Theorem E — input-table leak while the Encoder is frozen (untied MiniCPM5)

MiniCPM5 does **not share** the input table and the output head: \(E_{\mathrm{in}},W_{\mathrm{head}}\in\mathbb{R}^{V\times d}\) are two matrices.

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E_{\mathrm{in}},\qquad
z=X^{42}W_{\mathrm{head}}^{\top}/\alpha,\quad \alpha=1,\ s_{\mathrm{emb}}=1.
\]

**Theorem E.** If \(\theta_{\mathrm{enc}}\) is frozen while \(E_{\mathrm{in}}\) is trainable, then \(\partial L/\partial E_{\mathrm{in}}\) is nonzero through \(X^{0}\) (as long as Decoder CE can still depend on the prefix through the cache, or some path before B2 is not detached), so \(X^{0}\) drifts. Even if gradients come back only through the **head**: in YOCO the head does not directly update \(E_{\mathrm{in}}\) (untied), but **any** update to \(E_{\mathrm{in}}\) while the Encoder is frozen feeds the frozen stack an \(X^{0}\) that is no longer the MiniCPM5 input distribution. Theorem A’s “one residual stream” no longer holds for the input distribution.

**Corollary E1 (freeze boundary).** The input table **must be frozen** in B0/B1. The head is unbound from the input table, so:

| Policy | Action | Use |
| --- | --- | --- |
| **freeze_embed_and_head (B0 default)** | Freeze both \(E_{\mathrm{in}}\) and \(W_{\mathrm{head}}\) | Warm up new modules; logits still go through the MiniCPM5 head |
| **freeze_embed_train_head (B1 default)** | Freeze \(E_{\mathrm{in}}\), \(W_{\mathrm{head}}\) **trainable** | LP-FT: adapt the head to frozen features first, then move the backbone |
| train_embed_while_frozen (forbidden) | Encoder frozen, still train \(E_{\mathrm{in}}\) | Theorem E leak |

B0 freezes the head: new modules are random, gate is still 0→0.3; do not change the classification surface at the same time. When B1 unfreezes the Decoder it **may** train lm_head — the head is no longer bound to the input table, so it will not pull \(X^{0}\) off-distribution. \(E_{\mathrm{in}}\) unfreezes only in B2. **Never train the input embedding while the Encoder is frozen.**

ULMFiT (Howard & Ruder 2018) is “unfreeze from the top down”: B0 new modules → B1 unfreeze Decoder + head → B2 unfreeze Encoder+\(E_{\mathrm{in}}\). LP-FT is “train the head (or new interface) until it can use frozen features, then move the backbone”. YOCO’s new interface is cross-attn + \(W_K,W_V\), not a randomly initialized classification head — the gate rises from 0, so at the start of B0 the forward pass is still Theorem A. Both literatures support **move the new interface first, then the reader (including the head), last the writer + input table**; neither supports training \(E_{\mathrm{in}}\) in B0/B1.

---

## 4. Freeze boundaries (B0 / B1 / B2)

`scripts/param_budget.py` `FREEZE_BOUNDARIES` (C1 locked-in recipe):

| Substage | Tokens (default) | Frozen | Trainable | detach | embedding | gate |
| --- | ---: | --- | --- | --- | --- | --- |
| **B0** | 8B | Encoder, Decoder backbone, \(E_{\mathrm{in}}\), lm_head | cross-attn, \(W_K/W_V\), new LN, gate | yes | freeze_embed_and_head | 0→0.3 |
| **B1** | 27B | Encoder, \(E_{\mathrm{in}}\) | Decoder self-attn + MoE, cross-attn, \(W_K/W_V\), **lm_head** | yes | freeze_embed_train_head | →1 |
| **B2** | 15B | — | all (including \(E_{\mathrm{in}}\) and head) | **no** | train_untied | 1 |

B0’s tokens align with the existing 5–10B gate ramp (8B here; gate only rises to 0.3, avoiding pulling \(g\) to 1 on a frozen backbone and overfitting cross-attn to frozen features — the symmetric failure of LP-FT “a random head forces the backbone to change features”: here the head is good and the new branch is random).

B1 unfreezes the reader **and** lm_head. Writer and input table are a **frozen virtual-group MoE Encoder** (≈ MiniCPM5-16, Proposition F).

B2 ≥ 10B: Encoder experts start specializing only from this point and need load entropy; write/read need to co-adapt. If quality is unstable, lengthen B2, not B0.

**Do not:** greedily add layers (2 layers → freeze → add 2 more). There is no stable compute-saving evidence on LLMs; the end still needs a joint stage, and it also breaks the 16/26 cut.

### C1 timeline (locked in)

```
Phase A  Offline: 16/26 split, gate=0, Encoder+Decoder both virtual-group MoE,
         add cross-attn and W_K/W_V. Total params are 12.25B from here on.
B0  8B   Encoder / Decoder backbone / E_in / lm_head all frozen. Train only new modules.
         cache = X^16.detach() @ W_{K,V}. g: 0→0.3.
         Encoder forward = frozen MoE copy ≈ MiniCPM5 1–16 (Proposition F).
B1  27B  Unfreeze Decoder + lm_head. Encoder + E_in still frozen, still detach. g→1.
         Reader learns to use frozen memory; writer and input table do not change.
B2  15B  Drop detach, unfreeze Encoder + E_in, smaller LR.
         Encoder experts start specializing here; write/read co-adapt.
```

C1’s MoE initialization is compatible with freezing: virtual-group guarantees equality with dense at the conversion instant; freezing routing and experts keeps that equality until B2. Hash-MoE on the Encoder is redundant in B0/B1 (routing is already frozen solid); use it when B2 unfreezes.

---

## 5. Virtual-group freeze ⇒ approximate Theorem A

**Proposition F.** After Phase A virtual-group upsampling of a stack, at the conversion instant \(f_{\mathrm{MoE}}=f_{\mathrm{dense}}\) (one copy per shard is hit by top-\(k\)). If that stack’s experts and routing are then frozen, the equality holds numerically. Therefore C1’s Encoder in B0/B1 ≈ MiniCPM5 layers 1–16.

The Decoder is upsampled the same way in Phase A: B0 freezes the Decoder backbone ⇒ the Decoder also keeps the virtual-group equality until B1 unfreezes. B1 is the MoE recovery segment; B2 is when the writer joins.

**Hash-MoE.** The Encoder is already frozen, so hash routing on the Encoder is redundant. Put it in the first few steps of the B2 unfreeze. Decoder Hash-MoE is still useful in B1.

---

## 6. Sensitivity: delayed Encoder upsampling (not in the locked-in recipe)

The script still computes FLOPs for “B0/B1 Encoder stays MiniCPM5 dense” as a control. **It is not in the recipe.**

Per-layer dense SwiGLU \(37.75\mathrm{M} < 8\times 12.58\mathrm{M}=100.7\mathrm{M}\) active experts, so 16 layers \(0.75\mathrm{B}<1.76\mathrm{B}\). On the same 8+27+15B split that control is 997 H100-h (75%), about 4pp cheaper than C1, but it requires a second Encoder virtual-group at B2. The locked-in recipe does not adopt it.

---

## 7. FLOPs ledger (aligned with `--staged`)

Middle tier, emb and head each counted once, 50B token envelope, 40% MFU:

| Approach | H100-h | vs joint |
| --- | ---: | ---: |
| Train both stacks together | 1,325 | 100% |
| Freeze Encoder, train Decoder (full run; write/read do not co-adapt) | 987 | 75% |
| Train only new modules | 720 | 54% |
| **C1 unfreeze curriculum 8+27+15B (locked in)** | **1,046** | **79%** |
| Independent 50B+50B+20B stitch | 1,940 | 146% |
| Encoder as an LM for 25B then joint 25B | 874 | 66% (quality gamble) |

The independent stitch is more expensive for the same reason as before: each stack pays 6NT on 50B, plus stitch recovery; Theorem A’s representation alignment is thrown away. Caching Encoder hidden states for 50B tokens × \(d\) × 2 bytes ≈ 205 TB; you cannot skip the Encoder forward by “storing Encoder outputs”.

---

## 8. Optimizer and activation memory (may be worth more than FLOPs)

Accounting: bf16 weights 2 B/param; trainable adds bf16 gradients 2 B + Adam \(m,v\) fp32 totaling 8 B ⇒ trainable 12 B/param, frozen 2 B/param. Excludes activations, excludes fp32 master. Cache projections counted in storage (12.248B rather than the published 12.25B stack total). Numbers are **C1**.

| Stage | Trainable | Frozen | Adam \(m,v\) | Weights+grads+Adam | vs B2 Adam |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.22B | 12.03B | 1.75 GB | 26.69 GB | 2% |
| B1 | 7.60B | 4.65B | 60.82 GB | 100.52 GB | **62%** |
| B2 | 12.25B | 0 | 97.99 GB | 146.98 GB | 100% |

B1 Adam state is 62% of joint (Decoder total params + lm_head + cache projections / all total params).

**Activations:** after detach, Encoder 16 layers need not be kept for backprop. Layer-count model: leave \(26/42\approx 62\%\), save about 38% activation memory. Not the old ledger’s \(24/40=60\%\).

Muon stores one momentum copy only on 2D matrices, so the B1/B2 optimizer gap is slightly smaller than Adam’s 8 B/param; the direction is unchanged.

This is why §15.3 ranks the unfreeze curriculum ahead of NVFP4: it cuts backward FLOPs, Adam state, and activations at once. The published product is **C1+NVFP4 = 571** (C1 bf16 1,046 is operand ledger only; C1+FP8 729 is Hopper/Ada fallback). It does not replace freezing.

---

## 9. Token-split sensitivity

Envelope fixed at 50B. Script `--curriculum` (C1 column is locked in; dense-enc column is sensitivity):

| split | C1 vs joint | Legal |
| --- | ---: | --- |
| **Locked-in 8+27+15** | **79%** | yes |
| Short B2 5+35+10 | 78% | yes (B2 on the 10B floor) |
| Long B2 10+20+20 | 81% | yes (quality first) |
| Short B0 5+25+20 | 83% | yes |
| Skip B0 0+35+15 | 82% | yes (misses new-module warmup) |
| **B2=0 (8+42+0)** | 71% | **no** |

All legal splits under C1 are ≤85% of joint (max 83%). B2=0 is cheapest, but the Encoder never rewrites memory for the Decoder’s queries — PDSA’s “no write-time signal” lands directly on that path. The locked-in recipe does not adopt a B2 shorter than 10B.

---

## 10. Composition with Phase C / D

Phase C step 1 (published default, no KDA): freeze the trunk, train only the Lightning Indexer; the target is **in-layer** indexer-distribution alignment to dense attention (KL/MSE), not CE through the YOCO cache.

If `use_kda`: B already implements the 3:1 graph (still windowed); C step 0 is `C-kda` (only light the majority KDA layers; CSA/HCA still windowed), indexer/CSA/HCA come later; C total is still 25e9. Do not weld KDA into a `use_kda=False` overlay only at C.

- Encoder indexer supervision is inside Encoder layers; it does not need CE backprop into the Encoder, and **need not** drop B0/B1 detach just to train the indexer; C happens **after** B2, when detach has already been turned off for one round.
- Decoder indexer likewise: supervision is inside cross-attn / CSA layers.
- C’s FLOPs shape matches B0 (new modules + Decoder activation backprop + Encoder forward only), but \(N_{\mathrm{new}}\) is the indexer rather than 26 layers of cross-attn, so smaller.
- **Do not** make C’s “freeze the trunk” into a permanent Encoder freeze: M1’s Encoder CSA may still need light adaptation at long context (Phase D). After C alignment, a small joint step.

Phase D long context: default both stacks unfrozen, small LR. If memory is tight, the Encoder can be frozen again for a stretch (C1-style freeze-enc), but from 128K on cross-attn is already the bottleneck (budget doc §6); freezing the writer means giving up write-side adaptation — use only as a stopgap.

Phase F/G domain-expert distillation is orthogonal to this pretraining curriculum.

---

## 11. Failure modes (avoid these before implementing)

| Failure | Source | Avoid |
| --- | --- | --- |
| Independent stitch more expensive, cut drifts | Treat two stacks as two LMs | B0/B1/B2 only; Theorem A |
| Encoder backprop sneaks back | No detach, or \(W_K\) before detach | Theorem D code order |
| Encoder frozen but \(X^0\) drifting | Train input \(E_{\mathrm{in}}\) | Theorem E: freeze input table in B0/B1; B1 may train only the head |
| Encoder experts occupy slots without specializing | Phase A MoE both stacks then freeze Encoder | **Accept**: Encoder experts specialize only at B2≥15B; do not revert to independent LMs for this |
| Write/read ceiling stuck | B2=0 or B2≪10B | Default 15B; if unstable, lengthen B2 |
| B0 \(g\to1\) overfits frozen features | New branch random, backbone frozen | B0 only to \(g=0.3\) |
| Count early-exit into curriculum FLOPs | Training has no early-exit | Architecture doc Theorem C; curriculum uses full-length 6NT |
| Encoder Hash-MoE placed in B0 | Encoder already frozen | Hash-MoE follows Encoder unfreeze (B2) |
| Think the Encoder forward is skipped | CE is at the Decoder top | Forward must run; what is saved is backward and memory |

Theory **cannot** rule out: Encoder load collapse from too-short B2, a loss spike at the unfreeze instant, 12B recovery missing MiniCPM5 95%. Those are experiments; on a spike, roll back that substage and drop LR — do not revert to independent LMs.

---

## 12. Claim ledger

`python3 scripts/param_budget.py --verify` adds, beyond the middle tier’s 22:

| Claim | Result |
| --- | --- |
| **Locked-in recipe is C1** | PASS |
| C1 curriculum ≤85% joint | PASS (79%; middle-tier ledger) |
| B0/B1 detach, B2 does not detach | PASS |
| B0/B1 freeze input table; B0 freeze head, B1 train head | PASS |
| B1 Adam ≈ Decoder/total params ~62% | PASS |
| detach keeps 26/42 layer activations | PASS (62%) |
| All legal 50B splits ≤85% joint | PASS (C1 max 83%) |
| B2=0 cheaper but illegal | PASS |
| Cache projections are B0 new modules | PASS (1.05M) |
| Default B2 ≥10B | PASS (15B) |
| Encoder dense FFN < MoE FFN (sensitivity, not in the locked-in recipe) | PASS (0.75B < 1.76B) |
| delayed-enc control cheaper (sensitivity, not in the locked-in recipe) | PASS (75% < 79%) |

C1 ≤85% is still in the middle-tier ledger (79%). FP8 fallback 13 claims in [`FP8_THEORY.md`](FP8_THEORY.md); NVFP4 16 claims in [`NVFP4_THEORY.md`](NVFP4_THEORY.md); combined `--verify` **63/63**. Wall-clock is locked in as **C1+NVFP4 = 571**.

---

## 13. Plan revisions (this PR)

1. §4.0: **C1 locked in** (MoE both stacks first, freeze Encoder in B0/B1; detach, freeze \(E_{\mathrm{in}}\), B1 may train lm_head, \(W_K/W_V\)). Delayed Encoder upsampling is not in the recipe.
2. Phase A: virtual-group **each** of Encoder and Decoder. Split **16/26**.
3. §15.3 lever #5: C1 saves about 21% of Phase B FLOPs, B1 Adam 62%, activations ~38%.
4. Do not write a training-code skeleton (per user request: close the theory loop first).
5. **Wall-clock locked as C1+NVFP4**: 571 H100-h ([`NVFP4_THEORY.md`](NVFP4_THEORY.md)). Joint bf16, C1+FP8, and all-stage 2.0× are not locked in.

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
