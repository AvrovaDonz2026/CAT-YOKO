# CAT-YOKO FP8 theory verification

> **This document is the Hopper / Ada fallback ledger, no longer the Phase B published wall-clock.** The published dtype is **C1+NVFP4 = 571 H100-h**, see [`NVFP4_THEORY.md`](NVFP4_THEORY.md). The 729 / 1.5× / `fp8_moe` numbers below remain recomputable as the path when there are no FP4 tensor cores.
>
> Division of labor with the previous three docs: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks middle-tier parameters / 6NT / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) checks the C1 unfreeze curriculum; **this document checks the Hopper FP8 fallback**: which pieces of middle-tier C1 training can go FP8, how much more wall-clock can be cut, and which modules must stay high precision.
> The spec is still the middle tier: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. Base MiniCPM5-2B (Llama GQA). C1 freeze boundaries are unchanged. No new architecture, no training skeleton.
> Executable assertions: `python3 scripts/param_budget.py --verify` (including FP8 fallback claims), `--fp8`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **6NT is invariant, Amdahl upper bounds, and compatibility with Theorems A/E**; it cannot prove that FP8 kernels on 12B actually saturate 1.5×. That is an L1 experiment.

---

## 0. Conclusion (read this first)

**The Phase B wall-clock recipe was previously locked as C1+FP8: 729 H100-h, 55% of joint bf16. It has been demoted to a Hopper/Ada fallback; the published value is 571 in [`NVFP4_THEORY.md`](NVFP4_THEORY.md).**

Joint bf16 (1,325) is only the 100% baseline, not a path to run. C1 bf16 (1,046) is the operation-count ledger and still counts 6NT in bf16. The mixed FP8 policy in this document is the **Hopper/Ada fallback**; on Blackwell run [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Applying 1.5× to every stage (697) and the 2× peak (523) are not locked.

A **substantial fraction** of middle-tier training is MoE expert GEMM: **90.6%** of stored parameters and **81.9%** of the 6NT count. That block (plus frozen Encoder inference GEMM) goes FP8. FP8 **does not change** Kaplan \(6N_{\mathrm{act}}T\); it only raises the effective throughput of completing those FLOPs on H100.

| Approach | 50B tok H100-h | vs joint bf16 | Role |
| --- | ---: | ---: | --- |
| Train both stacks together, bf16 | 1,325 | 100% | baseline |
| C1 unfreeze curriculum, bf16 | 1,046 | 79% | operation-count ledger |
| **C1+FP8** (B0 student bf16; B1/B2 allowed GEMMs and frozen Encoder forward 1.5×) | **729** | **55%** | **Hopper/Ada fallback** |
| Apply 1.5× to every C1 stage (B0 student FP8 too) | 697 | 53% | sensitivity |
| C1 × peak 2× | 523 | 39% | upper bound, not published |

Several more nails:

1. **6NT is independent of dtype** (Theorem G). Plan §15.1's 1,325 / 1,046 remain the bf16 operation-count ledger; FP8 only enters the wall-clock column. After untied, the plan's accounting matches a full forward; the emb×2 1,413 is no longer listed separately.
2. **Publish 1.5×, do not publish 2×.** H100 SXM FP8 peak ≈ 1979 TFLOPS, ~2.0× bf16 989. MoE active share \(f=81.9\%\), GEMM 2× Amdahl is **1.69×**, which still supports 1.5×. 12B expert GEMMs (\(d=2048\), \(d_{\mathrm{moe}}=2048\), 12.58M per expert) are far smaller than DeepSeek-671B; dispatch/combine, scale, and small-kernel occupancy all eat the peak. MoE+attn \(f=95.8\%\) gives 1.92×; that is an upper bound.
3. **B0 student stays bf16.** The gate climbs from 0 to 0.3, hugging Theorem A's residual neighborhood; new modules are randomly initialized, so do not change compute precision here. Frozen Encoder forward is pure inference GEMM and can go FP8 already in B0.
4. **B1+B2 cover 84% of the 50B token envelope and 89% of C1 FLOPs.** That is the "substantial fraction": the stages that actually run for a long time use a low-precision student. B0 is only 11.0% of C1 FLOPs, so leaving the student in bf16 barely eats the gain (729 vs all-stages 697, a gap of ~32 H100-h).
5. **High-precision whitelist (this fallback document's historical set includes lm_head)**: input \(E_{\mathrm{in}}\), RMSNorm, router, gate, Lightning Indexer, attn softmax. Published NVFP4 **removes lm_head and attn QKV/O from the must-stay-high-precision set**. Theorem E still forbids quantizing the input table.
6. **L0 and Phase C indexer stay bf16.** Tiny correctness does not mix precision; indexer alignment is layer-internal KL on small tensors.
7. **Muon Newton-Schulz remains fp32**, orthogonal to network GEMM FP8. V4-style FP4 expert storage is a later option and is not in this recipe.
8. **C1 and FP8 multiply, they do not add.** C1 first cuts 21% of FLOPs, then multiplies the remaining wall by 1.5× (B0 excepted). Relative to joint bf16: \(79\% \times\) (B1/B2 1.5×, B0 near 1×) ≈ **55%**. That product is the fallback wall-clock **C1+FP8 = 729**.

Claim ledger: the 13 FP8 fallback claims remain in `--verify`; published totals are in [`NVFP4_THEORY.md`](NVFP4_THEORY.md) (**63/63**).

---

## 1. Theorem G: dtype does not change 6NT

Kaplan / Hoffmann accounting (same as the budget document):

\[
F = 6\,N_{\mathrm{act}}T = \underbrace{2N_{\mathrm{act}}T}_{\text{forward}} + \underbrace{2N_{\mathrm{act}}T}_{\text{activation backward}} + \underbrace{2N_{\mathrm{act}}T}_{\text{weight gradient}}.
\]

This is multiply-add count, not joules, and not seconds. Swapping expert projections from bf16 to FP8 E4M3/E5M2 does not change \(F\). What changes is

\[
H_{\text{wall}} = \frac{F}{\eta_{\mathrm{bf16}}}\, /\, S,\qquad
\eta_{\mathrm{bf16}} = 4.0\times 10^{14}\ \text{FLOPS (40% MFU)}.
\]

\(S=1\) is the bf16 column of plan §15.1; C1's \(F_{\mathrm{C}}\) is 21% less than joint, so bf16 wall-clock drops from 1,325 to 1,046. FP8 only enters \(S\).

H100 SXM:

\[
\frac{\text{FP8 peak}}{\text{bf16 peak}} = \frac{1.979\times 10^{15}}{9.89\times 10^{14}} \approx 2.00.
\]

If MFU on FP8 is still 40%, effective throughput is 7.92×10¹⁴, \(S=2\). That does not hold at 12B: expert matrices are small, MoE communication is not accelerated, and scale has overhead. The published value locks **\(S=1.5\)**.

---

## 2. "Substantial fraction": MoE GEMM is 81.9% of 6NT

One full middle-tier forward \(N_{\mathrm{fwd}}=6.36\mathrm{B}\) (emb and head counted once each):

| Block | Active parameters | Share of 6NT |
| --- | ---: | ---: |
| MoE experts (top-\(k\) SwiGLU) | 5.21B | **81.9%** |
| Self-attention + cross-attn projections | 0.61B | 9.7% |
| untied embedding + lm_head | 0.53B | 8.4% |
| RMSNorm / router / softmax | (ignored by 6NT) | — |

On the storage side MoE is still **90.6%** (882 expert slots, including replicas that never enter top-\(k\)). Training FLOPs follow activations, so "substantial fraction" is counted as **81.9%**. GQA compressed attention from the old placeholder 24% down to 9.7%, so the MoE share rose accordingly — Amdahl only gets better, not worse.

If only MoE GEMM becomes 2× and the rest stays 1×:

\[
S_{\mathrm{MoE}} = \frac{1}{0.819/2 + 0.181} \approx 1.69.
\]

1.69 is above the published 1.5; the gap is left for dispatch/combine and 12B small kernels. If attn projections go FP8 as well, \(f=95.8\%\), \(S\approx 1.92\) — that is an upper bound, **not written into §15.1**.

Frozen Encoder forward (B0/B1) is the inference form of the same GEMMs: no weight gradient, no activation backward, the safest FP8 cut in the whole graph. In B0 it is only a small fraction of that stage's FLOPs.

---

## 3. C1+FP8 fallback (stacked on C1, freeze boundaries unchanged)

This is the published recipe. Student = tensors that receive gradients. Frozen Encoder GEMMs are a separate column. Applying 1.5× to every stage (B0 student FP8 too) is not locked.

| Stage | student | Frozen Encoder GEMM | Reason |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | tiny correctness; NaN attribution must be clean |
| **B0** | **bf16** | **fp8** | gate 0→0.3 hugs Theorem A; frozen stack is inference |
| **B1** | **fp8** | **fp8** | Decoder allowed linear GEMMs (including lm_head / QKV/O) |
| **B2** | **fp8** | n/a | both stacks unfrozen; the same GEMM set remains FP8 |
| C | bf16 | fp8 | indexer KL is local, tensors are small |

B1+B2 = 27B+15B = 42B / 50B = **84%** of the token envelope. That is 89% of C1 FLOPs. B0's 8B is only **11.0%** of C1 FLOPs, so leaving the student in bf16 barely hurts wall-clock.

Scaling: prefer DeepSeek-V3-style **tile scaling** (activations 1×128, weights 128×128); use delayed scaling when the kernel is missing. Master weights stay bf16; Adam \(m,v\) stay fp32; Muon orthogonalization stays fp32. FP8 is the **GEMM compute dtype**, not the only storage dtype.

---

## 4. Modules that must stay high precision

`FP8_KEEP_HIGH_PREC`:

| Module | Why it cannot be FP8 |
| --- | --- |
| Input \(E_{\mathrm{in}}\) | Theorem E: when B0/B1 freeze the Encoder, \(X^0=s_{\mathrm{emb}}\,\mathrm{onehot}\,E_{\mathrm{in}}\) is the frozen stack's input distribution. An FP8 table would add quantization noise into 16 frozen MiniCPM5 residual layers. |
| lm_head | untied output head; trainable in B1, but still the classification surface, not on the FP8 GEMM whitelist. |
| RMSNorm | scale-sensitive under zero-centering + weight decay; DeepSeek also leaves LN in high precision. |
| router | expert choice is discrete; a few ULPs on the logits change top-\(k\). Same for aux-loss-free bias. |
| gate | Theorem A's bypass; B0's \(g\in[0,0.3]\) must be smooth. |
| indexer | Phase C must align dense attention distributions; an FP8 scoring head would dirty the KL target. |
| attn softmax | exp + normalize is stable only in bf16/fp32; what goes FP8 is the QK/AV GEMM, not softmax. |

Hash-MoE is `token_id → expert_id` and has no learned router GEMM; if an affine sits beside it, it still follows the whitelist.

---

## 5. Wall-clock ledger (fallback C1+FP8)

Let \(F_0,F_1,F_2\) be B0/B1/B2 Kaplan FLOPs, and \(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) the B0 Encoder forward. **Locked C1+FP8**:

\[
H_{\mathrm{FP8}}
= H(F_0 - F_{\mathrm{enc}}^{0})
+ \frac{H(F_{\mathrm{enc}}^{0})}{1.5}
+ \frac{H(F_1)}{1.5}
+ \frac{H(F_2)}{1.5}
\approx 729\ \text{H100-h}.
\]

Relative to joint bf16 1,325: **55%**. Relative to C1 bf16 1,046: about **30%** more wall-clock cut. **729 is the Hopper/Ada fallback wall-clock, not the published value.**

Sensitivity (not locked):

| Assumption | H100-h | vs joint |
| --- | ---: | --- |
| B0 student 1.5× as well | 697 | 53% |
| All-stage MoE-Amdahl 1.69× | 617 | 47% |
| Peak 2× | 523 | 39% |

Do not write 2× into the plan. If measured \(S<1.3\) on 12B, fall B1/B2 back to bf16; C1's 1,046 remains.

Frozen Encoder weights 4.38B: bf16 8.76 GB → FP8 storage 4.38 GB. This is the inference replica; master can still be bf16. What is saved is bandwidth, not 6NT.

---

## 6. Orthogonal to Muon / VRAM / FP4

- **Muon**: Newton-Schulz does ~5 fp32 iterations on the momentum matrix; that is not the same path as network FP8 GEMM. The §13 "Muon → FP8" ladder becomes: L0 stays bf16; Muon may open in B1 in parallel with FP8, each with its own fallback switch.
- **C1 VRAM leverage remains**: B1 Adam 62%, detach activations ~38% (leaving 26/42). FP8 further cuts GEMM activation bandwidth; it does not replace freeze.
- **FP4 expert storage** (V4): inference/checkpoint option. Training still needs a bf16 master and an fp32 optimizer; the gain is small at 12B, not in the recipe.
- **Phase D**: on long context the attention quadratic term grows (budget document §6). Consider attn-projection FP8 / FP8 KV then; this document only locks Phase B at 4K.

That is why §15.3 ranks the unfreeze curriculum ahead of FP8: C1 changes the operation count, FP8 changes operations per second. Cut \(F\) first, then multiply \(S\).

---

## 7. Failure modes (avoid before implementation)

| Failure | Source | Avoid |
| --- | --- | --- |
| Thinking FP8 reduces 6NT | Writing throughput as FLOPs | Theorem G; split columns in §15.1 |
| Publishing 2× | Copying the H100 peak | 12B Amdahl 1.69; lock 1.5× |
| B0 student FP8 dirtying Theorem A | gate neighborhood + new modules | B0 student bf16; FP8 only frozen Encoder forward |
| FP8 input table drifting the frozen Encoder | Training/quantizing \(E_{\mathrm{in}}\) | whitelist + Theorem E |
| router FP8 changing top-\(k\) | discrete choice | router / bias high precision |
| L0 mixed precision | tiny cannot be attributed | L0 bf16 |
| indexer FP8 alignment failure | Phase C KL | C indexer bf16 |
| Using FP8 to replace C1 | dtype ≠ freeze | C1 first, then FP8 |
| FP4 experts as training weights | no master | FP4 as later storage only |

What theory **cannot** rule out: 12B expert GEMM actually \(S\approx 1.2\), a loss spike when B1 switches to FP8, E4M3 activation overflow. Those are L1 experiments; on a spike fall that stage's student dtype back, do not change C1 freeze boundaries, and do not revert to two independent LMs.

---

## 8. Claim ledger

`python3 scripts/param_budget.py --verify` adds, beyond middle-tier 22 + curriculum 12:

| Claim | Result |
| --- | --- |
| FP8 does not change Kaplan 6NT | PASS |
| H100 FP8 peak ≈ 2× bf16 | PASS |
| Conservative wall-clock 1.5× | PASS |
| MoE ≥70% of 6NT | PASS (81.9%) |
| MoE-only Amdahl ∈ [1.50, 1.75] | PASS (1.69×) |
| Locked wall-clock is C1+FP8 mixed | PASS |
| Published C1+FP8 ≈729 (≤60% of joint bf16) | PASS (729 / 1,325 = 55%) |
| B0 student = bf16 | PASS |
| B1/B2 fallback student = fp8 | PASS |
| B1+B2 ≥80% of 50B | PASS (84%) |
| B0 ≤15% of C1 FLOPs | PASS (11.0%) |
| \(E_{\mathrm{in}}\) / lm_head / router / LN / gate / indexer / softmax high precision | PASS |
| C1 bf16 hours are not rewritten by the FP8 policy | PASS (1,046) |

---

## 9. Plan revisions (this PR)

1. **Phase B wall-clock has been changed to C1+NVFP4 (571).** This document's 729 is Hopper/Ada fallback only; C1 bf16 1,046 is the operation-count ledger only.
2. §6 / §8: precision becomes the C1+FP8 module policy.
3. §15.1: the default Phase B row is C1+FP8 **729**; 1,325 is labeled joint bf16 baseline.
4. §15.3: the published product of levers #5+#6 is 729 (55%).
5. §13: L0 stays bf16; FP8 opens from B1.
6. Do not write a training-code skeleton (per the user request, close the theory first).

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
python3 scripts/param_budget.py --staged --curriculum --fp8
python3 -m unittest tests.test_param_budget
```
