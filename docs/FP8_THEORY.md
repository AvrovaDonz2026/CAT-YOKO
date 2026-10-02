# CAT-YOKO FP8 Theory Verification

> **This note is the Hopper / Ada fallback ledger. It is no longer the Phase B published wall-clock.** The published dtype is **C1+NVFP4 = 571 H100-h**; see [`NVFP4_THEORY.md`](NVFP4_THEORY.md). The 729 / 1.5× / `fp8_moe` numbers below remain recomputable as the path when there is no FP4 tensor core.
>
> Division of labor with the previous three notes: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks mid-tier parameters / 6NT / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) checks the C1 unfreeze curriculum; **this note checks the Hopper FP8 fallback**: which mid-tier C1 training blocks may run FP8, how much more wall-clock can be cut, and which modules must stay high precision.
> The spec is still the mid-tier graph: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. The base is MiniCPM5-2B (Llama GQA). The C1 freeze boundary does not change. No new architecture is introduced, and no training skeleton is written.
> Executable assertions: `python3 scripts/param_budget.py --verify` (includes the FP8 fallback claims), `--fp8`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **6NT is unchanged, an Amdahl upper bound, and compatibility with Theorems A/E**. It cannot prove that FP8 kernels on 12B actually reach 1.5×. That is an L1 experiment.

---

## 0. Bottom line (read this first)

**The Phase B wall-clock recipe was once locked to C1+FP8: 729 H100-h, 55% of joint bf16. It is now demoted to the Hopper/Ada fallback; the published value is 571 in [`NVFP4_THEORY.md`](NVFP4_THEORY.md).**

Joint bf16 (1,325) is only the 100% control, not the path to run. C1 bf16 (1,046) is the operand ledger and still counts 6NT in bf16. The mixed FP8 policy in this note is the **Hopper/Ada fallback**; Blackwell runs [`NVFP4_THEORY.md`](NVFP4_THEORY.md). All-phase 1.5× (697) and peak 2× (523) are not locked.

**A substantial share** of mid-tier training is MoE expert GEMM: **90.6%** of stored parameters and **81.9%** of counted 6NT. That block (plus frozen encoder inference GEMM) runs FP8. FP8 **does not change** Kaplan \(6N_{\mathrm{act}}T\); it only raises the effective throughput at which those FLOPs complete on an H100.

| Recipe | 50B tok H100-h | vs joint bf16 | Role |
| --- | ---: | ---: | --- |
| Train both stacks together, bf16 | 1,325 | 100% | Control |
| C1 unfreeze curriculum, bf16 | 1,046 | 79% | Operand ledger |
| **C1+FP8** (B0 student bf16; B1/B2 allowed GEMMs and frozen encoder forward 1.5×) | **729** | **55%** | **Hopper/Ada fallback** |
| C1 all phases at 1.5× (B0 student also FP8) | 697 | 53% | Sensitivity |
| C1 × peak 2× | 523 | 39% | Upper bound, not published |

These points also have to stay nailed down:

1. **6NT is independent of dtype** (Theorem G). Plan §15.1's 1,325 / 1,046 remain the bf16 operand ledger; FP8 only enters the wall-clock column. After untying, the plan convention matches a full forward and no longer lists a separate emb×2 1,413.
2. **Publish 1.5×, not 2×.** H100 SXM FP8 peak ≈ 1979 TFLOPS, ~2.0× bf16 989. MoE active share \(f=81.9\%\) with GEMM at 2× Amdahl is **1.69×**, which still supports 1.5×. 12B expert GEMMs (\(d=2048\), \(d_{\mathrm{moe}}=2048\), 12.58M per expert) are far smaller than DeepSeek-671B; dispatch/combine, scales, and small kernels all eat the peak. MoE+attn \(f=95.8\%\) gives 1.92×; that is an upper bound.
3. **B0 student stays bf16.** The gate climbs from 0 to 0.3, sitting on Theorem A's residual neighborhood; new modules are randomly initialized, so compute precision does not change here. Frozen encoder forward is pure inference GEMM, so B0 may already use FP8 there.
4. **B1+B2 cover 84% of the 50B token envelope and 89% of C1 FLOPs.** That is the "substantial share": the stages that actually run for a long time use a low-precision student. B0 is only 11.0% of C1 FLOPs, so leaving the student in bf16 barely spends the gain (729 vs all-phase 697, a gap of ~32 H100-h).
5. **`KEEP_HIGH_PREC` is embed / rms_norm / qk_norm / router / gate / indexer / attn_softmax.** The historical fallback article listed lm_head in this set. Published NVFP4 **takes lm_head and attn QKV/O off the must-high-precision list**; `lm_head` is a GEMM slot, not `KEEP_HIGH_PREC`. Theorem E still forbids quantizing the input table.
6. **L0 and the Phase C indexer stay bf16.** Tiny correctness does not mix precision; indexer alignment is layer-local KL on small tensors.
7. **Muon Newton-Schulz stays fp32**, orthogonal to network GEMM FP8. V4-style FP4 expert storage is a later option and does not enter this recipe.
8. **C1 and FP8 multiply; they do not add.** C1 first cuts 21% of FLOPs, then the remaining wall is multiplied by 1.5× (B0 excepted). Relative to joint bf16: \(79\% \times\) (B1/B2 1.5×, B0 near 1×) ≈ **55%**. That product is the fallback wall-clock **C1+FP8 = 729**.

Claim ledger: the 13 FP8 fallback claims remain in `--verify`; the published total is in [`NVFP4_THEORY.md`](NVFP4_THEORY.md) (**63/63**).

Ampere has no FP4 tensor core. **Ampere NVFP4 emulation is a speed trap**; do not run `Nvfp4Linear` emu there and call it the published recipe. Hopper/Ada use this FP8 fallback (or bf16 autocast). B200 uses TE NVFP4 FPROP on the frozen encoder; see [`NVFP4_THEORY.md`](NVFP4_THEORY.md).

---

## 1. Theorem G: dtype does not change 6NT

Kaplan / Hoffmann convention (same as the budget note):

\[
F = 6\,N_{\mathrm{act}}T = \underbrace{2N_{\mathrm{act}}T}_{\text{forward}} + \underbrace{2N_{\mathrm{act}}T}_{\text{activation bwd}} + \underbrace{2N_{\mathrm{act}}T}_{\text{weight grad}}.
\]

This is multiply-add count, not joules and not seconds. Swapping expert projections from bf16 to FP8 E4M3/E5M2 does not change \(F\). What changes is

\[
H_{\text{wall}} = \frac{F}{\eta_{\mathrm{bf16}}}\, /\, S,\qquad
\eta_{\mathrm{bf16}} = 4.0\times 10^{14}\ \text{FLOPS (40% MFU)}.
\]

\(S=1\) is the bf16 column in plan §15.1; C1 \(F_{\mathrm{C}}\) is 21% below joint, so bf16 wall-clock falls from 1,325 to 1,046. FP8 only enters \(S\).

H100 SXM:

\[
\frac{\text{FP8 peak}}{\text{bf16 peak}} = \frac{1.979\times 10^{15}}{9.89\times 10^{14}} \approx 2.00.
\]

If MFU stayed 40% on FP8, effective throughput would be 7.92×10¹⁴ and \(S=2\). That does not hold on 12B: expert matrices are small, MoE communication is not accelerated, and scaling has overhead. The published value is locked at **\(S=1.5\)**.

---

## 2. "Substantial share": MoE GEMM is 81.9% of 6NT

One full mid-tier forward is \(N_{\mathrm{fwd}}=6.36\mathrm{B}\) (embed and head each counted once):

| Block | Active parameters | Share of 6NT |
| --- | ---: | ---: |
| MoE experts (top-\(k\) SwiGLU) | 5.21B | **81.9%** |
| Self-attention + cross-attn projections | 0.61B | 9.7% |
| Untied embedding + lm_head | 0.53B | 8.4% |
| RMSNorm / router / softmax | (ignored in 6NT) | — |

On the storage side MoE is still **90.6%** (882 expert slots, including replicas that never enter top-\(k\)). Training FLOPs follow activations, so the "substantial share" is counted at **81.9%**. GQA compresses attention from the old placeholder 24% down to 9.7%, and the MoE share rises accordingly — Amdahl only gets better, not worse.

Replacing only MoE GEMM with 2× and leaving the rest at 1×:

\[
S_{\mathrm{MoE}} = \frac{1}{0.819/2 + 0.181} \approx 1.69.
\]

1.69 is above the published 1.5; the gap is reserved for dispatch/combine and 12B small kernels. If attn projections also go FP8, \(f=95.8\%\) and \(S\approx 1.92\) — that is an upper bound and **is not written into §15.1**.

Frozen encoder forward (B0/B1) is the inference form of the same GEMMs: no weight gradients, no activation backward, the safest FP8 cut in the whole graph. In B0 it is only a small fraction of that phase's FLOPs.

---

## 3. C1+FP8 fallback (stacked on C1; freeze boundary unchanged)

This is the fallback recipe, not the published wall-clock. Student = tensors that receive gradients. Frozen encoder GEMMs are a separate column. All-phase 1.5× (B0 student also FP8) is not locked.

| Phase | student | Frozen encoder GEMM | Reason |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | Tiny correctness; NaN attribution must be clean |
| **B0** | **bf16** | **fp8** | Gate 0→0.3 sits on Theorem A; the frozen stack is inference |
| **B1** | **fp8** | **fp8** | Decoder allowed linear GEMMs (including lm_head / QKV/O) |
| **B2** | **fp8** | n/a | Both stacks unfrozen; the same GEMM set stays FP8 |
| C | bf16 | fp8 | Indexer KL is local and tensors are small |

B1+B2 = 27B+15B = 42B / 50B = **84%** of the token envelope. That is 89% of C1 FLOPs. B0's 8B is only **11.0%** of C1 FLOPs, so leaving the student in bf16 barely hurts wall-clock.

Scaling: prefer DeepSeek-V3-style **tile scaling** (activations 1×128, weights 128×128); use delayed scaling when the kernel is missing. Master weights stay bf16; Adam \(m,v\) stay fp32; Muon orthogonalization stays fp32. FP8 is a **GEMM compute dtype**, not the only storage dtype.

---

## 4. Modules that must stay high precision

`FP8_KEEP_HIGH_PREC` = `KEEP_HIGH_PREC` = embed / rms_norm / qk_norm / router / gate / indexer / attn_softmax. **`lm_head` is a GEMM slot, not `KEEP_HIGH_PREC`.**

| Module | Why it cannot be FP8 |
| --- | --- |
| Input \(E_{\mathrm{in}}\) (`embed`) | Theorem E: when B0/B1 freeze the encoder, \(X^0=s_{\mathrm{emb}}\,\mathrm{onehot}\,E_{\mathrm{in}}\) is the frozen stack's input distribution. An FP8 table would add quantization noise to 16 frozen MiniCPM5 residual layers. |
| RMSNorm (`rms_norm`) | Scale-sensitive under zero-centering + weight decay; DeepSeek also leaves LN in high precision. |
| QK-Norm (`qk_norm`) | Same small-vector variance path as RMSNorm; fp32 then cast back. Not a GEMM. |
| router (`router`) | Expert choice is discrete; a few ULPs on the logits change top-\(k\). Aux-loss-free bias is the same. |
| gate (`gate`) | Theorem A's bypass; B0 \(g\in[0,0.3]\) must be smooth. |
| indexer (`indexer`) | Phase C has to match a dense attention distribution; an FP8 scoring head would dirty the KL target. |
| attn softmax (`attn_softmax`) | Exp + normalize is stable only in bf16/fp32; what may be FP8 is the QK/AV GEMM, not softmax. |

The historical fallback whitelist also listed **lm_head** (untied classifier; B1-trainable) as must-high-precision. The published set does not: B1 allowed linear GEMMs include lm_head / QKV/O, matching [`NVFP4_THEORY.md`](NVFP4_THEORY.md). Theorem E still governs the input table, not the head.

Hash-MoE is `token_id → expert_id`; there is no learned router GEMM. If an affine sits beside it, that affine still follows the whitelist.

---

## 5. Wall-clock ledger (fallback C1+FP8)

Let \(F_0,F_1,F_2\) be B0/B1/B2 Kaplan FLOPs, and \(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) the B0 encoder forward. **Locked C1+FP8 fallback**:

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
| B0 student also 1.5× | 697 | 53% |
| All-phase MoE-Amdahl 1.69× | 617 | 47% |
| Peak 2× | 523 | 39% |

Do not write 2× into the plan. If measured \(S<1.3\) on 12B, roll B1/B2 back to bf16; C1's 1,046 remains.

Frozen encoder weights 4.38B: bf16 8.76 GB → FP8 storage 4.38 GB. This is the inference replica; the master may stay bf16. What is saved is bandwidth, not 6NT.

---

## 6. Orthogonal to Muon / memory / FP4

- **Muon**: Newton-Schulz runs ~5 fp32 iterations on the momentum matrix; that is not the same path as network FP8 GEMMs. The §13 "Muon → FP8" ladder becomes: L0 stays bf16; Muon may turn on in B1 in parallel with FP8, each with its own rollback switch.
- **C1 memory leverage still holds**: B1 Adam 62%, detached activations ~38% (leaving 26/42). FP8 further cuts GEMM activation bandwidth; it does not replace freezing.
- **FP4 expert storage** (V4): an inference/checkpoint option. Training still needs a bf16 master and an fp32 optimizer; gain on 12B is small, so it does not enter the recipe.
- **Phase D**: attention's quadratic term grows at long context (budget note §6). Attn-projection FP8 / FP8 KV can wait until then; this note only locks 4K Phase B.

That is why §15.3 ranks the unfreeze curriculum ahead of FP8: C1 changes operation count, FP8 changes operations per second. Cut \(F\) first, then multiply by \(S\).

---

## 7. Failure modes (avoid these before implementation)

| Failure | Source | Avoid by |
| --- | --- | --- |
| Thinking FP8 reduces 6NT | Writing throughput as FLOPs | Theorem G; §15.1 as separate columns |
| Publishing 2× | Copying the H100 peak | 12B Amdahl 1.69; lock 1.5× |
| B0 student FP8 dirtying Theorem A | Gate neighborhood + new modules | B0 student bf16; FP8 only the frozen encoder forward |
| FP8 input table drifting the frozen encoder | Training/quantizing \(E_{\mathrm{in}}\) | Whitelist + Theorem E |
| Router FP8 changing top-\(k\) | Discrete choice | Router / bias high precision |
| L0 mixed precision | Tiny cannot attribute | L0 bf16 |
| Indexer FP8 alignment failure | Phase C KL | Phase C indexer bf16 |
| Using FP8 to replace C1 | dtype ≠ freeze | C1 first, then FP8 |
| FP4 experts as training weights | No master | FP4 is later storage only |
| Treating Ampere NVFP4 emu as this fallback | Emu is a speed trap | Hopper/Ada FP8 or bf16; B200 TE NVFP4 FPROP on the frozen encoder |

Theory **cannot** rule out: 12B expert GEMMs measuring \(S\approx 1.2\), a loss spike when B1 switches to FP8, or E4M3 activation overflow. Those are L1 experiments; on a spike, roll back that phase's student dtype, and do not change the C1 freeze boundary or return to two independent LMs.

---

## 8. Claim ledger

`python3 scripts/param_budget.py --verify` adds, on top of mid-tier 22 + curriculum 12:

| Claim | Result |
| --- | --- |
| FP8 does not change Kaplan 6NT | PASS |
| H100 FP8 peak ≈ 2× bf16 | PASS |
| Conservative wall-clock 1.5× | PASS |
| MoE ≥70% of 6NT | PASS (81.9%) |
| MoE-only Amdahl ∈ [1.50, 1.75] | PASS (1.69×) |
| Locked wall-clock is C1+FP8 mixed | PASS |
| Published C1+FP8 ≈729 (≤60% joint bf16) | PASS (729 / 1,325 = 55%) |
| B0 student = bf16 | PASS |
| B1/B2 fallback student = fp8 | PASS |
| B1+B2 ≥80% of 50B | PASS (84%) |
| B0 ≤15% of C1 FLOPs | PASS (11.0%) |
| \(E_{\mathrm{in}}\) / router / LN / gate / indexer / softmax high precision (`lm_head` is a GEMM) | PASS |
| C1 bf16 hours are not rewritten by the FP8 policy | PASS (1,046) |

---

## 9. Plan revisions (this PR)

1. **Phase B wall-clock is now C1+NVFP4 (571).** This note's 729 is Hopper/Ada fallback only; C1 bf16 1,046 is the operand ledger only.
2. §6 / §8: precision becomes the C1+FP8 module policy.
3. §15.1: the default Phase B row is C1+FP8 **729**; 1,325 is labeled joint bf16 control.
4. §15.3: the published product of levers #5+#6 is 729 (55%).
5. §13: L0 stays bf16; FP8 turns on from B1.
6. Do not write a training-code skeleton (per the user: close the theory loop first).

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
python3 scripts/param_budget.py --staged --curriculum --fp8
python3 -m unittest tests.test_param_budget
```
