# CAT-YOKO NVFP4 theory verification

> Division of labor with the previous four docs: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks middle-tier parameters / 6NT / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) checks the C1 unfreeze curriculum; [`FP8_THEORY.md`](FP8_THEORY.md) is the **Hopper / Ada fallback ledger** (C1+FP8 = 729 H100-h). **This document checks the published dtype**: every linear GEMM that does not have to be bf16/fp32 is cut down to NVIDIA NVFP4; target card **RTX PRO 6000 / 6000D Blackwell**.
> The spec is still the middle tier: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. Base MiniCPM5-2B (Llama GQA). C1 freeze boundaries are unchanged. **Phase B wall-clock is locked as C1+NVFP4.** This PR does not write Transformer Engine kernels.
> Executable assertions: `python3 scripts/param_budget.py --verify` (including NVFP4 claims), `--nvfp4`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **6NT is invariant, Amdahl upper bounds, compatibility with Theorems A/E, and the must-stay-high-precision set**; it cannot prove that 12B small experts on sm_120 actually saturate 2.0×. That is a Blackwell L1 experiment.

---

## 0. Conclusion (read this first)

**The Phase B wall-clock recipe is locked as C1+NVFP4: 571 H100-h, 43% of joint bf16.** The accounting unit remains H100-h: RTX PRO 6000 Server BF16 peak 1 PFLOP is the same order as H100 SXM 0.989 PFLOP, so hours at 40% MFU are comparable. 6000D is the same Blackwell family; VRAM/peak follow the datasheet of the day, and are not assumed 1:1 with the 96GB 6000.

Joint bf16 (1,325) is only the 100% baseline. C1 bf16 (1,046) is the operation-count ledger. C1+FP8 (729) is the **Hopper/Ada fallback when there are no FP4 tensor cores**, no longer the published wall-clock. What actually runs is **C1's token split × mixed NVFP4**: B0 student stays bf16; B1/B2 every "not required high-precision" linear GEMM + frozen Encoder forward go NVFP4, speedup locked at **2.0× vs bf16** (another ×1.33 on top of the old FP8 1.5×, at the low end of NVIDIA GB200/GB300 vs FP8 1.31–1.73×). Peak 4× (261) and all-stage 2× (523, B0 student NVFP4 too) are not locked.

The decision rule is one sentence: **if it is not required bf16, use NVFP4.** lm_head and attn QKV/O projections on the old FP8 whitelist are not required bf16, and are taken off the whitelist. V4-style FP4 **expert storage** is still not in the recipe (that is checkpoint/inference, not training GEMM).

| Approach | 50B tok H100-h | vs joint bf16 | Role |
| --- | ---: | ---: | --- |
| Train both stacks together, bf16 | 1,325 | 100% | baseline |
| C1 unfreeze curriculum, bf16 | 1,046 | 79% | operation-count ledger |
| C1+FP8 (Hopper/Ada fallback) | 729 | 55% | when there is no Blackwell |
| **C1+NVFP4** (B0 student bf16; remaining allowed GEMMs 2.0×) | **571** | **43%** | **locked** |
| Apply 2.0× to every C1 stage (B0 student NVFP4 too) | 523 | 39% | sensitivity |
| C1 × peak 4× | 261 | 20% | upper bound, not published |

Several more nails:

1. **6NT is independent of dtype** (Theorem G). NVFP4, like FP8, only enters the wall-clock column \(S\) and does not change Kaplan operation count.
2. **Publish \(S=2.0\) vs bf16, do not publish 4×.** RTX PRO 6000 Server: BF16 1 / FP8 2 / FP4 4 PFLOPS. Hardware peak vs this card's FP8 is 2×; NVIDIA MaxText end-to-end on GB200/GB300 is 1.31–1.73× vs FP8. 12B experts are only 12.58M, so take the low end \(1.5\times 1.33\approx 2.0\) vs bf16. MoE active share \(f=81.9\%\), GEMM 4× Amdahl is 2.59×; counting attn projections + lm_head as GEMM too (\(f=95.8\%\)) is 3.55× — that is an upper bound.
3. **B0 student stays bf16.** This is required: gate 0→0.3 hugs Theorem A, new modules are randomly initialized. Frozen Encoder forward is pure inference GEMM and can go NVFP4 already in B0.
4. **B1/B2 student is `nvfp4`, not `nvfp4_moe`.** Every allowed linear GEMM is cut: MoE expert FPROP/DGRAD/WGRAD, self-attn \(W_Q,W_K,W_V,W_O\), cross-attn \(W_Q,W_O\), cache \(W_K,W_V\), **untied lm_head**. NVIDIA MaxText by default quantizes only MLP and leaves attention blocks high precision; that is their conservative recipe. This repo, by "not required bf16", also includes QKV/O and lm_head. If B1 diverges, **the first fallback is QKV/O back to high precision (MaxText accounting)**, not changing C1, and not reverting to two LMs.
5. **The must-stay-high-precision set is shorter.** Keep only: input \(E_{\mathrm{in}}\) (table lookup + Theorem E), RMSNorm / QK-Norm (fp32), router (discrete top-\(k\)), YOCO scalar gate (Theorem A), Lightning Indexer, attn softmax / SDPA score+context. **lm_head is no longer on the whitelist.**
6. **L0 and Phase C indexer stay bf16.** Tiny correctness does not mix precision; indexer alignment is layer-internal KL on small tensors.
7. **Master weights stay bf16; Adam \(m,v\) stay fp32; Muon Newton-Schulz stays fp32.** NVFP4 is the GEMM compute dtype: FPROP/DGRAD/WGRAD eat NVFP4, emit BF16, fold into the fp32 master. The recipe is 16-element microblocks + FP8 E4M3 block scaling + per-tensor FP32 scale; weights 2D 16×16; RHT only on WGRAD; stochastic rounding on the gradient quantizer.
8. **C1 and NVFP4 multiply, they do not add.** C1 first cuts 21% of FLOPs, then multiplies the remaining wall by 2.0× (B0 student excepted). Relative to joint bf16: \(79\% \times\) (B1/B2 2.0×, B0 near 1×) ≈ **43%**. That product is the locked wall-clock **C1+NVFP4 = 571**.
9. **This PR does not implement TE / sm_120 kernels.** Without Blackwell: the same module policy falls back to an FP8 placeholder, then to bf16 autocast. RTX 4080 SUPER cannot run NVFP4. Official TE fused NVFP4 (RHT/SR) may be unavailable on sm_120; the published recipe still writes the full training recipe, and kernels are wired on the target card later.

Claim ledger: middle-tier 22 + curriculum 12 + FP8 fallback 13 + NVFP4 16, `--verify` **63/63** pass.

---

## 1. Theorem G still holds: dtype does not change 6NT

Same Kaplan ledger as [`FP8_THEORY.md`](FP8_THEORY.md) §1:

\[
F = 6\,N_{\mathrm{act}}T,\qquad
H_{\text{wall}} = \frac{F}{\eta_{\mathrm{bf16}}}\, /\, S,\qquad
\eta_{\mathrm{bf16}} = 4.0\times 10^{14}\ \text{FLOPS (40% MFU)}.
\]

NVFP4 only enters \(S\). C1's \(F_{\mathrm{C}}\) is 21% less than joint; bf16 wall-clock is still 1,046.

RTX PRO 6000 Blackwell Server Edition (NVIDIA launch page):

\[
\text{BF16 } 1\ \text{PFLOP},\quad
\text{FP8 } 2\ \text{PFLOPS},\quad
\text{FP4 } 4\ \text{PFLOPS},\quad
96\,\text{GB GDDR7},\ 1597\,\text{GB/s}.
\]

Vs this card's BF16, FP4 peak is 4×. Vs this card's FP8, peak is 2×. Published \(S\) does not use the peak. NVIDIA MaxText (2026-06-08) on GB200/GB300 for Llama 3 8B / 3.1 405B: **NVFP4 vs FP8 = 1.31×–1.73×**, loss gap +0.026 nats. 12B small experts take the low end, times the already-published FP8 1.5×:

\[
S_{\mathrm{pub}} = 1.5 \times \frac{2.0}{1.5} = 2.0
\quad\text{(vs bf16; vs FP8 = } 2.0/1.5 \approx 1.33\in[1.31,1.73]\text{)}.
\]

H100 SXM BF16 peak \(9.89\times 10^{14}\) and 6000's \(1.00\times 10^{15}\) differ by <2%, so **H100-h ≈ 6000-h** (same 40% MFU). 6000D does not get a separate wall-clock table.

---

## 2. What must stay bf16/fp32, what must be NVFP4

Student = tensors that receive gradients. Master storage is always bf16, independent of compute dtype.

### 2.1 Must stay high precision (not NVFP4)

| Module | Why it must |
| --- | --- |
| Input \(E_{\mathrm{in}}\) | Table lookup is not a GEMM. Theorem E: when B0/B1 freeze the Encoder, \(X^0\) is the frozen stack's input distribution; quantization noise would pour into 16 frozen MiniCPM5 residual layers. |
| RMSNorm / QK-Norm | Small-vector variance, fp32 then cast back. Not a GEMM. |
| attn softmax / SDPA score+context | exp + normalize is stable only in fp32. NVIDIA also writes that softmax exponentially amplifies QKᵀ quantization noise; score/context are separated from linear projections. |
| router + expert bias | Expert choice is discrete top-\(k\); 21-dim logits also do not align to NVFP4 16-element microblocks. A few ULPs change routing. |
| YOCO scalar gate | Theorem A's bypass; B0's \(g\in[0,0.3]\) must be smooth. Not a GEMM. |
| Lightning Indexer | Phase C must align dense attention distributions; a 4-bit scoring head would dirty the KL target. |
| B0 student | New modules randomly initialized + gate ramp. NVIDIA also observed "full NVFP4 diverges, last few layers need BF16". |
| L0 / tiny | Correctness attribution. |
| Teacher MiniCPM5 | Already-trained bf16 checkpoint. |
| Adam \(m,v\) / Muon NS | fp32 optimizer state, not network GEMM. |

Hash-MoE is `token_id → expert_id` and has no learned router GEMM.

### 2.2 Not required bf16 → published NVFP4

Old C1+FP8 only cut MoE experts + frozen Encoder forward; leaving lm_head / attn projections on the whitelist was conservative, not a numerical necessity. Taken off per the user accounting:

| Slot | Stage | Notes |
| --- | --- | --- |
| MoE expert SwiGLU (gate/up/down) FPROP/DGRAD/WGRAD | B1/B2; frozen Encoder forward from B0 | **81.9%** of counted 6NT |
| Self-attn \(W_Q,W_K,W_V,W_O\) | B1/B2; frozen Encoder forward from B0 | projections are linear GEMMs; softmax stays high precision |
| cross-attn \(W_Q,W_O\) | B1/B2 | in B0 they **are** the student, still bf16 |
| cache \(W_K,W_V\) | B1/B2 | B0 same student bf16 |
| **untied lm_head** | B1/B2 | \(d\times V=2048\times 130560\) is a large GEMM; Theorem E governs the input table, not the head |

NVIDIA MaxText: "the three GEMMs quantize only MLP to NVFP4; attention blocks (QKV, O, score/context) stay high precision". This recipe agrees that **score/context must stay high precision**, and disagrees that QKV/O and lm_head are required bf16. On divergence, first fall QKV/O back to high precision; lm_head can still be NVFP4.

---

## 3. C1+NVFP4 locked (stacked on C1, freeze boundaries unchanged)

| Stage | student | Frozen Encoder GEMM | Reason |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | tiny correctness |
| **B0** | **bf16** | **nvfp4** | Theorem A; frozen stack is inference |
| **B1** | **nvfp4** | **nvfp4** | Decoder allowed linear GEMMs + lm_head |
| **B2** | **nvfp4** | n/a | same GEMM set after both stacks unfreeze |
| C | bf16 | nvfp4 | indexer is local, tensors are small |

B1+B2 = 84% tokens / 89% C1 FLOPs. B0 is **11.0%** of C1 FLOPs, of which frozen Encoder forward is about 17% of B0; leaving the student in bf16 barely eats the NVFP4 gain (571 vs all-stages 523, a gap of ~48 H100-h).

Scaling: 16-element microblocks, weights 2D 16×16, RHT on WGRAD, stochastic rounding on gradients. Without a kernel, the same modules go through an FP8 placeholder, then bf16 autocast.

96GB: Encoder offload and CPU Adam in B0/B1 become **operationally optional**, not a dtype change. 32GB cards still follow the original offload path; they cannot run NVFP4.

---

## 4. Wall-clock ledger (locked C1+NVFP4)

Let \(F_0,F_1,F_2\) be B0/B1/B2 Kaplan FLOPs, and \(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) the B0 Encoder forward. **Locked**:

\[
H_{\mathrm{NVFP4}}
= H(F_0 - F_{\mathrm{enc}}^{0})
+ \frac{H(F_{\mathrm{enc}}^{0})}{2.0}
+ \frac{H(F_1)}{2.0}
+ \frac{H(F_2)}{2.0}
\approx 571\ \text{H100-h}.
\]

Relative to joint bf16 1,325: **43%**. Relative to C1 bf16 1,046: about **45%** more wall-clock cut. Relative to C1+FP8 729: about **22%** more wall-clock cut. **571 is the Phase B published wall-clock.**

Sensitivity (not locked):

| Assumption | H100-h | vs joint |
| --- | ---: | --- |
| B0 student 2.0× as well | 523 | 39% |
| All-stage MoE-Amdahl 4× GEMM (2.59×) | 403 | 30% |
| Peak 4× | 261 | 20% |

Do not write 4× into the plan. If measured \(S<1.7\) vs bf16 on 12B, first fall QKV/O back to high precision; if still not enough, fall B1/B2 student to FP8 or bf16. C1's 1,046 remains.

---

## 5. Orthogonal to FP8 / VRAM / V4 FP4 storage

- **C1+FP8 = 729** remains a recomputable fallback ledger, see [`FP8_THEORY.md`](FP8_THEORY.md). Hopper has no FP4 tensor cores.
- **C1 VRAM leverage remains**: B1 Adam 62%, detach activations ~38%. NVFP4 further cuts GEMM activation bandwidth; it does not replace freeze. 96GB only makes offload optional.
- **V4 FP4 expert storage**: inference/checkpoint option. Training still needs a bf16 master and an fp32 optimizer; not in the recipe. NVFP4 is a **compute** format, not that storage path.
- **Muon**: Newton-Schulz stays fp32, orthogonal to network NVFP4 GEMM.

Cut \(F\) first (C1), then multiply \(S\) (NVFP4).

---

## 6. Failure modes (avoid before implementation)

| Failure | Source | Avoid |
| --- | --- | --- |
| Thinking NVFP4 reduces 6NT | Writing throughput as FLOPs | Theorem G |
| Publishing 4× | Copying the 6000 peak | 12B small experts; lock 2.0× vs bf16 |
| B0 student NVFP4 dirtying Theorem A | gate neighborhood + new modules | B0 student bf16 |
| NVFP4 input table drifting the frozen Encoder | Training/quantizing \(E_{\mathrm{in}}\) | whitelist + Theorem E |
| router NVFP4 changing top-\(k\) | discrete choice + width 21 | router high precision |
| Quantizing softmax to NVFP4 | exponential amplification of QKᵀ noise | score/context fp32; only quantize linear projections |
| L0 mixed precision | tiny cannot be attributed | L0 bf16 |
| Treating NVFP4 as already live on a 4080 | Ada has no FP4 tensor cores | fall back to FP8 placeholder / bf16 |
| Using NVFP4 to replace C1 | dtype ≠ freeze | C1 first, then NVFP4 |
| Treating V4 FP4 storage as training weights | no master | storage as a later option |
| Copying MaxText "MLP only" as "required" | treating a conservative recipe as a numerical necessity | publish with QKV/O+lm_head; on divergence fall QKV/O |

What theory **cannot** rule out: TE fused kernels unavailable on sm_120, 12B expert GEMM actually \(S\approx 1.4\), a loss spike when B1 switches to NVFP4. Those are L1; on a spike fall that stage's student slots back, do not change C1 freeze boundaries.

---

## 7. Claim ledger

`python3 scripts/param_budget.py --verify` adds, beyond middle-tier 22 + curriculum 12 + FP8 fallback 13:

| Claim | Result |
| --- | --- |
| NVFP4 does not change Kaplan 6NT | PASS |
| RTX PRO 6000 peak 1/2/4 PFLOP | PASS |
| Conservative wall-clock 2.0× vs bf16 (not 4×) | PASS |
| Published NVFP4/FP8 ratio ∈ [1.31, 1.73] | PASS (1.33) |
| Locked wall-clock is C1+NVFP4 | PASS |
| Published C1+NVFP4 ≈571 (≤45% of joint bf16) | PASS (571 / 1,325 = 43%) |
| B0 student = bf16 | PASS |
| B1/B2 student = nvfp4 | PASS |
| lm_head is an NVFP4 GEMM (not in the must-stay-high-precision set) | PASS |
| attn softmax must stay high precision; QKV is not required bf16 | PASS |
| Must-stay-high-precision set does not include lm_head | PASS |
| C1 bf16 hours are not rewritten by the NVFP4 policy | PASS (1,046) |
| C1+FP8 fallback still ≈729 | PASS |
| 6000 BF16 peak ≈ H100 | PASS |
| MoE-only 4× Amdahl is sensitivity only | PASS |
| Allowed GEMMs are ≥95% of 6NT | PASS (95.8%, including lm_head) |

---

## 8. Plan revisions (this PR)

1. **Phase B wall-clock is locked as C1+NVFP4: 571 H100-h.** Joint bf16 1,325 is baseline only; C1 bf16 1,046 is the operation-count ledger only; C1+FP8 729 is demoted to Hopper/Ada fallback.
2. Precision policy: every linear GEMM that is not required bf16 is NVFP4. The whitelist no longer includes lm_head.
3. Target hardware: published wall-clock is accounted at RTX PRO 6000 magnitude; **the cards that actually eat NVFP4 training kernels are B200 / SM 10.0 / 10.3** (``TeNvfp4Linear`` + default ``NVFP4BlockScaling``). sm_120 uses ``Nvfp4Linear`` emulation. Attention topology is unchanged. See [`B200_TRAIN.md`](B200_TRAIN.md).
4. Do not change 16/26, C1, causal Encoder, or M2 default.

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --nvfp4
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
