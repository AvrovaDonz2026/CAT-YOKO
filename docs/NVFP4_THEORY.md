# CAT-YOKO NVFP4 Theory Verification

> Division of labor with the previous four notes: [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) checks mid-tier parameters / 6NT / KV; [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) checks causality and M1/M2/M3; [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) checks the C1 unfreeze curriculum; [`FP8_THEORY.md`](FP8_THEORY.md) is the **Hopper / Ada fallback ledger** (C1+FP8 = 729 H100-h). **This note checks the published dtype**: every linear GEMM that does not have to stay bf16/fp32 is cut to NVIDIA NVFP4; the target SKU is **RTX PRO 6000 / 6000D Blackwell**.
> The spec is still the mid-tier graph: 16/26, ≈12.25B / 2.03B-in / 4.33B-out. The base is MiniCPM5-2B (Llama GQA). The C1 freeze boundary does not change. **Phase B wall-clock is locked to C1+NVFP4.** This PR does not implement Transformer Engine kernels.
> Executable assertions: `python3 scripts/param_budget.py --verify` (includes the NVFP4 claims), `--nvfp4`; `python3 -m unittest tests.test_param_budget`.
> Theory can prove **6NT is unchanged, an Amdahl upper bound, compatibility with Theorems A/E, and the must-high-precision set**. It cannot prove that 12B small experts on sm_120 actually reach 2.0×. That is a Blackwell L1 experiment.

---

## 0. Bottom line (read this first)

**The Phase B wall-clock recipe is locked to C1+NVFP4: 571 H100-h, 43% of joint bf16.** The billing unit is still H100-h: RTX PRO 6000 Server BF16 peak is 1 PFLOP, the same order as H100 SXM 0.989 PFLOP, so hours remain comparable at 40% MFU. 6000D is the same Blackwell family; memory and peak follow the datasheet of the day, and this note does not assume a 1:1 match with the 96GB 6000.

Joint bf16 (1,325) is only the 100% control. C1 bf16 (1,046) is the operand ledger. C1+FP8 (729) is the **Hopper/Ada fallback when there are no FP4 tensor cores**, and it is no longer the published wall-clock. The run that actually has to execute is **C1 token splits × mixed NVFP4**: B0 student stays bf16; B1/B2 every linear GEMM that is not required to stay high precision, plus the frozen encoder forward, run NVFP4, with the speedup locked at **2.0× vs bf16** (the old FP8 1.5× times another ×1.33, which sits on the low end of NVIDIA GB200/GB300 NVFP4 vs FP8, 1.31–1.73×). Peak 4× (261) and all-phase 2× (523, B0 student also NVFP4) are not locked.

The decision rule is one sentence: **if it does not have to stay bf16, use NVFP4.** `lm_head` and attn QKV/O projections in the old FP8 whitelist are not required to stay bf16, so they come off that whitelist. V4-style FP4 **expert storage** still does not enter the recipe (that is checkpoint/inference storage, not a training GEMM).

| Recipe | 50B tok H100-h | vs joint bf16 | Role |
| --- | ---: | ---: | --- |
| Train both stacks together, bf16 | 1,325 | 100% | Control |
| C1 unfreeze curriculum, bf16 | 1,046 | 79% | Operand ledger |
| C1+FP8 (Hopper/Ada fallback) | 729 | 55% | When there is no Blackwell |
| **C1+NVFP4** (B0 student bf16; remaining allowed GEMMs 2.0×) | **571** | **43%** | **Locked** |
| C1 all phases at 2.0× (B0 student also NVFP4) | 523 | 39% | Sensitivity |
| C1 × peak 4× | 261 | 20% | Upper bound, not published |

These points also have to stay nailed down:

1. **6NT is independent of dtype** (Theorem G). NVFP4, like FP8, only enters the wall-clock column \(S\); it does not change Kaplan operation count.
2. **Publish \(S=2.0\) vs bf16; do not publish 4×.** RTX PRO 6000 Server: BF16 1 / FP8 2 / FP4 4 PFLOPS. Hardware peak vs this card's FP8 is 2×; NVIDIA MaxText on GB200/GB300 is 1.31–1.73× vs FP8 end to end. The 12B experts are only 12.58M each, so take the low end \(1.5\times 1.33\approx 2.0\) relative to bf16. MoE active share \(f=81.9\%\) with GEMM at 4× Amdahl is 2.59×; counting attn projections + lm_head as GEMM as well (\(f=95.8\%\)) is 3.55× — that is an upper bound.
3. **B0 student stays bf16.** This is required: the gate 0→0.3 ride sits on Theorem A, and the new modules are randomly initialized. Frozen encoder forward is pure inference GEMM, so B0 may already use NVFP4 there.
4. **B1/B2 student is `nvfp4`, not `nvfp4_moe`.** Every allowed linear GEMM is cut: MoE expert FPROP/DGRAD/WGRAD, self-attn \(W_Q,W_K,W_V,W_O\), cross-attn \(W_Q,W_O\), cache \(W_K,W_V\), and **untied lm_head**. NVIDIA MaxText by default only quantizes MLP and leaves the attention block in high precision; that is their conservative recipe. This repo follows "not required to stay bf16" and also takes QKV/O and lm_head. If B1 diverges, **the first rollback is QKV/O back to high precision (MaxText convention)**, not a change to C1 and not a return to two language models.
5. **The must-high-precision set is short.** `KEEP_HIGH_PREC` is **embed / rms_norm / qk_norm / router / gate / indexer / attn_softmax**: input \(E_{\mathrm{in}}\) (table lookup + Theorem E), RMSNorm / QK-Norm (fp32), router (discrete top-\(k\)), YOCO scalar gate (Theorem A), Lightning Indexer, and attn softmax / SDPA score+context. **`lm_head` is a GEMM slot, not `KEEP_HIGH_PREC`.**
6. **L0 and the Phase C indexer stay bf16.** Tiny correctness does not mix precision; indexer alignment is layer-local KL on small tensors.
7. **Master weights stay bf16; Adam \(m,v\) stay fp32; Muon Newton-Schulz stays fp32.** NVFP4 is a GEMM compute dtype: FPROP/DGRAD/WGRAD consume NVFP4, emit BF16, and fold into the fp32 master. The recipe is 16-element microblocks + FP8 E4M3 block scales + per-tensor FP32 scales; weights are 2D 16×16; RHT is applied only on WGRAD; stochastic rounding is applied on the gradient quantizer.
8. **C1 and NVFP4 multiply; they do not add.** C1 first cuts 21% of FLOPs, then the remaining wall is multiplied by 2.0× (B0 student excepted). Relative to joint bf16: \(79\% \times\) (B1/B2 2.0×, B0 near 1×) ≈ **43%**. That product is the locked wall-clock **C1+NVFP4 = 571**.
9. **This PR does not implement TE / sm_120 kernels.** With no Blackwell: the same module policy falls back to an FP8 placeholder, then to bf16 autocast. RTX 4080 SUPER cannot run NVFP4. Official TE fused NVFP4 (RHT/SR) may be unavailable on sm_120; the published recipe still writes the full training recipe, and the kernel is wired later on the target card. **Ampere `Nvfp4Linear` emulation is a speed trap** (no FP4 tensor cores; quant/dequant is slower than bf16). **B200 uses Transformer Engine NVFP4 FPROP on the frozen encoder** (`TeNvfp4Linear` + default `NVFP4BlockScaling`).

Claim ledger: mid-tier 22 + curriculum 12 + FP8 fallback 13 + NVFP4 16, `--verify` **63/63** pass.

---

## 1. Theorem G still holds: dtype does not change 6NT

Same Kaplan ledger as [`FP8_THEORY.md`](FP8_THEORY.md) §1:

\[
F = 6\,N_{\mathrm{act}}T,\qquad
H_{\text{wall}} = \frac{F}{\eta_{\mathrm{bf16}}}\, /\, S,\qquad
\eta_{\mathrm{bf16}} = 4.0\times 10^{14}\ \text{FLOPS (40% MFU)}.
\]

NVFP4 only enters \(S\). C1 \(F_{\mathrm{C}}\) is 21% below joint, so the bf16 wall-clock is still 1,046.

RTX PRO 6000 Blackwell Server Edition (NVIDIA launch page):

\[
\text{BF16 } 1\ \text{PFLOP},\quad
\text{FP8 } 2\ \text{PFLOPS},\quad
\text{FP4 } 4\ \text{PFLOPS},\quad
96\,\text{GB GDDR7},\ 1597\,\text{GB/s}.
\]

Relative to this card's BF16, FP4 peak is 4×. Relative to this card's FP8, peak is 2×. Published \(S\) does not use peak. NVIDIA MaxText (2026-06-08) on GB200/GB300 with Llama 3 8B / 3.1 405B: **NVFP4 vs FP8 = 1.31×–1.73×**, loss delta +0.026 nats. For 12B small experts take the low end, times the already published FP8 1.5×:

\[
S_{\mathrm{pub}} = 1.5 \times \frac{2.0}{1.5} = 2.0
\quad\text{(vs bf16; vs FP8 = } 2.0/1.5 \approx 1.33\in[1.31,1.73]\text{)}.
\]

H100 SXM BF16 peak \(9.89\times 10^{14}\) and the 6000's \(1.00\times 10^{15}\) differ by <2%, so **H100-h ≈ 6000-h** (same 40% MFU). 6000D does not get a separate wall-clock table.

---

## 2. What must stay bf16/fp32, and what must be NVFP4

Student = tensors that receive gradients. Master storage is always bf16, independent of compute dtype.

### 2.1 Must stay high precision (not NVFP4)

`KEEP_HIGH_PREC` = embed / rms_norm / qk_norm / router / gate / indexer / attn_softmax. `lm_head` is not in this set.

| Module | Why it must stay high precision |
| --- | --- |
| Input \(E_{\mathrm{in}}\) (`embed`) | Table lookup is not a GEMM. Theorem E: when B0/B1 freeze the encoder, \(X^0\) is the frozen stack's input distribution, and quantization noise would pour into 16 frozen MiniCPM5 residual layers. |
| RMSNorm / QK-Norm (`rms_norm`, `qk_norm`) | Small-vector variance; fp32 then cast back. Not a GEMM. |
| attn softmax / SDPA score+context (`attn_softmax`) | Exp + normalize is stable only in fp32. NVIDIA also writes that softmax exponentially amplifies QKᵀ quantization noise; score/context are separate from the linear projections. |
| router + expert bias (`router`) | Expert choice is discrete top-\(k\); a 21-wide logit also does not align to NVFP4's 16-element microblocks. A few ULPs change routing. |
| YOCO scalar gate (`gate`) | Theorem A's bypass; B0 \(g\in[0,0.3]\) must be smooth. Not a GEMM. |
| Lightning Indexer (`indexer`) | Phase C has to match a dense attention distribution; a 4-bit scoring head would dirty the KL target. |
| B0 student | New modules are randomly initialized and the gate is ramping. NVIDIA also observed that "full NVFP4 diverges; the last few layers should stay BF16". |
| L0 / tiny | Correctness attribution. |
| Teacher MiniCPM5 | An already-trained bf16 checkpoint. |
| Adam \(m,v\) / Muon NS | fp32 optimizer state, not a network GEMM. |

Hash-MoE is `token_id → expert_id`; there is no learned router GEMM.

### 2.2 Not required to stay bf16 → published NVFP4

The old C1+FP8 recipe only cut MoE experts + frozen encoder forward, and left lm_head / attn projections on the whitelist out of conservatism, not numerical necessity. Per the published rule they come off:

| Slot | Phase | Notes |
| --- | --- | --- |
| MoE expert SwiGLU (gate/up/down) FPROP/DGRAD/WGRAD | B1/B2; frozen encoder forward from B0 | **81.9%** of counted 6NT |
| Self-attn \(W_Q,W_K,W_V,W_O\) | B1/B2; frozen encoder forward from B0 | Projections are linear GEMMs; softmax stays high precision |
| Cross-attn \(W_Q,W_O\) | B1/B2 | In B0 they **are** the student, so they stay bf16 |
| Cache \(W_K,W_V\) | B1/B2 | Same as student bf16 in B0 |
| **untied lm_head** | B1/B2 | \(d\times V=2048\times 130560\) is a large GEMM; Theorem E governs the input table, not the head |

NVIDIA MaxText: "the three GEMMs quantize only MLP to NVFP4; the attention block (QKV, O, score/context) stays high precision". This recipe agrees that **score/context must stay high precision**, and does not agree that QKV/O and lm_head must stay bf16. On divergence, first roll QKV/O back to high precision; lm_head may stay NVFP4.

---

## 3. Locked C1+NVFP4 (stacked on C1; freeze boundary unchanged)

| Phase | student | Frozen encoder GEMM | Reason |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | Tiny correctness |
| **B0** | **bf16** | **nvfp4** | Theorem A; the frozen stack is inference |
| **B1** | **nvfp4** | **nvfp4** | Decoder allowed linear GEMMs + lm_head |
| **B2** | **nvfp4** | n/a | Same GEMM set after both stacks unfreeze |
| C | bf16 | nvfp4 | Indexer is local and tensors are small |

B1+B2 = 84% of tokens / 89% of C1 FLOPs. B0 is **11.0%** of C1 FLOPs, of which frozen encoder forward is about 17% of B0; leaving the student in bf16 barely spends the NVFP4 gain (571 vs all-phase 523, a gap of ~48 H100-h).

Scaling: 16-element microblocks, 2D 16×16 weights, RHT on WGRAD, stochastic rounding on gradients. With no kernel, the same modules take the FP8 placeholder, then bf16 autocast.

96GB: Encoder offload and CPU Adam in B0/B1 become **operationally optional**, not a dtype change. A 32GB card still follows the original offload path; it cannot run NVFP4.

B200 / SM 10.0 / 10.3 runs **TE NVFP4 FPROP on the frozen encoder** (`TeNvfp4Linear` + default `NVFP4BlockScaling`). Ampere `Nvfp4Linear` emulation is a speed trap: there are no FP4 tensor cores, so the extra quant/dequant is slower than bf16. Use `--no-nvfp4` on Ampere, or the Hopper/Ada FP8 fallback in [`FP8_THEORY.md`](FP8_THEORY.md); do not bill Ampere emu hours against the 571 figure.

---

## 4. Wall-clock ledger (locked C1+NVFP4)

Let \(F_0,F_1,F_2\) be B0/B1/B2 Kaplan FLOPs, and \(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) the B0 encoder forward. **Locked**:

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
| B0 student also 2.0× | 523 | 39% |
| All-phase MoE-Amdahl 4× GEMM (2.59×) | 403 | 30% |
| Peak 4× | 261 | 20% |

Do not write 4× into the plan. If measured \(S<1.7\) vs bf16 on 12B, first roll QKV/O back to high precision; if that is still not enough, roll B1/B2 student back to FP8 or bf16. C1's 1,046 remains.

---

## 5. Orthogonal to FP8 / memory / V4 FP4 storage

- **C1+FP8 = 729** is still a recomputable fallback ledger; see [`FP8_THEORY.md`](FP8_THEORY.md). Hopper has no FP4 tensor core.
- **C1 memory leverage still holds**: B1 Adam 62%, detached activations ~38%. NVFP4 further cuts GEMM activation bandwidth; it does not replace freezing. 96GB only makes offload optional.
- **V4 FP4 expert storage**: an inference/checkpoint option. Training still needs a bf16 master and an fp32 optimizer, so it does not enter the recipe. NVFP4 is a **compute** format, not that storage path.
- **Muon**: Newton-Schulz stays fp32, orthogonal to network NVFP4 GEMMs.

Cut \(F\) first (C1), then multiply by \(S\) (NVFP4).

---

## 6. Failure modes (avoid these before implementation)

| Failure | Source | Avoid by |
| --- | --- | --- |
| Thinking NVFP4 reduces 6NT | Writing throughput as FLOPs | Theorem G |
| Publishing 4× | Copying the 6000 peak | 12B small experts; lock 2.0× vs bf16 |
| B0 student NVFP4 dirtying Theorem A | Gate neighborhood + new modules | B0 student bf16 |
| NVFP4 input table drifting the frozen encoder | Training/quantizing \(E_{\mathrm{in}}\) | Whitelist + Theorem E |
| Router NVFP4 changing top-\(k\) | Discrete choice + width 21 | Router high precision |
| Quantizing softmax to NVFP4 | Exponential amplification of QKᵀ noise | score/context fp32; quantize only linear projections |
| L0 mixed precision | Tiny cannot attribute | L0 bf16 |
| Treating NVFP4 as live on a 4080 | Ada has no FP4 tensor core | Fall back to FP8 placeholder / bf16 |
| Treating Ampere `Nvfp4Linear` emu as the published kernel | Emu is a speed trap | `--no-nvfp4` on Ampere; B200 TE FPROP on the frozen encoder |
| Using NVFP4 to replace C1 | dtype ≠ freeze | C1 first, then NVFP4 |
| Treating V4 FP4 storage as training weights | No master | Storage is a later option |
| Copying MaxText "MLP only" as a numerical must | Treating a conservative recipe as necessity | Publish QKV/O+lm_head; roll back QKV/O on divergence |

Theory **cannot** rule out: TE fused kernels unavailable on sm_120, 12B expert GEMMs measuring \(S\approx 1.4\), or a loss spike when B1 switches to NVFP4. Those are L1; on a spike, roll back that phase's student slots, and do not change the C1 freeze boundary.

---

## 7. Claim ledger

`python3 scripts/param_budget.py --verify` adds, on top of mid-tier 22 + curriculum 12 + FP8 fallback 13:

| Claim | Result |
| --- | --- |
| NVFP4 does not change Kaplan 6NT | PASS |
| RTX PRO 6000 peak 1/2/4 PFLOP | PASS |
| Conservative wall-clock 2.0× vs bf16 (not 4×) | PASS |
| Published NVFP4/FP8 ratio ∈ [1.31, 1.73] | PASS (1.33) |
| Locked wall-clock is C1+NVFP4 | PASS |
| Published C1+NVFP4 ≈571 (≤45% joint bf16) | PASS (571 / 1,325 = 43%) |
| B0 student = bf16 | PASS |
| B1/B2 student = nvfp4 | PASS |
| lm_head is an NVFP4 GEMM (not in the must-high-precision set) | PASS |
| attn softmax must stay high precision; QKV is not required bf16 | PASS |
| Must-high-precision set excludes lm_head | PASS |
| C1 bf16 hours are not rewritten by the NVFP4 policy | PASS (1,046) |
| C1+FP8 fallback still ≈729 | PASS |
| 6000 BF16 peak ≈ H100 | PASS |
| MoE-only 4× Amdahl is sensitivity only | PASS |
| Allowed GEMMs are ≥95% of 6NT | PASS (95.8%, including lm_head) |

---

## 8. Plan revisions (this PR)

1. **Phase B wall-clock is locked to C1+NVFP4: 571 H100-h.** Joint bf16 1,325 is control only; C1 bf16 1,046 is the operand ledger only; C1+FP8 729 is demoted to the Hopper/Ada fallback.
2. Precision policy: every linear GEMM that is not required to stay bf16 is NVFP4. `KEEP_HIGH_PREC` is embed / rms_norm / qk_norm / router / gate / indexer / attn_softmax. The whitelist no longer contains lm_head; lm_head is a GEMM slot.
3. Target hardware: published wall-clock is billed at RTX PRO 6000 order of magnitude; **the card that actually eats NVFP4 training kernels is B200 / SM 10.0 / 10.3** (``TeNvfp4Linear`` + default ``NVFP4BlockScaling``), including **TE NVFP4 FPROP on the frozen encoder**. sm_120 uses ``Nvfp4Linear`` emulation. **Ampere NVFP4 emulation is a speed trap** — do not treat it as the published kernel. Attention topology is unchanged. See [`B200_TRAIN.md`](B200_TRAIN.md).
4. Do not change 16/26, C1, the causal encoder, or the M2 default.

Recompute:

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --nvfp4
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
