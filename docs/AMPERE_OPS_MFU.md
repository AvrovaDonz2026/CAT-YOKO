# Ampere BF16 operator theoretical MFU (RTX 3090)

The 3090 **dense BF16 tensor** peak is **71.16 TFLOPS** (no 2:4 sparsity; large GEMMs measure ~72.3). HBM 936 GB/s, ridge ≈ **76 FLOP/byte**. An operator's **theoretical MFU** is the roofline:

\[
\mathrm{MFU}_{\mathrm{theo}} = \min\bigl(1,\; I / r\bigr),\quad I=\mathrm{FLOPs}/\mathrm{bytes},\quad r=\mathrm{peak}/\mathrm{BW}.
\]

Approaching theoretical MFU means approaching this roofline, not forcing 40% Kaplan or 100% peak on a seq=128 probe graph. Flash at seq=4096, hd=128 has IO-aware intensity \(I \approx S/4 = 1024\), already compute-bound, theory ~100%; at seq=128, \(I=32\), theory ~42%, then kernel launch knocks it to the 1% range.

`torch._grouped_mm` **only runs on SM90+**. On Ampere the API exists and then raises `RuntimeError`; a per-layer try/except used to cost 2–3× more than padded bmm. `grouped_mm_available()` now gates on compute capability, so 3090 uses bmm.

FlexAttention sliding window is about 1.7 ms on this torch 2.8 / sm_86 stack versus SDPA 0.02 ms: a speed trap. CSA/HCA do not switch to it. Not a CSA CUDA kernel.

## Per-block (theory)

Roofline from `python3 -m cat_yoko.ampere_mfu` on CPU (relative to 71.16 TFLOPS; indexer relative to 35.58 TFLOPS FP32/TF32):

| Operator | Phase | bf16-probe theoretical MFU | 12B-shape theoretical MFU | Tuning (approach the roofline) |
| --- | --- | ---: | ---: | --- |
| fused QKV | A–E | 84.2% | 100% compute-bound | concat frozen weights once; no per-step `torch.cat` |
| dense Flash GQA | YOCO cross; window covers seq | 56.1% (then launch knocks it to ~1%) | 100%; measured ~87% of peak | Flash + `enable_gqa`, do not repeat KV |
| masked window | encoder `n_win<seq` | 37.4% (pays S×S mask) | B0 `n_win=8192≥seq` uses Flash; this row is the C/probe hole | seq&lt;320 uses Efficient only; longer uses cuDNN only; cache the static window mask |
| CSA union | C-topk | 74.8% | 100% intensity, but the fused mask kernel cannot reach Flash | same; **do not** `enable_gqa+attn_mask` (Ampere silently drops to the math kernel); repeat KV then Efficient/cuDNN |
| HCA concat | C-hca+ | 78.7% | same, longer k | cache static slot bias; do not move CSA/HCA onto FlexAttention |
| MoE bmm | B2+ | 13.0% (12 tokens/expert, bandwidth-bound) | 100% intensity; uniform experts measure ~84% of peak | forbid grouped_mm on Ampere; cache frozen expert `gate‖up` |
| indexer fp32 | C-index | 29.7% (FP32 peak) | 81.3% | scores stay fp32 (KEEP_HIGH_PREC); TF32 `high`; includes Q/K projections |

12B B0 training at `seq=4096`, `n_win=8192`: the encoder window covers the sequence, so it uses dense Flash, not the masked row. The probe sets `n_win<seq` on purpose so Theorem B's compression hole is visible.

## Measured (3090, 2026-09-19T17:28Z)

Calibrated GEMM **72.12 TFLOPS**. `grouped_mm=False`.

| Operator | Measured | vs roofline |
| --- | --- | --- |
| fused QKV 12B shape | 74.66 T | ~105% (boost; frozen concat-once) |
| Flash GQA seq=4096 | 60.45 T | **85%** |
| MoE uniform bmm E=20 n=409 | 57.66 T | **81%** |
| CSA union seq=512 | 36.73 T | 52% (mask kernel, not Flash) |
| HCA concat seq=512 | 40.34 T | 57% |
| masked window seq=512 | 17.99 T | 25% |
| indexer fp32 12B shape | 14.28 T | 50% of FP32 roofline (d_idx=64 skinny K) |
| all bf16-probe seq=128 | &lt;1 T | launch-bound; even 30–84% theory is unreachable |

Ledger: [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/).

Earlier on the same GPU, before kernel tuning: large GEMM 72.3T; Flash s=4096 62.9T; fused QKV slower with per-step cat; MoE 2–3× more expensive if it took the grouped_mm try/except path. FlexAttention sliding window ~1.7 ms vs SDPA 0.02 ms; unused.

## How to run

```bash
python3 -m cat_yoko.ampere_mfu
bash scripts/run_ampere_mfu.sh
# OUT=/root/autodl-tmp/bf16-verify/mfu/ledger.json
```

CPU prints the theory table only. Do not pull Ultra-FineWeb. Do not write a full-graph checkpoint. Do not `--save-full`.
