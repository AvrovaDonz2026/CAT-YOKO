# RTX 3090 Ampere BF16 operator MFU comparison

Relative to the **71.16 TFLOPS** dense BF16 / **936 GB/s** roofline, not Kaplan 40%. `grouped_mm_available()=False` (sm_86, SM90 gate).

2026-09-19T17:28:16Z, torch 2.8 / sm_86, calibrated GEMM **72.12 TFLOPS**.

| Operator | Shape | Measured TFLOPS | Theoretical MFU | Measured/theory (roof) |
| --- | --- | ---: | ---: | ---: |
| cuBLAS GEMM | 8192³ bf16 | 72.12 | 100% | 101% (boost) |
| fused QKV | 12B shape frozen concat-once | 74.66 | 100% | ~105% |
| fused QKV | probe seq=128 | 0.58 | 84% | ~1% launch wall |
| Flash GQA | 12B seq=4096 hd=128 | 60.45 | 100% | **85%** |
| Flash GQA | probe seq=128 | 0.16 | 56% | ~0.4% launch wall |
| masked window | seq=512 cuDNN | 17.99 | 100% intensity | 25% (S×S mask, not Flash) |
| CSA union | seq=512 | 36.73 | 100% intensity | 52% |
| HCA concat | seq=512 k_len=576 | 40.34 | 100% intensity | 57% |
| MoE padded bmm | E=20 n=409 | 57.66 | 100% | **81%** |
| indexer fp32 | 12B S=4096 d=64 | 14.28 | 81% (FP32 peak) | 50% (skinny K) |
| grouped_mm | — | n/a | 0 | Ampere does not enable it |

Probe seq=128 is entirely launch-bound; approaching the roofline is not meaningful. 12B B0 with the window covering the sequence uses Flash. CSA/HCA must pay the mask on the Theorem B hole; without flash-attn / a CSA CUDA kernel they cannot reach Flash. An earlier run mistakenly enabled `enable_gqa+attn_mask`; Ampere **does not error and silently uses the math kernel**. This was changed to repeat KV. The 17:26 run in the log is the trap; 17:28 is after the fix.

Files: [`ledger.json`](ledger.json), [`ampere_mfu.log`](ampere_mfu.log). Spec: [`docs/AMPERE_OPS_MFU.md`](../../../../docs/AMPERE_OPS_MFU.md).
