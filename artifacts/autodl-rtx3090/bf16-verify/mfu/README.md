# RTX 3090 Ampere BF16 operator MFU cross-check

Roofline versus **71.16 TFLOPS** dense BF16 / **936 GB/s**, not Kaplan 40%. `grouped_mm_available()=False` (sm_86, SM90 gate).

2026-09-19T17:28:16Z, torch 2.8 / sm_86, calibration GEMM **72.12 TFLOPS**.

| Operator | Shape | Measured TFLOPS | Theoretical MFU | Measured/theoretical (roof) |
| --- | --- | ---: | ---: | ---: |
| cuBLAS GEMM | 8192³ bf16 | 72.12 | 100% | 101% (boost) |
| fused QKV | 12B-shape frozen concat-once | 74.66 | 100% | ~105% |
| fused QKV | probe seq=128 | 0.58 | 84% | ~1% launch-bound |
| Flash GQA | 12B seq=4096 hd=128 | 60.45 | 100% | **85%** |
| Flash GQA | probe seq=128 | 0.16 | 56% | ~0.4% launch-bound |
| masked window | seq=512 cuDNN | 17.99 | 100% intensity | 25% (S×S mask, not Flash) |
| CSA union | seq=512 | 36.73 | 100% intensity | 52% |
| HCA concat | seq=512 k_len=576 | 40.34 | 100% intensity | 57% |
| MoE padded bmm | E=20 n=409 | 57.66 | 100% | **81%** |
| indexer fp32 | 12B S=4096 d=64 | 14.28 | 81% (FP32 peak) | 50% (skinny K) |
| grouped_mm | — | n/a | 0 | Ampere does not enable it |

2026-09-20T01:02Z sliding-window remeasure (calibration GEMM 72.09 T). Tiling by `n_win` and 256-wide fat tiles at seq=512 are **both slower than S×S**; they only match or slightly beat it from seq=2048. Crossover table: [`band_xover.log`](band_xover.log). The code gate is `BANDED_SEQ_MIN=2048`. B0 covering windows still use Flash.

| Operator | Shape | Measured TFLOPS | roof |
| --- | --- | ---: | ---: |
| masked window | seq=512 | 18.89 | 26.5% |
| fat tiles 256 | seq=512 n_win=32 | 2.02 | 2.8% (disabled) |
| masked window | seq=2048 | 31.78 | 44.7% |
| fat tiles 256 | seq=2048 n_win=32 | 32.50 | 45.7% |

Probe seq=128 is launch-bound across the board; chasing the roofline is meaningless there. 12B B0 covering windows use Flash. CSA/HCA on the Theorem B hole must pay the mask; without flash-attn / without writing a CSA CUDA kernel they cannot reach Flash. The first run accidentally enabled `enable_gqa+attn_mask`; Ampere **does not error and silently uses the math kernel**. That is now repeat KV. The 17:26 log is the trap; 17:28 is the fix.

Files: [`ledger.json`](ledger.json), [`ampere_mfu.log`](ampere_mfu.log), [`band_xover.log`](band_xover.log). Spec: [`docs/AMPERE_OPS_MFU.md`](../../../../docs/AMPERE_OPS_MFU.md).
