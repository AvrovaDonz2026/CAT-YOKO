# RTX 3090 Ampere BF16 算子 MFU 对拍

相对 **71.16 TFLOPS** dense BF16 / **936 GB/s** 的 roofline，不是 Kaplan 40%。`grouped_mm_available()=False`（sm_86，SM90 门）。

2026-09-19T17:28:16Z，torch 2.8 / sm_86，校准 GEMM **72.12 TFLOPS**。

| 算子 | 形状 | 实测 TFLOPS | 理论 MFU | 实测/理论（roof） |
| --- | --- | ---: | ---: | ---: |
| cuBLAS GEMM | 8192³ bf16 | 72.12 | 100% | 101%（boost） |
| fused QKV | 12B 形冻结 concat-once | 74.66 | 100% | ~105% |
| fused QKV | probe seq=128 | 0.58 | 84% | ~1% launch 墙 |
| Flash GQA | 12B seq=4096 hd=128 | 60.45 | 100% | **85%** |
| Flash GQA | probe seq=128 | 0.16 | 56% | ~0.4% launch 墙 |
| masked window | seq=512 cuDNN | 17.99 | 100% 强度 | 25%（S×S mask，非 Flash） |
| CSA union | seq=512 | 36.73 | 100% 强度 | 52% |
| HCA concat | seq=512 k_len=576 | 40.34 | 100% 强度 | 57% |
| MoE padded bmm | E=20 n=409 | 57.66 | 100% | **81%** |
| indexer fp32 | 12B S=4096 d=64 | 14.28 | 81%（FP32 峰值） | 50%（瘦 K） |
| grouped_mm | — | n/a | 0 | Ampere 不开 |

探针 seq=128 全部 launch 墙，逼近 roofline 没有意义。12B B0 窗盖满走 Flash。CSA/HCA 在 Theorem B 洞上必须付 mask，没有 flash-attn / 不写 CSA CUDA kernel 就上不去 Flash。第一次跑曾误开 `enable_gqa+attn_mask`，Ampere **不报错、静默 math 核**；已改成 repeat KV。日志里 17:26 那次是陷阱，17:28 是修正后。

文件：[`ledger.json`](ledger.json)、[`ampere_mfu.log`](ampere_mfu.log)。规格见 [`docs/AMPERE_OPS_MFU.md`](../../../../docs/AMPERE_OPS_MFU.md)。
