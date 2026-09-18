# B0 发布档（6000D，8e9 tokens）

权重不进 GitHub。最新 overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

这是 C1 B0 **发布信封**（`--tokens 8e9`，`seq=4096`）进行中的 trainable overlay，不是 `--try`。
不要覆盖 `checkpoints/b0/` 里那份 32 步 MiniCPM5 overlay。

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 1480 |
| tokens_in_phase | 5,933,056 |
| seq | 4096 |
| sha256 | `33442cffc09d62a05c31c3b80e3aea454a3d102a30d71bc96a100e61f32d3c86` |
| 机器 | AutoDL RTX 6000D sm_120 |
| 运行时 | torch `2.15.0.dev20260918+cu130`（`venv-nightly`）+  batched MoE / frozen NVFP4 cache |
| 说明 | DummyStream；student bf16；冻结 encoder GEMM NVFP4 仿真；同阶段 resume 可接 |

2026-09-18T18:06Z 从 step 1400 同阶段重启：新算子 + nightly。吞吐约 1350 → **1670 tok/s**。原生 `float4` `copy_` 在 nightly 上仍失败，GEMM 继续 E2M1/16 仿真。

启动：`scripts/run_b0_full_autodl.sh`。日志：[`artifacts/autodl-rtx6000d/b0-full/`](../../artifacts/autodl-rtx6000d/b0-full/)。
