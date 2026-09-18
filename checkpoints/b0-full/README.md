# B0 发布档（6000D，8e9 tokens）

权重不进 GitHub。最新 overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

这是 C1 B0 **发布信封**（`--tokens 8e9`，`seq=4096`）进行中的 trainable overlay，不是 `--try`。
不要覆盖 `checkpoints/b0/` 里那份 32 步 MiniCPM5 overlay。

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 900 |
| tokens_in_phase | 3,557,376 |
| seq | 4096 |
| sha256 | `c52d01029ec25d2f77c11c05cf815773a65cda2d14284cc4b728a5f6c82a72fc` |
| 机器 | AutoDL RTX 6000D sm_120 |
| 说明 | DummyStream；student bf16；冻结 encoder GEMM NVFP4 仿真；同阶段 resume 可接 |

启动：`scripts/run_b0_full_autodl.sh`。日志：[`artifacts/autodl-rtx6000d/b0-full/`](../../artifacts/autodl-rtx6000d/b0-full/)。
