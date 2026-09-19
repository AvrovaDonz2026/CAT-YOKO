# B0 发布档（6000D，8e9 tokens）

权重不进 GitHub。最新 overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

这是 C1 B0 **发布信封**（`--tokens 8e9`，`seq=4096`）**进行中**的 trainable overlay，不是 `--try`，也还没跑完 8e9。
不要覆盖 `checkpoints/b0/` 里那份 32 步 MiniCPM5 overlay。

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 10680 |
| tokens_in_phase | 43,616,256（信封 8e9 的 ≈0.55%） |
| seq | 4096 |
| sha256 | `26c6c84193482fc273fb330b3d49c4af1f49579bc8b11ba6b64317749529ae07` |
| 机器 | AutoDL RTX 6000D sm_120 |
| 运行时 | torch `2.15.0.dev20260918+cu130`（`venv-nightly`）+ grouped MoE / fused QKV / chunked CE |
| 说明 | DummyStream；student bf16；冻结 encoder GEMM NVFP4 仿真；同阶段 resume 可接。实例随时可能没，这是落盘快照，不是终局。 |

2026-09-19T00:27Z 从 step **10560** 同阶段重启：jagged `grouped_mm` MoE + 融合 QKV + 分块 lm_head+CE。吞吐约 **1670 → 2820 tok/s**，mem **42092 → 56090 MiB**。`grouped_mm=True`，`te=True`，`te_nvfp4=False`（sm_120 上 TE NVFP4 Linear 仍失败，整进程禁用回 E2M1/16 仿真），`return_logits=False`。

启动：`scripts/run_b0_full_autodl.sh`。日志：[`artifacts/autodl-rtx6000d/b0-full/`](../../artifacts/autodl-rtx6000d/b0-full/)。
