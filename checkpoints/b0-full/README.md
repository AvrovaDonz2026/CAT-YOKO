# B0 发布档（B200，8e9 tokens）

权重不进 GitHub。最新 overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

这是 C1 B0 **发布信封**（`--tokens 8e9`，`seq=4096`）**进行中**的 trainable overlay，不是 `--try`，也还没跑完 8e9。
不要覆盖 `checkpoints/b0/` 里那份 32 步 MiniCPM5 overlay。

同一份说明会推到 Hub 文件夹：https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 26940 |
| tokens_in_phase | 130,041,856（信封 8e9 的 ≈1.63%） |
| seq | 4096 |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 可训练张量 | 132（无 Adam） |
| 机器 | Vast NVIDIA B200 SM 10.0 |
| 运行时 | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 cubin |
| 吞吐 / 显存 | ~15.7k tok/s，Trainer 137967 MiB（micro-batch=2） |
| 说明 | DummyStream；student bf16；冻结 encoder GEMM 硬件 NVFP4 FPROP；同阶段 resume 可接。不是终局。`MICRO_BATCH=1` 可回退。训练进行中约每 10 分钟覆盖 Hub 同路径。 |

上一份 6000D 快照是 step **16020**（`b5763b98…`）。本文件覆盖 Hub 同路径。

同阶段 B0，**不要** resume `checkpoints/b0/` 那份 32 步 `--try`。

```bash
bash scripts/run_b200.sh
# 或
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

Trainer：MiniCPM5 上采样 → overlay `trainable.pt` → 恢复 `step` / `tokens_in_phase` / stream / RNG。
