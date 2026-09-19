# B0 发布档（6000D，8e9 tokens）

权重不进 GitHub。最新 overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

这是 C1 B0 **发布信封**（`--tokens 8e9`，`seq=4096`）**进行中**的 trainable overlay，不是 `--try`，也还没跑完 8e9。
不要覆盖 `checkpoints/b0/` 里那份 32 步 MiniCPM5 overlay。

AutoDL RTX 6000D（`connect.weste.seetacloud.com`）**即将释放**。下一台从本文件接，不要假定那台盘还在。

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 16020 |
| tokens_in_phase | 65,488,896（信封 8e9 的 ≈0.82%） |
| seq | 4096 |
| sha256 | `b5763b98aafa2990d753934b181fdf5731d54ec1c25099be95cedc0dc28fb52b` |
| 可训练张量 | 132（无 Adam） |
| 机器 | AutoDL RTX 6000D sm_120（快照时） |
| 运行时 | torch `2.15.0.dev20260918+cu130` + grouped MoE / fused QKV / chunked CE |
| 吞吐 / 显存 | ~2820 tok/s，56090 MiB |
| 说明 | DummyStream；student bf16；冻结 encoder GEMM NVFP4 仿真；同阶段 resume 可接。不是终局。 |

## 下一台接训

同阶段 B0，**不要** resume `checkpoints/b0/` 那份 32 步 `--try`。

```bash
# 1) 代码：GitHub cursor/nvfp4-train-6000d-02c6（git fetch 若挂则 tar/scp overlay）
# 2) MiniCPM5-2B-Base → /root/autodl-tmp/hf/MiniCPM5-2B-Base（HF_ENDPOINT=https://hf-mirror.com 或 ModelScope）
# 3) Hub overlay：
#    git clone git@hf.co:AvrovaDonz/CAT-YOKO /tmp/cat-yoko-hf
#    mkdir -p /root/autodl-tmp/runs/b0-full
#    cp /tmp/cat-yoko-hf/checkpoints/b0-full/trainable.pt /root/autodl-tmp/runs/b0-full/
# 4) nightly venv：bash scripts/upgrade_torch_te_nightly_autodl.sh
# 5) 启动（脚本会 --resume /root/autodl-tmp/runs/b0-full）：
cd /root/autodl-tmp/CAT-YOKO
bash scripts/run_b0_full_autodl.sh
```

Trainer：MiniCPM5 上采样 → overlay `trainable.pt` → 恢复 `step` / `tokens_in_phase` / stream / RNG。不要连已释放的 `connect.westc.seetacloud.com`。

启动：`scripts/run_b0_full_autodl.sh`。日志：[`artifacts/autodl-rtx6000d/b0-full/`](../../artifacts/autodl-rtx6000d/b0-full/)。
