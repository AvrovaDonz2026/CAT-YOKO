# Phase B checkpoints

权重**不进 GitHub**。overlay / 全图 / shard 发 [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)。本目录只留指针和（`--try` 的）小日志。进度：[`docs/STATUS.md`](../docs/STATUS.md)。

| 路径 | 内容 | 大约体积 | 去向 |
| --- | --- | --- | --- |
| [`b0/`](b0/) | 6000D `--try` 32 步新模块 overlay | ~419MiB | Hub `checkpoints/b0/` |
| [`b0-nvfp4-try/`](b0-nvfp4-try/README.md) | 6000D NVFP4 wrap `--try` 2 步 | ~419MiB | Hub |
| [`b0-full/`](b0-full/README.md) | **发布信封 B0**（8e9，进行中） | ~419MiB | Hub `checkpoints/b0-full/` |
| [`b1/`](b1/README.md) | decoder + `lm_head` + 最终 RMSNorm | `--try` 后再上 Hub | Hub `checkpoints/b1/` |
| [`b2/`](b2/README.md) | 全模型 overlay | `--try` 后再上 Hub | Hub `checkpoints/b2/` |

12B 全图 `latest.pt` ≈ 23GiB，只写大盘，不要写进 git。

Hub 上 `checkpoints/b0/` **不是** 8B 包络。发布档进行中的 overlay 是 [`b0-full/`](b0-full/README.md)（B200 释放前 step **26940**，≈1.63% of 8e9；micro-batch=2）。

接发布档 B0（不要用 32 步 `--try`）：

```bash
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
python3 -m cat_yoko.b0 --resume /workspace/runs/b0-full \
  --upcycle-hf openbmb/MiniCPM5-2B-Base
# 或 SM100：bash scripts/run_b0_full_b200.sh
```

Trainer 会 MiniCPM5 上采样后再 overlay `trainable.pt`。B1：同一套 MiniCPM5 + **B0-full** overlay，再训 decoder。B2：MiniCPM5 填 encoder+embed，再叠 B1 overlay。

日志：[`artifacts/vast-b200/`](../artifacts/vast-b200/README.md)、[`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md)、[`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md)。

32GB 试跑（不能跑完信封；产物留本地 / Hub）：

```bash
python3 -m cat_yoko.b0 --try --save-dir /tmp/runs/b0
python3 -m cat_yoko.b1 --try --resume /tmp/runs/b0 --save-dir /tmp/runs/b1
python3 -m cat_yoko.b2 --try --resume /tmp/runs/b1 --save-dir /tmp/runs/b2
```
