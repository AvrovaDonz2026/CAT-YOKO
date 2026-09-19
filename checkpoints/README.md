# Phase B checkpoints

权重**不进 GitHub**。overlay / 全图 / shard 发 [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)。本目录只留日志和 meta。

| 文件 | 内容 | 大约体积 | 去向 |
| --- | --- | --- | --- |
| `b0/trainable.pt` | C1 B0 新模块（cross-attn / \(W_K,W_V\) / `ln_cross`） | 419MiB（`--try` 32 步） | Hub |
| [`b1/`](b1/README.md) | decoder + `lm_head` + 最终 RMSNorm overlay | `--try` 后再上 Hub | Hub `checkpoints/b1/` |
| [`b2/`](b2/README.md) | 全模型 overlay（解冻后 ≈ 全图） | `--try` 后再上 Hub | Hub [`checkpoints/b2/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2) |

12B 全图 `latest.pt` ≈ 23GiB，只写大盘（如 `/root/autodl-tmp`），不要写进 git。

Hub 上 `checkpoints/b0/` 是 32 步 `--try`（gate 0.301），**不是** 8B 包络。发布信封进行中的 overlay 在 [`b0-full/`](b0-full/README.md)（B200 step 18340，≈0.94% of 8e9）。

恢复 B0：

```bash
# 从 Hub 拉 overlay 再接 B1
git clone git@hf.co:AvrovaDonz/CAT-YOKO /tmp/cat-yoko-hf
python3 -m cat_yoko.b1 --resume /tmp/cat-yoko-hf/checkpoints/b0 \
  --upcycle-hf openbmb/MiniCPM5-2B-Base
```

Trainer 会 MiniCPM5 上采样后再 overlay `trainable.pt`。B1 接手：同一套 MiniCPM5 + B0 overlay，再训 decoder。B2 接手：MiniCPM5 填 encoder+embed，再叠 B1 overlay。训练日志：`train.log`；GPU 烟测 JSON：[`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md)、[`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md)。B1 Hub 指针：[`b1/README.md`](b1/README.md)。B2：[`b2/README.md`](b2/README.md)。

32GB 试跑（不能跑完 8B token；产物留在本地 / Hub，不进本仓）：

```bash
python3 -m cat_yoko.b0 --try --save-dir /root/autodl-tmp/runs/b0
python3 -m cat_yoko.b1 --try --resume /root/autodl-tmp/runs/b0 --save-dir /root/autodl-tmp/runs/b1
python3 -m cat_yoko.b2 --try --resume /root/autodl-tmp/runs/b1 --save-dir /root/autodl-tmp/runs/b2
```
