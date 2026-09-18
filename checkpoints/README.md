# Phase B checkpoints (Git LFS)

GitHub LFS 单文件上限 **5GiB**。12B 全图 `latest.pt` ≈ **23GiB**，不能进 git。

## HuggingFace（大权重）

GitHub LFS 单文件上限 **5GiB**；23GiB `latest.pt` 不进 git。以后的全图 overlay / 分片发到 [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)。本仓库只保留 LFS 能吃的 B0 `trainable.pt`。

发布产物：

| 文件 | 内容 | 大约体积 |
| --- | --- | --- |
| `b0/trainable.pt` | C1 B0 新模块（cross-attn / \(W_K,W_V\) / `ln_cross`） | 419MiB（`--try` 32 步；8B token 包络同结构约 0.44GiB） |
| `b1/` 同结构 | decoder + `lm_head` + 最终 RMSNorm | 需分片 |
| `b2/` 同结构 | 全模型 | 需 `shard-*.pt` ≤4GiB |

恢复 B0：

```bash
python3 -m cat_yoko.b1 --resume checkpoints/b0 --upcycle-hf openbmb/MiniCPM5-2B-Base
```

Trainer 会 MiniCPM5 上采样后再 overlay `trainable.pt`。

仓库里的 `checkpoints/b0/trainable.pt` 是 AutoDL RTX 4080 SUPER 上 `python3 -m cat_yoko.b0 --try` 的产物（32 步、seq=64、`--dummy-upcycle`；hf-mirror 拉 MiniCPM5-2B-Base 卡在 20MiB）。**不是** 8B token 包络。RTX 6000D 上真实 MiniCPM5 上采样的同结构 overlay 发到 [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)，不替换这份 dummy 对照。B1 接手仍走 MiniCPM5 上采样 + overlay。训练日志：`train.log`；GPU 烟测 JSON：[`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md)。

32GB 试跑（不能跑完 8B token）：

```bash
python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
python3 -m cat_yoko.b1 --try --resume checkpoints/b0 --save-dir checkpoints/b1
python3 -m cat_yoko.b2 --try --resume checkpoints/b1 --save-dir checkpoints/b2
```
