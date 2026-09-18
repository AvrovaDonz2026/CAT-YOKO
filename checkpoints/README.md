# Phase B checkpoints (Git LFS)

GitHub LFS 单文件上限 **5GiB**。12B 全图 `latest.pt` ≈ **23GiB**，不能进 git。

发布产物：

| 文件 | 内容 | 大约体积 |
| --- | --- | --- |
| `b0/trainable.pt` | C1 B0 新模块（cross-attn / \(W_K,W_V\) / `ln_cross`） | ~0.44GiB bf16 |
| `b1/` 同结构 | decoder + `lm_head` + 最终 RMSNorm | 需分片 |
| `b2/` 同结构 | 全模型 | 需 `shard-*.pt` ≤4GiB |

恢复 B0：

```bash
python3 -m cat_yoko.b1 --resume checkpoints/b0 --upcycle-hf openbmb/MiniCPM5-2B-Base
```

Trainer 会 MiniCPM5 上采样后再 overlay `trainable.pt`。

32GB 试跑（不能跑完 8B token）：

```bash
python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
python3 -m cat_yoko.b1 --try --resume checkpoints/b0 --save-dir checkpoints/b1
python3 -m cat_yoko.b2 --try --resume checkpoints/b1 --save-dir checkpoints/b2
```
