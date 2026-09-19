# B2 overlay（HuggingFace 指针）

权重**不进 GitHub**。文件在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2

这是 C1 **B2** 的落点（发布信封 `--tokens 15e9`，`seq=4096`；6000D 烟测用 `--try` 32 步、`seq=64`）。
不要覆盖 `checkpoints/b0/` / `checkpoints/b1/`。

| 项 | 值 |
| --- | --- |
| 发布信封 | 15e9 tokens，student **nvfp4**，全模型可训练 |
| `detach` | `False`（gate 恒 1.0） |
| 显存 | `--offload-blocks` + `--optim-cpu`，**`accum=1`**（逐层 Adam 不能累积） |
| `--try` 接手 | MiniCPM5 上采样 encoder+embed，再 overlay B1 `trainable.pt` |
| 产物 | Hub overlay / `shard-*.pt`；**不要** 23GiB `latest.pt` / `--save-full` |

启动（GPU 空闲且 B1 overlay 已在 `/root/autodl-tmp/runs/b1`）：

```bash
bash scripts/run_b2_try_autodl.sh
```

日志：[`artifacts/autodl-rtx6000d/b2/`](../../artifacts/autodl-rtx6000d/b2/)。
