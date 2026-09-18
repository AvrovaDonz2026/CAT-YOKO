# B1 overlay（Hub 指针）

权重**不进 GitHub**。文件在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b1

这是 C1 B1 的 **`trainable.pt` overlay**（decoder + untied `lm_head` + 最终 RMSNorm），不是 27e9 token 发布信封。

恢复：

```bash
# MiniCPM5 上采样（冻结 encoder + embed）+ B0 overlay + 本目录 B1 overlay
python3 -m cat_yoko.b1 --try \
  --resume /root/autodl-tmp/runs/b0-full \
  --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base \
  --save-dir /root/autodl-tmp/runs/b1
```

Trainer 先 MiniCPM5 上采样，再 `load_trainable_state` 叠 B0 的 cache / cross-attn；本阶段再训 decoder 栈。不要覆盖 Hub 上的 `checkpoints/b0/` 或 `checkpoints/b0-full/`。12B 全图 `latest.pt`（≈23GiB）不写 GitHub。

启动：[`scripts/run_b1_try_autodl.sh`](../../scripts/run_b1_try_autodl.sh)（`--try`：32 步、seq=64）。日志：[`artifacts/autodl-rtx6000d/b1/`](../../artifacts/autodl-rtx6000d/b1/)。
