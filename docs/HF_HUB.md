# HuggingFace Hub（大权重）

发布模型仓：[AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)

GitHub **不用 LFS**。代码和文档在 GitHub；`trainable.pt` / 全图 / shard / 未来 NVFP4 checkpoint 只走 Hub。

推送：[`scripts/push_to_hf.sh`](../scripts/push_to_hf.sh)。**不要**把 GitHub 整仓推进 Hub。

```bash
ssh -T git@hf.co
./scripts/push_to_hf.sh --dry-run
./scripts/push_to_hf.sh /root/autodl-tmp/runs/b0/trainable.pt
```

Clone：`git clone git@hf.co:AvrovaDonz/CAT-YOKO`

NVFP4 wrap 的 2 步 overlay 在 Hub `checkpoints/b0-nvfp4-try/trainable.pt`（不覆盖 32 步 `checkpoints/b0/`）。发布档 B0（8e9，seq=4096）overlay 走 `checkpoints/b0-full/`（B200 快照 step **25460**，约每 10 分钟覆盖同路径）。B200 续训（默认 micro-batch=2）见 [`docs/B200_TRAIN.md`](../docs/B200_TRAIN.md)。从 Vast 拉 overlay：[`scripts/pull_vast_b0_overlay.sh`](../scripts/pull_vast_b0_overlay.sh)（SSH Host `vast-b200`，不杀训练）。B1 `--try` overlay 走 `checkpoints/b1/`（decoder + `lm_head` + 最终 RMSNorm；权重不进 GitHub）。B2 `--try` 指针：GitHub [`checkpoints/b2/README.md`](../checkpoints/b2/README.md) → Hub `checkpoints/b2/`（全模型 overlay；resume B1 + MiniCPM5 encoder/embed）。日志进 GitHub `artifacts/autodl-rtx6000d/`。

Deploy SSH key 只放本机 `~/.ssh`（`HF_SSH_KEY` 可覆盖路径），在 https://huggingface.co/settings/keys 加公钥。**不要进 git**。

中国 AutoDL **下载**仍走 `HF_ENDPOINT=https://hf-mirror.com`（`scripts/autodl_env.sh`）。**上传**走 `huggingface.co` / `hf.co`。Key 若拷到 AutoDL，只放 `/root/autodl-tmp`。不要再连已释放的 `connect.westc.seetacloud.com`。
