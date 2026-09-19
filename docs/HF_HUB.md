# HuggingFace Hub（大权重）

发布模型仓：[AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)

GitHub **不用 LFS**。代码和文档在 GitHub；`trainable.pt` / 全图 / shard / 未来 NVFP4 checkpoint 只走 Hub。代码与派生权重 **Apache-2.0**（[`LICENSE`](../LICENSE)）；`push_to_hf.sh` 会把 `LICENSE` 一并推到 Hub。

推送：[`scripts/push_to_hf.sh`](../scripts/push_to_hf.sh)。**不要**把 GitHub 整仓推进 Hub。

```bash
ssh -T git@hf.co
./scripts/push_to_hf.sh --dry-run
./scripts/push_to_hf.sh checkpoints/b0-full/trainable.pt
```

`push_to_hf.sh` 每次都会带上 Hub 根卡片 `huggingface/README.md` 和文件夹说明 `checkpoints/b0-full/README.md`。

Clone：`git clone git@hf.co:AvrovaDonz/CAT-YOKO`

## 当前 B0 overlay

| 项 | 值 |
| --- | --- |
| Hub 文件 | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| Hub 说明 | [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md) |
| step | **26940** |
| tokens_in_phase | 130,041,856（≈1.63% of 8e9） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 刷新 | 训练进行中约每 10 分钟覆盖同路径；从 Vast 拉 overlay：[`scripts/pull_vast_b0_overlay.sh`](../scripts/pull_vast_b0_overlay.sh)（SSH Host `vast-b200`，不杀训练） |

NVFP4 wrap 的 2 步 overlay 在 Hub `checkpoints/b0-nvfp4-try/trainable.pt`（不覆盖 32 步 `checkpoints/b0/`）。B200 续训（默认 micro-batch=2）见 [`docs/B200_TRAIN.md`](../docs/B200_TRAIN.md)。B1 `--try` overlay 走 `checkpoints/b1/`（decoder + `lm_head` + 最终 RMSNorm；权重不进 GitHub）。B2 `--try` 指针：GitHub [`checkpoints/b2/README.md`](../checkpoints/b2/README.md) → Hub `checkpoints/b2/`（全模型 overlay；resume B1 + MiniCPM5 encoder/embed）。日志进 GitHub `artifacts/autodl-rtx6000d/`。

Deploy SSH key 只放本机 `~/.ssh`（`HF_SSH_KEY` 可覆盖路径），在 https://huggingface.co/settings/keys 加公钥。**不要进 git**。

中国 AutoDL **下载**仍走 `HF_ENDPOINT=https://hf-mirror.com`（`scripts/autodl_env.sh`）。**上传**走 `huggingface.co` / `hf.co`。Key 若拷到 AutoDL，只放 `/root/autodl-tmp`。不要再连已释放的 `connect.westc.seetacloud.com`。
