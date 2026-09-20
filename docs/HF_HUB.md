# HuggingFace Hub（大权重）

发布模型仓：[AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)

GitHub **不用 LFS**。代码、文档、日志在 GitHub；`trainable.pt` / 全图 / shard 只走 Hub。代码与派生权重 **Apache-2.0**（[`LICENSE`](../LICENSE)）。

进度口径：[`STATUS.md`](STATUS.md)。**不要**把 GitHub 整仓推进 Hub。

```bash
ssh -T git@hf.co
./scripts/push_to_hf.sh --dry-run
./scripts/push_to_hf.sh checkpoints/b0-full/trainable.pt
```

`push_to_hf.sh` 每次带上：

- 根卡片 `huggingface/README.md` → Hub `README.md`
- `checkpoints/b0-full/README.md`
- `LICENSE`

Clone：`git clone git@hf.co:AvrovaDonz/CAT-YOKO`

## 当前 B0 overlay

Vast B200 **已回收**。下表是 2026-09-19T06:44Z 释放前快照，不是终局。

| 项 | 值 |
| --- | --- |
| Hub 文件 | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| Hub 说明 | [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md) |
| step | **26940** |
| tokens_in_phase | 130,041,856（≈1.63% of 8e9） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 下一台 | 未知卡：`python3 -m cat_yoko.hw_recipe` + `bash scripts/run_b0_next.sh`。已知 SM100：[`B200_TRAIN.md`](B200_TRAIN.md)。`download_hub_overlay.py --name b0-full` |

[`scripts/pull_vast_b0_overlay.sh`](../scripts/pull_vast_b0_overlay.sh) 走 SSH Host `vast-b200`。那台已经回收，不要假设还能连。

其它 Hub 路径：

- `checkpoints/b0/` — 6000D `--try` 32 步，**不是** 8e9 信封
- `checkpoints/b0-nvfp4-try/` — 6000D NVFP4 wrap 2 步
- `checkpoints/b0-3090-bf16/` — RTX 3090 BF16 sibling 快照（step **28600** / sha256 `46ef6f99…`）。**不是**发布口径，不覆盖 `b0-full`
- `checkpoints/b1/`、`checkpoints/b2/` — 尚未上传（等 GPU）

日志：GitHub [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md)、[`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md)。

Deploy SSH key 只放本机 `~/.ssh`（`HF_SSH_KEY` 可覆盖路径），在 https://huggingface.co/settings/keys 加公钥。**不要进 git**。

中国机器 **下载**可走 `HF_ENDPOINT=https://hf-mirror.com`（`scripts/autodl_env.sh`）。**上传**走 `huggingface.co` / `hf.co`。不要再连已释放的 `connect.westc.seetacloud.com`。
