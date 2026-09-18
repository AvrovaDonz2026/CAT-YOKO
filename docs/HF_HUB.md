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

Deploy SSH key 只放本机 `~/.ssh`（`HF_SSH_KEY` 可覆盖路径），在 https://huggingface.co/settings/keys 加公钥。**不要进 git**。

中国 AutoDL **下载**仍走 `HF_ENDPOINT=https://hf-mirror.com`（`scripts/autodl_env.sh`）。**上传**走 `huggingface.co` / `hf.co`。Key 若拷到 AutoDL，只放 `/root/autodl-tmp`。不要再连已释放的 `connect.westc.seetacloud.com`。
