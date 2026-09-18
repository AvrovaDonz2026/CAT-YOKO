# HuggingFace Hub（AvrovaDonz/CAT-YOKO）

GitHub 只扛代码和 overlays；完整图、分片和以后的 NVFP4 权重走 HuggingFace LFS。

Hub 仓库：<https://huggingface.co/AvrovaDonz/CAT-YOKO>  
Git SSH：`git@hf.co:AvrovaDonz/CAT-YOKO`

推送脚本：[`scripts/push_to_hf.sh`](../scripts/push_to_hf.sh)（默认 `checkpoints/b0/trainable.pt` + 模型卡，staging `/tmp/cat-yoko-hf`）。**不要**把 GitHub 整仓推进 Hub。

## GitHub LFS vs HuggingFace LFS

| 仓 | 放什么 | 上限 |
| --- | --- | --- |
| **GitHub**（本仓） | 代码、文档、烟测 JSON、**overlays**（`trainable.pt` 等）**≤4GiB** | GitHub LFS **单文件 5GiB**。23GiB 全图 `latest.pt` **不能**进 git。分片也按 ≤4GiB 写，留余量。 |
| **HuggingFace** | 全图 / `shard-*.pt` / 未来 **NVFP4** checkpoint | Hub LFS 能吃 GitHub 吃不下的大文件。脚本对 **>5GiB** 只警告（GitHub LFS 不能收；HF 可以），**>50GiB** 直接拒绝。 |

下载 MiniCPM5 底座仍走镜像（见下）；**上传**只能打 `huggingface.co`，不要指望 `hf-mirror.com` 代发。

## Donz 加部署公钥

只贴 **公钥**（`*.pub`）。私钥不进本仓、不进 AutoDL overlay、不写进聊天记录。

1. 本机（或 `/root/autodl-tmp`，不要 overlay）生成专用 key，例如 `~/.ssh/id_ed25519_hf_cat_yoko`。
2. 把 **`.pub`** 加到 Donz 账号：<https://huggingface.co/settings/keys>（**user SSH keys**）。  
   仓库 Settings 里若有 Deploy / SSH 入口，也可以只授权 `AvrovaDonz/CAT-YOKO`。
3. 推送用 IdentityFile：`${HF_SSH_KEY:-$HOME/.ssh/id_ed25519_hf_cat_yoko}`。脚本不会打印私钥，也不会 `git add ~/.ssh/*`。

Token 上传可以，但不要把 token 写进仓库或 `scripts/autodl_env.sh`。

## 克隆

```bash
git clone git@hf.co:AvrovaDonz/CAT-YOKO
```

HTTPS 页面是 <https://huggingface.co/AvrovaDonz/CAT-YOKO>。国内拉权重仍应走镜像，不要拿这支 SSH 当下载默认。

## 推送（选中产物）

```bash
ssh -T git@hf.co
./scripts/push_to_hf.sh --dry-run
HF_SSH_KEY="${HF_SSH_KEY:-$HOME/.ssh/id_ed25519_hf_cat_yoko}" ./scripts/push_to_hf.sh
```

默认 payload：`checkpoints/b0/trainable.pt`，以及（若存在）`huggingface/README.md` → staging `README.md`、`huggingface/.gitattributes`。额外文件当参数传入。staging 默认 `/tmp/cat-yoko-hf`，与 GitHub checkout 分开；remote 名 `hf`。

## 中国 AutoDL

- **下载**：继续 `HF_ENDPOINT=https://hf-mirror.com`（`source scripts/autodl_env.sh`，缓存 `/root/autodl-tmp/hf`）。Xet/`cas-bridge.xethub.hf.co` 403 时 MiniCPM5 改 ModelScope，见 [`scripts/download_minicpm5.py`](../scripts/download_minicpm5.py)。
- **上传到 huggingface.co**：需要 SSH key（或 token）。镜像不能当上传入口。
- **不要把 key 放进 AutoDL overlay**（`/` 上那 30G）。若必须拷到机器上，放到 `/root/autodl-tmp`，用完记得权限 `600`。
- **不要重连旧 westc AutoDL**（不要再 SSH `connect.westc.seetacloud.com`；那张 4080 已释放）。
