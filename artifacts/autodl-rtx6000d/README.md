# AutoDL RTX 6000D（Blackwell sm_120）

实例 `autodl-container-8x4c4zmh8d-e96e943a`，SSH `connect.weste.seetacloud.com:34864`（**weste**，不是已释放的 westc）。
torch 2.8.0+cu128，驱动 595.71.05，CUDA 13.2。

| 项 | 实测 |
| --- | --- |
| GPU | NVIDIA RTX 6000D |
| compute | 12.0（sm_120） |
| 显存 | 85651 MiB（≈83.6 GiB） |
| overlay `/` | 30G — 不要放权重 |
| `/root/autodl-tmp` | 50G xfs |
| Hub | `HF_ENDPOINT=https://hf-mirror.com` |

NVFP4 配方的目标卡就是这张。本目录不实现 TE kernel；trainer 仍是 bf16 autocast placeholder。sm_120 只记录能力。

## HuggingFace 走 hf-mirror

```bash
source scripts/autodl_env.sh   # HF_ENDPOINT=https://hf-mirror.com，缓存 /root/autodl-tmp/hf
python3 scripts/download_minicpm5.py
python3 -m cat_yoko.b0 --try --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base \
  --save-dir /root/autodl-tmp/runs/b0
```

`openbmb/MiniCPM5-2B-Base` 的 `model.safetensors` 是 **5,033,557,128** 字节，sha256 `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`。
hf-mirror 的 `/resolve/` 会 302 到 `cas-bridge.xethub.hf.co`；这台机器对该 host 返回 **403**（这就是上一张 4080 卡在 20MiB 的原因）。`download_minicpm5.py` 失败后改走 ModelScope `OpenBMB/MiniCPM5-2B-Base`（同大小、同 sha）。Tokenizer 仍可从 hf-mirror git 拉（`tokenizer.json` ≈9.8MiB，不是 Xet）。

不要把 Hub 缓存在 overlay。不要把 SSH 密码写进本目录。
