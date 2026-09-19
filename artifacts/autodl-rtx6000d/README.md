# AutoDL RTX 6000D（Blackwell sm_120）

实例 `autodl-container-8x4c4zmh8d-e96e943a`，SSH 曾是 `connect.weste.seetacloud.com:34864`（**weste**，不是已释放的 westc）。**本机即将释放**；不要再连。下一台从 Hub [`checkpoints/b0-full/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full) 同阶段 resume B0。

发布档 B0 **已切到** `/root/autodl-tmp/venv-nightly`：torch `2.15.0.dev20260918+cu130`，驱动 595.71.05，CUDA 13.2。miniconda 里仍留着 2.8.0+cu128。

| 项 | 实测 |
| --- | --- |
| GPU | NVIDIA RTX 6000D |
| compute | 12.0（sm_120） |
| 显存 | 85651 MiB（≈83.6 GiB） |
| overlay `/` | 30G — 不要放权重 |
| `/root/autodl-tmp` | 50G xfs |
| Hub | `HF_ENDPOINT=https://hf-mirror.com` `HF_HUB_DISABLE_XET=1` |

NVFP4 配方的目标卡就是这张。Trainer 在 freeze 之后把允许的 ``nn.Linear`` 换成 ``Nvfp4Linear``（E2M1/16 仿真；有 TE 时走 ``NVFP4BlockScaling``）。注意力仍是因果 YOCO window + fp32 SDPA / qk_norm，不改拓扑。fused TE WGRAD/RHT 仍不是本仓硬依赖。

## HuggingFace 走 hf-mirror

```bash
source scripts/autodl_env.sh   # HF_ENDPOINT=https://hf-mirror.com，缓存 /root/autodl-tmp/hf
python3 scripts/download_minicpm5.py
python3 -m cat_yoko.b0 --try --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base \
  --save-dir /root/autodl-tmp/runs/b0
```

`openbmb/MiniCPM5-2B-Base` 的 `model.safetensors` 是 **5,033,557,128** 字节，sha256 `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`。
hf-mirror 的 `/resolve/` 会 302 到 `cas-bridge.xethub.hf.co`；这台机器对该 host 返回 **403**（这就是上一张 4080 卡在 20MiB 的原因）。`download_minicpm5.py` 失败后改走 ModelScope `OpenBMB/MiniCPM5-2B-Base`（同大小、同 sha）。Tokenizer 仍可从 hf-mirror git 拉（`tokenizer.json` ≈9.8MiB，不是 Xet）。

不要把 Hub 缓存在 overlay。不要把 SSH 密码或 HuggingFace deploy key 写进本目录。发布权重落到 HuggingFace [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)，不进 GitHub。

## B2 `--try`（等 GPU 空闲 + B1 overlay）

发布信封 15e9 tokens；6000D 先跑 [`scripts/run_b2_try_autodl.sh`](../../scripts/run_b2_try_autodl.sh)：`--try` 32 步、resume `/root/autodl-tmp/runs/b1`、MiniCPM5 上采样 encoder+embed、NVFP4 wrap 全部允许 GEMM、`--offload-blocks --optim-cpu --accum 1`。不要 `git fetch`，不要 50B 语料，不要 23GiB `--save-full`。指针：[`b2/`](b2/)、Hub [`checkpoints/b2/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2)。

## B0 `--try`（真实 MiniCPM5 上采样）

MiniCPM5-2B-Base sha256 `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a` 校验通过后，`python3 -m cat_yoko.b0 --try --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base` 32/32 步 exit 0。

| 项 | 值 |
| --- | --- |
| gate | 0.301 |
| last nll | 15.91（DummyStream 随机 token，不是评估） |
| peak | 24244 MiB |
| trainable | 219.21M / overlay 419MiB |
| overlay sha256 | `9012e5ac55c2f59ef7cacc34d5769444413d070116dbff0696c7b258b9aa0636` |

日志：`runs/gpu_and_b0.log`、`runs/b0/metrics.jsonl`、`gpu_smoke/tiny.json`。overlay 只进 HuggingFace，不进 GitHub。

## WESTE prep（2026-09-18 16:30Z）

hostname `autodl-container-8x4c4zmh8d-e96e943a`。GPU idle，0 MiB / 85651 MiB，cap 12.0。torch `2.8.0+cu128`。

Remote `/root/autodl-tmp/CAT-YOKO` HEAD was `39b6e2dc740bf1693beffb58ab2c5321140bdb06` (`cursor/nvfp4-c1-theory-02c6`) with a dirty NVFP4 working tree (files copied on top; not the train branch checkout). MiniCPM5-2B-Base is at `/root/autodl-tmp/hf/MiniCPM5-2B-Base` (`model.safetensors` 4.7G). Existing `/root/autodl-tmp/runs/b0` still has the 32-step overlay.

Disk at prep end: overlay `/` 1.8G/30G used (29G free); `/root/autodl-tmp` 7.4G/50G used (43G free).

`HF_ENDPOINT` and `HF_HUB_DISABLE_XET=1` are in `/root/autodl-tmp/cat-yoko-env.sh`, `/etc/profile.d/cat-yoko-hf.sh`, and `/etc/environment` (non-interactive SSH picks them up).

Transformer Engine: `transformer-engine==2.19.0` + `transformer_engine_cu12==2.19.0` installed. `import transformer_engine` works and `NVFP4BlockScaling` exists, but `import transformer_engine.pytorch` fails (`libtorch_cuda.so: undefined symbol: ncclCommWindowRegister`). Isolated `transformer-engine[pytorch]` also failed: pip tried to download torch 2.14, then `--no-build-isolation` compile died on missing `nccl_dev_cap.hpp` (not in torch 2.8). GPU test path is E2M1/16 emulation. Logs: `te_install.log`, `te_error_extract.txt`, `prep_status.txt`.

After `cursor/nvfp4-train-6000d-02c6` is on origin, overlay the tree with tar/scp (GitHub `git fetch` hangs). Published B0 is [`scripts/run_b0_full_autodl.sh`](../../scripts/run_b0_full_autodl.sh): 8e9 tokens, seq=4096, `--no-offload-encoder`, resume `/root/autodl-tmp/runs/b0` if present. tmux `b0-full`. Overlay → Hub `checkpoints/b0-full/`；日志 → [`b0-full/`](b0-full/)。不要覆盖 Hub 上 32 步 `checkpoints/b0/`。

B0 发布跑完、GPU 空闲后再跑 B1 `--try`：[`scripts/run_b1_try_autodl.sh`](../../scripts/run_b1_try_autodl.sh)（resume `runs/b0-full` 否则 `b0`，MiniCPM5 上采样，seq=64，32 步）。日志 → [`b1/`](b1/)。overlay → Hub `checkpoints/b1/`。

## PyTorch / TE nightly（NVFP4 GEMM）

torch **2.8.0+cu128** 暴露 `float4_e2m1fn_x2`，但 `copy_` 是 `NotImplemented`，cuBLAS NVFP4 GEMM 走不通。TE 2.19 的 `NVFP4BlockScaling` 在，可是 `transformer_engine.pytorch` 缺 `.so`（`ncclCommWindowRegister`）。

升级走**新 venv**，不碰正在跑的 B0 miniconda 进程：

```bash
bash scripts/upgrade_torch_te_nightly_autodl.sh   # tmux nightly-upgrade
# 探活：/root/autodl-tmp/venv-nightly/NVFP4_PROBE.json
# 同阶段 resume（先 ln 最新 trainable_step → trainable.pt）：
PY=/root/autodl-tmp/venv-nightly/bin/python bash scripts/run_b0_full_autodl.sh
```

Nightly 目标：`https://download.pytorch.org/whl/nightly/cu130` 上当天的 torch 2.15.dev + `transformer_engine[pytorch,core-cu13]`。TE 文档把 **NVFP4 训练 kernel 写成 SM100/103**；sm_120 是尽力探测（`disable_rht` / `disable_2d_quantization`）。探活失败就继续 E2M1/16 仿真，不改 C1 配方。

**已切换（2026-09-18T18:06Z）：** B0 从 step 1400 overlay 同阶段 resume，进程是 nightly Python。`transformer_engine.pytorch` 2.19 可以 import；`float4_e2m1fn_x2` 的 `copy_` 仍失败，所以 GEMM 还是仿真。探活 JSON：[`nvfp4/NVFP4_PROBE_nightly.json`](nvfp4/NVFP4_PROBE_nightly.json)。吞吐约 1350 → 1670 tok/s。

**已切换（2026-09-19T00:27Z）：** 从 step **10560** 同阶段 resume，新算子进内存图。`grouped_mm=True`，`te=True`，`te_nvfp4=False`（TE NVFP4 Linear 在 sm_120 上失败后整进程禁用），`return_logits=False`。吞吐约 **1670 → 2820 tok/s**，mem **42092 → 56090 MiB**。

**释放前快照（2026-09-19T02:42Z）：** Hub overlay step **16020**，`tokens_in_phase=65,488,896`（≈0.82%），sha256 `b5763b98…`。B0 未跑完 8e9。B200 已从该 overlay 续训；当前 Hub 钉见 [`checkpoints/b0-full/README.md`](../../checkpoints/b0-full/README.md)（step **26760**）。

## NVFP4 wrap 烟测（2026-09-18）

tiny CUDA wrap + 因果 window 通过；12B B0 `--try` 2 步 `nvfp4=True`，wrap 2815 个 Linear，peak 34442 MiB。日志在 [`nvfp4/`](nvfp4/)。**419MiB overlay 只上 HuggingFace** [`checkpoints/b0-nvfp4-try/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-nvfp4-try)，不覆盖 32 步 `checkpoints/b0/trainable.pt`。Transformer Engine 2.19 cu12 装上了，但 `transformer_engine.pytorch` 因 `ncclCommWindowRegister` 导不进，本跑走 E2M1/16 仿真。
