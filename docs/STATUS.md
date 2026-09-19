# 现状（2026-09-19）

这是 GitHub 上的**训练进度口径**。规格旋钮仍以 [`FROZEN_SPEC.md`](FROZEN_SPEC.md) 为准；权重以 HuggingFace 为准。

## 一句话

CAT-YOKO-12B 按 **C1+NVFP4** 在训 **B0**（新模块、冻 encoder、8e9 DummyStream）。**还没跑完**。Vast B200 已回收；最新 overlay 在 Hub。下一台同阶段 resume，不要从头、不要叠 32 步 `--try`。

## 进度

| 项 | 值 |
| --- | --- |
| 阶段 | C1 **B0**（encoder 冻结，只训新模块 ≈219.21M / 132 张量） |
| 信封 | 8e9 tokens，`seq=4096`，DummyStream（未拉 50B Ultra-FineWeb） |
| Hub overlay | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856（≈1.63% of 8e9） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 机器（已回收） | Vast NVIDIA B200 SM 10.0；2026-09-19T06:44Z 拷盘，随后 SSH 拒绝 |
| 运行时 | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 cubin |
| 吞吐 | **micro-batch=2**，~15.6–15.7k tok/s，8192 tok/step，Trainer ~138GiB |
| 精度 | student / 冻 decoder **bf16**；冻 encoder GEMM **NVFP4 FPROP** |
| 许可 | Apache-2.0（代码、派生权重、MiniCPM5 底座） |
| B1 / B2 | 未开。等 B0 信封或空闲 GPU 再 `--try` |

## 机器沿革

| 机器 | 角色 | 停在 |
| --- | --- | --- |
| RTX 4080 SUPER | 图 / `--try` 烟测 | 已释放；日志 [`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md) |
| RTX 6000D sm_120 | 发布档 B0 开跑（NVFP4 **仿真**） | step **16020**，`tokens_in_phase=65,488,896`；日志 [`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md) |
| Vast B200 SM 10.0 | 发布档 B0 续训（硬件 NVFP4 FPROP） | step **26940**；日志 [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |

同阶段 resume：`tokens_in_phase` 按 8192/step 接着加。step **22100** 时是 90,392,576。

## 下一台怎么接

权重不在 GitHub。先 Hub overlay，再 MiniCPM5 上采样：

```bash
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh          # SM100；默认 micro-batch=2
# MICRO_BATCH=1 bash scripts/run_b0_full_b200.sh
```

细节：[`B200_TRAIN.md`](B200_TRAIN.md)、[`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md)、[`HF_HUB.md`](HF_HUB.md)。

**不要**

- resume Hub `checkpoints/b0/` 那份 32 步 `--try`
- `--save-full` / 23GiB `latest.pt`（尤其 32GiB 容器盘）
- 为 B1/B2 `--try` 在 B0 还占 GPU 时抢卡
- 实现 Megatron EP/TP 循环、CSA、Phase C indexer
- 把 50B Ultra-FineWeb 拉进仓库或小盘
- 再连已释放的 AutoDL `westc` / `weste`
- 把 SSH 密码、deploy key 写进 git

## 产物放哪

| 东西 | 去向 |
| --- | --- |
| 代码、理论、指针、日志 | GitHub 本仓（**不用 LFS**） |
| `trainable.pt` / 全图 / shard | [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO) |
| 模型卡源 | [`huggingface/README.md`](../huggingface/README.md) → Hub 根 `README.md` |

`scripts/push_to_hf.sh` 会带上根卡片、`checkpoints/b0-full/README.md` 和 `LICENSE`。
