# B200 开训

目标卡 **NVIDIA B200 / SM 10.0**（同族 B100/B300 为 SM 10.3）。Transformer Engine：NVFP4 **训练** kernel 只在 SM 10.0 / 10.3。默认 `NVFP4BlockScaling()` = 权重 2D scaling + WGRAD 上 RHT + 梯度随机舍入。不要把 6000D 的 `disable_rht` / `disable_2d` 套到这张卡上。

sm_120（RTX PRO 6000D）走 `Nvfp4Linear` E2M1/16 仿真，没有 SM100 TMEM/UMMA 训练 kernel。

**上次 Vast B200 已回收**（2026-09-19，SSH 拒绝）。进度见 [`STATUS.md`](STATUS.md)。下一台从 Hub overlay 接，不要假设旧 SSH Host 还在。

## 从 Hub overlay 接 B0

6000D 停在 step **16020**。B200 释放前钉：

| 项 | 值 |
| --- | --- |
| Hub | https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full |
| step | **26940** |
| tokens_in_phase | 130,041,856（8e9 的 ≈1.63%） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| micro-batch | 2（`MICRO_BATCH=1` 回退） |
| 日志 | [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |

文件夹说明与 GitHub [`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md) 同步。

```bash
bash scripts/run_b200.sh
# 或分步：
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

启动参数：**seq=4096**，`--micro-batch 2`，`--no-offload-encoder`，`--no-grad-ckpt`，`--no-save-full`，DummyStream，不拉 50B Ultra-FineWeb。

Trainer：MiniCPM5 上采样 → overlay `trainable.pt` → 恢复 `step` / `tokens_in_phase` / stream / RNG。同阶段不要 resume `checkpoints/b0/` 的 32 步 `--try`。

## 上次实测（B200，已回收）

nvcc 12.9 cubin，**micro-batch=2**：中位约 **15.7k tok/s**（~520ms/step，8192 tok/step），Trainer **~138GiB**，nvidia-smi ~141/183GiB。相对 micro-batch=1 的 12.4k tok/s / 96GiB，吞吐 **+27%**（步时约 1.57×，不是 2×）。step **22100** 时 `tokens_in_phase=90,392,576`，之后每步 +8192。

Decoder 按 C1 仍是 bf16；encoder NVFP4 FPROP。相对 B200 BF16 2.25 PFLOPS，Kaplan B0 账大约 **14.5% MFU**：Encoder 前向只占 B0 FLOPs 的 **17%**。不要把冻结 Decoder 也套 NVFP4——激活会量化进 bf16 student，动定理 A。

热路径：`collapse_doc_ids`、向量化 `pad_packed_counts`、`repeat_interleave(..., output_size=)`、CE chunk 缓存、一步一次 D2H 的 nll/aux、CUDA SDPA 优先 Flash/cuDNN（`sdpa_kernel` 传 **list**）、冻结 MoE 不算 z-loss。

`CAT_YOKO_TE_NVFP4=0` 强制仿真。`FORCE_SM120=1` 才允许把 B200 启动脚本跑在 sm_120 上。

## 算子

| 槽 | B200（SM100） | 6000D（sm_120） |
| --- | --- | --- |
| attn QKV/O、cross Q/O、cache KV、lm_head、shared SwiGLU | `TeNvfp4Linear`：wrap 时 `copy_` 一次进 `te.Linear`，`te.autocast(recipe=NVFP4BlockScaling())` | `Nvfp4Linear` 仿真 |
| fused QKV / gate+up / cache KV | 冻住：一次 fused `te.Linear`。可训练：顺序 native TE；`cache_k`/`cache_v` 一次 cat GEMM | 仿真 fused cat |
| MoE routed experts | `te.GroupedLinear`；SM100 默认 RHT 把 expert token 数补到 **64**。冻住 gate+up 合成一次 GroupedLinear，且 `weight{i}.requires_grad=False`。禁止 stack TE master 进 bf16 `grouped_mm` | permute + `grouped_mm` 仿真 |
| router / embed / RMSNorm / qk_norm / SDPA | 高精度 | 同左 |
| 注意力拓扑 | 因果 YOCO window + GQA 16/2/128；不实现 CSA | 同左 |

leading dim 不是 16 的倍数时 **pad 零行再切回**，仍走 SM100 kernel。单次 TE 异常只记那一发 shape，不关整进程 NVFP4。B0/B1 `detach_cache` 时 encoder 整段 `torch.no_grad()`，冻住的 `TeNvfp4Linear` 不建 WGRAD 图。

Master 仍是 bf16 Parameter。`state_dict` 键仍是 `q_proj.weight`。

实测：torch 2.11.0+cu128，源码编 TE 2.19 `@stable` SM100。**FPROP** 可用。**WGRAD** 在 cuBLAS 12.8.4.1 上 `CUBLAS_STATUS_NOT_SUPPORTED`。B0 冻 encoder 只需要 FPROP。

## TE 安装

PyPI `transformer-engine-torch` 预编译 `.so` 在 torch 2.11 上会 `undefined symbol: CUDAErrorLogCapture`。必须 `scripts/build_te_from_source.sh`（`--no-build-isolation --no-deps`，`NVTE_CUDA_ARCHS=100`）。

CUTLASS SM100A `stg.256` 需要 **nvcc ≥ 12.9**。CUDA 12.8 编出来的 core 会在 B0 尺寸（4096×2048）quantize 时把 GPU 打挂。装 `cuda-nvcc-12-9 libcublas-dev-12-9 cuda-nvtx-12-9`。`MAX_JOBS` 默认 `nproc`。`te_linear_ready` 用 **4096×2048**。CUDA 12.9 `lib64` 进 `LD_LIBRARY_PATH`。源码安装后先改 dist-info，再从 `/tmp` import。

## B1 / B2

等 B0 信封走完，或另有空闲 GPU 做 `--try`（不要和正在占卡的 B0 抢）：

```bash
bash scripts/run_b1_try_b200.sh
bash scripts/run_b1_full_b200.sh
bash scripts/run_b2_try_b200.sh
bash scripts/run_b2_full_b200.sh
```

B200 上默认 **不** CPU offload Encoder / 逐层 offload / CPU Adam。

## 磁盘

Vast 基础镜像容器盘经常只有 **32GiB overlay**。`/workspace` 不一定是持久卷。stop/start 保容器盘；recycle/destroy 会清。不要写 23GiB `latest.pt`。MiniCPM5 ≈5GiB + torch/TE + overlay 必须挤进这 32GiB。

不要把 SSH 密码、deploy key 写进仓库。
