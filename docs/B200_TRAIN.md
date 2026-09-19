# B200 开训

目标卡 **NVIDIA B200 / SM 10.0**（同族 B100/B300 为 SM 10.3）。Transformer Engine 文档：NVFP4 **训练** kernel 只在 SM 10.0 / 10.3。默认配方 `NVFP4BlockScaling()` = 权重 2D scaling + WGRAD 上 RHT + 梯度随机舍入。不要把 6000D 的 `disable_rht` / `disable_2d` 套到这张卡上。

sm_120（RTX PRO 6000D）继续走 `Nvfp4Linear` E2M1/16 仿真。那张卡没有 SM100 的 TMEM/UMMA 训练 kernel。

## 算子

| 槽 | B200（SM100） | 6000D（sm_120） |
| --- | --- | --- |
| attn QKV/O、cross Q/O、cache KV、lm_head、shared SwiGLU | `TeNvfp4Linear`：wrap 时 `copy_` 一次进 `te.Linear`，`te.autocast(recipe=NVFP4BlockScaling())` | `Nvfp4Linear` 仿真 |
| fused QKV / gate+up / cache KV | 冻住：一次 fused `te.Linear`（copy-once 拼接 out 维）。可训练：顺序 native TE；`cache_k`/`cache_v` 一次 cat GEMM | 仿真 fused cat |
| MoE routed experts | `te.GroupedLinear`；SM100 默认 RHT 把 expert token 数补到 **64**（`disable_rht` / sm_120 仍是 16）。冻住 gate+up 合成一次 GroupedLinear，且 `weight{i}.requires_grad=False`。禁止 stack TE master 进 bf16 `grouped_mm` | permute + `grouped_mm` 仿真 |
| router / embed / RMSNorm / qk_norm / SDPA | 高精度，不变 | 同左 |
| 注意力拓扑 | 因果 YOCO window + GQA 16/2/128；不实现 CSA | 同左 |

leading dim 不是 16 的倍数时 **pad 零行再切回**，仍走 SM100 kernel，不掉回 STE。单次 TE 异常只记那一发 shape，**不会**把 SM100 整进程关掉。B0/B1 `detach_cache` 时 encoder 整段 `torch.no_grad()`，冻住的 `TeNvfp4Linear` 也不再给 TE 建 WGRAD 图。

Master 仍是 bf16 Parameter。`state_dict` 键仍是 `q_proj.weight`。Adam 打 TE Parameter（与 `nn.Linear` 别名）。

实测（torch 2.11.0+cu128，源码编 TE 2.19 `@stable` SM100）：**FPROP** 16×128 / 冻权重 dX 可用。**WGRAD** 在 cuBLAS 12.8.4.1 上 `CUBLAS_STATUS_NOT_SUPPORTED`。B0 冻 encoder 只需要 FPROP。Probe 把 `te_nvfp4_linear` 记成 FPROP，WGRAD 单独落 `te_nvfp4_linear_wgrad`。

PyPI `transformer-engine-torch` 预编译 `.so` 在 torch 2.11 上会 `undefined symbol: CUDAErrorLogCapture`。必须 `scripts/build_te_from_source.sh`（`--no-build-isolation --no-deps`，`NVTE_CUDA_ARCHS=100`），只留一份 SM100 `libtransformer_engine.so`，并把源码 metapackage 的 `Version` 钉成与 `transformer-engine-cu12` 一致。

CUTLASS 把 SM100A `stg.256` 编进 kernel 的条件是 **nvcc ≥ 12.9**。CUDA 12.8 编出来的 core：16×128 FPROP 能过，B0 尺寸（4096×2048）的 NVFP4 quantize 会打 `CUTE_ARCH_STORE256_SM100A_ENABLED` 然后把 GPU 打挂。Vast 上 `apt-get install cuda-nvcc-12-9 libcublas-dev-12-9 cuda-nvtx-12-9`（12.9 nvcc 默认没有 cuBLAS / NVTX）。`MAX_JOBS` 默认 `nproc`（这台 192 vCPU），不要钉死 8。`te_linear_ready` 用 **4096×2048** 探（B0 seq），不再用 16×128 / 256×2048。启动脚本和源码编译脚本都把 CUDA 12.9 `lib64` 放进 `LD_LIBRARY_PATH`。源码安装后先改 dist-info（去掉 `+git` Version、`Root-Is-Purelib: true`），再从 `/tmp` import，不要在 TE 源码目录里 import（会撞 PyPI sanity check）。

## 从 Hub overlay 接 B0

上一台 6000D 停在 step **16020**。B200 已续到 step **21500**，`tokens_in_phase=87,934,976`（8e9 的 ≈1.10%）。权重在 https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full 。

```bash
# 仓库根目录。Vast 默认 /venv/main + /workspace
bash scripts/run_b200.sh
# 或分步：
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

启动参数：**seq=4096**，`--no-offload-encoder`，`--no-grad-ckpt`（约 179GiB HBM），`--no-save-full`，DummyStream，不拉 50B Ultra-FineWeb。

当前 B0（nvcc 12.9 cubin，micro-batch=1）：约 **11.8k tok/s**（~350ms/step），HBM ~96/183GiB。Decoder 按 C1 仍是 bf16；encoder NVFP4 FPROP。Hub overlay step **21500**。热路径：`collapse_doc_ids`、向量化 `pad_packed_counts`（无每层 host list）、`repeat_interleave(..., output_size=)`、CE chunk 缓存、一步一次 D2H 的 nll/aux。不要为 B1/B2 `--try` 杀掉正在跑的 B0。

环境变量 `CAT_YOKO_TE_NVFP4=0` 可强制仿真。`FORCE_SM120=1` 才允许把 B200 启动脚本跑在 sm_120 上。

## B1 / B2

等 B0 信封走完（或 GPU 空闲要烟测）：

```bash
bash scripts/run_b1_try_b200.sh    # 32 步 seq=64
bash scripts/run_b1_full_b200.sh   # 27e9
bash scripts/run_b2_try_b200.sh
bash scripts/run_b2_full_b200.sh   # 15e9
```

B200 上默认 **不** CPU offload Encoder / 逐层 offload / CPU Adam。

## 磁盘

Vast 基础镜像容器盘经常只有 **32GiB overlay**。`/workspace` **不一定是持久卷**（`vast-capabilities` 里 `workspace_is_volume`）。stop/start 保容器盘；recycle/destroy 会清。不要写 23GiB `latest.pt`。MiniCPM5 ≈5GiB + torch/TE + overlay 必须挤进这 32GiB。

不要把 SSH 密码、deploy key 写进仓库。
