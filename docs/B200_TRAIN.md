# B200 开训

目标卡 **NVIDIA B200 / SM 10.0**（同族 B100/B300 为 SM 10.3）。Transformer Engine 文档：NVFP4 **训练** kernel 只在 SM 10.0 / 10.3。默认配方 `NVFP4BlockScaling()` = 权重 2D scaling + WGRAD 上 RHT + 梯度随机舍入。不要把 6000D 的 `disable_rht` / `disable_2d` 套到这张卡上。

sm_120（RTX PRO 6000D）继续走 `Nvfp4Linear` E2M1/16 仿真。那张卡没有 SM100 的 TMEM/UMMA 训练 kernel。

## 算子

| 槽 | B200（SM100） | 6000D（sm_120） |
| --- | --- | --- |
| attn QKV/O、cross Q/O、cache KV、lm_head、shared SwiGLU | `TeNvfp4Linear`：wrap 时 `copy_` 一次进 `te.Linear`，`te.autocast(recipe=NVFP4BlockScaling())` | `Nvfp4Linear` 仿真 |
| MoE routed experts | `te.GroupedLinear`（有则用之），否则 serial `te.Linear`。禁止把 TE 权重 stack 进 bf16 `grouped_mm` | permute + `grouped_mm` 仿真 |
| fused QKV / gate+up | 顺序 native TE GEMM | 仿真 fused cat |
| router / embed / RMSNorm / qk_norm / SDPA | 高精度，不变 | 同左 |
| 注意力拓扑 | 因果 YOCO window + GQA 16/2/128；不实现 CSA | 同左 |

单次非法 shape（例如 leading dim 不是 16 的倍数）只跳过那一发，**不会**把 SM100 TE 整进程关掉。

Master 仍是 bf16 Parameter。`state_dict` 键仍是 `q_proj.weight`。Adam 打 TE Parameter（与 `nn.Linear` 别名）。

## 从 Hub overlay 接 B0

上一台 6000D 停在 step **16020**，`tokens_in_phase=65,488,896`（8e9 的 ≈0.82%）。权重在 https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full 。

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
