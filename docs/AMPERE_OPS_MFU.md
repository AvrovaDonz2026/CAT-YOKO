# Ampere BF16 算子理论 MFU（RTX 3090）

3090 的 **dense BF16 tensor** 峰值是 **71.16 TFLOPS**（无 2:4 稀疏；大 GEMM 实测约 72.3）。HBM 936 GB/s，ridge ≈ **76 FLOP/byte**。某个算子的**理论 MFU** 是 roofline：

\[
\mathrm{MFU}_{\mathrm{theo}} = \min\bigl(1,\; I / r\bigr),\quad I=\mathrm{FLOPs}/\mathrm{bytes},\quad r=\mathrm{peak}/\mathrm{BW}.
\]

逼近理论 MFU 是逼近这条 roofline，不是在 seq=128 的探针图上硬凑 40% Kaplan 或 100% 峰值。Flash 在 seq=4096、hd=128 时 IO-aware 强度 \(I \approx S/4 = 1024\)，已经算力墙，理论 ~100%；seq=128 时 \(I=32\)，理论约 42%，再被 kernel launch 打到 1% 量级。

`torch._grouped_mm` **只在 SM90+ 能跑**。Ampere 上 API 在、一调就 `RuntimeError`，以前每层 try/except 比直接 padded bmm 贵 2–3×。现在 `grouped_mm_available()` 按 compute capability 关门，3090 走 bmm。

FlexAttention 滑窗在这套 torch 2.8 / sm_86 上约 1.7 ms，对照 SDPA 0.02 ms，是速度陷阱。CSA/HCA 不换上去。不是 CSA CUDA kernel。

## Layout（激活，不是 overlay 权重）

12B `qk_norm=True`。旧路径是 fused QKV → `view+transpose` 成 `[B,H,S,D]` 再 RMSNorm / RoPE。RMSNorm 和 `torch.cat` 式 `rotate_half` 会把 Flash 要的 **BSHD 内存序**（stride `(S·H·D, D, H·D, 1)`）打成 packed BHSD，每层多一次 Q/K 拷贝；`o_proj` 前再 `contiguous().view` 又拷一次。

现在：

- qk_norm + RoPE 留在连续 `[B,S,H,D]`（最后一维是 `head_dim`）
- SDPA 只拿 `transpose` 视图，不 `.contiguous()`
- `o_proj` 用 `reshape`；Flash 若写出同一 BSHD 内存，这步是 view
- fat-tile 滑窗仍要 packed BHSD（`unfold` 沿 S），只在带状路径上 pack；B0 覆盖窗不走这条
- 冻结 MoE `bmm(x, W.transpose(1,2))` 保持 **TN**（K stride-1）。不要把 `W.T` contiguous 成 NN
- **不**把 3 个 QKV Linear 收成一个 fused Linear 模块（Hub overlay 仍是 132 张量）

3090 正在跑的 B0 进程不重测；Flash GQA seq=4096 上次 ~85% MFU 仍是对照。这是激活 layout，不改 step 26940 权重。

## 各块（理论）

CPU 上 `python3 -m cat_yoko.ampere_mfu` 打出的 roofline（相对 71.16 TFLOPS；indexer 相对 35.58 TFLOPS FP32/TF32）：

| 算子 | 阶段 | bf16-probe 理论 MFU | 12B 形理论 MFU | 调整（逼近 roofline） |
| --- | --- | ---: | ---: | --- |
| fused QKV | A–E | 84.2% | 100% 算力墙 | 冻结权重 concat 一次，不再每步 `torch.cat` |
| dense Flash GQA | YOCO cross；窗盖满 seq | 56.1%（再被 launch 打到 ~1%） | 100%；实测可到 ~87% 峰值 | Flash + `enable_gqa`，不 repeat KV；qk_norm/RoPE 在 BSHD |
| masked window | encoder `n_win<seq` | 37.4%（旧 S×S 行） | B0 `n_win=8192≥seq` 走 Flash | **fat tiles 256**（仅 `seq≥2048`）；seq=512 上 256 宽仍 0.14× S×S；CSA union 仍是 S×S |
| CSA union | C-topk | 74.8% | 100% 强度，但 fused mask 核到不了 Flash | 同上；**不要** `enable_gqa+attn_mask`（Ampere 静默掉 math 核），repeat KV 后走 Efficient/cuDNN |
| HCA concat | C-hca+ | 78.7% | 同左，k 更长 | 静态槽 bias 缓存；不把 CSA/HCA 换上 FlexAttention |
| MoE bmm | B2+ | 13.0%（每专家 12 token，带宽墙） | 100% 强度；均匀专家实测 ~84% 峰值 | Ampere 禁止 grouped_mm；冻专家缓存 `gate‖up` |
| indexer fp32 | C-index | 29.7%（FP32 峰值） | 81.3% | 分数仍 fp32（KEEP_HIGH_PREC）；TF32 `high`；含 Q/K 投影 |

12B B0 训练 `seq=4096`、`n_win=8192`：encoder 窗盖满，走 dense Flash，不是 masked 行。探针故意 `n_win<seq`，Theorem B 的压缩洞才露出来。滑窗在 `n_win<seq` 且 `seq≥2048` 时走 **256 宽 fat tiles**。按 `n_win` 切 32×64 是 launch 陷阱；256 宽在 seq=512 上仍是 0.14× S×S，seq=4096 / n_win=32 才到 2.93×。不是 FlexAttention，不是 CSA kernel。短序列仍走一张 S×S mask。CSA/HCA 的 union / concat 仍要付 S×S mask。

3090 上续训 B0：**resume Hub overlay，写到独立目录**，不要覆盖 `checkpoints/b0-full` / Hub step **26940**。Ampere 没有 FP4 tensor core，续训加 `--no-nvfp4`（C1+NVFP4 发布墙钟结论不变，只是这张卡跑 BF16）。ZeRO-3 只解决装得下。

## GPU 持续吃满（3090 ZeRO-3）

算子本身（Flash GQA / fused QKV / MoE TN bmm）已经在算力墙上。占用掉到 **0%** 来自步进空窗，不是 kernel 太慢：

1. **ZeRO 预取被 live cap 掐掉**（默认 `stage3_max_live_parameters=1e9`）——一层冻结 MoE ~0.75e9，再加 0.5e9 预取桶就超了，预取静默不跑，每步 GPU 空约 1s。现 `max_live/reuse=2e9`，预取桶 **5e8**。`persistence` 留 **1e6**：5e6 会把全部 2048×2048 Q/O 钉在卡上，3090 第一步 `persistent all_gather` 在 47.34/47.41 GiB OOM。专家 12.6e6 仍 offload。
2. **CPU Adam 太慢**——ZeRO-3 把 student 切成碎片，torch AdamW 每步 ~1s，nvidia-smi 打成 0%。走 **DeepSpeedCPUAdam**（仍是两组 decay / no-decay，不 `zero_force` 压扁）。算子要 JIT：3090 脚本把 miniconda `ninja` 放进 PATH，并用系统 `libstdc++`（GLIBCXX_3.4.30）preload。装不上就回退 torch AdamW，banner 仍是 `adam=ds-cpu`。
3. **每步 D2H**（nll / DS grad-norm / moe_utilization / `cuda.get_rng_state_all`）。`LOG_EVERY=20`；CUDA RNG 只在存盘时拍。
4. **torch inductor 32 workers** 抢 CPU。`TORCH_COMPILE_DISABLE=1`。`CUDA_DEVICE_MAX_CONNECTIONS=8`。
5. DummyStream `doc_ids` 留 host；下一批 H2D 和 backward 重叠。MoE `counts.max()` 侧流藏进共享专家。
6. **ZeRO 预热**：wrap 之后、计时循环之前跑一次合成 batch 的 fwd+bwd（**不** `engine.step()`，RNG 复原）。把 allgather 轨迹记录和 kernel JIT 从第一步 404 tok/s 挪走。中间那长段 ~647 tok/s 不是冷启动，是预取被 live cap 掐掉 + torch CPU Adam；预热补不回那一段，occupancy v2 已经拉回 ~719。

不要为了吃满去关 param offload（49GiB 卡塞不下 12B+激活）。不要 FlexAttention / CSA kernel。`SAVE_EVERY=200` 的 gather 空窗仍在，约 18 分钟一次。

## 滑窗交叉（3090，2026-09-20T01:06Z，H=16 hd=128 equal-head）

| seq | n_win | S×S | fat 256 | 相对 S×S |
| --- | ---: | ---: | ---: | ---: |
| 512 | 32 | 0.07 ms | 0.53 ms | **0.14×**（禁用） |
| 1024 | 32 | 0.19 ms | 0.55 ms | **0.35×**（禁用） |
| 2048 | 32 | 0.63 ms | 0.54 ms | 1.17×（门槛） |
| 4096 | 32 | 2.27 ms | 0.78 ms | **2.93×** |
| 4096 | 256 | 2.27 ms | 1.07 ms | 2.13× |
| 4096 | 8192 覆盖 | 2.08 ms mask | 1.18 ms Flash | 覆盖窗走 Flash |

`BANDED_SEQ_MIN=2048`。B0 训练窗盖满，不走这条。

## 实测（3090，2026-09-19T17:28Z；滑窗 2026-09-20 重测）

校准 GEMM **72.12 TFLOPS**。`grouped_mm=False`。

| 算子 | 实测 | 相对 roofline |
| --- | --- | --- |
| fused QKV 12B 形 | 74.66 T | ~105%（boost；冻结 concat-once） |
| Flash GQA seq=4096 | 60.45 T | **85%** |
| MoE 均匀 bmm E=20 n=409 | 57.66 T | **81%** |
| CSA union seq=512 | 36.73 T | 52%（mask 核，非 Flash） |
| HCA concat seq=512 | 40.34 T | 57% |
| masked window seq=512 | 17.99 T | 25% |
| banded fat-tile seq=512 n_win=32（已禁用） | 2.02 T | 2.8%（0.14× S×S，故 `BANDED_SEQ_MIN=2048`） |
| masked window seq=2048 | 31.78 T | 45% |
| banded fat-tile seq=2048 n_win=32 | 32.50 T | 46%（1.17× S×S） |
| indexer fp32 12B 形 | 14.28 T | 50% of FP32 roofline（d_idx=64 瘦 K） |
| 全部 bf16-probe seq=128 | &lt;1 T | launch 墙，理论 30–84% 也够不着 |

Ledger：[`artifacts/autodl-rtx3090/bf16-verify/mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/)。

先前同卡、调算子前：大 GEMM 72.3T；Flash s=4096 62.9T；fused QKV 每步 cat 更慢；MoE 若走 grouped_mm try/except 贵 2–3×。FlexAttention 滑窗 ~1.7 ms vs SDPA 0.02 ms，不用。

## 怎么跑

```bash
python3 -m cat_yoko.ampere_mfu
bash scripts/run_ampere_mfu.sh
# OUT=/root/autodl-tmp/bf16-verify/mfu/ledger.json

# 3090 同阶段续训（独立 SAVE；只读 resume Hub overlay；不要 push 覆盖 Hub）
# STEPS=8 bash scripts/run_b0_ampere_3090.sh
```

CPU 只打理论表。不要拉 Ultra-FineWeb。不要写完整图 checkpoint。不要 `--save-full`。不要覆盖 Hub `b0-full`。
