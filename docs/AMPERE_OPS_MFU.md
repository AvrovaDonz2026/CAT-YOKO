# Ampere BF16 算子理论 MFU（RTX 3090）

3090 的 **dense BF16 tensor** 峰值是 **71.16 TFLOPS**（无 2:4 稀疏；大 GEMM 实测约 72.3）。HBM 936 GB/s，ridge ≈ **76 FLOP/byte**。某个算子的**理论 MFU** 是 roofline：

\[
\mathrm{MFU}_{\mathrm{theo}} = \min\bigl(1,\; I / r\bigr),\quad I=\mathrm{FLOPs}/\mathrm{bytes},\quad r=\mathrm{peak}/\mathrm{BW}.
\]

逼近理论 MFU 是逼近这条 roofline，不是在 seq=128 的探针图上硬凑 40% Kaplan 或 100% 峰值。Flash 在 seq=4096、hd=128 时 IO-aware 强度 \(I \approx S/4 = 1024\)，已经算力墙，理论 ~100%；seq=128 时 \(I=32\)，理论约 42%，再被 kernel launch 打到 1% 量级。

`torch._grouped_mm` **只在 SM90+ 能跑**。Ampere 上 API 在、一调就 `RuntimeError`，以前每层 try/except 比直接 padded bmm 贵 2–3×。现在 `grouped_mm_available()` 按 compute capability 关门，3090 走 bmm。

FlexAttention 滑窗在这套 torch 2.8 / sm_86 上约 1.7 ms，对照 SDPA 0.02 ms，是速度陷阱。CSA/HCA 不换上去。不是 CSA CUDA kernel。

## 各块（理论）

CPU 上 `python3 -m cat_yoko.ampere_mfu` 打出的 roofline（相对 71.16 TFLOPS；indexer 相对 35.58 TFLOPS FP32/TF32）：

| 算子 | 阶段 | bf16-probe 理论 MFU | 12B 形理论 MFU | 调整（逼近 roofline） |
| --- | --- | ---: | ---: | --- |
| fused QKV | A–E | 84.2% | 100% 算力墙 | 冻结权重 concat 一次，不再每步 `torch.cat` |
| dense Flash GQA | YOCO cross；窗盖满 seq | 56.1%（再被 launch 打到 ~1%） | 100%；实测可到 ~87% 峰值 | Flash + `enable_gqa`，不 repeat KV |
| masked window | encoder `n_win<seq` | 37.4%（付 S×S mask） | B0 `n_win=8192≥seq` 走 Flash；此行是 C/探针洞 | seq&lt;320 只用 Efficient；更长只用 cuDNN；静态窗 mask 缓存 |
| CSA union | C-topk | 74.8% | 100% 强度，但 fused mask 核到不了 Flash | 同上；**不要** `enable_gqa+attn_mask`（Ampere 静默掉 math 核），repeat KV 后走 Efficient/cuDNN |
| HCA concat | C-hca+ | 78.7% | 同左，k 更长 | 静态槽 bias 缓存；不把 CSA/HCA 换上 FlexAttention |
| MoE bmm | B2+ | 13.0%（每专家 12 token，带宽墙） | 100% 强度；均匀专家实测 ~84% 峰值 | Ampere 禁止 grouped_mm；冻专家缓存 `gate‖up` |
| indexer fp32 | C-index | 29.7%（FP32 峰值） | 81.3% | 分数仍 fp32（KEEP_HIGH_PREC）；TF32 `high`；含 Q/K 投影 |

12B B0 训练 `seq=4096`、`n_win=8192`：encoder 窗盖满，走 dense Flash，不是 masked 行。探针故意 `n_win<seq`，Theorem B 的压缩洞才露出来。

## 实测（3090）

把 `scripts/run_ampere_mfu.sh` 的 ledger 拉回后填 [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/)。看 `achieved_mfu / theo_mfu`（`frac_of_roofline`），不是 Kaplan 40%。

先前同卡探针（调算子前）：大 GEMM **72.3 TFLOPS**；Flash s=4096 **62.9T ≈ 87% 峰值**；masked s=128 Efficient 优于 cuDNN；fused QKV 冻结缓存优于每步 cat；MoE 均匀 bmm ~61T。grouped_mm 在 sm_86 上是陷阱。

## 怎么跑

```bash
python3 -m cat_yoko.ampere_mfu
bash scripts/run_ampere_mfu.sh
# OUT=/root/autodl-tmp/bf16-verify/mfu/ledger.json
```

CPU 只打理论表。不要拉 Ultra-FineWeb。不要写完整图 checkpoint。不要 `--save-full`。
