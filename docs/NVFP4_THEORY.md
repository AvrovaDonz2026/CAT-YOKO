# CAT-YOKO NVFP4 理论验证

> 与前四篇分工：[`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) 核中间档参数 / 6NT / KV；[`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) 核因果与 M1/M2/M3；[`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) 核 C1 解冻课程；[`FP8_THEORY.md`](FP8_THEORY.md) 是 **Hopper / Ada 回退账本**（C1+FP8 = 729 H100-h）。**这篇核发布 dtype**：凡是不必 bf16/fp32 的线性 GEMM 都砍成 NVIDIA NVFP4；目标卡 **RTX PRO 6000 / 6000D Blackwell**。
> 规格仍是中间档：16/26，≈12.25B / 2.03B-in / 4.33B-out。底座 MiniCPM5-2B（Llama GQA）。C1 冻结边界不变。**Phase B 墙钟按 C1+NVFP4 定稿。** 本 PR 不写 Transformer Engine kernel。
> 可执行断言：`python3 scripts/param_budget.py --verify`（含 NVFP4 claim）、`--nvfp4`；`python3 -m unittest tests.test_param_budget`。
> 理论能证明的是 **6NT 不变、Amdahl 上界、与定理 A/E 的兼容、必须高精度集合**；不能证明 12B 小专家在 sm_120 上真能吃满 2.0×。那是 Blackwell L1 实验。

---

## 0. 结论（先看这个）

**Phase B 墙钟配方按 C1+NVFP4 定稿：571 H100-h，联合 bf16 的 43%。** 记账单位仍是 H100-h：RTX PRO 6000 Server 的 BF16 峰值 1 PFLOP 与 H100 SXM 0.989 PFLOP 同量级，40% MFU 下小时数可比。6000D 是同族 Blackwell，显存/峰值以当时 datasheet 为准，不假设与 96GB 6000 1:1。

联合 bf16（1,325）只是 100% 对照。C1 bf16（1,046）是操作数账。C1+FP8（729）是 **没有 FP4 tensor core 时的 Hopper/Ada 回退**，不再是发布墙钟。真正要跑的是 **C1 的 token 切分 × 混合 NVFP4**：B0 student 仍 bf16；B1/B2 所有「非必须高精度」线性 GEMM + 冻结 Encoder 前向走 NVFP4，加速比锁 **2.0× vs bf16**（相对旧 FP8 1.5× 再 ×1.33，落在 NVIDIA GB200/GB300 相对 FP8 的 1.31–1.73× 低端）。峰值 4×（261）和全阶段 2×（523，B0 student 也 NVFP4）不定稿。

判别规则只有一句：**不必须 bf16 的，就用 NVFP4。** 旧 FP8 白名单里的 lm_head 与 attn QKV/O 投影不是必须 bf16，从白名单拿掉。V4 式 FP4 **专家存储**仍然不进配方（那是存盘/推理，不是训练 GEMM）。

| 做法 | 50B tok H100-h | vs 联合 bf16 | 角色 |
| --- | ---: | ---: | --- |
| 两栈一起训，bf16 | 1,325 | 100% | 对照 |
| C1 解冻课程，bf16 | 1,046 | 79% | 操作数账 |
| C1+FP8（Hopper/Ada 回退） | 729 | 55% | 无 Blackwell 时 |
| **C1+NVFP4**（B0 student bf16；其余允许的 GEMM 2.0×） | **571** | **43%** | **定稿** |
| C1 全阶段套 2.0×（B0 student 也 NVFP4） | 523 | 39% | 敏感性 |
| C1 × 峰值 4× | 261 | 20% | 上界，不发布 |

还必须钉死的几条：

1. **6NT 与 dtype 无关**（定理 G）。NVFP4 与 FP8 一样只进墙钟列 \(S\)，不改 Kaplan 操作数。
2. **发布 \(S=2.0\) vs bf16，不发布 4×。** RTX PRO 6000 Server：BF16 1 / FP8 2 / FP4 4 PFLOPS。相对 FP8 的硬件峰值是 2×；NVIDIA MaxText 在 GB200/GB300 上端到端是 1.31–1.73× vs FP8。12B 专家只有 12.58M，取低端 \(1.5\times 1.33\approx 2.0\) 相对 bf16。MoE 活跃份额 \(f=81.9\%\)、GEMM 4× 的 Amdahl 是 2.59×；把 attn 投影 + lm_head 也算进 GEMM（\(f=95.8\%\)）是 3.55×——那是上界。
3. **B0 student 保持 bf16。** 这是必须：gate 0→0.3 贴着定理 A，新模块随机初始化。冻结 Encoder 前向是纯推理 GEMM，B0 就可以 NVFP4。
4. **B1/B2 student 是 `nvfp4`，不是 `nvfp4_moe`。** 允许的线性 GEMM 全部砍掉：MoE 专家 FPROP/DGRAD/WGRAD、自attn \(W_Q,W_K,W_V,W_O\)、cross-attn \(W_Q,W_O\)、cache \(W_K,W_V\)、**untied lm_head**。NVIDIA MaxText 默认只量化 MLP、注意力块留高精度；那是他们的保守配方。本仓库按「不必须 bf16」把 QKV/O 与 lm_head 也纳入。若 B1 发散，**第一回退是 QKV/O 回到高精度（MaxText 口径）**，不是改 C1，也不是改回两个 LM。
5. **必须高精度的集合变短。** 只留：输入 \(E_{\mathrm{in}}\)（查表 + 定理 E）、RMSNorm / QK-Norm（fp32）、router（离散 top-\(k\)）、YOCO 标量 gate（定理 A）、Lightning Indexer、attn softmax / SDPA 的 score+context。**lm_head 不再在白名单里。**
6. **L0 与 Phase C indexer 保持 bf16。** tiny 正确性不混精度；indexer 对齐是小张量上的层内 KL。
7. **Master 权重仍 bf16；Adam \(m,v\) 仍 fp32；Muon Newton-Schulz 仍 fp32。** NVFP4 是 GEMM 计算 dtype：FPROP/DGRAD/WGRAD 吃 NVFP4，吐 BF16，折进 fp32 master。配方是 16 元微块 + FP8 E4M3 块缩放 + 每张量 FP32 缩放；权重 2D 16×16；RHT 只打 WGRAD；随机舍入打梯度量化器。
8. **C1 与 NVFP4 相乘，不是相加。** C1 先砍 21% FLOPs，再在剩下的墙上乘 2.0×（B0 student 例外）。相对联合 bf16：\(79\% \times\)（B1/B2 2.0×，B0 近 1×）≈ **43%**。这条积就是定稿墙钟 **C1+NVFP4 = 571**。
9. **本 PR 不实现 TE / sm_120 kernel。** 无 Blackwell 时：同一套模块策略退回 FP8 placeholder，再退回 bf16 autocast。RTX 4080 SUPER 不能跑 NVFP4。sm_120 上官方 TE 融合 NVFP4（RHT/SR）可能不可用；发布配方仍写完整训练 recipe，kernel 以后在目标卡上接。

Claim ledger：中间档 22 + 课程 12 + FP8 回退 13 + NVFP4 16，`--verify` **63/63** 通过。

---

## 1. 定理 G 仍成立：dtype 不改 6NT

与 [`FP8_THEORY.md`](FP8_THEORY.md) §1 同一条 Kaplan 账：

\[
F = 6\,N_{\mathrm{act}}T,\qquad
H_{\text{wall}} = \frac{F}{\eta_{\mathrm{bf16}}}\, /\, S,\qquad
\eta_{\mathrm{bf16}} = 4.0\times 10^{14}\ \text{FLOPS（40% MFU）}.
\]

NVFP4 只进入 \(S\)。C1 的 \(F_{\mathrm{C}}\) 相对联合少 21%，bf16 墙钟仍是 1,046。

RTX PRO 6000 Blackwell Server Edition（NVIDIA 发布页）：

\[
\text{BF16 } 1\ \text{PFLOP},\quad
\text{FP8 } 2\ \text{PFLOPS},\quad
\text{FP4 } 4\ \text{PFLOPS},\quad
96\,\text{GB GDDR7},\ 1597\,\text{GB/s}.
\]

相对本卡 BF16，FP4 峰值 4×。相对本卡 FP8，峰值 2×。发布 \(S\) 不用峰值。NVIDIA MaxText（2026-06-08）在 GB200/GB300 上 Llama 3 8B / 3.1 405B：**NVFP4 vs FP8 = 1.31×–1.73×**，loss 差 +0.026 nats。12B 小专家取低端，乘上已经发布的 FP8 1.5×：

\[
S_{\mathrm{pub}} = 1.5 \times \frac{2.0}{1.5} = 2.0
\quad\text{（相对 bf16；相对 FP8 = } 2.0/1.5 \approx 1.33\in[1.31,1.73]\text{）}.
\]

H100 SXM BF16 峰值 \(9.89\times 10^{14}\) 与 6000 的 \(1.00\times 10^{15}\) 相差 <2%，所以 **H100-h ≈ 6000-h**（同一 40% MFU）。6000D 不另出一张墙钟表。

---

## 2. 哪些必须 bf16/fp32，哪些必须 NVFP4

Student = 接收梯度的张量。Master 存储永远 bf16，与计算 dtype 无关。

### 2.1 必须留高精度（非 NVFP4）

| 模块 | 为什么必须 |
| --- | --- |
| 输入 \(E_{\mathrm{in}}\) | 查表不是 GEMM。定理 E：B0/B1 冻结 Encoder 时 \(X^0\) 是冻结栈的输入分布，量化噪声会灌进 16 层冻结 MiniCPM5 残差。 |
| RMSNorm / QK-Norm | 小向量方差，fp32 再 cast 回。不是 GEMM。 |
| attn softmax / SDPA score+context | 指数 + 归一只在 fp32 稳定。NVIDIA 也写明 softmax 会指数放大 QKᵀ 量化噪声；score/context 与线性投影分开。 |
| router + expert bias | 专家选择是离散 top-\(k\)；21 维 logit 也不对齐 NVFP4 的 16 元微块。几个 ULP 就会改路由。 |
| YOCO 标量 gate | 定理 A 的旁路；B0 的 \(g\in[0,0.3]\) 必须平滑。不是 GEMM。 |
| Lightning Indexer | Phase C 要对齐稠密注意力分布；4-bit 打分头会把 KL 目标弄脏。 |
| B0 student | 新模块随机初始化 + gate 爬坡。NVIDIA 也观察到「全 NVFP4 会发散、末几层要留 BF16」。 |
| L0 / tiny | 正确性归因。 |
| Teacher MiniCPM5 | 已训好的 bf16 检查点。 |
| Adam \(m,v\) / Muon NS | fp32 优化器状态，不是网络 GEMM。 |

Hash-MoE 是 `token_id → expert_id`，没有学到的 router GEMM。

### 2.2 不是必须 bf16 → 发布 NVFP4

旧 C1+FP8 只砍 MoE 专家 + 冻结 Encoder 前向，lm_head / attn 投影留在白名单里是保守，不是数值必然。按用户口径拿掉：

| 槽 | 阶段 | 备注 |
| --- | --- | --- |
| MoE 专家 SwiGLU（gate/up/down）FPROP/DGRAD/WGRAD | B1/B2；冻结 Encoder 前向从 B0 起 | 占计入 6NT 的 **81.9%** |
| 自attn \(W_Q,W_K,W_V,W_O\) | B1/B2；冻结 Encoder 前向从 B0 起 | 投影是线性 GEMM；softmax 仍高精度 |
| cross-attn \(W_Q,W_O\) | B1/B2 | B0 里它们**就是** student，仍 bf16 |
| cache \(W_K,W_V\) | B1/B2 | B0 同 student bf16 |
| **untied lm_head** | B1/B2 | \(d\times V=2048\times 130560\) 是大 GEMM；定理 E 管的是输入表，不管 head |

NVIDIA MaxText：「三个 GEMM 只对 MLP 量化到 NVFP4；注意力块（QKV、O、score/context）留高精度」。本配方同意 **score/context 必须高精度**，不同意把 QKV/O 与 lm_head 当成必须 bf16。发散时先把 QKV/O 退回高精度，lm_head 仍可 NVFP4。

---

## 3. C1+NVFP4 定稿（叠在 C1 上，不改冻结边界）

| 阶段 | student | 冻结 Encoder GEMM | 理由 |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | tiny 正确性 |
| **B0** | **bf16** | **nvfp4** | 定理 A；冻结栈是推理 |
| **B1** | **nvfp4** | **nvfp4** | Decoder 允许的线性 GEMM + lm_head |
| **B2** | **nvfp4** | n/a | 两栈解冻后同样的 GEMM 集合 |
| C | bf16 | nvfp4 | indexer 局部、张量小 |

B1+B2 = 84% token / 89% C1 FLOPs。B0 占 C1 FLOPs **11.0%**，其中冻结 Encoder 前向约占 B0 的 17%；student 留 bf16 几乎不吃掉 NVFP4 收益（571 vs 全阶段 523，差 ~48 H100-h）。

缩放：16 元微块、权重 2D 16×16、WGRAD 上 RHT、梯度随机舍入。无 kernel 时同一套模块走 FP8 placeholder，再走 bf16 autocast。

96GB：B0/B1 的 Encoder offload 与 CPU Adam 变成**操作可选**，不是 dtype 变更。32GB 卡仍按原 offload 路径；它跑不了 NVFP4。

---

## 4. 墙钟账本（定稿 C1+NVFP4）

令 \(F_0,F_1,F_2\) 为 B0/B1/B2 的 Kaplan FLOPs，\(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) 为 B0 的 Encoder 前向。**定稿**：

\[
H_{\mathrm{NVFP4}}
= H(F_0 - F_{\mathrm{enc}}^{0})
+ \frac{H(F_{\mathrm{enc}}^{0})}{2.0}
+ \frac{H(F_1)}{2.0}
+ \frac{H(F_2)}{2.0}
\approx 571\ \text{H100-h}.
\]

相对联合 bf16 1,325：**43%**。相对 C1 bf16 1,046：再砍约 **45%** 墙钟。相对 C1+FP8 729：再砍约 **22%** 墙钟。**571 是 Phase B 发布墙钟。**

敏感性（不定稿）：

| 假设 | H100-h | vs 联合 |
| --- | ---: | --- |
| B0 student 也 2.0× | 523 | 39% |
| 全阶段 MoE-Amdahl 4× GEMM（2.59×） | 403 | 30% |
| 峰值 4× | 261 | 20% |

不把 4× 写进计划。12B 上若实测 \(S<1.7\) vs bf16，先把 QKV/O 退回高精度；仍不够再把 B1/B2 student 退到 FP8 或 bf16。C1 的 1,046 仍在。

---

## 5. 与 FP8 / 显存 / V4 FP4 存储正交

- **C1+FP8 = 729** 仍是可复算的回退账，见 [`FP8_THEORY.md`](FP8_THEORY.md)。Hopper 没有 FP4 tensor core。
- **C1 显存杠杆仍在**：B1 Adam 62%、detach 激活 ~38%。NVFP4 再砍 GEMM 激活带宽；不替代冻结。96GB 只是让 offload 变成可选。
- **V4 FP4 专家存储**：推理/存盘选项。训练期还要 bf16 master 与 fp32 优化器，不进配方。NVFP4 是**计算**格式，不是那条存储路径。
- **Muon**：Newton-Schulz 仍 fp32，与网络 NVFP4 GEMM 正交。

先砍 \(F\)（C1），再乘 \(S\)（NVFP4）。

---

## 6. 失败模式（实现前先避开）

| 失败 | 来源 | 避免 |
| --- | --- | --- |
| 以为 NVFP4 减少 6NT | 把吞吐写成 FLOPs | 定理 G |
| 发布 4× | 抄 6000 峰值 | 12B 小专家；锁 2.0× vs bf16 |
| B0 student NVFP4 弄脏定理 A | gate 邻域 + 新模块 | B0 student bf16 |
| NVFP4 输入表漂冻结 Encoder | 训/量化 \(E_{\mathrm{in}}\) | 白名单 + 定理 E |
| router NVFP4 改 top-\(k\) | 离散选择 + 宽 21 | router 高精度 |
| 把 softmax 量化成 NVFP4 | 指数放大 QKᵀ 噪声 | score/context fp32；只量化线性投影 |
| L0 混精度 | tiny 无法归因 | L0 bf16 |
| 在 4080 上当 NVFP4 已生效 | Ada 无 FP4 tensor core | 回退 FP8 placeholder / bf16 |
| 用 NVFP4 替代 C1 | dtype ≠ 冻结 | 先 C1 再 NVFP4 |
| 把 V4 FP4 存储当训练权重 | 无 master | 存储后期可选项 |
| 抄 MaxText 只量化 MLP 当「必须」 | 把保守配方当成数值必然 | 发布含 QKV/O+lm_head；发散再退 QKV/O |

理论**不能**排除的：sm_120 上 TE 融合 kernel 不可用、12B 专家 GEMM 实际 \(S\approx 1.4\)、B1 切 NVFP4 时 loss 尖峰。那些是 L1；尖峰就回退该阶段的 student 槽位，不要改 C1 冻结边界。

---

## 7. Claim ledger

`python3 scripts/param_budget.py --verify` 在中间档 22 + 课程 12 + FP8 回退 13 之外增加：

| Claim | 结果 |
| --- | --- |
| NVFP4 不改 Kaplan 6NT | PASS |
| RTX PRO 6000 峰值 1/2/4 PFLOP | PASS |
| 保守墙钟 2.0× vs bf16（不是 4×） | PASS |
| 发布 NVFP4/FP8 比 ∈ [1.31, 1.73] | PASS（1.33） |
| 定稿墙钟是 C1+NVFP4 | PASS |
| 发布 C1+NVFP4 ≈571（≤45% 联合 bf16） | PASS（571 / 1,325 = 43%） |
| B0 student = bf16 | PASS |
| B1/B2 student = nvfp4 | PASS |
| lm_head 是 NVFP4 GEMM（不在必须高精度集合） | PASS |
| attn softmax 必须高精度；QKV 不是必须 bf16 | PASS |
| 必须高精度集合不含 lm_head | PASS |
| C1 bf16 小时不被 NVFP4 策略改写 | PASS（1,046） |
| C1+FP8 回退仍 ≈729 | PASS |
| 6000 BF16 峰值 ≈ H100 | PASS |
| MoE-only 4× Amdahl 只作敏感性 | PASS |
| 允许的 GEMM 占 6NT ≥95% | PASS（95.8%，含 lm_head） |

---

## 8. 对计划的修订（本 PR）

1. **Phase B 墙钟按 C1+NVFP4 定稿：571 H100-h。** 联合 bf16 1,325 只作对照；C1 bf16 1,046 只作操作数账；C1+FP8 729 降为 Hopper/Ada 回退。
2. 精度策略：不必须 bf16 的线性 GEMM 全部 NVFP4。白名单不再包含 lm_head。
3. 目标硬件：RTX PRO 6000 / 6000D。本 PR **不**接 TE kernel。
4. 不改 16/26、C1、因果 Encoder、M2 默认。

复算：

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --nvfp4
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
