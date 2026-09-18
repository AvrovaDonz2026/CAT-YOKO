# CAT-YOKO FP8 理论验证

> 与前三篇分工：[`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) 核中间档参数 / 6NT / KV；[`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) 核因果与 M1/M2/M3；[`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) 核 C1 解冻课程；**这篇核 dtype**：中间档 C1 训练里哪一块可以走 FP8、墙钟能再砍多少、哪些模块必须留高精度。
> 规格仍是中间档：16/24，≈12B / 2.3B-in / 4.5B-out。C1 定稿不变。不引入新架构，不写训练骨架。
> 可执行断言：`python3 scripts/param_budget.py --verify`（含 FP8 claim）、`--fp8`；`python3 -m unittest tests.test_param_budget`。
> 理论能证明的是 **6NT 不变、Amdahl 上界、与定理 A/E 的兼容**；不能证明 12B 上 FP8 kernel 真能吃满 1.5×。那是 L1 实验。

---

## 0. 结论（先看这个）

中间档训练的 **相当一部分** 是 MoE 专家 GEMM：占存储参数的 **84.6%**、占计入 6NT 的 **71.5%**。这块（外加冻结 Encoder 的推理 GEMM）可以走 FP8。FP8 **不改** Kaplan \(6N_{\mathrm{act}}T\)，只提高 H100 上完成这些 FLOPs 的有效吞吐。

| 做法 | 50B tok H100-h | vs 联合 bf16 |
| --- | ---: | ---: |
| 两栈一起训，bf16 | 1,354 | 100% |
| C1 解冻课程，bf16 | 1,090 | 81% |
| **C1 + FP8 定稿策略**（B0 student bf16；B1/B2 MoE GEMM 与冻结 Encoder 前向 1.5×） | **761** | **56%** |
| C1 全阶段套 1.5×（B0 student 也 FP8；**不定稿**） | 726 | 54% |
| C1 × 峰值 2×（上界，**不发布**） | 545 | 40% |

还必须钉死的几条：

1. **6NT 与 dtype 无关**（定理 G）。计划 §15.1 的 1,413 / 1,354 / 1,090 仍是 bf16 操作数账本；FP8 只进墙钟列。
2. **发布 1.5×，不发布 2×。** H100 SXM FP8 峰值 ≈ 1979 TFLOPS，是 bf16 989 的 ~2.0×。MoE 活跃份额 \(f=71.5\%\)、GEMM 2× 的 Amdahl 是 **1.56×**，正好托住 1.5×。12B 专家 GEMM（\(d=2304\)，\(d_{\mathrm{moe}}=2048\)，单专家 14.16M）远小于 DeepSeek-671B，dispatch/combine、scale、小 kernel 占用都会把峰值吃掉。MoE+attn 的 \(f=95.6\%\) 给出 1.92×，那是上界。
3. **B0 student 保持 bf16。** gate 从 0 爬到 0.3，贴着定理 A 的残差邻域；新模块随机初始化，不在这里换计算精度。冻结 Encoder 前向是纯推理 GEMM，B0 就可以 FP8。
4. **B1+B2 覆盖 50B 信封的 84% token、C1 FLOPs 的 89%。** 这就是「相当一部分」：真正长时间跑的阶段走 `fp8_moe`。B0 只占 C1 FLOPs 的 11%，student 留 bf16 几乎不吃掉 FP8 收益（761 vs 全阶段 726，差 ~35 H100-h）。
5. **高精度白名单**：tied \(E\)、RMSNorm、router、gate、Lightning Indexer、attn softmax。定理 E（冻结 Encoder 时 \(X^0\) 不能漂）直接禁止 FP8 输入表。
6. **L0 与 Phase C indexer 保持 bf16。** tiny 正确性不混精度；indexer 对齐是小张量上的层内 KL。
7. **Muon Newton-Schulz 仍是 fp32**，与网络 GEMM 的 FP8 正交。V4 式 FP4 专家存储是后期可选项，不进本配方。
8. **C1 与 FP8 相乘，不是相加。** C1 先砍 19% FLOPs，再在剩下的墙上乘 1.5×（B0 例外）。相对联合 bf16：\(81\% \times\)（B1/B2 1.5×，B0 近 1×）≈ **56%**。

Claim ledger：中间档 22 + 课程 12 + FP8 12，`--verify` **46/46** 通过。

---

## 1. 定理 G：dtype 不改 6NT

Kaplan / Hoffmann 口径（与预算篇相同）：

\[
F = 6\,N_{\mathrm{act}}T = \underbrace{2N_{\mathrm{act}}T}_{\text{前向}} + \underbrace{2N_{\mathrm{act}}T}_{\text{激活反传}} + \underbrace{2N_{\mathrm{act}}T}_{\text{权重梯度}}.
\]

这是乘加次数，不是 joule，也不是秒。把专家投影从 bf16 换成 FP8 E4M3/E5M2，\(F\) 不变。变的是

\[
H_{\text{wall}} = \frac{F}{\eta_{\text{bf16}}}\, /\, S,\qquad
\eta_{\text{bf16}} = 4.0\times 10^{14}\ \text{FLOPS（40% MFU）}.
\]

\(S=1\) 就是计划 §15.1 的 bf16 列；C1 的 \(F_{\mathrm{C}}\) 相对联合少 19%，所以 bf16 墙钟从 1,354 降到 1,090。FP8 只进入 \(S\)。

H100 SXM：

\[
\frac{\text{FP8 peak}}{\text{bf16 peak}} = \frac{1.979\times 10^{15}}{9.89\times 10^{14}} \approx 2.00.
\]

若 MFU 在 FP8 上仍是 40%，有效吞吐 7.92×10¹⁴，\(S=2\)。12B 上不成立：专家矩阵小、MoE 通信不加速、scale 有开销。发布值锁 **\(S=1.5\)**。

---

## 2. 「相当一部分」：MoE GEMM 占 6NT 的 71.5%

中间档一次完整前向 \(N_{\mathrm{fwd}}=6.50\mathrm{B}\)（emb 计一次）：

| 块 | 激活参数 | 占 6NT |
| --- | ---: | ---: |
| MoE 专家（top-\(k\) SwiGLU） | 4.64B | **71.5%** |
| 自注意力 + cross-attn 投影 | 1.57B | 24.2% |
| tied embedding | 0.28B | 4.4% |
| RMSNorm / router / softmax | （6NT 忽略） | — |

存储侧 MoE 仍是 **84.6%**（720 专家槽，含未入 top-\(k\) 的副本）。训练 FLOPs 走的是激活，所以「相当一部分」按 **71.5%** 计。

只把 MoE GEMM 换成 2×、其余 1×：

\[
S_{\mathrm{MoE}} = \frac{1}{0.715/2 + 0.285} \approx 1.56.
\]

1.56 略高于发布的 1.5，差额留给 dispatch/combine 与 12B 小 kernel。若连 attn 投影也 FP8，\(f=95.6\%\)，\(S\approx 1.92\)——这是上界，**不写入 §15.1**。

冻结 Encoder 的前向（B0/B1）是同一批 GEMM 的推理形态：无权重梯度、无激活反传，是整张图里最安全的 FP8 切口。B0 里它约占该阶段 FLOPs 的 18%。

---

## 3. 定稿策略（叠在 C1 上，不改冻结边界）

Student = 接收梯度的张量。冻结 Encoder 的 GEMM 单独一列。

| 阶段 | student | 冻结 Encoder GEMM | 理由 |
| --- | --- | --- | --- |
| L0 | bf16 | bf16 | tiny 正确性；NaN 归因必须干净 |
| **B0** | **bf16** | **fp8** | gate 0→0.3 贴着定理 A；冻结栈是推理 |
| **B1** | **fp8_moe** | **fp8** | Decoder 专家 GEMM 是 27B token 的大头 |
| **B2** | **fp8_moe** | n/a | 两栈都解冻，专家 GEMM 仍 FP8 |
| C | bf16 | fp8 | indexer KL 局部、张量小 |

B1+B2 = 27B+15B = 42B / 50B = **84%** 的 token 信封。对应 C1 FLOPs 的 89%。B0 的 8B 只占 C1 FLOPs **11.4%**，student 留 bf16 几乎不损害墙钟。

缩放：优先 DeepSeek-V3 式 **tile scaling**（激活 1×128、权重 128×128）；kernel 没有时用 delayed scaling。Master 权重仍 bf16；Adam \(m,v\) 仍 fp32；Muon 正交化仍 fp32。FP8 是 **GEMM 计算 dtype**，不是唯一存储 dtype。

---

## 4. 必须保持高精度的模块

`FP8_KEEP_HIGH_PREC`：

| 模块 | 为什么不能 FP8 |
| --- | --- |
| tied \(E\) | 定理 E：B0/B1 冻结 Encoder 时 \(X^0=s_{\mathrm{emb}}\,\mathrm{onehot}\,E\) 是冻结栈的输入分布。FP8 表会给 16 层冻结 MiniCPM 残差加量化噪声。 |
| RMSNorm | 零中心 + weight decay 的尺度敏感；DeepSeek 也把 LN 留在高精度。 |
| router | 专家选择是离散的；logit 上几个 ULP 就会改 top-\(k\)。aux-loss-free 偏置同样。 |
| gate | 定理 A 的旁路；B0 的 \(g\in[0,0.3]\) 必须平滑。 |
| indexer | Phase C 要对齐稠密注意力分布；FP8 打分头会把 KL 目标弄脏。 |
| attn softmax | 指数 + 归一只在 bf16/fp32 稳定；FP8 的是 QK/AV GEMM，不是 softmax。 |

Hash-MoE 是 `token_id → expert_id`，没有学到的 router GEMM；若旁边还有仿射，仍走白名单。

---

## 5. 墙钟账本（C1 × FP8）

令 \(F_0,F_1,F_2\) 为 B0/B1/B2 的 Kaplan FLOPs，\(F_{\mathrm{enc}}^{0}=2N_{\mathrm{enc}}T_0\) 为 B0 的 Encoder 前向。定稿策略：

\[
H_{\mathrm{FP8}}
= H(F_0 - F_{\mathrm{enc}}^{0})
+ \frac{H(F_{\mathrm{enc}}^{0})}{1.5}
+ \frac{H(F_1)}{1.5}
+ \frac{H(F_2)}{1.5}
\approx 761\ \text{H100-h}.
\]

相对联合 bf16 1,354：**56%**。相对 C1 bf16 1,090：再砍约 **30%** 墙钟。

敏感性（不定稿）：

| 假设 | H100-h | vs 联合 |
| --- | ---: | ---: |
| B0 student 也 1.5× | 726 | 54% |
| 全阶段 MoE-Amdahl 1.56× | ~700 | 52% |
| 峰值 2× | 545 | 40% |

不把 2× 写进计划。12B 上若实测 \(S<1.3\)，回退 B1/B2 到 bf16，C1 的 1,090 仍在。

冻结 Encoder 权重 4.50B：bf16 9.00 GB → FP8 存储 4.50 GB。这是推理副本；master 仍可 bf16。省的是带宽，不是 6NT。

---

## 6. 与 Muon / 显存 / FP4 正交

- **Muon**：Newton-Schulz 在动量矩阵上做 ~5 次 fp32 迭代，与网络 FP8 GEMM 不是同一条路径。§13 的「Muon → FP8」阶梯改成：L0 保持 bf16；Muon 可在 B1 与 FP8 并行开，各留回退开关。
- **C1 显存杠杆仍在**：B1 Adam 60%、detach 激活 ~40%。FP8 再砍 GEMM 激活带宽；不替代冻结。
- **FP4 专家存储**（V4）：推理/存盘选项。训练期还要 bf16 master 与 fp32 优化器，12B 上收益小，不进配方。
- **Phase D**：长上下文上注意力二次项变大（预算篇 §6）。那时再考虑 attn 投影 FP8 / FP8 KV；本篇只锁 4K 的 Phase B。

这就是 §15.3 把解冻课程排在 FP8 前面的原因：C1 改的是操作数，FP8 改的是每秒操作数。先砍 \(F\)，再乘 \(S\)。

---

## 7. 失败模式（实现前先避开）

| 失败 | 来源 | 避免 |
| --- | --- | --- |
| 以为 FP8 减少 6NT | 把吞吐写成 FLOPs | 定理 G；§15.1 分列 |
| 发布 2× | 抄 H100 峰值 | 12B Amdahl 1.56；锁 1.5× |
| B0 student FP8 弄脏定理 A | gate 邻域 + 新模块 | B0 student bf16；只 FP8 冻结 Encoder 前向 |
| FP8 输入表漂冻结 Encoder | 训/量化 tied \(E\) | 白名单 + 定理 E |
| router FP8 改 top-\(k\) | 离散选择 | router / 偏置高精度 |
| L0 混精度 | tiny 无法归因 | L0 bf16 |
| indexer FP8 对齐失败 | Phase C KL | C 的 indexer bf16 |
| 用 FP8 替代 C1 | dtype ≠ 冻结 | 先 C1 再 FP8 |
| FP4 专家当训练权重 | 无 master | FP4 只作后期存储 |

理论**不能**排除的：12B 专家 GEMM 实际 \(S\approx 1.2\)、B1 切 FP8 时 loss 尖峰、E4M3 激活溢出。那些是 L1 实验；尖峰就回退该阶段的 student dtype，不要改 C1 冻结边界，也不要改回两个独立 LM。

---

## 8. Claim ledger

`python3 scripts/param_budget.py --verify` 在中间档 22 + 课程 12 之外增加：

| Claim | 结果 |
| --- | --- |
| FP8 不改 Kaplan 6NT | PASS |
| H100 FP8 峰值 ≈ 2× bf16 | PASS |
| 保守墙钟 1.5× | PASS |
| MoE ≥70% of 6NT | PASS（71.5%） |
| MoE-only Amdahl ∈ [1.50, 1.75] | PASS（1.56×） |
| C1 FP8 定稿策略 ≤60% 联合 bf16 | PASS（761 / 1,354 = 56%） |
| B0 student = bf16 | PASS |
| B1/B2 student = fp8_moe | PASS |
| B1+B2 ≥80% of 50B | PASS（84%） |
| B0 ≤15% of C1 FLOPs | PASS（11.4%） |
| tied \(E\) / router / LN / gate / indexer / softmax 高精度 | PASS |
| C1 bf16 小时不被 FP8 策略改写 | PASS（1,090） |

---

## 9. 对计划的修订（本 PR）

1. §6 / §8：精度从「bf16；FP8 视 kernel 逐步启用」改为本篇定稿策略。
2. §15.1：6NT 表仍是 bf16；增 C1 bf16 与 C1+FP8 墙钟行。脚注 1.5–2× 改为 **发布 1.5×，2× 只作峰值上界**。
3. §15.3 杠杆 #6：C1 × FP8 定稿把 Phase B 从 1,090 收到 **761 H100-h（联合 bf16 的 56%）**。
4. §13：L0 保持 bf16；FP8 从 B1 开；Muon 正交化与 FP8 GEMM 分开回退。
5. 不写训练代码骨架（按用户要求，理论先闭环）。

复算：

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
python3 scripts/param_budget.py --staged --curriculum --fp8
python3 -m unittest tests.test_param_budget
```
