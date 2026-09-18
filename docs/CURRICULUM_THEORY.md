# CAT-YOKO 解冻课程理论验证

> 与前两篇分工：[`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) 核中间档参数 / FLOPs / KV；[`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) 核因果与 M1/M2/M3；**这篇核 Phase B 的可训练子集**——冻结边界、梯度在 cache 处截断、tied embedding、延迟 Encoder MoE、优化器/激活显存、token 切分。
> 规格仍是中间档：16/24，≈12B / 2.3B-in / 4.5B-out。不引入新架构，不把两栈拆成两个独立 LM。
> 可执行断言：`python3 scripts/param_budget.py --verify`（含课程 claim）、`--staged`、`--curriculum`；`python3 -m unittest tests.test_param_budget`。
> 理论能证明的是 **FLOPs / 显存 / 梯度流 / 与定理 A 的兼容**；不能证明 12B 上采样后的质量。质量仍走 L0→L1 实验，B2 不够就加长 B2。

---

## 0. 结论（先看这个）

「先分开训两个模型再焊在一起」**更贵且破坏定理 A**。能省的是同一套切开权重上的 **B0 → B1 → B2 解冻课程**。

| 做法 | 50B tok H100-h | vs 联合 |
| --- | ---: | ---: |
| 两栈一起训（基线） | 1,354 | 100% |
| C1：两栈都先 MoE，B0/B1 冻 Encoder | 1,090 | **81%** |
| **C2：Encoder 保持 MiniCPM dense 到 B2 再上采样（推荐）** | **1,044** | **77%** |
| 独立 50B+50B+20B 拼接 | 2,054 | **152%（更贵）** |

还必须钉死的几条：

1. **梯度在全局 cache 处 `.detach()`。** Encoder 无权重梯度、无激活梯度；前向省不掉（YOCO 的 CE 在 Decoder 顶）。这是定理 D。
2. **Tied embedding 必须随 Encoder 冻结，或解绑后只训 LM head。** MiniCPM 的 \(E\) 同时是输入表和输出头；B0/B1 若训练 tied \(E\)，冻结 Encoder 的输入分布会漂，定理 A 的残差流不再是「同一条」。这是定理 E。默认 **freeze_tied**。
3. **\(W_K,W_V\) 是新模块**，挂在 `encoder.detach()` **之后**，B0 就可训。它们不在已发布的 12.05B 栈合计里（上界 \(2d^2=10.62\mathrm{M}\)）。
4. **C1 在 Phase A 就把 Encoder 做成 MoE 再冻住，专家副本直到 B2 才特化**——白付 MoE 前向。C2 让 Encoder 在 B0/B1 保持 MiniCPM dense（每层 39.81M < 7×14.16M），定理 A 对 Encoder **精确**成立到 B2，前向更便宜。B2 再用 virtual-group 上采样 Encoder。
5. **FLOPs 约省 19–23%；Adam 状态在 B1 只有联合的 60%；detach 丢掉 Encoder 激活约 40%。** 塞进更少卡时，显存杠杆可能比 FLOPs 杠杆更有用。
6. **B2 不能为 0**（write/read 永不共同适应，PDSA 已警告）。默认 B2 = 15B ≥ 10B。质量不稳加长 B2，不要改回两个独立 LM。
7. Phase C「冻主干、只训 indexer」叠在 B2 **之后**；indexer 对齐是层内 KL，不是穿过 cache 的 LM 反传。

Claim ledger：中间档 22 条 + 课程 11 条，` --verify` **33/33** 通过。

---

## 1. 形式化：一条计算图，三个可训练子集

长度 \(n\) 的因果 LM，记号与架构篇 §1 相同：

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E,\quad
X^{\ell}=\mathrm{SelfDec}^{\ell}(X^{\ell-1}),\ \ell=1..L_e,
\]

\[
(\hat K,\hat V)=\big(\mathrm{sg}(X^{L_e})W_K,\;\mathrm{sg}(X^{L_e})W_V\big)
\quad\text{（B0/B1；}\mathrm{sg}=\texttt{.detach()}\text{）},
\]

\[
X^{\ell}=\mathrm{CrossDec}^{\ell}(X^{\ell-1},\hat K,\hat V),\ \ell=L_e+1..L_e+L_d,
\quad
z=X^{L_e+L_d}E^{\top}/\alpha.
\]

B2 去掉 \(\mathrm{sg}\)，\(E\) 解冻。

Kaplan 口径（与预算篇相同：前向 \(2NT\) + 反向激活 \(2NT\) + 反向权重 \(2NT\)）：

| 子集 | 前向 | 反向激活 | 反向权重 |
| --- | --- | --- | --- |
| 冻结、且 \(\mathrm{sg}\) 截断 | \(2N\) | 0 | 0 |
| 冻结、但仍在 Decoder 残差链上（B0 的 Decoder 骨干） | \(2N\) | \(2N\) | 0 |
| 可训练 | \(2N\) | \(2N\) | \(2N\) |

所以：

\[
\begin{aligned}
F_{\mathrm{joint}}&=6\,N_{\mathrm{fwd}}T,\\
F_{\mathrm{freeze\text{-}enc}}&=\big(2(N_{\mathrm{emb}}+N_{\mathrm{enc}})+6N_{\mathrm{dec}}\big)T,\\
F_{\mathrm{new\text{-}mod}}&=\big(2N_{\mathrm{fwd}}+2N_{\mathrm{new}}+2N_{\mathrm{dec}}\big)T.
\end{aligned}
\]

\(N_{\mathrm{new}}=L_d A_{\times}+2d^2\)（24 层 cross-attn + cache 投影上界）。C2 的 \(N_{\mathrm{enc}}\) 换成 16 层 MiniCPM dense：

\[
N_{\mathrm{enc}}^{\mathrm{dense}}=L_e(A_{\mathrm{self}}+3d\,d_{\mathrm{ff}})=1.06\mathrm{B}
\quad<\quad
N_{\mathrm{enc}}^{\mathrm{MoE,act}}=2.01\mathrm{B}.
\]

课程

\[
F_{\mathrm{C}}=F_{\mathrm{new\text{-}mod}}(T_0)+F_{\mathrm{freeze\text{-}enc}}(T_1)+F_{\mathrm{joint}}(T_2),\quad T_0+T_1+T_2=50\mathrm{B}.
\]

C1 用 MoE Encoder；C2 在 \(T_0,T_1\) 用 dense Encoder，\(T_2\) 仍用 MoE（B2 边界上采样）。

---

## 2. 定理 D — 梯度在 detached cache 处停止

**定理 D.** 设 \((\hat K,\hat V)\) 由 \(\mathrm{sg}(X^{L_e})\) 右乘 \(W_K,W_V\) 得到，损失函数 \(L\) 只通过 Decoder 与 \((\hat K,\hat V)\) 依赖输入。则

\[
\frac{\partial L}{\partial\theta_{\mathrm{enc}}}=0,\qquad
\frac{\partial L}{\partial X^{\ell}}=0\quad(\ell\le L_e).
\]

因而 Encoder 激活不必为反传保留。

**证明.** \(\mathrm{sg}\) 把 \(X^{L_e}\) 当作常数。链式法则在 cache 处断开。\(W_K,W_V\) 若在 \(\mathrm{sg}\) **之后**，仍可得到 \(\partial L/\partial W_K\)。□

**实现约束（冻结边界，不是启发式）：**

```
X_Le = encoder(embed(x))
X_Le = X_Le.detach()          # B0/B1
K, V = X_Le @ W_K, X_Le @ W_V # 新模块，可训练
```

若先投影再 detach，\(W_K,W_V\) 被冻进 Encoder，B0 训不到 YOCO 接口。若既不 detach 也不把 Encoder `requires_grad=False`，激活显存省不掉，还可能让某个漏标的 Encoder 参数吃到梯度。

B2 必须**去掉** detach，否则 write/read 无法共同适应。

---

## 3. 定理 E — tied embedding 泄漏

MiniCPM 用同一份 \(E\in\mathbb{R}^{V\times d}\)：

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E,\qquad
z=X^{40}E^{\top}/\alpha,\quad \alpha=d/\mathrm{dim\_model\_base}=9.
\]

**定理 E.** 若 \(\theta_{\mathrm{enc}}\) 冻结而 \(E\) 可训练，则 \(\partial L/\partial E\) 经 LM head 非零，于是 \(X^{0}\) 漂移。冻结 Encoder 在**已经不是 MiniCPM 输入分布**的 \(X^{0}\) 上计算 \(X^{16}\)。定理 A 的「同一条残差流」对输入分布不再成立。

**推论 E1（冻结边界）。** B0/B1 二者择一：

| 策略 | 做法 | 选用 |
| --- | --- | --- |
| **freeze_tied（默认）** | \(E\) 随 Encoder 冻；logits 仍走 MiniCPM head | 保守；Decoder 被压在原 logit 空间，利于恢复 |
| **untie_train_head（LP-FT）** | \(E_{\mathrm{in}}\) 冻，\(E_{\mathrm{out}}=E\) 的克隆可训 | Kumar et al. 2022：先探头再全微调，减少特征扭曲 |
| train_tied（禁止） | Encoder 冻、tied \(E\) 仍训 | 定理 E 泄漏 |

ULMFiT（Howard & Ruder 2018）是「从顶向下解冻」：B0 新模块 → B1 解冻 Decoder → B2 解冻 Encoder+\(E\)。LP-FT 是「先把头（或新接口）训到能用冻结特征，再动骨干」。YOCO 的新接口是 cross-attn + \(W_K,W_V\)，不是随机初始化的分类头——gate 从 0 升，B0 开始时前向仍是定理 A。两种文献都支持 **先动新接口、再动 reader、最后动 writer**，不支持 B0/B1 训练 tied \(E\)。

---

## 4. 冻结边界（B0 / B1 / B2）

`scripts/param_budget.py` 的 `FREEZE_BOUNDARIES`（C1/C2 只改 Encoder FFN，不改谁可训）：

| 子阶段 | token（默认） | 冻结 | 可训练 | detach | tied \(E\) | gate |
| --- | ---: | --- | --- | --- | --- | --- |
| **B0** | 8B | Encoder、Decoder 骨干、tied \(E\) | cross-attn、\(W_K/W_V\)、新 LN、gate | 是 | freeze_tied | 0→0.3 |
| **B1** | 27B | Encoder、tied \(E\) | Decoder self-attn + MoE、cross-attn、\(W_K/W_V\) | 是 | freeze_tied | →1 |
| **B2** | 15B | — | 全部（含 \(E\)） | **否** | train_tied | 1 |

B0 的 token 与已有 5–10B gate 爬坡对齐（此处取 8B，gate 只升到 0.3，避免在冻结骨干上把 \(g\) 拉到 1 而导致 cross-attn 过拟合冻结特征——这是 LP-FT「随机头逼骨干改特征」的对称失败：这里头是好的，新分支是随机的）。

B1 解冻 reader。Writer 仍是 MiniCPM-16（C2 精确，C1 virtual-group 近似）。

B2 ≥ 10B：Encoder 专家（C2 才刚出现）需要负载熵；write/read 需要共同适应。质量不稳加长 B2，而不是加长 B0。

**不要做：** 贪心逐层加层（2 层→冻→再加 2 层）。LLM 上没有稳定省算力的证据，结尾仍要联合，还破坏 16/24 切点。

---

## 5. Virtual-group 冻结 ⇒ 近似定理 A（C1）；延迟上采样 ⇒ 精确定理 A（C2）

**命题 F.** Phase A 对某栈做 virtual-group 上采样后，转换瞬间 \(f_{\mathrm{MoE}}=f_{\mathrm{dense}}\)（每片一份副本被 top-\(k\) 命中）。若该栈的专家与路由随后冻结，等式在数值上保持。因此 C1 在 B0/B1 的 Encoder ≈ MiniCPM 第 1–16 层。

C2 更强：B0/B1 **根本不上采样 Encoder**，Encoder **就是** MiniCPM 第 1–16 层。定理 A 对 Encoder 精确成立，直到 B2 边界。

Decoder 在 Phase A 上采样（两变体相同）：B0 冻 Decoder 骨干 ⇒ Decoder 也保持 virtual-group 等式，直到 B1 解冻。B1 是 MoE 恢复段；B2 才让 writer 参与。

**Hash-MoE.** C1 的 Encoder 已冻，Encoder 上的哈希路由是多余的。C2 的 Encoder 在 B2 才变成 MoE，Hash-MoE bootstrap 只放在 B2 开头若干步。Decoder Hash-MoE 仍在 B1 有用。

---

## 6. 命题 G — 延迟 Encoder MoE（C2）严格更省，且理论更干净

每层 MiniCPM dense SwiGLU \(3d\cdot5760=39.81\mathrm{M}\)。中间档 Encoder 激活专家 \(7\times14.16\mathrm{M}=99.12\mathrm{M}\)。16 层：

\[
1.06\mathrm{B}=N_{\mathrm{enc}}^{\mathrm{dense}}<N_{\mathrm{enc}}^{\mathrm{MoE,act}}=2.01\mathrm{B}.
\]

C2 的 \(T_0,T_1\) 前向更便宜；\(T_2\) 与 C1 同为 MoE 联合。故同一 split 下 \(F_{\mathrm{C2}}<F_{\mathrm{C1}}\)。脚本：1,044 vs 1,090 H100-h（77% vs 81%）。

C2 还避免「冻结的专家副本占着 4.50B Encoder 权重却不特化」：B0/B1 只存 16× dense FFN（加注意力 ≈1.06B），B2 才展开成 18 专家/层。

**C2 的 Phase A 修订：** 只对 **Decoder** 做 virtual-group；Encoder 保持 dense。B2 开始时对 Encoder 做同样的 virtual-group（0 token，离线），然后进入短联合。首层 dense 惯例仍可在这次上采样时采用（Enc routed 17→19 补回 12.05B，预算篇 §9）。

C1 是「Phase A 已经两栈都 upcycled」时的回退，不是默认。

---

## 7. FLOPs 账本（与 `--staged` 对齐）

中间档、emb 计一次、50B token 信封、40% MFU：

| 做法 | H100-h | vs 联合 |
| --- | ---: | ---: |
| 两栈一起训 | 1,354 | 100% |
| 冻 Encoder、训 Decoder（全程；write/read 不共同适应） | 1,035 | 76% |
| 同上，Encoder 为 dense（C2 的 B1 极限） | 969 | 72% |
| 只训新模块 | 779 | 58% |
| **C1 解冻课程 8+27+15B** | **1,090** | **81%** |
| **C2 延迟 Encoder MoE 8+27+15B** | **1,044** | **77%** |
| 独立 50B+50B+20B 拼接 | 2,054 | 152% |
| Encoder 当 LM 25B 再联合 25B | 916 | 68%（质量赌博） |

独立拼接更贵的原因不变：两栈各付 50B 的 6NT，再加拼接恢复；定理 A 的表示对齐被扔掉。50B token × \(d\) × 2 bytes 的 Encoder 隐状态缓存 ≈ 230 TB，不能靠「存 Encoder 输出」逃掉前向。

---

## 8. 优化器与激活显存（可能比 FLOPs 更值）

记账：bf16 权重 2 B/参；可训练另加 bf16 梯度 2 B + Adam \(m,v\) fp32 共 8 B ⇒ 可训练 12 B/参，冻结 2 B/参。不含激活、不含 fp32 master。Cache 投影计入存储（12.06B 而非 12.05B）。

| 阶段 | 变体 | 可训练 | 冻结 | Adam \(m,v\) | 权重+梯度+Adam | vs B2 Adam |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | C1 | 0.52B | 11.54B | 4.16 GB | 29.32 GB | 4% |
| B0 | **C2** | 0.52B | **8.10B** | 4.16 GB | **22.44 GB** | 4% |
| B1 | C1 | 7.27B | 4.78B | 58.18 GB | 96.84 GB | **60%** |
| B1 | **C2** | 7.27B | **1.34B** | 58.18 GB | **89.96 GB** | **60%** |
| B2 | 同 | 12.06B | 0 | 96.46 GB | 144.68 GB | 100% |

B1 的 Adam 状态是联合的 60%（Decoder 总参 / 全体总参，含 cache 投影）。C2 还少存 Encoder 专家权重：B1 冻结参数 1.34B vs C1 的 4.78B。

**激活：** detach 之后 Encoder 16 层不必保留给反传。层数模型：留下 \(24/40=60\%\)，约省 40% 激活显存。C2 的 Encoder 层内 FFN 也更窄（dense 39.81M vs 激活 MoE 99M），前向峰值更低。

Muon 只在 2D 矩阵上存一份动量，B1/B2 的优化器差距会略小于 Adam 的 8 B/参，方向不变。ZeRO 仍会切分冻结参数；C2 的冻结 Encoder 更小，切完也更小。

这就是 §15.3 把解冻课程排在 FP8 前面的原因：它同时砍反向 FLOPs、Adam 状态、激活。

---

## 9. Token 切分敏感性

信封固定 50B。脚本 `--curriculum`：

| split | C1 vs 联合 | C2 vs 联合 | 合法 |
| --- | ---: | ---: | --- |
| **默认 8+27+15** | 81% | **77%** | 是 |
| 短 B2 5+35+10 | 79% | 75% | 是（B2 贴着 10B 下限） |
| 长 B2 10+20+20 | 82% | 79% | 是（质量优先） |
| 短 B0 5+25+20 | 84% | 81% | 是 |
| 跳过 B0 0+35+15 | 84% | 80% | 是（少了新模块热身） |
| **B2=0（8+42+0）** | 73% | 69% | **否** |

所有合法切分在 C1/C2 下都 ≤85% 联合。B2=0 最便宜，但 Encoder 从不对 Decoder 的 query 改写记忆——PDSA「无写入时信号」直接打在这条路上。默认不采用短于 10B 的 B2。

---

## 10. 与 Phase C / D 的复合

Phase C 第 1 步：冻主干，只训 Lightning Indexer，目标是 **层内** indexer 分布对齐稠密注意力（KL/MSE），不是穿过 YOCO cache 的 CE。

- Encoder indexer 的监督在 Encoder 层内，不需要 CE 反传到 Encoder，也**不必**为了训 indexer 而撤掉 B0/B1 的 detach；C 发生在 B2 **之后**，那时 detach 已经关掉过一轮。
- Decoder indexer 同理，监督在 cross-attn / CSA 层内。
- C 的 FLOPs 形态与 B0 相同（新模块 + Decoder 激活反传 + Encoder 只前向），但 \(N_{\mathrm{new}}\) 是 indexer 而不是 24 层 cross-attn，更小。
- **不要**把 C 的「冻主干」做成永远冻 Encoder：M1 的 Encoder CSA 在长上下文（Phase D）仍可能需要轻微适应。C 对齐完再小步联训。

Phase D 长上下文：默认两栈都解冻、小 LR。若显存不够，可对 Encoder 再冻一段（C1 式 freeze-enc），但 128K 起 cross-attn 已是瓶颈（预算篇 §6），冻 writer 等于放弃写侧适应，只当权宜。

Phase F/G 的分域专家蒸馏与这套预训练课程正交。

---

## 11. 失败模式（实现前先避开）

| 失败 | 来源 | 避免 |
| --- | --- | --- |
| 独立拼接更贵、切点漂移 | 两栈当两个 LM | 只用 B0/B1/B2；定理 A |
| Encoder 反传偷偷回来 | 没 detach，或 \(W_K\) 在 detach 前 | 定理 D 的代码顺序 |
| 冻结 Encoder 但 \(X^0\) 在漂 | 训 tied \(E\) | 定理 E：freeze_tied 或解绑 head |
| Encoder 专家占坑不特化 | C1 过早 upcycle 再冻 | **默认 C2** |
| write/read 上限卡住 | B2=0 或 B2≪10B | 默认 15B；不稳加长 B2 |
| B0 上 \(g\to1\) 过拟合冻结特征 | 新分支随机、骨干冻 | B0 只到 \(g=0.3\) |
| 把 early-exit 算进课程 FLOPs | 训练没有 early-exit | 架构篇定理 C；课程用全长 6NT |
| Encoder Hash-MoE 放在 B0 | C1 已冻 / C2 还不是 MoE | Hash-MoE 跟 MoE 解冻走 |
| 以为省掉 Encoder 前向 | CE 在 Decoder 顶 | 前向必跑；省的是反向与显存 |

理论**不能**排除的：B2 太短导致 Encoder 负载坍塌、解冻瞬间 loss 尖峰、12B 恢复不到 MiniCPM 95%。那些是实验；尖峰就回退该子阶段、降 LR，不要改回独立 LM。

---

## 12. Claim ledger

`python3 scripts/param_budget.py --verify` 在中间档 22 条之外增加：

| Claim | 结果 |
| --- | --- |
| C2 同 split 下严格少于 C1 | PASS（77% < 81%） |
| C2 ≤80% 联合 50B | PASS（77%） |
| Encoder dense 激活 FFN < MoE 激活 FFN | PASS（1.06B < 2.01B） |
| B0/B1 detach，B2 不 detach | PASS |
| B0/B1 tied \(E\) = freeze_tied | PASS |
| B1 Adam ≈ Decoder/总参 ~60% | PASS |
| detach 保留 24/40 层激活 | PASS |
| 所有合法 50B split ≤85% 联合 | PASS（C1 max 84%；C2 max 81%） |
| B2=0 更便宜但非法 | PASS |
| cache 投影是 B0 新模块 | PASS（10.62M） |
| 默认 B2 ≥10B | PASS（15B） |

C1 ≤85% 仍在中间档账本里（81%）。

---

## 13. 对计划的修订（本 PR）

1. §4.0：冻结边界表（detach、tied \(E\)、\(W_K/W_V\)）；**默认 C2**；C1 为已两栈 upcycle 的回退。
2. Phase A：C2 **只上采样 Decoder**；Encoder 上采样挪到 B2 边界。
3. §15.3 杠杆 #5：写明 FLOPs ~77–81%、B1 Adam 60%、激活 ~40%。
4. 不写训练代码骨架（按用户要求，理论先闭环）。

复算：

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --staged --curriculum
python3 -m unittest tests.test_param_budget
```
