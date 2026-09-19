# CAT-YOKO 解冻课程理论验证

> 与前两篇分工：[`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) 核中间档参数 / FLOPs / KV；[`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md) 核因果与 M1/M2/M3；**这篇核 Phase B 的可训练子集**——冻结边界、梯度在 cache 处截断、untied embedding、**C1 定稿配方**、优化器/激活显存、token 切分。延迟 Encoder MoE 只作敏感性对照。
> 规格仍是中间档：16/26，≈12.25B / 2.03B-in / 4.33B-out。底座 MiniCPM5-2B（Llama GQA，untied）。不引入新架构，不把两栈拆成两个独立 LM。
> 可执行断言：`python3 scripts/param_budget.py --verify`（含课程 claim）、`--staged`、`--curriculum`、`--fp8`、`--nvfp4`；`python3 -m unittest tests.test_param_budget`。
> 理论能证明的是 **FLOPs / 显存 / 梯度流 / 与定理 A 的兼容**；不能证明 12B 上采样后的质量。质量仍走 L0→L1 实验，B2 不够就加长 B2。
> NVFP4 墙钟（不改 6NT；Phase B 定稿 **C1+NVFP4 = 571 H100-h**）见 [`NVFP4_THEORY.md`](NVFP4_THEORY.md)。Hopper/Ada 回退 C1+FP8 = 729 见 [`FP8_THEORY.md`](FP8_THEORY.md)。

---

## 0. 结论（先看这个）

「先分开训两个模型再焊在一起」**更贵且破坏定理 A**。能省的是同一套切开权重上的 **B0 → B1 → B2 解冻课程**。冻结边界按 **C1** 定稿；Phase B **墙钟按 C1+NVFP4 定稿（571 H100-h）**。

| 做法 | 50B tok H100-h | vs 联合 | 角色 |
| --- | ---: | ---: | --- |
| 两栈一起训（联合 bf16） | 1,325 | 100% | 对照 |
| C1：两栈都先 MoE，B0/B1 冻 Encoder | 1,046 | 79% | 操作数账 |
| **C1+FP8（Hopper/Ada 回退）** | 729 | 55% | 无 Blackwell |
| **C1+NVFP4** | **571** | **43%** | **定稿墙钟** |
| 独立 50B+50B+20B 拼接 | 1,940 | **146%（更贵）** | 不要做 |

还必须钉死的几条：

1. **梯度在全局 cache 处 `.detach()`。** Encoder 无权重梯度、无激活梯度；前向省不掉（YOCO 的 CE 在 Decoder 顶）。这是定理 D。
2. **输入 embedding 必须随 Encoder 冻结；永远不要在 Encoder 冻结时训输入表。** MiniCPM5 **untied**：\(E_{\mathrm{in}}\) 与 lm_head 是两份矩阵。B0 冻输入表 **和** head；B1 **可以训 lm_head**（输入表仍冻）。这是定理 E。禁止 `train_embed_while_frozen`。
3. **\(W_K,W_V\) 是新模块**，挂在 `encoder.detach()` **之后**，B0 就可训。它们不在已发布的 12.25B 栈合计里（\(2\cdot d\cdot 256=1.05\mathrm{M}\)）。
4. **配方是 C1。** Phase A 对两栈都做 virtual-group MoE，B0/B1 冻 Encoder：一次离线手术、从 token 0 就是发布的 12.25B 中间档、B2 只解冻不改结构。冻结期间 Encoder ≈ MiniCPM5 1–16（命题 F）。B0/B1 付 MoE Encoder 前向（1.76B vs dense 0.75B），专家副本到 B2 才特化——所以 B2 默认 15B。推迟 Encoder 上采样（原 C2）不进配方，只留在 `--curriculum` 敏感性表。
5. **C1 相对联合约省 21% FLOPs；Adam 状态在 B1 只有联合的 62%；detach 丢掉 Encoder 激活约 38%（留下 26/42 ≈ 62%）。** 塞进更少卡时，显存杠杆可能比 FLOPs 杠杆更有用。
6. **B2 不能为 0**（write/read 永不共同适应，PDSA 已警告）。默认 B2 = 15B ≥ 10B。质量不稳加长 B2，不要改回两个独立 LM。
7. Phase C「冻主干、只训 indexer」叠在 B2 **之后**；indexer 对齐是层内 KL，不是穿过 cache 的 LM 反传。
8. **墙钟敲死为 C1+NVFP4。** 不必须 bf16 的线性 GEMM 走 NVFP4，B0 student / L0 / indexer 保持 bf16。发布 **571 H100-h（联合 bf16 的 43%）**。见 [`NVFP4_THEORY.md`](NVFP4_THEORY.md)。

Claim ledger：中间档 22 条 + 课程 12 条 + FP8 回退 13 条 + NVFP4 16 条，`--verify` **63/63** 通过。

---

## 1. 形式化：一条计算图，三个可训练子集

长度 \(n\) 的因果 LM，记号与架构篇 §1 相同：

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E_{\mathrm{in}},\quad
X^{\ell}=\mathrm{SelfDec}^{\ell}(X^{\ell-1}),\ \ell=1..L_e,
\]

\[
(\hat K,\hat V)=\big(\mathrm{sg}(X^{L_e})W_K,\;\mathrm{sg}(X^{L_e})W_V\big)
\quad\text{（B0/B1；}\mathrm{sg}=\texttt{.detach()}\text{）},
\]

\[
X^{\ell}=\mathrm{CrossDec}^{\ell}(X^{\ell-1},\hat K,\hat V),\ \ell=L_e+1..L_e+L_d,
\quad
z=X^{L_e+L_d}W_{\mathrm{head}}^{\top}/\alpha,\quad \alpha=1.
\]

B2 去掉 \(\mathrm{sg}\)，\(E_{\mathrm{in}}\) 与 \(W_{\mathrm{head}}\) 都解冻。MiniCPM5 没有 μP，\(s_{\mathrm{emb}}=1\)，\(\alpha=1\)。

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

\(N_{\mathrm{new}}=L_d A_{\times}+2d\cdot 256\)（26 层 cross-attn Q/O + cache 投影 \(1.05\mathrm{M}\)）。定稿 C1 的 \(N_{\mathrm{enc}}\) 是 MoE 激活 \(1.76\mathrm{B}\)（不含 emb）。

课程

\[
F_{\mathrm{C}}=F_{\mathrm{new\text{-}mod}}(T_0)+F_{\mathrm{freeze\text{-}enc}}(T_1)+F_{\mathrm{joint}}(T_2),\quad T_0+T_1+T_2=50\mathrm{B}.
\]

C1 全程用 MoE Encoder（B0/B1 冻结，B2 解冻）。

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

## 3. 定理 E — 冻结 Encoder 时输入表泄漏（untied MiniCPM5）

MiniCPM5 **不共享** 输入表与输出头：\(E_{\mathrm{in}},W_{\mathrm{head}}\in\mathbb{R}^{V\times d}\) 是两份矩阵。

\[
X^{0}=s_{\mathrm{emb}}\,\mathrm{onehot}(x)E_{\mathrm{in}},\qquad
z=X^{42}W_{\mathrm{head}}^{\top}/\alpha,\quad \alpha=1,\ s_{\mathrm{emb}}=1.
\]

**定理 E.** 若 \(\theta_{\mathrm{enc}}\) 冻结而 \(E_{\mathrm{in}}\) 可训练，则 \(\partial L/\partial E_{\mathrm{in}}\) 经 \(X^{0}\) 非零（只要 Decoder 的 CE 还能通过 cache 依赖前缀，或 B2 之前某条没 detach 的路径），于是 \(X^{0}\) 漂移。即便梯度只通过 **head** 回来：在 YOCO 里 head 不直接更新 \(E_{\mathrm{in}}\)（untied），但 **任何** 在 Encoder 冻结期间对 \(E_{\mathrm{in}}\) 的更新都会让冻结栈吃到已经不是 MiniCPM5 输入分布的 \(X^{0}\)。定理 A 的「同一条残差流」对输入分布不再成立。

**推论 E1（冻结边界）。** 输入表在 B0/B1 **必须冻**。head 与输入表解绑，所以：

| 策略 | 做法 | 选用 |
| --- | --- | --- |
| **freeze_embed_and_head（B0 默认）** | \(E_{\mathrm{in}}\) 与 \(W_{\mathrm{head}}\) 都冻 | 新模块热身；logits 仍走 MiniCPM5 head |
| **freeze_embed_train_head（B1 默认）** | \(E_{\mathrm{in}}\) 冻，\(W_{\mathrm{head}}\) **可训** | LP-FT：先把头适配冻结特征，再动骨干 |
| train_embed_while_frozen（禁止） | Encoder 冻、仍训 \(E_{\mathrm{in}}\) | 定理 E 泄漏 |

B0 冻 head：新模块随机、gate 还在 0→0.3，不要同时改分类面。B1 解冻 Decoder 时 **可以** 训 lm_head——head 已不与输入表绑定，不会把 \(X^{0}\) 拽偏。B2 才解冻 \(E_{\mathrm{in}}\)。**永远不要在 Encoder 冻结时训练输入 embedding。**

ULMFiT（Howard & Ruder 2018）是「从顶向下解冻」：B0 新模块 → B1 解冻 Decoder + head → B2 解冻 Encoder+\(E_{\mathrm{in}}\)。LP-FT 是「先把头（或新接口）训到能用冻结特征，再动骨干」。YOCO 的新接口是 cross-attn + \(W_K,W_V\)，不是随机初始化的分类头——gate 从 0 升，B0 开始时前向仍是定理 A。两种文献都支持 **先动新接口、再动 reader（含 head）、最后动 writer + 输入表**，不支持 B0/B1 训练 \(E_{\mathrm{in}}\)。

---

## 4. 冻结边界（B0 / B1 / B2）

`scripts/param_budget.py` 的 `FREEZE_BOUNDARIES`（C1 定稿）：

| 子阶段 | token（默认） | 冻结 | 可训练 | detach | embedding | gate |
| --- | ---: | --- | --- | --- | --- | --- |
| **B0** | 8B | Encoder、Decoder 骨干、\(E_{\mathrm{in}}\)、lm_head | cross-attn、\(W_K/W_V\)、新 LN、gate | 是 | freeze_embed_and_head | 0→0.3 |
| **B1** | 27B | Encoder、\(E_{\mathrm{in}}\) | Decoder self-attn + MoE、cross-attn、\(W_K/W_V\)、**lm_head** | 是 | freeze_embed_train_head | →1 |
| **B2** | 15B | — | 全部（含 \(E_{\mathrm{in}}\) 与 head） | **否** | train_untied | 1 |

B0 的 token 与已有 5–10B gate 爬坡对齐（此处取 8B，gate 只升到 0.3，避免在冻结骨干上把 \(g\) 拉到 1 而导致 cross-attn 过拟合冻结特征——这是 LP-FT「随机头逼骨干改特征」的对称失败：这里头是好的，新分支是随机的）。

B1 解冻 reader **和** lm_head。Writer 与输入表是 **冻结的 virtual-group MoE Encoder**（≈ MiniCPM5-16，命题 F）。

B2 ≥ 10B：Encoder 专家从这一刻才开始特化，需要负载熵；write/read 需要共同适应。质量不稳加长 B2，而不是加长 B0。

**不要做：** 贪心逐层加层（2 层→冻→再加 2 层）。LLM 上没有稳定省算力的证据，结尾仍要联合，还破坏 16/26 切点。

### C1 时间线（定稿）

```
Phase A  离线：16/26 切开，gate=0，Encoder+Decoder 都 virtual-group MoE，
         加 cross-attn 与 W_K/W_V。此后总参就是 12.25B。
B0  8B   Encoder / Decoder 骨干 / E_in / lm_head 全冻。只训新模块。
         cache = X^16.detach() @ W_{K,V}。g: 0→0.3。
         Encoder 前向 = 冻结的 MoE 副本 ≈ MiniCPM5 1–16（命题 F）。
B1  27B  解冻 Decoder + lm_head。Encoder + E_in 仍冻，仍 detach。g→1。
         Reader 学会用冻结记忆；writer 与输入表不改。
B2  15B  去掉 detach，解冻 Encoder + E_in，LR 更小。
         Encoder 专家从这里才开始特化；write/read 共同适应。
```

C1 的 MoE 初始化与冻结兼容：virtual-group 保证转换瞬间等于 dense；冻住路由和专家，这个等式一直维持到 B2。Encoder 上的 Hash-MoE 在 B0/B1 是多余的（路由已经冻死），放到 B2 解冻时再用。

---

## 5. Virtual-group 冻结 ⇒ 近似定理 A

**命题 F.** Phase A 对某栈做 virtual-group 上采样后，转换瞬间 \(f_{\mathrm{MoE}}=f_{\mathrm{dense}}\)（每片一份副本被 top-\(k\) 命中）。若该栈的专家与路由随后冻结，等式在数值上保持。因此 C1 在 B0/B1 的 Encoder ≈ MiniCPM5 第 1–16 层。

Decoder 在 Phase A 同样上采样：B0 冻 Decoder 骨干 ⇒ Decoder 也保持 virtual-group 等式，直到 B1 解冻。B1 是 MoE 恢复段；B2 才让 writer 参与。

**Hash-MoE.** Encoder 已冻，Encoder 上的哈希路由是多余的。放到 B2 解冻开头若干步。Decoder Hash-MoE 仍在 B1 有用。

---

## 6. 敏感性：推迟 Encoder 上采样（不定稿）

脚本仍计算「B0/B1 Encoder 保持 MiniCPM5 dense」的 FLOPs，作为对照，**不进配方**。

每层 dense SwiGLU \(37.75\mathrm{M} < 8\times 12.58\mathrm{M}=100.7\mathrm{M}\) 激活专家，故 16 层 \(0.75\mathrm{B}<1.76\mathrm{B}\)。同一 8+27+15B split 下该对照是 997 H100-h（75%），比 C1 再省约 4pp，但要在 B2 做第二次 Encoder virtual-group。定稿不采用。

---

## 7. FLOPs 账本（与 `--staged` 对齐）

中间档、emb 与 head 各计一次、50B token 信封、40% MFU：

| 做法 | H100-h | vs 联合 |
| --- | ---: | ---: |
| 两栈一起训 | 1,325 | 100% |
| 冻 Encoder、训 Decoder（全程；write/read 不共同适应） | 987 | 75% |
| 只训新模块 | 720 | 54% |
| **C1 解冻课程 8+27+15B（定稿）** | **1,046** | **79%** |
| 独立 50B+50B+20B 拼接 | 1,940 | 146% |
| Encoder 当 LM 25B 再联合 25B | 874 | 66%（质量赌博） |

独立拼接更贵的原因不变：两栈各付 50B 的 6NT，再加拼接恢复；定理 A 的表示对齐被扔掉。50B token × \(d\) × 2 bytes 的 Encoder 隐状态缓存 ≈ 205 TB，不能靠「存 Encoder 输出」逃掉前向。

---

## 8. 优化器与激活显存（可能比 FLOPs 更值）

记账：bf16 权重 2 B/参；可训练另加 bf16 梯度 2 B + Adam \(m,v\) fp32 共 8 B ⇒ 可训练 12 B/参，冻结 2 B/参。不含激活、不含 fp32 master。Cache 投影计入存储（12.248B 而非发布的 12.25B 栈合计）。数字为 **C1**。

| 阶段 | 可训练 | 冻结 | Adam \(m,v\) | 权重+梯度+Adam | vs B2 Adam |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.22B | 12.03B | 1.75 GB | 26.69 GB | 2% |
| B1 | 7.60B | 4.65B | 60.82 GB | 100.52 GB | **62%** |
| B2 | 12.25B | 0 | 97.99 GB | 146.98 GB | 100% |

B1 的 Adam 状态是联合的 62%（Decoder 总参 + lm_head + cache 投影 / 全体总参）。

**激活：** detach 之后 Encoder 16 层不必保留给反传。层数模型：留下 \(26/42\approx 62\%\)，约省 38% 激活显存。不是旧账本的 \(24/40=60\%\)。

Muon 只在 2D 矩阵上存一份动量，B1/B2 的优化器差距会略小于 Adam 的 8 B/参，方向不变。

这就是 §15.3 把解冻课程排在 NVFP4 前面的原因：它同时砍反向 FLOPs、Adam 状态、激活。发布积是 **C1+NVFP4 = 571**（C1 bf16 1,046 只作操作数账；C1+FP8 729 为 Hopper/Ada 回退），不替代冻结。

---

## 9. Token 切分敏感性

信封固定 50B。脚本 `--curriculum`（C1 列是定稿；dense-enc 列是敏感性）：

| split | C1 vs 联合 | 合法 |
| --- | ---: | --- |
| **定稿 8+27+15** | **79%** | 是 |
| 短 B2 5+35+10 | 78% | 是（B2 贴着 10B 下限） |
| 长 B2 10+20+20 | 81% | 是（质量优先） |
| 短 B0 5+25+20 | 83% | 是 |
| 跳过 B0 0+35+15 | 82% | 是（少了新模块热身） |
| **B2=0（8+42+0）** | 71% | **否** |

所有合法切分在 C1 下都 ≤85% 联合（最大 83%）。B2=0 最便宜，但 Encoder 从不对 Decoder 的 query 改写记忆——PDSA「无写入时信号」直接打在这条路上。定稿不采用短于 10B 的 B2。

---

## 10. 与 Phase C / D 的复合

Phase C 第 1 步（发布默认、无 KDA）：冻主干，只训 Lightning Indexer，目标是 **层内** indexer 分布对齐稠密注意力（KL/MSE），不是穿过 YOCO cache 的 CE。

若 `use_kda`：第 0 步是 `C-kda`（只点亮多数 KDA 层，CSA/HCA 仍滑窗），indexer/CSA/HCA 靠后；C 合计仍 25e9。

- Encoder indexer 的监督在 Encoder 层内，不需要 CE 反传到 Encoder，也**不必**为了训 indexer 而撤掉 B0/B1 的 detach；C 发生在 B2 **之后**，那时 detach 已经关掉过一轮。
- Decoder indexer 同理，监督在 cross-attn / CSA 层内。
- C 的 FLOPs 形态与 B0 相同（新模块 + Decoder 激活反传 + Encoder 只前向），但 \(N_{\mathrm{new}}\) 是 indexer 而不是 26 层 cross-attn，更小。
- **不要**把 C 的「冻主干」做成永远冻 Encoder：M1 的 Encoder CSA 在长上下文（Phase D）仍可能需要轻微适应。C 对齐完再小步联训。

Phase D 长上下文：默认两栈都解冻、小 LR。若显存不够，可对 Encoder 再冻一段（C1 式 freeze-enc），但 128K 起 cross-attn 已是瓶颈（预算篇 §6），冻 writer 等于放弃写侧适应，只当权宜。

Phase F/G 的分域专家蒸馏与这套预训练课程正交。

---

## 11. 失败模式（实现前先避开）

| 失败 | 来源 | 避免 |
| --- | --- | --- |
| 独立拼接更贵、切点漂移 | 两栈当两个 LM | 只用 B0/B1/B2；定理 A |
| Encoder 反传偷偷回来 | 没 detach，或 \(W_K\) 在 detach 前 | 定理 D 的代码顺序 |
| 冻结 Encoder 但 \(X^0\) 在漂 | 训输入 \(E_{\mathrm{in}}\) | 定理 E：B0/B1 冻输入表；B1 只许训 head |
| Encoder 专家占坑不特化 | Phase A 两栈都 MoE 再冻 Encoder | **接受**：B2≥15B 才让 Encoder 专家特化；不要因此改回独立 LM |
| write/read 上限卡住 | B2=0 或 B2≪10B | 默认 15B；不稳加长 B2 |
| B0 上 \(g\to1\) 过拟合冻结特征 | 新分支随机、骨干冻 | B0 只到 \(g=0.3\) |
| 把 early-exit 算进课程 FLOPs | 训练没有 early-exit | 架构篇定理 C；课程用全长 6NT |
| Encoder Hash-MoE 放在 B0 | Encoder 已冻 | Hash-MoE 跟 Encoder 解冻走（B2） |
| 以为省掉 Encoder 前向 | CE 在 Decoder 顶 | 前向必跑；省的是反向与显存 |

理论**不能**排除的：B2 太短导致 Encoder 负载坍塌、解冻瞬间 loss 尖峰、12B 恢复不到 MiniCPM5 95%。那些是实验；尖峰就回退该子阶段、降 LR，不要改回独立 LM。

---

## 12. Claim ledger

`python3 scripts/param_budget.py --verify` 在中间档 22 条之外增加：

| Claim | 结果 |
| --- | --- |
| **定稿配方是 C1** | PASS |
| C1 课程 ≤85% 联合 | PASS（79%；中间档账本） |
| B0/B1 detach，B2 不 detach | PASS |
| B0/B1 冻输入表；B0 冻 head、B1 训 head | PASS |
| B1 Adam ≈ Decoder/总参 ~62% | PASS |
| detach 保留 26/42 层激活 | PASS（62%） |
| 所有合法 50B split ≤85% 联合 | PASS（C1 max 83%） |
| B2=0 更便宜但非法 | PASS |
| cache 投影是 B0 新模块 | PASS（1.05M） |
| 默认 B2 ≥10B | PASS（15B） |
| Encoder dense FFN < MoE FFN（敏感性，不定稿） | PASS（0.75B < 1.76B） |
| delayed-enc 对照更便宜（敏感性，不定稿） | PASS（75% < 79%） |

C1 ≤85% 仍在中间档账本里（79%）。FP8 回退 13 条见 [`FP8_THEORY.md`](FP8_THEORY.md)；NVFP4 16 条见 [`NVFP4_THEORY.md`](NVFP4_THEORY.md)；合计 `--verify` **63/63**。墙钟定稿是 **C1+NVFP4 = 571**。

---

## 13. 对计划的修订（本 PR）

1. §4.0：**C1 定稿**（两栈都先 MoE，B0/B1 冻 Encoder；detach、冻 \(E_{\mathrm{in}}\)、B1 可训 lm_head、\(W_K/W_V\)）。推迟 Encoder 上采样不进配方。
2. Phase A：对 Encoder、Decoder **各自** virtual-group。切分 **16/26**。
3. §15.3 杠杆 #5：C1 约省 21% Phase B FLOPs、B1 Adam 62%、激活 ~38%。
4. 不写训练代码骨架（按用户要求，理论先闭环）。
5. **墙钟敲死为 C1+NVFP4**：571 H100-h（[`NVFP4_THEORY.md`](NVFP4_THEORY.md)）。联合 bf16、C1+FP8 与全阶段 2.0× 不定稿。

复算：

```bash
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
