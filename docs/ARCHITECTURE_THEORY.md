# CAT-YOKO 架构理论验证

> 与 [`THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md) 分工：那篇核**中间档参数 / FLOPs / KV**；这篇核**数据流、因果、感受野、全局 cache 接口**。规格仍是中间档：Encoder 16L / Decoder 24L，CSA+HCA+8K 滑窗，主目标 128K–256K。
> 可执行断言：`python3 scripts/arch_verify.py --verify` 与 `python3 -m unittest tests.test_arch_verify`。
> 理论能证明的是**自洽、因果、复杂度与信息流**；不能证明 12B 上采样后的质量。质量仍走 L0→L1 实验。

---

## 0. 结论（先看这个）

架构在因果与残差切分上是自洽的，但计划文本把**三件不同的机制**收成了一句话「CSA/HCA 压缩全局 cache」。必须拆开，否则实现会在错误的位置做 top-k。

| 机制 | 发生位置 | query 是谁 | 作用 |
| --- | --- | --- | --- |
| **M1** Encoder CSA/HCA | self-decoder 每一层 | **当前输入 token** | 更便宜地写每 token 表示 |
| **M2** 全局 cache 再池化 | Encoder 顶 → \(\hat K,\hat V\) | 无（write-first） | 把 \(N\) 槽压成 \(N/m\)；**可选** |
| **M3** Decoder 侧 indexer / 校准回退 | cross-attn | **生成 query** | 真正的 query-aware 检索 |

YOCO 的全局性来自 **M3 读全部（或所选）cache 槽**，不来自 encoder 滑窗感受野。16 层 × 8K 窗堆叠感受野只有 **131072**，盖不住 256K；这**不妨碍** 256K 检索，因为 token 0 的 cache 槽仍然在，decoder 可以直接读。

还必须钉死的几条：

1. **gate≈0 时，16/24 切分 ≡ 原 40 层残差流**（差一个尚未打开的 cross-attn）。这是 MiniCPM 热启合法的理由。
2. **CSA/HCA 压缩支路必须排除自身块**，否则块内未来 token 泄漏；**滑窗补这个洞**。充分条件：\(n_{\mathrm{win}}\ge m'\)。8K ≥ 128，成立。
3. **Early-exit 只属于推理 prefill**。训练是两条栈都跑全部 token；不要把 34% 激活份额误当成训练 FLOPs。
4. **Decoder 自注意力在训练时也看到全长序列。** 「生成序列通常不长」只描述推理。长上下文训练里 decoder self-attn 必须是滑窗/KDA，全局混合交给 cross-attn。
5. Encoder 16 层做不了「3 层 bootstrap + CSA:HCA=1:1」。冻结为 **2× sliding + 7 CSA + 7 HCA**（对齐 V4-Flash 的 2 层滑窗 bootstrap）。

Claim ledger：14/14 通过。

---

## 1. 形式化数据流

长度 \(n\) 的序列，隐状态 \(X^{0}\in\mathbb{R}^{n\times d}\) 为 embedding（含 MiniCPM `scale_emb`）。

\[
\begin{aligned}
X^{\ell} &= \mathrm{SelfDec}^{\ell}(X^{\ell-1}), && \ell=1,\ldots,L_e=16,\\
(\hat K,\hat V) &= \big(X^{L_e}W_K,\; X^{L_e}W_V\big), && \hat K,\hat V\in\mathbb{R}^{n\times d_{\mathrm{kv}}},\\
X^{\ell} &= \mathrm{CrossDec}^{\ell}(X^{\ell-1},\hat K,\hat V), && \ell=L_e+1,\ldots,L_e+L_d.
\end{aligned}
\]

每个 CrossDec 块：

\[
\begin{aligned}
U &= X + \mathrm{SelfAttn}_{\le n_{\mathrm{win}}}(X),\\
Z &= U + g\cdot \mathrm{CrossAttn}(Q=UW_Q,\; K=\hat K,\; V=\hat V),\\
X' &= Z + \mathrm{MoE}(Z).
\end{aligned}
\]

\(g\in[0,1]\) 是 Phase A/B 的 cross-attn gate。因果 mask：self-attn 与 cross-attn 的 query \(t\) 都不得看见位置 \(>t\)。

外部行为是因果 LM：logits 来自 \(X^{L_e+L_d}\) 的 tied head。这就是 YOCO 说的「看起来像 decoder-only，只缓存一次」。

---

## 2. 定理 A — gate=0 时切分等价于原残差流

**定理 A.** 若 (i) \(\mathrm{SelfDec}\) 与 \(\mathrm{CrossDec}\) 的 self-attn+FFN 就是原 MiniCPM 第 \(1..16\) 与第 \(17..40\) 层，(ii) \(g=0\)，(iii) 尚未 MoE 化、尚未把注意力换成 CSA，则对任意输入，CAT-YOKO 的 \(X^{40}\) 等于原 MiniCPM 的 \(X^{40}\)。

**证明.** \(g=0\) 时 CrossDec 退化为 SelfAttn+FFN。残差输入为 \(X^{16}\)，正是 MiniCPM 第 17 层的输入。按层归纳即得。□

推论：

- Phase A「cross-attn 旁路」不是启发式，是**切点合法**的充分条件。
- MoE 上采样、CSA 替换各自打破等价，必须分步（Phase A 切分 → B 恢复 → C 稀疏化），与 §13 去风险阶梯一致。
- Decoder 第 1 层（全局第 17 层）的残差已经是 \(X^{16}\)；打开 \(g\) 之后，cross-attn 读的也是 \(X^{16}\) 的投影。所以 \(g\) 从 0 升到 1 是在**同一份记忆上增加「按位置混合前缀」的通路**，不是突然接入一个外来 encoder。

打开 \(g\) 之后多出来的能力：self-attn 若是滑窗，位置 \(t\) 的残差只含局部；cross-attn 允许 \(t\) 混合 \(\hat K_{1:t}\)，即 **\(X^{16}\) 的因果前缀**。这正是 YOCO 用一层记忆换全局感受野的机制。

---

## 3. 三件机制：不要把 M1 当成 M3

计划 §2.0 写「self-decoder 的 CSA/HCA 产出更紧凑的全局 cache」。字面会让人以为 Encoder 做完 CSA，cache 槽数已经是 \(N/m\)。**不是。**

### M1 — Encoder 层内 CSA/HCA

每一层把该层的 KV 沿序列压缩，再让**当前层的 query（输入 token）** 去选。输出仍是 \(n\) 条隐状态。顶层 \(X^{L_e}\) 仍是 \(n\times d\)。  
对 decoder 而言，这是 **write-time 语境化**：写 cache 时用的 query 是「这段输入自己」，不是用户稍后的问题。PDSA 的「无写入时信号」直接打在这里——Encoder CSA 的 top-k 救不了「事后才被问到」的针。

### M2 — 全局 cache 再池化（可选）

\[
\hat K' = \mathrm{Pool}_m(\hat K)\in\mathbb{R}^{(n/m)\times d_{\mathrm{kv}}}.
\]

这才真正减少 **YOCO 那一份** cache 的槽数。预算篇 §7 的 0.29 GB（\(m=4\)）和 0.14 GB（÷8）属于 M2，不属于 M1。M2 是 write-first，没有 decoder query。

### M3 — Decoder 侧选择（真正的检索）

生成 query \(q_t\) 对 \(\hat K_{1:t}\)（或 M2 后的槽）做 dense / top-k / 校准回退。这才是 query-aware。128K–256K 上未压缩 cross-attn 压过 decoder MLP（预算篇 §6），所以 **M3 或 M2 至少要有一个**；推荐 M3（可回退），M2 作省显存的加项。

**架构约束.** 实现上 cross-attn 默认先做成因果 dense（合法、对应定理 A 的连续放松），再在 Phase C 加 M3。不要把 Encoder Lightning Indexer 的 top-k 权重复用到 decoder query 上——两个 query 分布不同。

---

## 4. 定理 B — CSA/HCA 因果 + 滑窗补洞

记压缩率 \(m\)，query 位置 \(t\)（0-index）。自身块号 \(b=\lfloor t/m\rfloor\)。块 \(s\) 覆盖 token \([sm,(s+1)m)\)。

**压缩支路可见性（V4 §2.3.1）**

\[
\mathcal{S}_{\mathrm{comp}}(t)=\{s:s < \lfloor t/m\rfloor\}.
\]

**引理 B1（压缩支路不泄漏未来）.** 若 \(s\in\mathcal{S}_{\mathrm{comp}}(t)\)，则块 \(s\) 内最大下标 \((s+1)m-1 \le m\lfloor t/m\rfloor-1 \le t-1\)。CSA 的重叠支路 \(C^b\) 用的是上一块，最新 token 不更晚。故压缩支路因果。

**引理 B2（自身块有洞）.** 自身块内 \(\le t\) 的 token（含自己）不在 \(\mathcal{S}_{\mathrm{comp}}\)。自身块内 \(>t\) 的 token 若被纳入压缩键，会泄漏未来——这就是必须整块排除的原因。

**滑窗**

\[
\mathcal{W}(t)=\{p:\max(0,t-n_{\mathrm{win}}+1)\le p\le t\}.
\]

**定理 B（补洞）.** 若 \(n_{\mathrm{win}}\ge m\)，则自身块内所有 \(\le t\) 的 token 都在 \(\mathcal{W}(t)\) 里。因此

\[
\mathcal{V}(t)=\mathcal{W}(t)\;\cup\;\bigcup_{s\in\mathcal{S}_{\mathrm{comp}}(t)}[sm,(s+1)m)
\]

因果，且自身块过去对 query 可见。

CAT-YOKO：\(n_{\mathrm{win}}=8192\)，\(m=4\)，\(m'=128\)，\(8192\ge 128\)。HCA 同样适用。若有人把窗降到 \(<128\) 还保留 HCA \(m'=128\)，补洞失败——这是消融 `n_win∈{2K,4K,8K}` 的**下限不是 0、至少 128** 的理由。2K/4K/8K 都安全。

**Indexer.** top-\(k\) 只允许从 \(\mathcal{S}_{\mathrm{comp}}(t)\) 里删，不许加。稀疏化掉点是召回问题，不是因果问题；所以 Phase C 要先做 indexer 对齐。

有限 \(n=64\) 上的穷举见 `scripts/arch_verify.py`（CSA/HCA/YOCO 零泄漏，top-k 是子集）。

---

## 5. 定理 C — YOCO cross-attn 因果与 early-exit

**因果.** query \(t\) 可读 cache 槽 \(\{0,\ldots,t\}\)。Encoder 是因果的，\(\hat K_j\) 只依赖 token \(\le j\)，故 \(j\le t\) 时 \(\hat K_j\) 不含未来。

**推理 prefill early-exit.** 提示 \(x_{0:n-1}\) 的全局 cache 在 \(X^{L_e}_{0:n-1}\) 算完后就闭合。要出第一个生成 token，只需 **decoder 在位置 \(n-1\) 上跑一次**（读已写好的 cache），不必对 \(n\) 个提示位置跑 \(L_d\) 层。这是 YOCO Table 1 的 early-exit：省的是 \(O(L_d n)\) 的 decoder prefill，不是 decoder 的最后一位。

**训练没有 early-exit.** 每个位置都有 CE，decoder 必须在全部 \(t=0..n-1\) 上前向。训练 FLOPs 用 \(N_{\mathrm{fwd}}^{\mathrm{act}}\)（预算篇），不能用 34% 的 encoder 份额去估 Phase B。

---

## 6. 感受野：全局从哪来

| 路径 | 感受野 |
| --- | --- |
| Encoder 纯滑窗堆叠 | \(L_e\cdot n_{\mathrm{win}}=16\cdot 8192=131072\) |
| Encoder CSA/HCA | 写时即可看压缩长程（有损） |
| Decoder 自注意力 | \(n_{\mathrm{win}}=8192\)（**不能**跨 YOCO 切点与 encoder 窗相加） |
| Decoder cross-attn | 因果前缀长度 \(t+1\)（或 M3 的 \(k\)） |

YOCO 原文 self-decoder 就是滑窗或 retention：encoder **不必**全局混合。token 0 的槽仍在，decoder 在 \(t=256\mathrm{K}\) 也能读它。256K 主目标**不依赖** encoder RF ≥ 256K。

CSA/HCA 在 encoder 里的真正作用是：(a) 训练/prefill 时降低 encoder 注意力二次项（只在 \(n\gg 8K\) 时发生，预算篇 §5）；(b) 可选地让写入的表示带一点长程语境。它不是 256K 能检索的充分条件——充分条件是 **M3 的全局读**。

Decoder 窗不能和 encoder 窗叠感受野：切点之后 decoder 的 self-attn 只看 decoder 残差流的局部，远距必须走 \(\hat K\)。

---

## 7. 共享全局 cache 的信息瓶颈

Decoder-only：第 \(\ell\) 层的键来自该层隐状态，共 \(L_d\) 份彼此不同的记忆。  
YOCO：\(L_d\) 层读**同一份** \(\hat K(X^{L_e})\)，每层只有 \(W_Q^{\ell}\) 不同。

记忆张量从 \(O(L_d n d_{\mathrm{kv}})\) 降到 \(O(n d_{\mathrm{kv}})\)，因子 \(L_d=24\)。这是显存定理，也是表达力赌注：多层不能再「改写」键，只能换查询。YOCO 在 1M needle 上近似满分，说明对检索型任务这份记忆够用；它**不**证明多层推理/改写（agent 状态、版本化）够用——那是计划 §14 Tier 3 可训练 lifecycle 的动机，不是 CSA 的动机。

M2 再沿序列池化，瓶颈从 \(n\) 降到 \(n/m\)。PDSA 已经量过 write-first 选择会丢「无写入时信号」的针，所以 M2 默认不要比 \(m=4\) 更狠；÷8 只作 stretch。

---

## 8. 16/24 非对称

YOCO 原文 \(L/2+L/2\)。CAT-YOKO 取 16/24，对应「更轻的 writer、更重的 reader」：

- Prefill / 长输入绑定 writer（2.29B，34% 计划口径激活）。
- 生成期把算力留给 reader（4.49B）在记忆上做更多层的 \(W_Q\) 查询。
- MiniCPM 前 16 层作 writer、后 24 层作 reader，与定理 A 的切点一致（前低层特征、后高层处理）。

理论**不唯一决定** 16/24。12/28 会更便宜 prefill、更弱记忆；20/20 更接近 YOCO 原文。这是 §7 消融项，不是错误。16/24 与「输入轻、输出重」的中间档叙事一致即可。

---

## 9. Encoder 层调度：为什么是 2/7/7 而不是「前 3 层 bootstrap」

16 层要同时满足：(i) 前几层接近 MiniCPM 稠密注意力以便热启，(ii) CSA:HCA=1:1。

- 2 层 sliding bootstrap → 余 14 层 → **7 CSA + 7 HCA**。
- 3 层 bootstrap → 余 13 层 → **无法 1:1**。

V4-Flash 的 `compress_ratios` 以两个 `0`（sliding）开头；V4-Pro 文本是 2× HCA bootstrap。对 MiniCPM 上采样，**sliding bootstrap 更近原 MHA**（只是加窗），优于一上来 HCA \(m'=128\)。

**冻结：** Encoder `layer_types` =

```
sliding, sliding, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca, csa, hca
```

Decoder self-attn：全部 sliding（或以后的 KDA 混合），**不要**默认在 decoder self-attn 上再铺 CSA——全局已经在 cross-attn。Decoder 上的 CSA 是额外复杂度，且与 YOCO「decoder self 用高效局部注意力」重复。

计划原文「前 3 层 sliding/HCA」与「2× HCA bootstrap」并列表述，已在本节冻结为上面这一条。

---

## 10. MoE、Hash-MoE、μP（架构侧）

- 非对称激活来自**两个栈的层数与 top-k**，不是同一层上按 token 改 k。与定理 A 兼容：FFN 被换成 MoE 后等价被打破，所以 virtual-group 上采样要单独恢复（Phase B）。
- 每栈首层 dense：路由在第 1 层不稳定（DeepSeekMoE 惯例），与注意力 bootstrap 同构——都是「先别上最险的归纳偏置」。预算篇：首层 dense 后 Enc routed 17→19 补回 12.05B。
- Hash-MoE：冻结 `token_id→expert_id`，无学习路由，不进入注意力因果。Encoder 的 Hash-MoE 放到 B2 解冻之后（课程篇 §5）。
- μP：残差乘子锁 \(1.4/\sqrt{40}\)（预算篇 §9）。与 YOCO 切分正交：切的是层，不是尺度。

mHC / Muon / MTP 不进入本篇因果核验。mHC 是残差谱约束，关了不影响 YOCO/CSA 合法性。

---

## 11. KDA 与 PDSA 放在哪

KDA 是**每层一个固定大小的循环状态**，与 YOCO 的 KV 槽正交。它不能提供 M3：状态对所有未来 query 是同一份 gist，正是 PDSA 说的 write-first。计划 §2.5 的定位（效率 + 粗覆盖，不是中段精确检索）与本篇一致。

若在 Encoder 上做 3:1 KDA:CSA，是在 **M1** 里用线性层换 CSA 层，减少的是 encoder 二次项与每层 KV；**不**减少 YOCO 全局 cache 槽数（那是 M2），也**不**给出 decoder 侧 query-aware（那是 M3）。混合比例仍用 §7 消融，本篇只禁止「上了 KDA 就等于保中段」。

PDSA 校准回退是 M3 的门控，不是第四种注意力。阈值必须在目标长度上校准（计划 §14），短上下文校准会让回退永不触发——与预算篇「128K 起 cross-attn 才成为瓶颈」同一长度尺度。

---

## 12. 从理论推出的失败模式（实现前先避开）

| 失败 | 来源 | 避免 |
| --- | --- | --- |
| 块内未来泄漏 | 压缩支路看见自身块 | 定理 B：排除自身块 + \(n_{\mathrm{win}}\ge m'\) |
| 256K 检索为 0 | 误以为必须靠 encoder RF | 保证 M3 能读到 token 0 的槽 |
| Decode 被 cross-attn 打死 | 只有 M1、没有 M2/M3 | 128K 起开 M3 或 \(m=4\) 的 M2 |
| 训练 FLOPs 少算 3× | 把 early-exit 用到训练 | 训练两条栈全长 |
| Decoder 长训 \(O(n^2)\) | 以为「生成短」所以 decoder 全注意力 | 训练时 decoder self 也是窗 |
| 热启崩 | \(g=1\) 或一上来 CSA top-k | 定理 A：\(g=0\)；Phase C 再稀疏 |
| 16 层 1:1 排不下 | 3 层 bootstrap | 冻结 2/7/7 |
| Indexer 复用错 query | 把 M1 的 indexer 接到 M3 | 两套 query，两套（或后加的）indexer |

理论**不能**排除的：MoE 负载坍塌、indexer 对不齐、lost-in-the-middle（位置偏置，IN2/FILM + RoPE/NoPE）、12B 恢复不到 MiniCPM 95%。那些是实验。

---

## 13. Claim ledger

`python3 scripts/arch_verify.py --verify`：

| Claim | 结果 |
| --- | --- |
| 16+24=40 | PASS |
| \(n_{\mathrm{win}}\ge m,m'\) | PASS（8192≥128） |
| Encoder 调度 2 sliding + 7 CSA + 7 HCA | PASS |
| 窗-only encoder RF = 131072 < 256K | PASS（且不需要 ≥256K） |
| 共享 cache = 1 writer × 24 readers | PASS |
| Early-exit 仅推理 | PASS（条文） |
| n=64 CSA/HCA/YOCO 因果、自身块不走压缩、窗补洞、top-k 子集 | PASS |
| M1 ≠ M2 ≠ M3 | PASS（条文；测试检查槽数与 query 集不同） |

---

## 14. 对计划的修订（本 PR）

1. §2.0：CSA/HCA 不自动压 YOCO 槽数；全局性来自 cross-attn。
2. §2.3：Encoder `layer_types` 冻结为 2/7/7 sliding→CSA/HCA；Decoder self 默认全滑窗。
3. §2.0「生成序列通常不长」：标明仅推理；训练 decoder self 仍是窗。
4. 与预算篇衔接：128K–256K 必须有 M2 或 M3；默认推 M3。

---

## 15. 分训再合并（算力，不是因果）

两栈**独立当 LM 训再拼接**会破坏定理 A 的表示对齐，且 50B+50B+拼接比联合 50B **更贵**（约 1.5×）。能省的是同一套切开权重上的**解冻课程**。

定理 D：B0/B1 在 \(X^{16}\) 上 `.detach()`，Encoder 无权重/激活梯度；\(W_K,W_V\) 挂在 detach 之后，是新模块。  
定理 E：tied \(E\) 必须随 Encoder 冻结（或解绑只训 head），否则冻结 Encoder 的输入分布会漂。

定稿 **C1**：Phase A 两栈都 MoE，B0/B1 冻 Encoder（virtual-group 冻结 ⇒ Encoder ≈ MiniCPM-16；B2 才让 Encoder 专家特化）。

数字：C1 约 **81%** 联合 50B；独立拼接 **152%**。B1 Adam 状态约联合的 **60%**。冻结边界见 [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md) 与训练计划 §4.0。FP8 再把 C1 墙钟从 1,090 收到 **761 H100-h**（[`FP8_THEORY.md`](FP8_THEORY.md)）。`python3 scripts/param_budget.py --staged --curriculum --fp8`。

复算：

```bash
python3 scripts/arch_verify.py --verify
python3 -m unittest tests.test_arch_verify
python3 scripts/param_budget.py --verify                 # 中间档 + 解冻课程 + FP8 账本
python3 scripts/param_budget.py --staged --curriculum --fp8
```
