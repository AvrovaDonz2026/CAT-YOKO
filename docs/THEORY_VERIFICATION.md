# CAT-YOKO 中间档理论验证

> 规格冻结为训练计划默认档：`CAT-YOKO-12B`，Encoder 激活 ≈2.03B / 输入 token，Decoder 激活 ≈4.33B / 输出 token。
> 底座是 Apache-2.0 **MiniCPM5-2B**（Llama GQA），不是 MiniCPM-2B。
> 本文只做**可复算的理论核对**（参数、FLOPs、KV、复杂度、无 μP），不引入新架构。
> 数字源：`python3 scripts/param_budget.py --full`；断言：`python3 scripts/param_budget.py --verify` 与 `python3 -m unittest tests.test_param_budget`。
> 架构（因果、切分等价、M1/M2/M3）见 [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)。
> 解冻课程（C1 定稿：冻结边界、定理 D/E）见 [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)。
> NVFP4：Phase B 墙钟定稿 **C1+NVFP4 = 571 H100-h**（定理 G；不改 6NT）见 [`NVFP4_THEORY.md`](NVFP4_THEORY.md)。Hopper/Ada 回退 C1+FP8 = 729 见 [`FP8_THEORY.md`](FP8_THEORY.md)。

---

## 0. 结论（先看这个）

中间档在**计划采用的记账口径**下是自洽的：

| 量 | 计划宣称 | 精确值 | 判定 |
| --- | ---: | ---: | --- |
| 总参数 | ≈12B | **12.25B** | 通过 |
| Encoder 激活 / 输入 token | ≈2.03B | **2.03B** | 通过 |
| Decoder 激活 / 输出 token | ≈4.33B | **4.33B** | 通过 |
| 50B tok 训练 | ~1,325 H100-h | **1,325 H100-h** | 通过 |
| 1M KV（decoder-only MHA / GQA-2 / MLA / YOCO+MLA / ÷8） | 344 / 43 / 48 / 1.15 / 0.14 GB | **344.06 / 43.01 / 48.38 / 1.15 / 0.14 GB** | 通过 |

计划文本相对旧 MiniCPM-2B 账本需要改、但不改规格的几处：

1. **专家稀疏度**是 **38.1%（8/21）/ 52.4%（11/21）**，不是 7/18、9/18，也不是 37%/53%。
2. **注意力占总参 ~5.0%**（GQA 16/2）；MoE 仍占 **90.6%**。不要把 MiniCPM-2B 占位注意力的 13%、或 24B 时期的 ~7% 搬过来。KDA 仍然参数中性。
3. 示意图应是 **≈2.03B / ≈4.33B**，层数 **16+26=42**，不是 16/24 与 2.3/4.5。
4. §16 的「YOCO+MLA+÷8 → 0.14 GB」把**全局 cache 再沿序列压 8×**当成已实现；CSA 的 \(m=4\) 只给出 **÷4 → 0.29 GB**。÷8 是额外设计选择，不是 CSA 的必然推论。

理论层面两个会改变训练/推理优先级的结论：

- **8K 未压缩滑窗主导长上下文注意力 FLOPs**。`index_topk` 从 256 改到 512 只动约 3% 的 score+AV 量；CSA/HCA 相对 dense 的节省只在 \(n \gg 8\mathrm{K}\) 出现。Phase B 在 4K 上保持稠密/滑窗，在复杂度上是对的——短上下文上 CSA concat 甚至略贵于 dense。
- **未压缩的 decoder cross-attn 在 128K 起压过 decoder MLP**（128K 约 3.2×，256K 约 6.5×）。中间档要兑现「128K–256K 可用」，全局 cache 必须走压缩或 query-aware 选择，而不能把 encoder 顶层 raw KV 原样交给 26 层 cross-attn。

---

## 1. 符号与底座

底座取 `openbmb/MiniCPM5-2B` 的 `config.json`（Llama GQA，Apache-2.0）：

\[
d=2048,\quad V=130560,\quad n_h=16,\quad n_{\mathrm{kv}}=2,\quad d_h=128,\quad L_0=42,\quad d_{\mathrm{ff}}^{\mathrm{dense}}=6144.
\]

**没有 MiniCPM-2B 的 μP。** `scale_emb=1`，`logit_scale=d/\mathrm{dim\_model\_base}=2048/2048=1`，残差乘子 \(1\)（不是 \(1.4/\sqrt{L}\)）。

YOCO 拆分：Encoder（self-decoder）\(L_e=16\)，Decoder（cross-decoder）\(L_d=26\)。DeepSeekMoE 单专家

\[
E = 3\,d\,d_{\mathrm{moe}} = 3\cdot 2048\cdot 2048 = 12.58\mathrm{M}.
\]

Untied embedding + LM head

\[
\mathrm{Emb} = Vd = 130560\cdot 2048 = 0.2674\mathrm{B},\qquad
\mathrm{LM_{head}} = Vd = 0.2674\mathrm{B}.
\]

**发布自注意力**是 MiniCPM5 GQA（16 Q / 2 KV），不是 MiniCPM-2B 的 1.25×MHA 占位：

\[
A_{\mathrm{self}} = 2d^{2} + 2d\cdot(n_{\mathrm{kv}}d_h) = 9.437\mathrm{M}.
\]

**Cross-attn** 只计 Q/O（K/V 来自 YOCO cache）：\(A_{\times}=2d^{2}=8.389\mathrm{M}\)。Cache 投影 \(W_K,W_V\)：\(2\cdot d\cdot 256=1.05\mathrm{M}\)，不进已发布的 12.25B 栈合计。

每 token、每栈的激活（计划口径：输入表计入 Encoder，untied head 计入 Decoder）：

\[
\begin{aligned}
N_{\mathrm{enc}}^{\mathrm{act}} &= \mathrm{Emb} + L_e A_{\mathrm{self}} + L_e(n_s^{e}+k^{e})E,\\
N_{\mathrm{dec}}^{\mathrm{act}} &= \mathrm{LM_{head}} + L_d(A_{\mathrm{self}}+A_{\times}) + L_d(n_s^{d}+k^{d})E.
\end{aligned}
\]

一次完整前向（输入表与 head 各算一次）：

\[
N_{\mathrm{fwd}}^{\mathrm{act}} = \mathrm{Emb} + \mathrm{LM_{head}} + (N_{\mathrm{enc}}^{\mathrm{act}}-\mathrm{Emb}) + (N_{\mathrm{dec}}^{\mathrm{act}}-\mathrm{LM_{head}}).
\]

untied 之后计划口径 \(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}}\) **等于** \(N_{\mathrm{fwd}}^{\mathrm{act}}\)（两张表各计一次，不再出现 tied 时的 emb×2）。下文仍分列，数字相同。

---

## 2. 882 专家槽守恒：三档同总参、同专家数、不同激活

三档都把 **882 个专家实例** 分给 16+26 层，**专家布局固定 1+20**，只改 top-\(k\)：

| 档位 | Enc \(n_s+N_r\), top-\(k\) | Dec \(n_s+N_r\), top-\(k\) | 专家实例 | 总参 |
| --- | --- | --- | ---: | ---: |
| 省算力 | 1+20, \(k=4\) | 1+20, \(k=6\) | \(16\cdot21+26\cdot21=882\) | 12.25B |
| **中间（默认）** | **1+20, \(k=7\)** | **1+20, \(k=10\)** | **882** | **12.25B** |
| 近-dense | 1+20, \(k=12\) | 1+20, \(k=16\) | 882 | 12.25B |

这是中间档「能在不改总参的前提下调训练成本」的代数原因：总参由 routed 数决定，训练 FLOPs 由 top-\(k\) 决定。砍总参不会砍训练成本。三档专家数相同，所以总参相同不是巧合——只旋 top-\(k\)。

精确激活与稀疏度（专家稀疏度 \(=(n_s+k)/(n_s+N_r)\)）：

| 档位 | Enc 激活 | Dec 激活 | \(N_{\mathrm{fwd}}^{\mathrm{act}}\) | 计划口径 \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) | 稀疏度 enc/dec | 50B tok H100-h（计划口径） |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 省算力 | 1.43B | 3.02B | 4.45B | 4.45B | 23.8% / 33.3% | 926 |
| **中间** | **2.03B** | **4.33B** | **6.36B** | **6.36B** | **38.1% / 52.4%** | **1,325** |
| 近-dense | 3.04B | 6.29B | 9.33B | 9.33B | 61.9% / 81.0% | 1,943 |

计划表里的 ~930 / ~1,325 / ~1,940 是对 1.43+3.02、2.03+4.33、3.04+6.29 的四舍五入；精确值如上。中间档稀疏度应写成 **38% / 52%**（8/21、11/21）。

固定部分（三档相同，GQA 注意力）：

| 块 | 参数 |
| --- | ---: |
| Embedding（untied 输入表） | 0.267B |
| LM head（untied） | 0.267B |
| Enc 自注意力 \(16\times 9.437\mathrm{M}\) | 0.151B |
| Dec 自注意力 \(26\times 9.437\mathrm{M}\) | 0.245B |
| Dec cross-attn \(26\times 8.389\mathrm{M}\) | 0.218B |
| MoE 专家 \(882\times 12.58\mathrm{M}\) | 11.10B |
| **合计** | **12.25B** |

注意力合计 \(0.151+0.245+0.218=0.614\mathrm{B}\)，占 12.25B 的 **5.0%**；MoE **90.6%**。GQA 把注意力从 MiniCPM-2B 占位的 ~13% 压下来；KDA 替换部分 CSA/HCA 层仍然只动注意力块（量级 \(10^{-1}\mathrm{B}\)），**不必为了加 KDA 而改 12B 总参目标**。

---

## 3. 中间档逐项展开

Encoder（16 层，1 shared + 20 routed，top-\(k=7\)）：

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 16\cdot 8\cdot 12.58\mathrm{M} = 1.611\mathrm{B},\\
N_{\mathrm{enc}}^{\mathrm{act}} &= 0.267 + 0.151 + 1.611 = 2.029\mathrm{B},\\
\mathrm{stack_{enc}} &= 0.151 + 16\cdot 21\cdot 12.58\mathrm{M} = 4.379\mathrm{B}.
\end{aligned}
\]

Decoder（26 层，1 shared + 20 routed，top-\(k=10\)，含 cross-attn Q/O）：

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 26\cdot 11\cdot 12.58\mathrm{M} = 3.599\mathrm{B},\\
N_{\mathrm{dec}}^{\mathrm{act}} &= 0.267 + 0.245 + 0.218 + 3.599 = 4.330\mathrm{B},\\
\mathrm{stack_{dec}} &= 0.245 + 0.218 + 26\cdot 21\cdot 12.58\mathrm{M} = 7.334\mathrm{B}.
\end{aligned}
\]

\[
N_{\mathrm{total}} = 0.267 + 0.267 + 4.379 + 7.334 = 12.247\mathrm{B},\quad
N_{\mathrm{fwd}}^{\mathrm{act}} = 6.359\mathrm{B}.
\]

与计划「≈12B / ≈2.03B-in / ≈4.33B-out」一致。

---

## 4. 训练算力（6NT）与 Chinchilla 位置

Kaplan / Hoffmann 口径：训练 FLOPs \(\approx 6 N_{\mathrm{act}} T\)（前向 2NT + 反向 4NT，忽略注意力二次项）。H100 有效算力取计划 §15.1 的 \(4.0\times 10^{14}\) FLOPS（约 40% MFU）。

中间档、\(T=50\mathrm{B}\)：

| \(N_{\mathrm{act}}\) 口径 | \(N\) | FLOPs | H100-h | A100-h |
| --- | ---: | ---: | ---: | ---: |
| 计划（enc+dec） | 6.36B | \(1.91\times 10^{21}\) | **1,325** | 4,239 |
| 完整前向（与计划相同，untied） | 6.36B | \(1.91\times 10^{21}\) | 1,325 | 4,239 |
| 仅 Encoder（prefill / early-exit） | 2.03B | \(6.09\times 10^{20}\) | 423 | 1,353 |

1,325 对上计划的 ~1,325。那是 **联合 bf16** 的对照，不是 Phase B 发布墙钟。发布值是 **C1+NVFP4 = 571 H100-h**（联合 bf16 的 43%）。C1+FP8 = 729 是 Hopper/Ada 回退。NVFP4 不改这张 6NT 表。见 [`NVFP4_THEORY.md`](NVFP4_THEORY.md)。

**这不是 Chinchilla 预训练。** Hoffmann 最优大约 \(20 N\) tokens（dense）。50B / 6.36B ≈ **7.9 token / 激活参数**，属于上采样恢复 + 继续训练，不是从零训 12B。把 50–150B 写成 Phase B 恢复预算是对的；把它理解成「12B 已经训充分」则过满。

二次注意力项在 4K 主训练上相对 6NT 很小（§6），在 Phase D 的 32K–128K 上不可忽略——这正是 CSA/HCA 的训练期价值，而不只是推理期价值。

---

## 5. 注意力复杂度：8K 滑窗才是大头

CSA / HCA 每个 query 实际参加 core attention 的 KV 条数（concat = 压缩分支 + 未压缩滑窗，因果）：

\[
\begin{aligned}
k_{\mathrm{CSA}}(n) &= \min(k_{\mathrm{index}}, \lfloor n/m\rfloor) + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{HCA}}(n) &= \lfloor n/m'\rfloor + \min(n_{\mathrm{win}}, n),\\
k_{\mathrm{dense}}(n) &= n,
\end{aligned}
\]

其中 \(m=4\)，\(m'=128\)，\(n_{\mathrm{win}}=8192\)，\(k_{\mathrm{index}}=256\)（计划 256–512 的下沿）。

| \(n\) | dense | CSA | HCA | CSA/dense |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 4,096 | 4,352 | 4,128 | **1.06** |
| 8,192 | 8,192 | 8,448 | 8,256 | **1.03** |
| 32,768 | 32,768 | 8,448 | 8,448 | 0.26 |
| 131,072 | 131,072 | 8,448 | 9,216 | 0.064 |
| 262,144 | 262,144 | 8,448 | 10,240 | 0.032 |
| 1,048,576 | 1,048,576 | 8,448 | 16,384 | 0.008 |

要点：

1. **\(n \le 8\mathrm{K}\) 时 CSA/HCA 不省 FLOPs**（concat 甚至略多于 dense）。Phase B 在 4K 上先跑稠密/滑窗、Phase C 再稀疏化，与复杂度曲线一致，不是单纯的「训练稳定性」偏好。
2. **过了 8K 之后，滑窗 8192 条几乎钉死 CSA 的 \(k\)**。`index_topk=256` 相对 512：\(k=8448\) vs \(8704\)，约 **3%**。长上下文消融应优先扫 `n_win ∈ {2K,4K,8K}`，而不是把 `index_topk` 当成主成本旋钮。
3. HCA 在 1M 上 \(k=\lfloor 10^6/128\rfloor+8192=16384\)，仍远小于 dense，但已经是 CSA 的约 2×；HCA 层不宜作为 1M 上的多数层。

Score+AV 用 \(4\,d\,n\,k\,L\)（两段 matmul × 2 flop/MAC）。中间档一次长度为 \(n\) 的前向，MLP 项 \(2 N_{\mathrm{fwd}}^{\mathrm{act}} n\) 对比注意力二次项：

| \(n\) | MLP 前向 | 42L dense | Encoder 16L CSA/HCA 交错 | Enc + Dec 26L 滑窗 | Dec 26L **未压缩** cross-attn |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | \(5.2\times10^{13}\) | \(5.8\times10^{12}\) | \(2.3\times10^{12}\) | \(5.9\times10^{12}\) | \(3.6\times10^{12}\) |
| 32K | \(4.2\times10^{14}\) | \(3.7\times10^{14}\) | \(3.6\times10^{13}\) | \(9.3\times10^{13}\) | \(2.3\times10^{14}\) |
| 128K | \(1.7\times10^{15}\) | \(5.9\times10^{15}\) | \(1.5\times10^{14}\) | \(3.8\times10^{14}\) | \(3.7\times10^{15}\) |
| 256K | \(3.3\times10^{15}\) | \(2.4\times10^{16}\) | \(3.2\times10^{14}\) | \(7.8\times10^{14}\) | \(1.5\times10^{16}\) |

读表：

- 4K：注意力二次项 ≪ MLP，6NT 是好近似。
- 32K：42L dense 已经和 MLP 同量级；encoder 压缩后仍可比 MLP 小一个数量级。
- **128K 起，未压缩 cross-attn 单独就超过整网 MLP**。YOCO 若把全局 cache 存成 raw 顶层 KV，decoder 会把 CSA/HCA 在 encoder 侧省下的东西在 cross-attn 里花回去。

---

## 6. Prefill early-exit 与 decode 瓶颈

YOCO（arXiv 2405.05254）：prefill 只需跑完 self-decoder 即可写出全局 \(\hat K,\hat V\)，cross-decoder 在生成第一个 token 时才进入。中间档：

| | 值 |
| --- | --- |
| 层数比 \(L_e/(L_e+L_d)\) | 16/42 = **38%** |
| 激活比 \(N_{\mathrm{enc}}^{\mathrm{act}}/(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}})\) | 2.03/6.36 = **32%** |

所以「输入侧更轻」不只是层数减半：中间档的非对称 MoE 再把 prefill 算力压到全模型前向的约三分之一。这与「主交付 128K–256K、超长输入是 encoder 的活」一致。

Decode 一步（decoder MLP 前向 \(2 N_{\mathrm{dec}}^{\mathrm{act}}\) vs 对长度为 \(n\) 的全局 cache 做 cross-attn）：

| \(n\) | dec-MLP | xattn 全量 | 全量 / MLP | xattn CSA \(m=4\) | xattn CSA top-\(k\)+8K 窗 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | \(8.66\times10^9\) | \(6.98\times10^9\) | 0.81× | 0.20× | 0.21× |
| **128K** | \(8.66\times10^9\) | \(2.79\times10^{10}\) | **3.22×** | 0.81× | **0.21×** |
| **256K** | \(8.66\times10^9\) | \(5.58\times10^{10}\) | **6.45×** | 1.61× | **0.21×** |
| 1M | \(8.66\times10^9\) | \(2.23\times10^{11}\) | 25.8× | 6.45× | 0.21× |

**中间档要在 128K–256K 上可用，全局 cache 不能是未压缩 raw KV。** 最低限度：沿序列做 \(m=4\) 压缩（256K 上 cross-attn 仍约 1.6× MLP）。要让 decode 重新由 MLP 主导，需要对全局 cache 做 CSA 式 top-\(k\)（或 PDSA 校准回退），把每 query 的 \(k\) 钉在 \(n_{\mathrm{win}}+k_{\mathrm{index}}\approx 8.4\mathrm{K}\)。这与计划 §14「query-aware 优先于 write-first、校准稀疏回退」同方向，而且是算力约束，不只是召回约束。

---

## 7. KV cache：§16 的数字是对的，假设需要写清

bf16、内容字节（不含 allocator padding）。「1M」= \(10^6\) token（不是 \(2^{20}\)）。GB = \(10^9\) B。

MHA：每层每 token \(2d\) 个元素（K 和 V）。**GQA-2**：2 个 KV 头（\(n_{\mathrm{kv}}=2\)，\(d_h=128\)，每 token \(2\times 256\) 元素）。MLA-576：DeepSeek-V3 式 latent \(k_{\mathrm{lora}}+d_{\mathrm{rope}}=512+64\)。

\[
\mathrm{bytes} = n \cdot L_{\mathrm{cache}} \cdot d_{\mathrm{kv}} \cdot 2.
\]

| 配方 | 8K | 32K | 128K | 256K | 1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| decoder-only MHA 42L | 2.82 | 11.27 | 45.10 | 90.19 | **344.06** |
| decoder-only GQA-2 42L | 0.35 | 1.41 | 5.64 | 11.27 | **43.01** |
| decoder-only MLA-576 42L | 0.40 | 1.59 | 6.34 | 12.68 | **48.38** |
| **YOCO + MLA-576（1 层全局）** | 0.01 | 0.04 | 0.15 | 0.30 | **1.15** |
| YOCO + MLA + 序列÷8 | — | — | 0.02 | 0.04 | **0.14** |
| YOCO + MLA + CSA \(m=4\) | — | — | 0.04 | 0.08 | **0.29** |
| YOCO + MiniCPM5 GQA-2（1 层全局） | 0.01 | 0.03 | 0.13 | 0.27 | 1.02 |
| Decoder 26×8K 窗（MLA） | 0.25 | 0.25 | 0.25 | 0.25 | **0.25** |
| Encoder 16×8K 窗（MLA） | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |

8K/32K/128K/256K 列为 \(8192/32768/131072/262144\)；1M 列为 \(10^6\)（与计划 §16 相同）。

核对计划 §16：

- 344 / 43 / 48 / 1.15 / 0.14 GB：**通过**（小数 GB、\(n=10^6\)、GQA=2 头、MLA=576、42 层）。
- 「+≈0.2 GB 的 8K 滑窗」：Decoder 26 层精确 **0.25 GB**，量级通过。Encoder 自己还有约 0.15 GB 的 8K 窗（YOCO 原文也承认 self-decoder 有常数 cache）。128K–256K 上全局 cache（0.15–0.30 GB）和 8K 窗（0.25+0.15 GB）同量级——再一次说明 **8K 窗不是免费的**。
- **÷8 不是 CSA \(m=4\) 的推论。** \(m=4\) 给出 0.29 GB @ 1M。0.14 GB 需要额外的序列压缩（更重的 HCA 式全局池化、或对全局 cache 再做一层压缩）。应在计划里标成「可选 stretch」，避免 impl 时按 CSA 默认实现却以为已经对上 0.14 GB。

主目标 128K–256K：YOCO+MLA 全局 cache **0.15–0.30 GB**，即使加上两侧 8K 窗也远小于 decoder-only MHA 的 45–90 GB。1M 推理在中等硬件上可行，这一条理论成立。

---

## 8. 发布 GQA vs MiniCPM5 原生 CSA/HCA（敏感性，不改规格）

发布账本就是 MiniCPM5 GQA 16/2，**不再用 1.25×MHA 占位**。DeepSeek-V4-Flash 用 `head_dim=512`、`q_lora_rank=1024`（\(d=4096\)）。不能把这组维数搬到 MiniCPM5 的 \(d=2048\)。敏感性模型用 MiniCPM5 原生 MQA-128（\(d_h=128\)，单 KV 头）+ LoRA-Q 512 + grouped output，按 V4 论文 §2.3 计 CSA/HCA 投影：

| 每层自注意力 | 参数 |
| --- | ---: |
| MiniCPM5 MHA \(4d^2\)（未采用） | 16.78M |
| **发布 GQA 16/2** | **9.437M** |
| CSA MQA-128（含 indexer） | 10.00M |
| HCA MQA-128 | 8.41M |
| CSA/HCA 平均 | 9.20M |

详细模型与发布 GQA **几乎同大**（9.20M vs 9.44M）。若将来冻结为 MQA-128 CSA/HCA：

- 总参落到约 **12.24B**，激活约 **2.03B / 4.32B**（注意力也在激活里）。
- 与 12.25B / 2.03 / 4.33 在两位小数内对齐，不必为 CSA 改 routed。
- 若还想微调激活，应加 top-\(k\) 而不是加 routed。

**在投影维数冻结前，继续用 GQA 16/2 作为中间档规格。** 详细模型只说明：换成 CSA/HCA 不会突然把 12B 撑破。

---

## 9. 首层 dense（Phase A）与「不要 μP」

计划 Phase A：「每栈首层保留 dense」。当前 §3 表按**全部 MoE** 记账。把每栈第 1 层换成 MiniCPM5 dense SwiGLU（\(3\cdot d\cdot 6144=37.75\mathrm{M}\)）：

| | 全 MoE（规格表） | 首层 dense、Nr 不变 |
| --- | ---: | ---: |
| 总参 | 12.25B | **11.79B** |
| Enc 激活 | 2.03B | 1.97B |
| Dec 激活 | 4.33B | 4.23B |

激活仍在 ≈2.0 / ≈4.3 的四舍五入里。总参差 0.45B。补回 ~12.25B 且不改 top-\(k\)：Encoder/Decoder routed **20→21**（其余不变）→ 总参 **12.30B**，激活仍是 1.97 / 4.23。建议在 Phase A 落地时采用「Enc 1+21 / Dec 1+21、首层 dense」，对外仍称中间档。

**MiniCPM5 是 Llama，没有 μP 缩放。**

- `scale_emb=1`，logits 不除以 9，残差乘子是恒等 \(1\)。
- **不要**把 MiniCPM-2B 的 \(1.4/\sqrt{40}\) 或 \(1.4/\sqrt{42}\) 搬过来。拆成 16+26 之后也不要按新栈深重算残差。
- 切的是层，不是尺度。

---

## 10. 信息论约束（PDSA）与中间档的关系

PDSA（Zou & Donz，arXiv 2606.28876）给出一条与参数预算正交、但和 §6 的 decode 瓶颈同方向的约束：

- **无写入时信号**：query-independent 可写性在静态文本上接近随机（AUC 0.63–0.66）。HCA / KDA / CSA 的压缩步都是 write-first，会系统性丢掉「写入时无信号、query 时才被命中」的块。
- 因此 **必须留一条到低压缩 / 原始 KV 的 query 时回退**。§6 显示这条回退在 128K–256K 上也是算力上该走的路：把 cross-attn 的 \(k\) 从 \(n\) 收到 \(n_{\mathrm{win}}+k_{\mathrm{index}}\)，decode 才回到 MLP 主导。
- 中间档把精确检索预算给了 Decoder 的 cross-attn（4.33B 激活、26 层），把长输入压缩预算给了 Encoder（2.03B、16 层）。这与「query-aware 读发生在 decode、write-first 压缩发生在 prefill」一致。不要为了「再省一点 prefill」把 encoder 改成纯 HCA/KDA 而没有 CSA 锚点 + 回退。

线性注意力不能修 lost-in-the-middle（计划 §2.5 已澄清）；中间档也不依赖 KDA 来保中段召回。IN2/FILM + 校准回退才是中段路径。

---

## 11. Claim ledger

对计划已发布数字的机械化核对（`scripts/param_budget.py --verify`，GQA 16/2、全 MoE、中间档）：

| Claim | 结果 |
| --- | --- |
| 总参 ≈12B | PASS（12.25B） |
| Enc 激活 ≈2.03B | PASS（2.03B） |
| Dec 激活 ≈4.33B | PASS（4.33B） |
| Emb / Enc attn / Dec attn / cross ≈ 0.27 / 0.15 / 0.25 / 0.22B | PASS |
| 专家稀疏度 8/21、11/21 | PASS（38.1%/52.4%） |
| 三档共享 882 专家槽 | PASS |
| 50B tok ≈1325 H100-h | PASS（1325） |
| 1M KV 344 / 43 / 48 / 1.15 / 0.14 GB | PASS |
| 26×8K 窗 ≈0.25 GB | PASS（0.25 GB） |
| 无 μP：logit_scale = 1；残差恒等 | PASS |
| freeze-enc ≈75% 联合；独立拼接 146%；C1 课程 79% | PASS（见课程篇） |

解冻课程另 12 条（定稿 C1、detach、untied \(E\)、Adam 62%、合法 split ≤85% 等）与中间档 22 条、FP8 回退 13 条、NVFP4 16 条合计 **`--verify` 63/63**，见 [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)、[`FP8_THEORY.md`](FP8_THEORY.md)、[`NVFP4_THEORY.md`](NVFP4_THEORY.md)。

文本层（不进 `--verify`，已在上文展开）：

| 文本 | 处理 |
| --- | --- |
| MiniCPM-2B / \(d=2304\) / \(V=122753\) / 16/24 | 已改为 MiniCPM5-2B / \(d=2048\) / \(V=130560\) / 16+26 |
| 720 专家槽、1+17 布局 | 已改为 882 槽、两栈都 1+20，只改 top-\(k\) |
| 稀疏度 37%/53% 或 7/18、9/18 | 已改为 38.1%/52.4%（8/21、11/21） |
| 注意力 ~7% 或 ~13% | 已改为 ~5.0%（GQA） |
| 示意图 3B/6B 或 2.3/4.5 | 已改为 2.03B/4.33B |
| 全局 cache ÷8 | 已标为可选；CSA \(m=4\) 对应 ÷4 → 0.29 GB |
| μP logits /9、残差 \(1.4/\sqrt{40}\) | 已改为恒等 1 |
| 首层 dense 未进规格表 | 总参落到 11.79B；用 Enc/Dec \(N_r=21\) 补回 12.30B |

---

## 12. 对实现的约束（从理论读出来的，不是新功能）

1. **规格继续锁中间档**：Enc 16L、1+20、top-\(k=7\)；Dec 26L、1+20、top-\(k=10\)；发布注意力 GQA 16/2。Phase A 若首层 dense，两栈 routed 都调到 21。
2. **全局 cache 必须压缩或可选**。否则 128K–256K decode 被 26 层 cross-attn 主导，YOCO+CSA 的 encoder 收益会被吐回去。
3. **`n_win=8K` 是第一成本旋钮**；`index_topk` 不是。消融按计划 §7 做 {2K,4K,8K}。
4. **不要引入 MiniCPM-2B μP**，也不要按新栈深重算残差。
5. **4K 主训练不要指望 CSA 省算力**；稀疏化放在 Phase C、长上下文放在 Phase D，与 FLOPs 曲线一致。
6. 投影维数冻结后，用 `python3 scripts/param_budget.py --attn csa_mqa64 --full` 重跑；CSA/HCA 与 GQA 几乎同参，不要改层数拆分。
7. **Phase B 按 C1 定稿**（两栈先 MoE、冻 Encoder、detach cache、冻输入表、B1 可训 lm_head、B2≥10B）。不要把两栈当独立 LM 再拼接。
8. **Phase B 墙钟按 C1+NVFP4 定稿**（571 H100-h；不必须 bf16 的线性 GEMM；B0 student / L0 / indexer / embed / LN / router / softmax 高精度）。发布 2.0× vs bf16，不改 6NT，不发布 4×。联合 bf16 只作对照。C1+FP8 729 为 Hopper/Ada 回退。

复算命令：

```bash
python3 scripts/param_budget.py              # 中间档摘要
python3 scripts/param_budget.py --tier all   # 三档对照
python3 scripts/param_budget.py --full       # KV / 复杂度 / 无 μP / Nr 回搜
python3 scripts/param_budget.py --verify     # 规格 + 解冻课程 + FP8 回退 + NVFP4 断言
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 -m unittest tests.test_param_budget
```
