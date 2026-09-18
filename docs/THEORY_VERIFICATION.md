# CAT-YOKO 中间档理论验证

> 规格冻结为训练计划默认档：`CAT-YOKO-12B`，Encoder 激活 ≈2.3B / 输入 token，Decoder 激活 ≈4.5B / 输出 token。
> 本文只做**可复算的理论核对**（参数、FLOPs、KV、复杂度、μP 一致性），不引入新架构。
> 数字源：`python3 scripts/param_budget.py --full`；断言：`python3 scripts/param_budget.py --verify` 与 `python3 -m unittest tests.test_param_budget`。
> 架构（因果、切分等价、M1/M2/M3）见 [`ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)。
> 解冻课程（C1 定稿：冻结边界、定理 D/E）见 [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)。
> FP8：Phase B 墙钟定稿 **C1+FP8 = 761 H100-h**（定理 G；不改 6NT）见 [`FP8_THEORY.md`](FP8_THEORY.md)。

---

## 0. 结论（先看这个）

中间档在**计划采用的记账口径**下是自洽的：

| 量 | 计划宣称 | 精确值 | 判定 |
| --- | ---: | ---: | --- |
| 总参数 | ≈12B | **12.05B** | 通过 |
| Encoder 激活 / 输入 token | ≈2.3B | **2.29B** | 通过 |
| Decoder 激活 / 输出 token | ≈4.5B | **4.49B** | 通过 |
| 50B tok 训练 | ~1,400 H100-h | **1,413 H100-h** | 通过 |
| 1M KV（decoder-only MHA / GQA-4 / MLA / YOCO+MLA / ÷8） | 369 / 41 / 46 / 1.15 / 0.14 GB | **368.64 / 40.96 / 46.08 / 1.15 / 0.14 GB** | 通过 |

需要改计划文本、但不改规格的几处（**本 PR 已改 `docs/TRAINING_PLAN.md`**）：

1. **专家稀疏度**写成了 37% / 53%，精确值是 **38.9%（7/18）/ 50.0%（9/18）**。
2. **注意力占总参 ~7%** 是 24B 时期的比例；12B 下占位注意力是 **13.0%**（MoE 仍占 **84.6%**，KDA 仍然参数中性）。
3. §2 示意图仍写 ≈3B / ≈6B，应改为 **≈2.3B / ≈4.5B**。
4. §16 的「YOCO+MLA+÷8 → 0.14 GB」把**全局 cache 再沿序列压 8×**当成已实现；CSA 的 \(m=4\) 只给出 **÷4 → 0.29 GB**。÷8 是额外设计选择，不是 CSA 的必然推论。

理论层面两个会改变训练/推理优先级的结论：

- **8K 未压缩滑窗主导长上下文注意力 FLOPs**。`index_topk` 从 256 改到 512 只动约 3% 的 score+AV 量；CSA/HCA 相对 dense 的节省只在 \(n \gg 8\mathrm{K}\) 出现。Phase B 在 4K 上保持稠密/滑窗，在复杂度上是对的——短上下文上 CSA concat 甚至略贵于 dense。
- **未压缩的 decoder cross-attn 在 128K 起压过 decoder MLP**（128K 约 3.2×，256K 约 6.5×）。中间档要兑现「128K–256K 可用」，全局 cache 必须走压缩或 query-aware 选择，而不能把 encoder 顶层 raw KV 原样交给 24 层 cross-attn。

---

## 1. 符号与底座

底座取 `openbmb/MiniCPM-2B-sft-bf16` 的 `config.json`：

\[
d=2304,\quad V=122753,\quad n_h=36,\quad d_h=64,\quad L_0=40,\quad d_{\mathrm{ff}}^{\mathrm{dense}}=5760.
\]

μP 常量：`scale_emb=12`，`dim_model_base=256`，`scale_depth=1.4`。

YOCO 拆分：Encoder（self-decoder）\(L_e=16\)，Decoder（cross-decoder）\(L_d=24\)。DeepSeekMoE 单专家

\[
E = 3\,d\,d_{\mathrm{moe}} = 3\cdot 2304\cdot 2048 = 14.16\mathrm{M}.
\]

Tied embedding

\[
\mathrm{Emb} = Vd = 122753\cdot 2304 = 0.2828\mathrm{B}.
\]

**占位自注意力**（计划 §3，CSA/HCA/MLA 维数冻结前）：\(A_{\mathrm{self}} = 1.25\cdot 4d^2 = 26.54\mathrm{M}\)。
**Cross-attn 上界**：\(A_{\times} = 4d^2 = 21.23\mathrm{M}\)。

每 token、每栈的激活（计划口径，embedding 计入该栈）：

\[
\begin{aligned}
N_{\mathrm{enc}}^{\mathrm{act}} &= \mathrm{Emb} + L_e A_{\mathrm{self}} + L_e(n_s^{e}+k^{e})E,\\
N_{\mathrm{dec}}^{\mathrm{act}} &= \mathrm{Emb} + L_d(A_{\mathrm{self}}+A_{\times}) + L_d(n_s^{d}+k^{d})E.
\end{aligned}
\]

一次完整前向（embedding 只算一次，YOCO 因果 LM 训练的正确口径）：

\[
N_{\mathrm{fwd}}^{\mathrm{act}} = \mathrm{Emb} + (N_{\mathrm{enc}}^{\mathrm{act}}-\mathrm{Emb}) + (N_{\mathrm{dec}}^{\mathrm{act}}-\mathrm{Emb}).
\]

计划 §3 / §15 的 GPU-hour 用的是 \(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}}\)（embedding 计两次）。下文两种都报。

---

## 2. 720 专家槽守恒：三档同总参、不同激活

三档都把 **720 个专家实例** 分给 16+24 层，只改 shared / routed / top-\(k\)：

| 档位 | Enc \(n_s+N_r\), top-\(k\) | Dec \(n_s+N_r\), top-\(k\) | 专家实例 | 总参 |
| --- | --- | --- | ---: | ---: |
| 省算力 | 1+20, \(k=3\) | 1+15, \(k=4\) | \(16\cdot21+24\cdot16=720\) | 12.05B |
| **中间（默认）** | **1+17, \(k=6\)** | **1+17, \(k=8\)** | **\(16\cdot18+24\cdot18=720\)** | **12.05B** |
| 近-dense | 2+10, \(k=8\) | 2+20, \(k=12\) | \(16\cdot12+24\cdot22=720\) | 12.05B |

这是中间档「能在不改总参的前提下调训练成本」的代数原因：总参由 routed 数决定，训练 FLOPs 由 top-\(k\) 决定。砍总参不会砍训练成本。

精确激活与稀疏度（专家稀疏度 \(=(n_s+k)/(n_s+N_r)\)）：

| 档位 | Enc 激活 | Dec 激活 | \(N_{\mathrm{fwd}}^{\mathrm{act}}\) | 计划口径 \(N_{\mathrm{enc}}+N_{\mathrm{dec}}\) | 稀疏度 enc/dec | 50B tok H100-h（计划口径） |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 省算力 | 1.61B | 3.13B | 4.46B | 4.74B | 19.0% / 31.2% | 988 |
| **中间** | **2.29B** | **4.49B** | **6.50B** | **6.78B** | **38.9% / 50.0%** | **1,413** |
| 近-dense | 2.97B | 6.19B | 8.88B | 9.16B | 83.3% / 63.6% | 1,908 |

计划表里的 ~950 / ~1,400 / ~1,900 是对 1.6+3.1、2.3+4.5、3.0+6.2 的四舍五入；精确值如上。中间档稀疏度应写成 **39% / 50%**，不是 37% / 53%。

固定部分（三档相同，占位注意力）：

| 块 | 参数 |
| --- | ---: |
| Embedding (tied) | 0.28B |
| Enc 自注意力 \(16\times 26.54\mathrm{M}\) | 0.42B |
| Dec 自注意力 \(24\times 26.54\mathrm{M}\) | 0.64B |
| Dec cross-attn \(24\times 21.23\mathrm{M}\) | 0.51B |
| MoE 专家 \(720\times 14.16\mathrm{M}\) | 10.19B |
| **合计** | **12.05B** |

注意力占位合计 \(0.42+0.64+0.51=1.57\mathrm{B}\)，占 12.05B 的 **13.0%**；MoE **84.6%**。24B 时期注意力约 6.6%–7%，那句话不能原样搬到 12B。KDA 替换部分 CSA/HCA 层仍然只动注意力块（即使全换成更小的线性核，量级也只是 \(10^{-1}\mathrm{B}\)），**不必为了加 KDA 而改 12B 总参目标**。

---

## 3. 中间档逐项展开

Encoder（16 层，1 shared + 17 routed，top-\(k=6\)）：

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 16\cdot 7\cdot 14.16\mathrm{M} = 1.585\mathrm{B},\\
N_{\mathrm{enc}}^{\mathrm{act}} &= 0.283 + 0.425 + 1.585 = 2.293\mathrm{B},\\
\mathrm{stack_{enc}} &= 0.425 + 16\cdot 18\cdot 14.16\mathrm{M} = 4.502\mathrm{B}.
\end{aligned}
\]

Decoder（24 层，1 shared + 17 routed，top-\(k=8\)，含 cross-attn）：

\[
\begin{aligned}
\mathrm{FFN_{act}} &= 24\cdot 9\cdot 14.16\mathrm{M} = 3.057\mathrm{B},\\
N_{\mathrm{dec}}^{\mathrm{act}} &= 0.283 + 0.637 + 0.510 + 3.057 = 4.487\mathrm{B},\\
\mathrm{stack_{dec}} &= 0.637 + 0.510 + 24\cdot 18\cdot 14.16\mathrm{M} = 7.262\mathrm{B}.
\end{aligned}
\]

\[
N_{\mathrm{total}} = 0.283 + 4.502 + 7.262 = 12.047\mathrm{B},\quad
N_{\mathrm{fwd}}^{\mathrm{act}} = 6.497\mathrm{B}.
\]

与计划「≈12B / ≈2.3B-in / ≈4.5B-out」一致。

---

## 4. 训练算力（6NT）与 Chinchilla 位置

Kaplan / Hoffmann 口径：训练 FLOPs \(\approx 6 N_{\mathrm{act}} T\)（前向 2NT + 反向 4NT，忽略注意力二次项）。H100 有效算力取计划 §15.1 的 \(4.0\times 10^{14}\) FLOPS（约 40% MFU）。

中间档、\(T=50\mathrm{B}\)：

| \(N_{\mathrm{act}}\) 口径 | \(N\) | FLOPs | H100-h | A100-h |
| --- | ---: | ---: | ---: | ---: |
| 计划（enc+dec，emb×2） | 6.78B | \(2.03\times 10^{21}\) | **1,413** | 4,520 |
| 完整前向（emb×1） | 6.50B | \(1.95\times 10^{21}\) | 1,354 | 4,331 |
| 仅 Encoder（prefill / early-exit） | 2.29B | \(6.88\times 10^{20}\) | 478 | 1,529 |

1,413 对上计划的 ~1,400。那是 **联合 bf16、emb×2** 的对照，不是 Phase B 发布墙钟。发布值是 **C1+FP8 = 761 H100-h**（联合 bf16 emb×1 的 56%）。FP8 不改这张 6NT 表。见 [`FP8_THEORY.md`](FP8_THEORY.md)。

**这不是 Chinchilla 预训练。** Hoffmann 最优大约 \(20 N\) tokens（dense）。50B / 6.5B ≈ **7.7 token / 激活参数**，属于上采样恢复 + 继续训练，不是从零训 12B。把 50–150B 写成 Phase B 恢复预算是对的；把它理解成「12B 已经训充分」则过满。

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

| \(n\) | MLP 前向 | 40L dense | Encoder 16L CSA/HCA 交错 | Enc + Dec 24L 滑窗 | Dec 24L **未压缩** cross-attn |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | \(5.3\times10^{13}\) | \(6.2\times10^{12}\) | \(2.6\times10^{12}\) | \(6.3\times10^{12}\) | \(3.7\times10^{12}\) |
| 32K | \(4.3\times10^{14}\) | \(4.0\times10^{14}\) | \(4.1\times10^{13}\) | \(1.0\times10^{14}\) | \(2.4\times10^{14}\) |
| 128K | \(1.7\times10^{15}\) | \(6.3\times10^{15}\) | \(1.7\times10^{14}\) | \(4.1\times10^{14}\) | \(3.8\times10^{15}\) |
| 256K | \(3.4\times10^{15}\) | \(2.5\times10^{16}\) | \(3.6\times10^{14}\) | \(8.4\times10^{14}\) | \(1.5\times10^{16}\) |

读表：

- 4K：注意力二次项 ≪ MLP，6NT 是好近似。
- 32K：40L dense 已经和 MLP 同量级；encoder 压缩后仍可比 MLP 小一个数量级。
- **128K 起，未压缩 cross-attn 单独就超过整网 MLP**。YOCO 若把全局 cache 存成 raw 顶层 KV，decoder 会把 CSA/HCA 在 encoder 侧省下的东西在 cross-attn 里花回去。

---

## 6. Prefill early-exit 与 decode 瓶颈

YOCO（arXiv 2405.05254）：prefill 只需跑完 self-decoder 即可写出全局 \(\hat K,\hat V\)，cross-decoder 在生成第一个 token 时才进入。中间档：

| | 值 |
| --- | --- |
| 层数比 \(L_e/(L_e+L_d)\) | 16/40 = **40%** |
| 激活比 \(N_{\mathrm{enc}}^{\mathrm{act}}/(N_{\mathrm{enc}}^{\mathrm{act}}+N_{\mathrm{dec}}^{\mathrm{act}})\) | 2.29/6.78 = **34%** |

所以「输入侧更轻」不只是层数减半：中间档的非对称 MoE 再把 prefill 算力压到全模型前向的约三分之一。这与「主交付 128K–256K、超长输入是 encoder 的活」一致。

Decode 一步（decoder MLP 前向 \(2 N_{\mathrm{dec}}^{\mathrm{act}}\) vs 对长度为 \(n\) 的全局 cache 做 cross-attn）：

| \(n\) | dec-MLP | xattn 全量 | 全量 / MLP | xattn CSA \(m=4\) | xattn CSA top-\(k\)+8K 窗 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32K | \(8.97\times10^9\) | \(7.25\times10^9\) | 0.81× | 0.20× | 0.21× |
| **128K** | \(8.97\times10^9\) | \(2.90\times10^{10}\) | **3.23×** | 0.81× | **0.21×** |
| **256K** | \(8.97\times10^9\) | \(5.80\times10^{10}\) | **6.46×** | 1.62× | **0.21×** |
| 1M | \(8.97\times10^9\) | \(2.32\times10^{11}\) | 25.8× | 6.46× | 0.21× |

**中间档要在 128K–256K 上可用，全局 cache 不能是未压缩 raw KV。** 最低限度：沿序列做 \(m=4\) 压缩（256K 上 cross-attn 仍约 1.6× MLP）。要让 decode 重新由 MLP 主导，需要对全局 cache 做 CSA 式 top-\(k\)（或 PDSA 校准回退），把每 query 的 \(k\) 钉在 \(n_{\mathrm{win}}+k_{\mathrm{index}}\approx 8.4\mathrm{K}\)。这与计划 §14「query-aware 优先于 write-first、校准稀疏回退」同方向，而且是算力约束，不只是召回约束。

---

## 7. KV cache：§16 的数字是对的，假设需要写清

bf16、内容字节（不含 allocator padding）。「1M」= \(10^6\) token（不是 \(2^{20}\)）。GB = \(10^9\) B。

MHA：每层每 token \(2d\) 个元素（K 和 V）。GQA-4：4 个 KV 头（不是 \(36/4=9\)）。MLA-576：DeepSeek-V3 式 latent \(k_{\mathrm{lora}}+d_{\mathrm{rope}}=512+64\)。

\[
\mathrm{bytes} = n \cdot L_{\mathrm{cache}} \cdot d_{\mathrm{kv}} \cdot 2.
\]

| 配方 | 8K | 32K | 128K | 256K | 1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| decoder-only MHA 40L | 3.02 | 12.08 | 48.32 | 96.64 | **368.64** |
| decoder-only GQA-4 40L | 0.34 | 1.34 | 5.37 | 10.74 | **40.96** |
| decoder-only MLA-576 40L | 0.38 | 1.51 | 6.04 | 12.08 | **46.08** |
| **YOCO + MLA-576（1 层全局）** | 0.01 | 0.04 | 0.15 | 0.30 | **1.15** |
| YOCO + MLA + 序列÷8 | — | — | 0.02 | 0.04 | **0.14** |
| YOCO + MLA + CSA \(m=4\) | — | — | 0.04 | 0.08 | **0.29** |
| YOCO + MiniCPM-native MQA-64 | — | — | 0.03 | 0.07 | 0.26 |
| Decoder 24×8K 窗（MLA） | 0.23 | 0.23 | 0.23 | 0.23 | **0.23** |
| Encoder 16×8K 窗（MLA） | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |

8K/32K/128K/256K 列为 \(8192/32768/131072/262144\)；1M 列为 \(10^6\)（与计划 §16 相同）。

核对计划 §16：

- 369 / 41 / 46 / 1.15 / 0.14 GB：**通过**（小数 GB、\(n=10^6\)、GQA=4 头、MLA=576）。
- 「+≈0.2 GB 的 24 层 8K 滑窗」：精确 **0.23 GB**，量级通过。Encoder 自己还有约 0.15 GB 的 8K 窗（YOCO 原文也承认 self-decoder 有常数 cache）。128K–256K 上全局 cache（0.15–0.30 GB）和 8K 窗（0.23+0.15 GB）同量级——再一次说明 **8K 窗不是免费的**。
- **÷8 不是 CSA \(m=4\) 的推论。** \(m=4\) 给出 0.29 GB @ 1M。0.14 GB 需要额外的序列压缩（更重的 HCA 式全局池化、或对全局 cache 再做一层压缩）。应在计划里标成「可选 stretch」，避免 impl 时按 CSA 默认实现却以为已经对上 0.14 GB。

主目标 128K–256K：YOCO+MLA 全局 cache **0.15–0.30 GB**，即使加上两侧 8K 窗也远小于 decoder-only MHA 的 47–94 GB。1M 推理在中等硬件上可行，这一条理论成立。

---

## 8. 占位注意力 vs MiniCPM 原生 CSA/HCA（敏感性，不改规格）

DeepSeek-V4-Flash 用 `head_dim=512`、`q_lora_rank=1024`（\(d=4096\)）。不能把这组维数搬到 MiniCPM 的 \(d=2304\)。敏感性模型用 MiniCPM 原生 MQA-64（\(d_h=64\)，单 KV 头）+ LoRA-Q 512 + grouped output，按 V4 论文 §2.3 计 CSA/HCA 投影：

| 每层自注意力 | 参数 |
| --- | ---: |
| MiniCPM MHA \(4d^2\) | 21.23M |
| 计划占位 \(1.25\times\mathrm{MHA}\) | **26.54M** |
| CSA MQA-64（含 indexer） | 10.29M |
| HCA MQA-64 | 8.86M |
| CSA/HCA 平均 | 9.57M |

详细模型比占位**更小**，不是更大。若将来冻结为 MQA-64 CSA/HCA：

- 总参落到约 **11.37B**，激活约 **2.02B / 4.08B**（注意力也在激活里）。
- 补回 12.05B 只加 routed 专家即可（top-\(k\) 不动则激活几乎不动）：脚本给出 Enc 1+17 / Dec 1+19。
- 若还想把激活拉回 2.3/4.5，应加 top-\(k\) 而不是加 routed。

**在投影维数冻结前，继续用 1.25×MHA 占位作为中间档规格。** 详细模型只说明：占位是保守上界，不会在实现 CSA 之后突然把 12B 撑破。

---

## 9. 首层 dense（Phase A）与 μP 拆栈

计划 Phase A：「每栈首层保留 dense」。当前 §3 表按**全部 MoE** 记账。把每栈第 1 层换成 MiniCPM dense SwiGLU（\(3\cdot d\cdot 5760=39.81\mathrm{M}\)）：

| | 全 MoE（规格表） | 首层 dense、Nr 不变 |
| --- | ---: | ---: |
| 总参 | 12.05B | **11.62B** |
| Enc 激活 | 2.29B | 2.23B |
| Dec 激活 | 4.49B | 4.40B |

激活仍在 ≈2.3 / ≈4.5 的四舍五入里。总参差 0.43B。补回 12.05B 且不改 top-\(k\)：Encoder routed **17→19**（其余不变）→ 总参 **12.04B**，激活仍是 2.23 / 4.40。建议在 Phase A 落地时采用「Enc 1+19 / Dec 1+17、首层 dense」，对外仍称中间档。

μP 残差乘子原式是 \(\mathrm{scale\_depth}/\sqrt{L_0}=1.4/\sqrt{40}=0.2214\)。拆成 16+24 之后：

- **必须继续用 \(1.4/\sqrt{40}\)**。权重是在这个尺度上训出来的。
- 若改成 \(1.4/\sqrt{16}=0.350\) 或 \(1.4/\sqrt{24}=0.286\)，继承来的残差会被放大 1.29–1.58×，正是计划警告的「μP 缩放丢失 → 数值漂移」。
- Logits 除以 \(d/\mathrm{dim\_model\_base}=2304/256=9\)；`scale_emb=12`。二者与层数无关，拆栈后保持即可。

---

## 10. 信息论约束（PDSA）与中间档的关系

PDSA（Zou & Donz，arXiv 2606.28876）给出一条与参数预算正交、但和 §6 的 decode 瓶颈同方向的约束：

- **无写入时信号**：query-independent 可写性在静态文本上接近随机（AUC 0.63–0.66）。HCA / KDA / CSA 的压缩步都是 write-first，会系统性丢掉「写入时无信号、query 时才被命中」的块。
- 因此 **必须留一条到低压缩 / 原始 KV 的 query 时回退**。§6 显示这条回退在 128K–256K 上也是算力上该走的路：把 cross-attn 的 \(k\) 从 \(n\) 收到 \(n_{\mathrm{win}}+k_{\mathrm{index}}\)，decode 才回到 MLP 主导。
- 中间档把精确检索预算给了 Decoder 的 cross-attn（4.5B 激活、24 层），把长输入压缩预算给了 Encoder（2.3B、16 层）。这与「query-aware 读发生在 decode、write-first 压缩发生在 prefill」一致。不要为了「再省一点 prefill」把 encoder 改成纯 HCA/KDA 而没有 CSA 锚点 + 回退。

线性注意力不能修 lost-in-the-middle（计划 §2.5 已澄清）；中间档也不依赖 KDA 来保中段召回。IN2/FILM + 校准回退才是中段路径。

---

## 11. Claim ledger

对计划已发布数字的机械化核对（`scripts/param_budget.py --verify`，占位注意力、全 MoE、中间档）：

| Claim | 结果 |
| --- | --- |
| 总参 ≈12B | PASS（12.05B） |
| Enc 激活 ≈2.3B | PASS（2.29B） |
| Dec 激活 ≈4.5B | PASS（4.49B） |
| Emb / Enc attn / Dec attn / cross ≈ 0.28 / 0.42 / 0.64 / 0.51B | PASS |
| 专家稀疏度 7/18、9/18 | PASS（计划文本已改为 38.9%/50.0%） |
| 三档共享 720 专家槽 | PASS |
| 50B tok ≈1400 H100-h | PASS（1413） |
| 1M KV 369 / 41 / 46 / 1.15 / 0.14 GB | PASS |
| 24×8K 窗 ≈0.2 GB | PASS（0.23 GB） |
| μP logits 缩放 = 9；残差保持 \(1.4/\sqrt{40}\) | PASS |
| freeze-enc ≈76% 联合；独立拼接 152%；C1 课程 81% | PASS（见课程篇） |

解冻课程另 12 条（定稿 C1、detach、tied \(E\)、Adam 60%、合法 split ≤85% 等）与中间档 22 条、FP8 13 条合计 **`--verify` 47/47**，见 [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)、[`FP8_THEORY.md`](FP8_THEORY.md)。

文本层（不进 `--verify`，已在上文展开）：

| 文本 | 处理 |
| --- | --- |
| 稀疏度 37%/53% | 已改为 38.9%/50.0% |
| 注意力 ~7% | 已改为 ~13%（12B） |
| §2 示意图 3B/6B | 已改为 2.3B/4.5B |
| 全局 cache ÷8 | 已标为可选；CSA \(m=4\) 对应 ÷4 → 0.29 GB |
| §11「对齐到 24.0B / 3.0 / 6.0」 | 已改为 12.05B / 2.3 / 4.5 |
| 首层 dense 未进规格表 | 激活仍 ≈2.3/4.5；总参用 Enc \(N_r=19\) 补回 |

---

## 12. 对实现的约束（从理论读出来的，不是新功能）

1. **规格继续锁中间档**：Enc 16L、1+17、top-\(k=6\)；Dec 24L、1+17、top-\(k=8\)；占位注意力 1.25×MHA。Phase A 若首层 dense，只把 Enc routed 调到 19。
2. **全局 cache 必须压缩或可选**。否则 128K–256K decode 被 24 层 cross-attn 主导，YOCO+CSA 的 encoder 收益会被吐回去。
3. **`n_win=8K` 是第一成本旋钮**；`index_topk` 不是。消融按计划 §7 做 {2K,4K,8K}。
4. **μP 残差不要按新栈深重算**。
5. **4K 主训练不要指望 CSA 省算力**；稀疏化放在 Phase C、长上下文放在 Phase D，与 FLOPs 曲线一致。
6. 投影维数冻结后，用 `python3 scripts/param_budget.py --attn csa_mqa64 --full` 重跑，用 routed / top-\(k\) 补回 12.05B 与 2.3/4.5，不要改层数拆分。
7. **Phase B 按 C1 定稿**（两栈先 MoE、冻 Encoder、detach cache、freeze_tied、B2≥10B）。不要把两栈当独立 LM 再拼接。
8. **Phase B 墙钟按 C1+FP8 定稿**（761 H100-h；B1/B2 MoE GEMM + 冻结 Encoder 前向；B0 student / L0 / indexer / 白名单高精度）。发布 1.5×，不改 6NT，不发布 2×。联合 bf16 只作对照。

复算命令：

```bash
python3 scripts/param_budget.py              # 中间档摘要
python3 scripts/param_budget.py --tier all   # 三档对照
python3 scripts/param_budget.py --full       # KV / 复杂度 / μP / Nr 回搜
python3 scripts/param_budget.py --verify     # 规格 + 解冻课程 + FP8 断言
python3 scripts/param_budget.py --staged --curriculum --fp8
python3 -m unittest tests.test_param_budget
```
