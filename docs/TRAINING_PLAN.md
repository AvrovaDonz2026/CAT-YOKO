# CAT-YOKO 训练计划：基于 MiniCPM-2B 上采样的 Causal Encoder-Decoder（YOCO 式）混合注意力 MoE

> 目标：以 OpenBMB **MiniCPM-2B** 为底座，训练一个 **Causal Encoder-Decoder（YOCO / "You Only Cache Once" 式 decoder-decoder）** 模型：
> **≈12B 总参数**（已按算力预算从 24B 下调）；非对称激活 **默认：Encoder 处理输入 ≈2.3B 激活/token、Decoder 生成输出 ≈4.5B 激活/token**
> （更省算力档 ≈1.6B/3.1B、近-dense 档 ≈3.0B/6.2B 见 §3）；
> 注意力用 **DeepSeek-V4-Flash 式 CSA + HCA 压缩注意力 + 8K 大滑动窗口**；原生长上下文（YOCO 单一全局 KV cache）。
>
> ⚠️ **关键**：总参数主要影响**显存/存储**；**训练算力 ∝ 激活参数 × tokens**。要真正降训练成本必须降**激活**（选更省算力档），而不是只降总参。
>
> 本文是可执行的工程训练计划。**实现默认已敲死**在 [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)
> / `cat_yoko.config.CATYokoConfig.middle_12b()`：因果 16/24、C1 全 MoE、C1+FP8、Phase B 滑窗 + 门控
> cross-attn、AdamW；**KDA / mHC / MTP / Muon / 首层 dense / M2 不是发布默认**。本文其余档位与消融是敏感性，不是训练代码的开关默认。

---

## 0. 需求澄清与关键假设

你的原始描述里有几处需要显式确认的点，本计划先按下面的解释推进，如与你的意图不符请指出：

| 你的表述 | 本计划的解释 | 备注 / 可调整项 |
| --- | --- | --- |
| `基于 openbmb 的 minicpm2b` | 底座 = **MiniCPM-2B**（dense，40 层，hidden 2304，FFN 5760，36 头，vocab 122753，tie embedding） | 也可换 `MiniCPM-2B-128k` 变体作为长上下文底座 |
| `和 deepseekv4.1 flash 一样` | 对标 **DeepSeek-V4-Flash** 的架构范式（CSA/HCA 混合注意力 + DeepSeekMoE + mHC + Muon + MTP + Hash-MoE bootstrap） | V4-Flash 官方为 284B/13B decoder-only；我们做的是**同架构、缩小到 12B 且改造为 YOCO encoder-decoder（非对称激活）的复刻** |
| `Causal-Encoder-Decoder，输入激活 3b，输出激活 6b`；后续 **`砍到12B`** | **YOCO 式 decoder-decoder**：**Encoder=self-decoder** 处理输入并产出**单一全局 KV cache**；**Decoder=cross-decoder** 生成输出并对该全局 cache 做 cross-attention。**总参数 ≈12B**（命名 `CAT-YOKO-12B`）；激活默认 **中间档（~2.3B-in/~4.5B-out）**，更省档 1.6/3.1、近-dense 档 3.0/6.2 见 §3 | 激活的非对称性来自**两个物理不同的栈**（encoder 较小、decoder 较大），不是变 top-k。见 §2、§3 |
| `大滑窗注意力 8k` | 每个 CSA/HCA 层保留的**未压缩滑窗分支** `n_win = 8192` | DeepSeek-V4 默认 `n_win=128`，8K 是明显放大，成本更高但局部保真更好 |
| `CSA HCA` | **Compressed Sparse Attention** + **Heavily Compressed Attention**（DeepSeek-V4 的两种压缩注意力，层间交错） | 见 §2 |

> ✅ **本轮已确认规格**：Causal Encoder-Decoder（YOCO 式）；**总参数 12B**（从 24B 下调）；激活默认 **≈2.3B-in / ≈4.5B-out**。
> **发布配方已敲死**（[`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md) / `cat_yoko.config`）：因果 16/24、C1 全 MoE、C1+FP8 墙钟 761 H100-h、Phase B 滑窗+门控 cross-attn、M2/KDA/mHC/MTP/Muon 关。
> Encoder **不是**双向。仓库名 **CAT-YOKO** 中的 "YOKO" 即对应 **YOCO**。

> ⚠️ **重要现实提示（务必先读）**：完整复刻这一架构并从 32T 级别数据预训练是**前沿实验室量级**的工程。
> 以 MiniCPM-2B 为底座做**上采样（upcycling）+ 继续训练**能把成本降到几百 B token 量级，
> 但仍需要几十~上百张 H100/H800 级 GPU 的持续算力。本计划按"**上采样改造 + 继续预训练**"路线设计，
> 这是在可控预算内得到该架构可用模型的唯一现实路径（从零 32T 预训练不在推荐范围）。

---

## 1. 底座与目标规格

### 1.1 MiniCPM-2B（底座，来自官方 config）

| 项 | 值 |
| --- | --- |
| 层数 `num_hidden_layers` | 40 |
| 隐藏维 `hidden_size` | 2304 |
| FFN 中间维 `intermediate_size` | 5760（SwiGLU） |
| 注意力头 `num_attention_heads` | 36（MHA，`num_key_value_heads=36`） |
| head_dim | 64 |
| 词表 `vocab_size` | 122753 |
| 激活 | SiLU / SwiGLU |
| 位置编码 | RoPE |
| Embedding | **tie（输入/输出共享）** |
| 非词嵌入参数 | ≈2.4B（总含 emb ≈2.7B） |
| 特殊设计 | **μP 风格缩放**（`scale_emb`、`scale_depth`、`dim_model_base`）+ **WSD 学习率调度** |

> MiniCPM 的 μP 缩放常量（embedding 乘子、residual `scale_depth/√L` 中的 \(L=40\)、logits 除以 `d/dim_model_base=9`）
> 在拆成 16+24 之后**必须保持原值**（不要改成 √16 / √24），否则继承权重的残差尺度会漂。核对见理论验证 §9。

### 1.2 目标模型 `CAT-YOKO-12B`（Causal Encoder-Decoder / YOCO 式）

| 项 | 推荐值（默认中间档） | 说明 |
| --- | --- | --- |
| 架构 | **YOCO decoder-decoder**：Encoder(self-decoder) → 全局 KV cache → Decoder(cross-decoder) | 见 §2 |
| 总参数 | **≈12B** | 见 §3 预算 |
| **Encoder** 激活/输入 token | **≈2.3B**（档位可选 1.6/2.3/3.0） | 16 层，MoE，CSA/HCA+8K 滑窗，产出全局 cache |
| **Decoder** 激活/输出 token | **≈4.5B**（档位可选 3.1/4.5/6.2） | 24 层，MoE，自注意力 + **cross-attn 到全局 cache** |
| 隐藏维 | 2304（沿用底座，enc/dec 一致以便共享 emb 与热启） | |
| FFN | **DeepSeekMoE** 细粒度专家，`moe_intermediate_size=2048` | Enc: 1 shared+17 routed, top-k 6；Dec: 1 shared+17 routed, top-k 8（默认档） |
| 注意力 | **CSA/HCA 混合 + 8K 滑窗**；enc 内含长程压缩，dec 的 cross-attn 复用单一全局 cache；**可选叠加 KDA 线性注意力做 3:1 三路混合** | §2 / §2.5 |
| KV cache | **单一全局 cache（You Only Cache Once）** + 压缩 → O(N) 级显存 | 长上下文关键收益 |
| 残差 | **mHC**（可选，先用普通残差跑通） | 稳定性增强 |
| 训练目标 | 主 CE + **MTP** 辅助头（可选） | |
| 优化器 | **Muon（2D 权重）+ AdamW（emb/norm/router/bias）** | |
| 上下文 | **主交付目标 128K–256K 可用**；阶段式 4K → 8K → 32K → 128K → 256K；1M 为 stretch（推理支持 + needle） | 8K 滑窗 + 压缩长程 + YOCO |

---

## 2. 架构：YOCO 式 Causal Encoder-Decoder + CSA/HCA 压缩注意力

### 2.0 总体骨架（decoder-decoder / YOCO）

CAT-YOKO 由两个因果栈组成，行为上等价于一个 decoder-only Transformer，但"只缓存一次"：

```
输入 tokens ─► [Encoder = Self-Decoder, 16 层, ≈2.3B 激活/token]
                 │  高效因果注意力（CSA/HCA + 8K 滑窗），逐层压缩长程
                 ▼
          顶层隐状态  ──►  产出【单一全局 KV cache  K̂, V̂】(You Only Cache Once)
                 │
                 ▼
        [Decoder = Cross-Decoder, 24 层, ≈4.5B 激活/token]
           每层 = 高效因果自注意力(生成序列, 滑窗)  +  Cross-Attn(→ K̂,V̂)  +  MoE-FFN
                 ▼
             RMSNorm ─► LM Head (tie emb) ─► 下一个 token
```

关键性质（来自 YOCO；形式化与因果证明见 [`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)）：
- **只缓存一次**：只有 encoder 顶层产出的一个全局 cache 被所有 cross-decoder 层复用，KV cache 显存从 `O(N·L)` 降到约 `O(N)`。Encoder 层内 CSA/HCA（**M1**）**不**自动减少这份 cache 的槽数；沿序列再池化（**M2**）才把槽数从 \(N\) 降到 \(N/m\)。
- **Prefill 可提前退出（仅推理）**：提示只需跑完 encoder 即可写出全局 cache，decoder 只在最后一位上跑一次出首 token。训练仍是两栈全长前向。这正是"**输入侧更轻（≈2.3B）**"的动机。中间档 encoder 激活份额约 34%（见 `docs/THEORY_VERIFICATION.md` §6）。
- **非对称激活来自两个物理栈**：encoder 较小（16 层、专家/激活更少 → ≈2.3B），decoder 较大（24 层、含 cross-attn、专家/激活更多 → ≈4.5B）。**不是**变 top-k。
- **保留全局注意力能力**：全局性来自 decoder **cross-attn 读 cache（M3）**，不是来自 encoder 滑窗堆叠感受野（16×8K=131072，盖不住 256K，也不需要盖住）。

> 设计取舍：YOCO 原论文 self-decoder 用 sliding-window 或 gated retention。我们把 self-decoder 的
> 层内注意力换成 **CSA/HCA + 8K 滑窗（M1）**，降低 encoder 在 \(n\gg 8K\) 时的二次项，并可选地对写入表示做长程混合。
> **更紧凑的全局 cache 是 M2（可选池化），生成时的 query-aware 检索是 M3（decoder 侧 indexer / 校准回退）**——与 M1 不是同一件事。
> Decoder **自注意力在训练时也看到全长序列**，必须用滑窗/KDA；「生成序列通常不长」只描述推理。长程一律走 cross-attn。

### 2.1 CSA / HCA / 8K 滑窗（注意力细节）

DeepSeek-V4 用**逐层交错的两种压缩注意力**替换 V3 的 MLA 全量注意力，核心目标是把长上下文注意力成本从
`O(L²)` 降到近似 `O(L·k)`，并大幅压缩 KV cache。这套压缩注意力用在 **Encoder(self-decoder)** 内，也用于其自注意力。三类层：

### 2.2 三种层类型（`layer_types`）

1. **Sliding-window（bootstrap 层）**：只做局部滑窗因果注意力，窗口 = `sliding_window`，无长程分支。Encoder **前 2 层**用它（冻结；16 层才能排下 7+7 CSA/HCA）。
2. **CSA（Compressed Sparse Attention）**
   - 把每 `m=4` 个 token 的 KV 压成 1 条（带可学习压缩权重 `Z` 与位置偏置，overlapping window）。
   - 用 **Lightning Indexer** 给 query 对压缩条目打分，取 **top-`index_topk`**（默认 512）条参与注意力（即在压缩序列上做 **DSA**）。
   - 额外拼接一条**未压缩滑窗 K/V 分支**（大小 `n_win`）保留局部细节。
3. **HCA（Heavily Compressed Attention）**
   - 把每 `m'=128` 个 token 压成 1 条（non-overlapping），**不做 indexer**，对全部压缩条目**稠密注意力**。
   - 同样拼接一条未压缩滑窗分支。

> 实现要点（与官方参考一致）：CSA/HCA 都是把 **raw 滑窗 K/V** 与 **压缩 K/V** 沿序列轴 `concat`，
> 构造一个组合 mask 后跑**一次**标准 masked attention；CSA 的 mask 经 `top_k` 过滤，HCA 的 mask 全可见。
> 二者只差在"压缩率"和"是否 top-k"。

### 2.3 本项目的注意力配置（推荐）

| 参数 | 推荐值 | 对应 DeepSeek-V4 名 |
| --- | --- | --- |
| `sliding_window` (`n_win`) | **8192** | 你要求的"8K 大滑窗"；且必须 \(\ge m'=128\) 才能补 HCA 自身块的洞 |
| Encoder `layer_types` | **2× sliding bootstrap，其后 CSA:HCA=1:1** → 16 层为 **2 sliding + 7 CSA + 7 HCA** | V4-Flash 以 2 层 sliding 开头；16 层做不了「3 bootstrap + 1:1」 |
| Decoder self-attn | **全部 sliding**（可选日后 KDA 混合）；**默认不再铺 CSA** | 全局混合已在 cross-attn（M3） |
| CSA 压缩率 `m` | 4 | `compress_rate_csa` |
| HCA 压缩率 `m'` | 128 | `compress_rate_hca` |
| `index_topk` (CSA, **M1**) | 256～512 | Encoder 层内 Lightning Indexer；**不要**复用到 decoder query |
| Decoder 侧选择（**M3**） | 128K 起需要：dense→top-k 或校准回退 | 真正的生成期检索；见架构理论 §3 |
| 注意力底座 | 建议保留 **MLA / MQA 模式**（KV 头共享）以省 KV cache | V4 基于 MLA 的 MQA 模式实现 DSA |

> **8K 滑窗的代价**：滑窗分支是未压缩的 `O(L·n_win)` 成本，`n_win=8192` 比默认 128 大 64×，
> 局部注意力开销显著上升。若长上下文吞吐吃紧，可在长程能力足够时把 `n_win` 回调到 2K–4K，
> 或仅在部分层用 8K 滑窗。建议在 §7 做 `n_win ∈ {2K,4K,8K}` 消融。

### 2.4 从 MiniCPM MHA 迁移到 CSA/HCA + Encoder/Decoder 的初始化

MiniCPM 是 40 层 decoder-only 标准 MHA，没有 MLA 潜在向量、压缩器/indexer，也没有 cross-attn。迁移思路：

- **拆成两个栈**：把 MiniCPM 的 40 层权重切给 Encoder(16) + Decoder(24)，或按需复制/截取（见 §4 Phase A）。
- **注意力主干**：`q/k/v/o` 投影继承 MHA 权重（若切到 MLA，用 SVD 把 K/V 投影分解为低秩 `W^{DKV}·W^{UK/UV}` 初始化，`q_lora_rank` 同理）。
- **Decoder 的 cross-attn**：`q` 投影可从对应自注意力 `q` 初始化；`k/v` 投影从 encoder 顶层 KV 空间对齐初始化（或随机小尺度）；**先旁路 cross-attn（gate≈0）再逐步打开**以稳定训练。gate=0 时 16/24 切分与原 40 层残差流等价（架构理论定理 A）。
- **新增模块**（压缩器 `W^{aKV}/W^{bKV}/W^{aZ}/W^{bZ}`、位置偏置 `B`、Lightning Indexer）：小尺度随机初始化，**先"稠密对齐"再"稀疏化"**（见 §4 Phase C）。

### 2.5 可选增强：加入 KDA 线性注意力（三路混合）

**动机与一个必须澄清的误解**：直觉上会觉得"加 KDA 线性注意力能保证 CSA/HCA 在上下文**中段**不丢信息"。
但文献结论其实相反——**线性注意力（含 KDA）恰恰是"中段精确召回"最弱的一环**：其固定大小 RNN 状态会发生
记忆碰撞，落在局部窗口外的 needle 容易丢失（arXiv 2507.06457、LoLA）。混合模型能恢复召回，靠的是**保留
full/稀疏注意力层**承载检索路径，而不是靠线性层。此外 **lost-in-the-middle 本质是位置偏置**（softmax 模型也有），
主要靠 RoPE/NoPE 校准缓解，加线性注意力并不能直接修它。**因此保中段精确检索的是 CSA 的 top-k 与少量 full 锚点，不是 KDA。**

**那为何仍推荐引入 KDA？** 两个互补收益：
1. **效率杠杆（主）**：Kimi Linear 用 **3:1 KDA:MLA** 混合，1M 上下文 KV cache ↓~75%、解码 ↑~6×，且质量不降反升。
   把多数层换成 KDA、少数层保留 CSA/HCA，可显著降低模型在长上下文的成本。
2. **对 CSA 选择性失败的"兜底覆盖"（次）**：CSA 风险在于 Lightning Indexer 的 top-k **漏选**中段相关 block。
   KDA 是**无 top-k、顺序敏感、每 token 都写入状态**的路径，提供 gist 级全序列覆盖，作为漏选时的安全网
   （注意是粗覆盖，不替代精确检索）。

**推荐配置（作为可选项，用消融定夺）**：
- **Encoder(self-decoder) 三路混合**：约 **3:1 的 KDA : (CSA/HCA)**，例如每 4 层 `[KDA, KDA, KDA, CSA]`，每隔几组插 1 层 HCA，
  并保留 **1–2 层高 `index_topk` 的 CSA 或真·full 注意力作为"召回锚点"**（hybrid-linear：gated-delta 类在 3:1~6:1 即达 Transformer 级召回）。
- **位置编码**：KDA 用学习衰减提供位置/近因信息；full/CSA 锚点层可考虑 **NoPE**（Kimi Linear 做法）。
- **Decoder(cross-decoder)**：自注意力用滑窗/KDA（训练时序列可以很长，不能改回全注意力）；跨段检索交给 cross-attn → 全局 cache（M3）。

**代价 / 注意**：
- KDA 需额外的 **DPLR chunked kernel** + **独立循环状态**管理（与 YOCO"只缓存一次"正交：YOCO 省 KV cache，KDA 状态是每层各自的小状态）。
- 混合改造已知坑：**模型可能学会忽略线性路径**；上采样初始化时让 KDA 与被替换的注意力做行为对齐，训练中监控各路径贡献。
- 参数上，KDA 层通常比 MHA/CSA 更省参（无大 KV 投影），把部分 CSA/HCA 层替换为 KDA 会略降每栈参数，需在 §3 脚本里按实际 KDA 维度重算并用 routed 专家数补回 12B。

> 结论：**值得加，但定位是"效率 + 兜底覆盖"，不是"保中段精确检索"**。是否上、以及 KDA:CSA:HCA 的确切比例，用 §7 消融决定。

---

## 3. 参数预算推导（`CAT-YOKO-12B`，Encoder-Decoder 拆分）

底座维度 `d=2304`、`vocab=122753`、tie embedding。Encoder 16 层、Decoder 24 层（共 40，沿用底座深度）。
`moe_intermediate_size=2048`（DeepSeek-V4 同量级、对硬件友好），单专家 SwiGLU ≈ `3·d·2048 ≈ 14.16M`。

### 3.1 预算表（12B 总参，激活三档可选）

单专家 SwiGLU ≈ `3·d·2048 ≈ 14.16M`；Embedding（tie，全局共享）≈0.283B；三档总参数均 ≈12B，仅激活/稀疏度不同。

| 档位 | Enc 专家(shared+routed, top-k) | Dec 专家(shared+routed, top-k) | Enc 激活/输入 | Dec 激活/输出 | 稀疏度 enc/dec | 训练算力(×50B tok) |
| --- | --- | --- | --- | --- | --- | --- |
| 省算力档 | 1+20，top-k 3 | 1+15，top-k 4 | 1.61B | 3.13B | 19.0% / 31.2% | 988 H100-h |
| **默认（中间档）** | **1+17，top-k 6** | **1+17，top-k 8** | **2.29B** | **4.49B** | **38.9% / 50.0%** | **1,413 H100-h** |
| 近-dense 档 | 2+10，top-k 8 | 2+20，top-k 12 | 2.97B | 6.19B | 83.3% / 63.6% | 1,908 H100-h |

固定部分（三档相同）：Enc 自注意力 ≈0.42B、Dec 自注意力 ≈0.64B、Dec cross-attn ≈0.51B、emb ≈0.28B。
三档 **专家实例都是 720**（只重分配 shared/routed/top-k），所以总参同为 12.05B、训练成本只随激活变。

> 🔑 **总参数 ≈ 显存/存储；训练算力 ∝ 激活 × tokens。** 三档总参都是 12.05B，但训练成本随**激活**变（988 → 1,413 → 1,908 H100-h @ 50B tok）。
> 默认取中间档（enc 2.29B / dec 4.49B）——"1.6 太少、3.0 太多"的折中。层数拆分（16/24）、`moe_intermediate_size`、
> 每栈 shared/routed/top-k 均为旋钮；注意力 ×1.25 为占位估计，定稿后用脚本重算并微调 routed 数对齐 12.05B。
> 另注：12B 下占位注意力占总参 **~13%**（MoE ~85%），**加 KDA 仍不改变总参预算**（见 §2.5）。逐项复算、KV/FLOPs/μP 与 claim ledger 见 [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md)。

### 3.2 复算脚本（`scripts/param_budget.py`，默认中间档）

```python
d, V, moe_int = 2304, 122753, 2048
emb    = V*d                       # tied, 全局共享
expert = 3*d*moe_int               # SwiGLU 单专家
attn   = int(1.25*4*d*d)           # 每层自注意力（用实际实现替换）
cross  = 4*d*d                     # 每层 cross-attn（仅 decoder）

# 默认中间档（12B / ~2.3B-in / ~4.5B-out）
Le, ns_e, tk_e, Nr_e = 16, 1, 6, 17   # Encoder = self-decoder
Ld, ns_d, tk_d, Nr_d = 24, 1, 8, 17   # Decoder = cross-decoder
# 省算力档: (1,3,20) / (1,4,15)   近-dense 档: (2,8,10) / (2,12,20)

enc_act = emb + attn*Le          + Le*(ns_e+tk_e)*expert
dec_act = emb + (attn+cross)*Ld  + Ld*(ns_d+tk_d)*expert
total   = emb + attn*Le + Le*(ns_e+Nr_e)*expert \
              + (attn+cross)*Ld + Ld*(ns_d+Nr_d)*expert
print(f"enc_active(input)={enc_act/1e9:.2f}B "
      f"dec_active(output)={dec_act/1e9:.2f}B total={total/1e9:.2f}B")
```

完整复算（KV / FLOPs / μP / 三档对照 / 规格断言）以 `scripts/param_budget.py` 为准，推导见 [`docs/THEORY_VERIFICATION.md`](THEORY_VERIFICATION.md)：

```bash
python3 scripts/param_budget.py --full
python3 scripts/param_budget.py --verify
python3 scripts/param_budget.py --fp8
```

---

## 4. 分阶段训练配方（核心）

整体思路：**上采样 + 手术式改造 + 分阶段继续训练**，用 MiniCPM-2B 已有能力做"暖启动"，
每次只引入一个大变化并让模型恢复，避免一次性改动太多导致坍塌。Token 预算是数量级建议，非日历时间。

```
Phase A  架构手术与初始化        —— 0 token（离线权重变换）
Phase B  上采样恢复性继续预训练   —— 50–150B token @ seq 4K（dense/滑窗注意力，先不稀疏）
Phase C  注意力稀疏化对齐         —— 20–50B token（Indexer 稠密对齐 → 开 top-k → 开 HCA 压缩）
Phase D  长上下文扩展            —— 20–60B token（8K→32K→128K，逐级 RoPE 缩放）
Phase E  WSD 退火 / 高质量数据    —— 20–50B token（LR 衰减段，堆数学/代码/长文）
Phase F  SFT                    —— 1–10B token（指令 + 长上下文 + 工具）
Phase G  RL（GRPO/可选 DPO）      —— 按域分批
（可选）  MTP 头联合训练           —— 从 Phase B 起挂一个 MTP 头，权重 0.1–0.3
```

### 4.0 分栈 / 分层训再合并？——可行，但默认不要拆成两个独立 LM

YOCO 不是 seq2seq：训练时 **同一条序列先后穿过 Encoder 和 Decoder**，loss 在 Decoder 顶。Encoder 与 Decoder 的表示在 MiniCPM 40 层里已经联合训过；定理 A（[`docs/ARCHITECTURE_THEORY.md`](ARCHITECTURE_THEORY.md)）说 gate=0 的 16/24 切分 **就是** 那条残差流。把两栈当成两个独立 LM 分别训再拼接，等于扔掉这份对齐。

完整冻结边界、梯度截断、tied embedding、优化器/激活显存与 split 敏感性见 [`docs/CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)。FP8 模块策略与墙钟见 [`docs/FP8_THEORY.md`](FP8_THEORY.md)。数字：`python3 scripts/param_budget.py --staged --curriculum --fp8`。

**Phase B 配方已按 C1+FP8 定稿**：C1 冻结边界（Phase A 两栈都 MoE → B0/B1 冻 Encoder → B2 短联合）× 混合 FP8。延迟 Encoder MoE、全阶段 1.5×、峰值 2× 只作敏感性，不进配方。联合 bf16 只作 100% 对照。

中间档 50B token、emb 计一次：

| 做法 | H100-h | vs 联合 50B |
| --- | ---: | ---: |
| 两栈一起训（联合 bf16 对照） | 1,354 | 100% |
| Encoder 冻结，只训 Decoder + cross-attn（全程；不共同适应） | 1,035 | 76% |
| 只训新模块（cross-attn + \(W_K/W_V\)；骨干冻结） | 779 | 58% |
| C1 解冻课程 8+27+15B（bf16 操作数账） | 1,090 | 81% |
| **C1+FP8（定稿）** | **761** | **56%** |
| Encoder 当独立 LM 50B + 冻 Encoder 训 Decoder 50B + 20B 拼接恢复 | 2,054 | **152%（更贵）** |
| Encoder 先当 LM 25B 再联合 25B | 916 | 68%（**质量赌博**：联合 token 减半是否够恢复） |

**结论：**

1. **「先分开训两个模型再焊在一起」不省算力。** 同样 50B/栈再加拼接恢复，是联合训练的 1.5×。把 Encoder 隐状态缓存下来给 Decoder 用也不现实（50B token × \(d\) × 2 bytes ≈ 230 TB）。
2. **「同一套切开的权重上，按可训练子集分层解冻」才省。** 省的是 Encoder 的反向（以及新模块阶段 Decoder 骨干的权重梯度），不是少跑 Encoder 前向——YOCO 的 CE 在 Decoder 上，Encoder 前向省不掉。
3. 冻结 Encoder 大约省 **24% FLOPs**，但 Encoder 不再为 Decoder 的 query 改写记忆（write/read 不共同适应）。PDSA 的「无写入时信号」也提示：只训 reader、永远冻 writer，检索上限会卡住。所以冻 Encoder 只能当 **Phase B 的中段**，结尾必须有一段短联合（B2 ≥ 10B，默认 15B）。
4. 只训 cross-attn / indexer（Phase A 热身、Phase C 第 1 步）最省（约 42%+），这是已经写进 Phase C 的做法，不是新发明。
5. 贪心逐层加层（2 层 → 冻 → 再加 2 层）在 LLM 上没有稳定省算力的证据，还要最终联合微调，**不做**。
6. DeepSeek 式「分域专家各自 SFT+RL 再蒸馏」只适用于 **Phase F/G 后训练**，不适用于这套 12B 预训练骨架。
7. **C1 冻结边界已定稿；墙钟再敲死为 C1+FP8。** Phase A 对两栈都 virtual-group MoE，B0/B1 冻 Encoder：一次离线手术，token 0 就是 12.05B 中间档，B2 只解冻。B1/B2 MoE GEMM + 冻结 Encoder 前向走 FP8，B0 student 保持 bf16。发布墙钟 **761 H100-h**。不把 Encoder 推迟到 B2 再上采样，也不把 1.5× 套到 B0 student 上（那是敏感性对照，不进配方）。

**冻结规则（定理 D/E，C1 定稿；实现时写进 trainer，不是口头约定）：**

| 项 | B0 / B1 | B2 |
| --- | --- | --- |
| 全局 cache | `X^{16}.detach()` 再乘 \(W_K,W_V\) | **去掉** detach |
| \(W_K,W_V\) | 新模块，可训练（上界 \(2d^2=10.62\mathrm{M}\)） | 可训练 |
| Tied embedding \(E\) | **冻结**（或解绑后只训 LM head / LP-FT） | 解冻 tied \(E\) |
| Encoder 权重 | 冻结（已是 virtual-group MoE） | 解冻；专家从此特化 |
| Decoder | B0 冻骨干、只训 cross-attn；B1 解冻 self-attn+MoE | 解冻 |

禁止：Encoder 冻结时仍训练 tied \(E\)（输入分布漂，定理 E）。

**定稿配方（预算紧时，用 C1+FP8 替换「Phase B 50B 全程联合 bf16」）：**

| 子阶段 | token | 可训练 | Encoder FFN | gate |
| --- | ---: | --- | --- | --- |
| B0 | 8B（5–10B） | 新模块（cross-attn、\(W_K/W_V\)、gate、新 LN）；骨干 + tied \(E\) 冻结 | 冻结的 virtual-group MoE | 0 → 0.3 |
| B1 | 27B（20–40B） | 解冻 Decoder；**Encoder + tied \(E\) 仍冻**；cache 仍 detach | 同上 | → 1 |
| B2 | 15B（10–20B） | 两栈都解冻，LR 更小 | 解冻；专家开始特化 | 1 |

总 token 仍约 50B。C1 操作数约 **81% 联合**；发布墙钟是 **C1+FP8 = 761 H100-h（联合 bf16 的 56%）**。Adam 状态在 B1 只有联合的 **60%**；detach 丢掉 Encoder 激活约 **40%**。质量不稳就把 B2 加长，而不是回去做两个独立 LM，也不是改回全程联合 bf16。Phase C 的「冻主干、只训 indexer」仍然叠在这套课程**后面**（层内 KL，不是穿过 cache 的 CE；indexer 保持 bf16）。

### Phase A — 架构手术与初始化（离线）

0. **切分为 Encoder / Decoder 两栈（YOCO 化）**：把 MiniCPM-2B 的 40 个 dense 层映射到 **Encoder 16 层 + Decoder 24 层**。
   推荐方案：Encoder 取底座**前 16 层**权重、Decoder 取**后 24 层**权重（保持层深语义）；共享同一份 tie embedding / LM head。
   Decoder 每层**新增 cross-attn 子层**（初始 gate≈0 旁路，见 §2.4），使初始前向≈原 decoder-only 行为，便于恢复。
1. **MoE 上采样（dense FFN → 细粒度 MoE）**，对 Encoder、Decoder **各自**执行，采用 Megatron-LM `upcycling_utils.py`（C1 定稿：两栈都在本阶段完成；B0/B1 再冻 Encoder，见 §4.0）：
   - 把 dense FFN 的中间维切成 G 段、每段复制成多个专家（**virtual-group 初始化**：保证转换瞬间 top-k 恰好选到每个分片的一份副本，等价于原 dense 函数）。
   - **权重缩放**：SwiGLU 专家投影按 `(E·G²/T)^(1/3)` 量级缩放（论文验证约降 1.5% loss）。
   - **路由**：`softmax-then-topK`（优于 topK-then-softmax）；亲和度打分用 **Sqrt(Softplus(·))**（V4 做法）。
   - 每栈**默认全 MoE**（C1：token 0 两栈都是 MoE）。DeepSeek 式「首层 dense」不进发布配方；需要时用 `--first-dense` 敏感性，再把 Enc routed 17→19 补占位账本。
   - **μP**：残差乘子继续用 `scale_depth/√40`，不要按 16/24 重算（见理论验证 §9）。
   - **Hash-MoE bootstrap**：Decoder 最前若干层 MoE 用冻结的 `token_id → expert_id` 哈希路由（V4 做法，稳定早期）。Encoder 在 B0/B1 已冻，Encoder 上的哈希路由多余，放到 B2 解冻时再用。
2. **注意力改造**：继承 `q/k/v/o`（或 SVD 到 MLA 低秩）；新增 CSA/HCA 压缩器、位置偏置、Lightning Indexer 用小尺度随机初始化。此阶段先把所有注意力层当作**稠密/滑窗**跑（不启用 top-k、不启用 HCA 压缩），等价于近似原注意力。
3. **保留 MiniCPM μP 缩放常量**（emb 乘子、`scale_depth`、logits 缩放）。
4. **可选 mHC**：先用普通残差跑通 Phase B/C，稳定后再切 mHC（把残差映射约束到 Birkhoff 多胞形/双随机矩阵，谱范数 ≤1）。

> Encoder/Decoder 交界：Encoder 顶层输出经一个（可学习的）投影 \(W_K,W_V\) 得到全局 `K̂,V̂` 供所有 cross-decoder 层复用（YOCO 单次缓存）。\(W_K,W_V\) 是新模块。B0/B1 必须 `X^{16}.detach()` **之后**再乘投影（定理 D）；B2 去掉 detach。

### Phase B — 上采样恢复性继续预训练

- **目的**：让 MoE 化 + 注意力改造 + encoder-decoder 化后的模型恢复语言建模能力（含 cross-attn 逐步打开）。
- 序列长度 4K，注意力仍为 dense/滑窗（未稀疏），数据用通用预训练混合（见 §5）。
- **逐步打开 cross-attn**：Decoder cross-attn 的 gate 从 0 线性升到 1。解冻课程下：B0（~8B）只升到 0.3，B1 升到 1，避免在冻结骨干上把 \(g\) 拉满。
- **蒸馏加速**：以 **MiniCPM-2B（dense，teacher）** 做 logit KD（KL(teacher‖student)，温度 1–2，权重 0.5→0 线性衰减），大幅缩短恢复期。
- **MoE 负载均衡**：aux-loss-free 偏置法（`e_score_correction_bias`，按各专家负载更新偏置，更新率如 1e-3）+ **轻量 sequence-wise balance loss**（权重 ~1e-3）防单序列极端不均衡。Encoder 专家从 B2 才开始更新，负载监控从 B2 起算 Encoder。
- 学习率：**WSD**（Warmup-Stable-Decay）——短 warmup（0.5–1B token），进入 stable 段（LR ≈ MiniCPM 预训练峰值的 30–50%，因为是继续训练）。此阶段保持 stable 不衰减。B2 解冻 Encoder 时 LR 再降一档。
- **Tied embedding**：B0/B1 冻结（或解绑后只训 LM head）；禁止 Encoder 冻着还训 tied \(E\)（定理 E）。
- **预算紧时不要改成两个独立 LM**：用 §4.0 已定稿的 **C1+FP8**（B0 新模块 → B1 冻 Encoder → B2 短联合；B1/B2 MoE GEMM FP8），同样 ~50B token，墙钟 **761 H100-h（联合 bf16 的 56%）**。

### Phase C — 注意力稀疏化对齐（关键、易翻车）

遵循 DeepSeek-V3.2 "先稠密暖启、再稀疏"的思路引入 DSA/压缩：

1. **Indexer 稠密对齐**：冻结主干，仅训练 Lightning Indexer，让其打分分布**对齐稠密注意力权重**（对 indexer 输出与真实注意力分布做 KL/MSE 对齐）。此步不改变主输出，只教 indexer "该选谁"。监督是**层内**的，叠在 B2 **之后**；不是穿过 YOCO cache 的 CE，也不要用这步永远冻住 Encoder。
2. **打开 CSA top-k**：把 CSA 层从"全可见"切到"top-`index_topk`"，小步继续训练让主干适应稀疏。
3. **打开 HCA 压缩**：启用 `m'=128` 强压缩 + 稠密压缩注意力。
4. **打开 8K 滑窗**：确认滑窗分支与压缩分支 concat/mask 正确，端到端联训。
- 每一步都监控 loss 尖峰；出现不稳定就回退该步、延长对齐或降低 LR。

### Phase D — 长上下文扩展

- 逐级提升训练序列长度：**8K → 32K → 128K**（如需更长可继续）。
- RoPE：按目标长度做频率缩放（NTK/YaRN 类）或直接长序列继续训练；MiniCPM-2B-128k 的 rope_scaling 可作参考。
- CSA/HCA 让长程注意力成本可控；8K 未压缩滑窗保证局部保真。
- 数据切到长文档 / 拼接长样本；用 needle & RULER 做过程监控。

### Phase E — WSD 退火（高质量数据）

- 进入 WSD 的 **Decay** 段：LR 快速（指数/1-sqrt）衰减到峰值的 ~1/100。
- 数据配比切向**高质量 + 数学 + 代码 + 长上下文 + 指令化**（MiniCPM 经验：退火段喂高质量数据收益最大）。

### Phase F — SFT

- 指令/多轮对话/长上下文/工具调用/代码/数学；打包到目标长度，loss 只在 response。
- 可按 DeepSeek-V4 的"**分域专家先各自 SFT+RL，再 on-policy 蒸馏成统一模型**"做，但本项目规模（12B）可先做单一混合 SFT。

### Phase G — RL

- **GRPO**（DeepSeek 系）为主，reward 覆盖数学可验证、代码可执行、指令遵循；可加 DPO 作为轻量偏好对齐。
- **长上下文可用度 RL（关键、且省算力）**：用**可验证的长上下文任务**做 RLVR，直接把"真读、真用长输入"奖励出来——比再喂海量长 token 便宜得多，专治 lost-in-the-middle / 长指令不遵循 / 多跳漏检：
  - **奖励信号**：长文 QA（RULER 式、needle 变体、多跳 HotpotQA 扩展）用 exact-match/F1 可验证；**grounding/引用奖励**（答案必须引用正确 passage/行，契合 §14 PDSA 证据选择）；长指令遵循（约束可检查）。
  - **课程**：在 128K→256K 逐级做 RL，把关键信息刻意放到**中段/长距**，强化中段召回（与 §12 的 IN2/FILM 训练互补）。
  - **省算力**：长 trace RL 每 rollout 成本高 → 用 **PS-PPO（prefix-sampling PPO）** 只回传采样前缀、无偏截断，显著降长序列 RL 的算力/显存；或对超长上下文用 PDSA 选证据后再 RL（缩短有效 rollout 长度）。
- RL 阶段注意 MoE 路由与稀疏注意力在长 rollout 下的稳定性。

---

## 5. 数据

| 阶段 | 主要数据 | 量级（token） |
| --- | --- | --- |
| B 恢复 | **OpenBMB**：Ultra-FineWeb en 60% / zh 30% + UltraData-Math L2 10% | 50B 信封（prepare 按 `--max-tokens` 切片） |
| C 稀疏化 | 与 B 同分布，偏长文档 | 20–50B |
| D 长上下文 | 长文档、书籍、代码仓库级拼接、合成长依赖任务 | 20–60B |
| E 退火 | 高质量精选 + 数学 + 代码 + 指令化 SFT 前体 | 20–50B |
| F SFT | **UltraChat** 等指令/多轮（不是 Phase B） | 1–10B |
| G RL | 可验证任务 prompt 集（数学/代码/agent） | prompt 级 |

要点：

- **Tokenizer 必须是 MiniCPM-2B**（`openbmb/MiniCPM-2B-sft-bf16`，`V=122753`）。不要用 MiniCPM3 / MiniCPM4 tokenizer 去喂 2B 上采样图。Ultra-FineWeb 是 MiniCPM4 时代网页过滤集，**要重新 tokenize**。
- 默认 mix `phase-b` 全是 OpenBMB。可选 `phase-b-code` 把 10% 换成 StarCoder（不是 OpenBMB；Ultra-FineWeb 论文评测 mix 用过 10% 代码）。
- UltraChat / 指令对话留给 Phase F/G，不进 Phase B。
- 实现：`python3 -m cat_yoko.prepare --mix phase-b --tokenizer openbmb/MiniCPM-2B-sft-bf16 --out data/phaseb.bin --max-tokens 1e8` → int32 packed mmap；`cat_yoko.train --data data/phaseb.bin --upcycle-hf openbmb/MiniCPM-2B-sft-bf16`。sidecar `*.bin.meta.json` 带 `eos_id`。仓库 **不检入语料**。
- 长上下文样本用文档拼接 + 合成"大海捞针/多跳"；严格去重与评测集去污。Ultra-FineWeb 许可证标 Apache 2.0，源网页版权仍按各站条款。

---

## 6. 优化器 / 超参 / 稳定性

| 项 | 推荐 |
| --- | --- |
| 优化器 | **发布默认 AdamW**（\(\beta=(0.9,0.95)\)，wd=0.1）。Muon 开关留着、**默认关**（见 [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)） |
| Muon | 对动量做 Newton-Schulz 正交化；配 **hybrid ZeRO** 实现（V4 做法）；lr 需单独调（通常比 Adam 大） |
| LR 调度 | **WSD**：warmup(0.5–1B) → stable → decay；继续训练峰值取底座预训练峰值的 0.3–0.5× |
| Batch | 全局 batch 随阶段增大（如 4M→16M token/step）；长上下文阶段用 seq packing |
| 精度 | **混合精度定稿**（[`docs/FP8_THEORY.md`](docs/FP8_THEORY.md)）：B1/B2 的 MoE 专家 GEMM + 冻结 Encoder 前向 GEMM 用 FP8；B0 student、L0、Phase C indexer、tied \(E\)、RMSNorm、router、gate、attn softmax 保持高精度。不改 6NT；墙钟按 1.5× 计。V4 式 FP4 专家存储是后期可选项，不进本配方 |
| 正则/稳定 | zero-centered & weight-decayed RMSNorm、router z-loss（轻）、grad clip 1.0 |
| MoE 均衡 | aux-loss-free 偏置更新 + 轻量 seq-balance loss；监控专家利用率/丢弃率 |
| MTP | 辅助头权重 0.1–0.3；可只在 B–E 用，推理可丢弃或用于投机解码 |
| μP | 保留 MiniCPM 的 emb/residual/logits 缩放常量 |

Muon 的 Newton-Schulz 正交化保持 fp32，与网络 FP8 GEMM 正交。不要在 L0 开 FP8。

---

## 7. 评测与消融

**能力评测**：MMLU / CMMLU / C-Eval（知识），GSM8K / MATH（数学），HumanEval / MBPP（代码），
BBH（推理），IFEval（指令遵循）。
**长上下文**：**RULER**、**Needle-in-a-Haystack**、LongBench；核对 8K 滑窗 + 压缩长程 + YOCO 全局 cache 在 32K/128K/1M 的检索保真（YOCO 报告 1M 近乎满分 needle）。
**效率**：单 token 推理 FLOPs、**KV cache 大小（YOCO 只缓存一次，应显著低于 decoder-only 基线）**、prefill 延迟（encoder early-exit 收益）、decode 吞吐。
**必做消融**：
1. **Encoder/Decoder 层数拆分**（如 16/24 vs 20/20 vs 12/28）对 2.3B/4.5B 激活与质量的影响；
2. `n_win ∈ {2K, 4K, 8K}` 对质量/吞吐的权衡；
3. CSA:HCA 层比例（1:1 vs 2:1 vs 3:1）；
4. `index_topk ∈ {128,256,512}`；
5. MoE 粒度/专家数（`moe_intermediate_size`、`n_routed`、`top_k`）对 12B 总量 / 各档激活目标的命中；
6. **YOCO decoder-decoder vs 等参数 decoder-only**（验证 KV cache / prefill 收益且不掉点）；
7. **是否引入 KDA 及 KDA:CSA:HCA 比例**（如纯 CSA/HCA vs 3:1 KDA混合 vs 6:1）——重点看 RULER/多跳中段召回是否**因加 KDA 而下降**（预期线性层会略降精确召回，需 full/CSA 锚点补偿）与长上下文吞吐/KV cache 收益；
8. 上采样 vs 从底座 dense 直接继续训练（验证 upcycling 收益）；
9. Muon vs AdamW；mHC vs 普通残差；cross-attn gate 渐开 vs 直接开；full 锚点层 NoPE vs RoPE。
10. **解冻课程**：C1（定稿）vs 全程联合 vs **非法** B2=0（课程篇 §9；看恢复 PPL 与 RULER，不是只看 FLOPs）。延迟 Encoder MoE 只作敏感性，不进配方。
11. **FP8**：B1 `fp8_moe` vs 全程 bf16（看恢复 PPL / 溢出，不是只看墙钟）；B0 student 误开 FP8 作负对照。白名单模块必须仍是高精度。

---

## 8. 基础设施

| 组件 | 建议 |
| --- | --- |
| 训练框架 | 本仓库参考实现是 **PyTorch**（`cat_yoko.train --backend torch`）。规模化走 **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** / Megatron-Core（MoE + EP/TP/PP/CP + upcycling）。映射：`cat_yoko.megatron.mapping.megatron_blueprint`；`--dump-megatron` 打 JSON。YOCO **不是** `GPTModel`。 |
| 并行 | `ParallelPlan`：TP/PP/EP/CP/SP。12B：TP ∈ {1,2,3,4,6,9,12,18,36}；**EP ∈ {1,17}**（17 质数）。PP>1 时 encoder|decoder 切在第 16 层（`pipeline_split_rank`）。长上下文用 Context/Sequence Parallel。 |
| 注意力 kernel | **FlashMLA** 稀疏 prefill/decode kernel（支撑 DSA，FP8 KV）；**NSA** 的 Triton kernel 可参考压缩+选择+滑窗三分支实现 |
| MoE kernel | 融合的 MoE dispatch/combine kernel（计算/通信/访存 overlap） |
| 精度 | bf16 master + 定稿 FP8 GEMM（§6 / [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md)）；确定性/可复现 kernel（可选） |
| 显存 | 张量级重计算（`--grad-ckpt`）、B0/B1 冻结 Encoder CPU offload、B2 逐层 offload、Adam 动量 CPU offload（`--optim-cpu`）；规模化再 ZeRO / 专家 offload |
| 推理 | vLLM / SGLang（已集成 DSA/FlashMLA 稀疏 kernel）用于评测与 RL rollout |

> 若无法自研 CSA/HCA kernel，**起步可用 HuggingFace `transformers` 的 `DeepseekV4` 参考实现**
> （`layer_types`、`compress_rates`、`sliding_window`、`index_topk` 等已暴露）跑通正确性与小规模训练，
> 再迁移到高性能 kernel 做规模化。

---

## 9. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 稀疏注意力训练不稳定 / 掉点 | 严格走 Phase C"稠密对齐→逐步稀疏"；indexer 先单独对齐；出问题即回退单步 |
| MoE 负载坍塌 / 专家闲置 | aux-loss-free 偏置 + seq-balance loss + Hash-MoE bootstrap + 监控利用率 |
| 上采样后能力回退 | virtual-group 初始化 + 权重缩放 + teacher 蒸馏 + LR 重置到较高 stable 段 |
| 8K 滑窗成本过高 | 消融回调 `n_win`；仅部分层用 8K；长程交给 CSA/HCA |
| μP 缩放丢失导致数值漂移 | 手术后保留 MiniCPM 全部缩放常量并单测前向尺度 |
| Muon 不收敛/超参陌生 | 先用 AdamW 跑通基线，再切 Muon 并单独扫 lr；保留回退开关 |
| kernel 缺失 | 先用 HF 参考实现验证正确性，再上高性能 kernel |
| FP8 溢出 / loss 尖峰 | B0 student 保持 bf16；B1 切 `fp8_moe` 时盯 NaN 与专家利用率；回退该阶段 dtype，不改 C1 冻结边界 |
| 长上下文外推差 | 分级 RoPE 缩放 + 长样本课程 + RULER 过程监控 |

---

## 10. 里程碑（以能力/预算计，不以日历计）

1. **M1 手术就绪**：离线得到 `CAT-YOKO-12B`（Encoder 16L / Decoder 24L）初始权重，前向数值尺度自检通过，cross-attn 旁路下短跑 loss 不发散。
2. **M2 恢复达标**：Phase B 后（cross-attn 全开），通用 benchmark 恢复到 MiniCPM-2B 的 ~95%+。
3. **M3 稀疏化达标**：Phase C 后开启 CSA top-k + HCA + 8K 滑窗，短上下文质量与 M2 基本持平，效率明显改善。
4. **M4 长上下文**：**128K–256K RULER/Needle 通过且质量可用（主目标）**；1M 作为 stretch（推理跑得起 + needle 能过即可，不追质量）；单 token FLOPs 与 **KV cache（YOCO 单缓存）** 显著低于 decoder-only 对照，prefill early-exit 收益兑现。
5. **M5 后训练**：SFT + GRPO 后，指令/数学/代码达到目标区间，产出可发布 checkpoint。

---

## 11. 立即可做的下一步

发布规格已敲死，见 [`docs/FROZEN_SPEC.md`](FROZEN_SPEC.md)。下一步是跑仓库里的 **12B 训练代码**（tiny 单测 → meta 12B 图 → `--dump-megatron` → 有卡再 FSDP / Megatron）。

1. `python3 -m unittest tests.test_param_budget tests.test_arch_verify tests.test_train tests.test_trainer tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload`
2. `python3 -m cat_yoko.prepare --mix local --local texts.jsonl --tokenizer dummy --config tiny --out /tmp/t.bin --max-tokens 256` 然后 `python3 -m cat_yoko.train --config tiny --phase B0 --steps 3 --accum 1 --data /tmp/t.bin`
3. 有 GPU：`python3 -m cat_yoko.gpu_smoke`（tiny）；`python3 -m cat_yoko.gpu_smoke --middle`（12B B0 一步，≥28GiB，bf16 直接建图）；`--middle --phase B1`（Encoder 卸载 + CPU Adam）；`--c1`（同一张 12B 图 B0→B1→B2）
4. `python3 -m cat_yoko.train --config 12b --meta`（数参数，不分配 24GB）
5. `python3 -m cat_yoko.train --config 12b --dump-megatron`（双栈 TransformerConfig JSON，不跑 Megatron）
6. 有网 + GPU 时：`pip install 'cat-yoko[data]'`，`prepare --mix phase-b --tokenizer openbmb/MiniCPM-2B-sft-bf16 --out data/phaseb.bin --max-tokens 1e8`，再 `--config 12b --phase B0 --upcycle-hf openbmb/MiniCPM-2B-sft-bf16 --data data/phaseb.bin --save-dir runs/b0 --dtype bf16 --grad-ckpt --device cuda --steps N` 按 C1+FP8 开训。B1/B2 用 `--resume` 接 `latest.pt`（权重 + packed 游标 + RNG；不恢复上一阶段 Adam / step）。也可用 `--c1 --save-dir runs/c1 --steps N` 在同一张图上连跑三阶段，写出 `runs/c1/{B0,B1,B2}/latest.pt`。12B 默认不存 Adam。单卡 32GB + ~62GiB host cgroup：B0 直接一步；B1 卸冻结 Encoder + CPU Adam（一步 smoke 走 ephemeral 动量）；B2 逐层 offload，backward 完一层就 clip+Adam（`--accum 1`）。规模化再 `--backend megatron`。不要在小 VM / CI 上下载 Ultra-FineWeb 或 12B 权重。4M global batch / 全参 GPU Adam 仍要多卡或 ZeRO。

不要再改 16/24、C1、C1+FP8、因果 Encoder、M2 默认。质量问题加长 B2 或回退 dtype，不改冻结边界。

---

## 12. 更多先进技术（按 价值/风险 分层，避免堆砌）

> 原则：新颖组件越多、训练越难。以下按"低风险先上、高风险后上/可选"排序，每个都应能独立开关与回退。

### Tier 1 — 低风险高收益（建议默认加）

- **QK-Norm**（对 query/key 做 RMSNorm）+ **z-loss**（router-z 抑制路由 logit 爆炸 + 输出 logit z-loss）+ **双 RMSNorm（pre+post，OLMo2/Gemma2 式）**：深层 + MoE + 稀疏注意力这种新颖栈的关键稳定器，成本极低。
- **文档感知注意力掩码**：packed 长序列内**不跨文档**注意，避免污染长上下文训练信号。
- **FIM（Fill-in-the-Middle）**：代码数据填中训练，提升补全/编辑能力。
- **IN2 / 信息密集型长上下文训练（FILM 类）**：合成"关键信息位于长文**中段**"的训练样本——**这才是修 lost-in-the-middle 的正解**，比加 KDA 直接有效（见 §2.5 的澄清）。

### Tier 2 — 中风险高收益（base 稳定后加）

- **MTP → 投机解码**：复用已挂的 MTP 头做 EAGLE 式自投机，推理提速；训练侧几乎零额外成本。
- **FP8 训练**（[`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) 定稿）：B1/B2 MoE 专家 GEMM + 冻结 Encoder 前向；B0 student / indexer / 白名单模块保持高精度。发布墙钟 1.5×，不发布 2×。**FP4 专家存储**是后期可选项，不进本配方。
- **attention logit soft-cap / QK-clip**（Gemma2 / Kimi）：抑制极端 logit，进一步稳训练。
- **RoPE/NoPE 校准与频率缩放（YaRN）**：长上下文外推 + 缓解位置偏置。

### Tier 3 — 谨慎 / 最后上（默认先不上）

- **mHC**（已在 §2，标可选）、**Muon**（留 AdamW 回退）、**3 路 KDA 混合**（§2.5）。
- **零计算 / 弹性 top-k 专家**：按 token 难度自适应激活量（潜在契合"2.3B/4.5B 非对称"，但复杂、易不稳，作为研究项）。
- **共享注意力块（Zamba 式）跨多层复用**：进一步省参/省 cache，但耦合强。

---

## 13. 训练难度管理：分级去风险（重要）

**核心：不要一次点亮所有新组件。** 组合创新（YOCO + CSA/HCA + 可选 KDA + DeepSeekMoE + Muon + mHC + MTP）的风险是叠乘的，
逐步引入 + 每步可回退 + 指标监控，是唯一稳妥路径。

**去风险阶梯：**
- **L0（tiny 正确性）**：小配置（`hidden 256, enc2L/dec2L, sliding=8, m=4, m'=8, index_topk=2`）验证：YOCO 数据流（encoder→全局 cache→cross-decoder）、CSA/HCA/KDA 的 mask 与 kernel、MoE 路由/均衡。只看"能不能对、会不会 NaN"。**精度 bf16，不开 FP8。**
- **L1（半规模去风险原型，≈3–6B）**：用**完整新颖架构栈**但**专家数减半**，在几十 B token 上跑通稳定性、上采样恢复曲线、稀疏化对齐、cross-attn 渐开。廉价的架构验证台。
- **L2（扩到 12B 目标）**：**MoE 专家数是最安全的扩展轴**——架构在 L1 验证后，半规模→12B 主要是加 routed 专家（+ 少量继续训练让新专家分化），风险远低于改架构。
- **每个新组件单独一步**：稀疏化 → KDA → mHC → Muon；**FP8 不进 L0**（tiny 保持 bf16），从 B1 开 MoE GEMM，与 Muon 正交化分开留回退开关。盯 loss 尖峰/专家利用率/召回指标；坏了就回退该步。

**难度—收益取舍速查：**

| 想省事/快出成果 | 想要极致长上下文效率 |
| --- | --- |
| 先 decoder-only + CSA/HCA（不上 YOCO/KDA/mHC/Muon），跑通再逐步加 | 全栈 YOCO + 3:1 KDA + CSA/HCA + FP8，但严格走 L0→L1→L2 |

---

## 14. 融合 PDSA 记忆管理（可训练生命周期 + 校准回退）

> 依据：**Memory-Managed Long-Context Attention**（Zou & Donz，arXiv `2606.28876`，本团队工作，下称 PDSA）。
> 该文是**围绕冻结 LLM 的推理/评估层记忆系统**（不是可训练架构），核心为：query-independent 写入器 +
> 硬边界生命周期（overwrite/protection/eviction，≤32 槽）+ query-aware 读取器 + **校准稀疏回退** + 冻结 LLM 从原始证据生成。
> 它与 CSA/HCA/KDA **正交、互补**：后者是 token 级状态压缩，PDSA 是语义单元级受管理记忆。
> PDSA 明确的"下一步"是把生命周期**做进模型、可训练**——CAT-YOKO 正好可作为该实例。

### 14.1 关键可迁移结论

- **"无写入时信号"边界（PDSA §5 实测）**：静态文本上，不看 query 的可写性判断 ≈ 随机（AUC 0.63–0.66 vs query-aware 0.89–0.97），纯 bounded memory 只能召回 ~0.56 黄金证据。**推论**：任何 **write-first 压缩**（HCA、KDA、甚至 CSA 的压缩步）都会系统性丢掉"写入时无信号、事后才被 query 命中"的信息 → **必须保留一条到低压缩/原始 KV 的 query 时回退**。
- **bounded 选择在长文上优于读全文（PDSA §4）**：8.2k 词时读全文反而掉分（lost-in-the-middle），≤10% 证据即达全文 F1 的 102–116%。佐证 CAT-YOKO"压缩 + 选择"路线方向正确。

### 14.2 分层集成方案（契合 §13 难度管理）

- **Tier 1（低风险，建议加）— 校准置信度门控的稀疏回退**：
  给 CSA 的 Lightning Indexer 增加**置信度信号**（如 top-k 得分的均值/熵）；低于阈值时**扩大 top-k 或回退到更少压缩/原始 KV 检索**。
  阈值**在部署长度 regime 上校准**（PDSA 记录的负结果：在短上下文校准会让回退永不触发、长文覆盖崩溃）。
  这是**用本团队自己的实测**对治前述"中段召回"担忧的原则化手段，且属推理期增量、可后加。
- **Tier 2（中风险）— query-aware 优先于 write-first**：层调度上**多用 CSA（query-aware 选择）**、审慎用 HCA/KDA（write-first）承担关键检索；把 HCA/KDA 定位为"廉价 gist 覆盖 + 兜底"，精确检索交给 CSA + full 锚点 + 回退。
- **Tier 3（研究级，base 稳定后）— 可训练 bounded editable memory lifecycle**：
  把 YOCO 的"只增全局 cache"升级为**有界、可编辑、带生命周期**的记忆：学习到的**写入器**（write/overwrite/protect/evict，按 key/salience）管理一个容量受限的记忆，decoder 的 cross-attn 读取它。
  收益：KV cache 真正有界 + **版本化/保护语义**（agent、长程任务差异化）；风险：switched-process 稳定性（PDSA Appendix H）、写入不稳定，需谨慎——属未解研究，单独里程碑推进，留符号化/冻结回退。

### 14.3 对应消融（补入 §7）

- 有/无**校准稀疏回退**在 RULER/多跳中段召回与长文 F1 上的差异；回退阈值**跨长度 regime 校准**的敏感性。
- **CSA 比例 ↑（query-aware） vs HCA/KDA 比例 ↑（write-first）**对"事后才被命中的信息"召回的影响。
- （研究项）可训练 editable memory vs 只增全局 cache：KV cache 上界、版本化任务正确性、稳定性。

> 定位提醒：PDSA 的贡献是**记忆管理**，不替代 CSA/HCA/KDA 的**状态压缩**；二者叠加才是完整方案。
> 不迁移其冻结-reader 评估台架与 32 槽具体数字（那是方法学证据，非架构）。

---

## 15. 算力预算估算与低预算路线（重要现实约束）

> 前提修正：本计划最初默认"几十~上百张 H100"。**若算力有限，24B 从头/重继续预训练不可行**，需重新排优先级：
> **先在小规模验证架构，有预算再放大**。要证明的是"架构创新（YOCO×CSA/HCA + KDA + PDSA 可训练生命周期）"，
> 不是"大模型规模"——相邻 hybrid-linear 分析在 340M/1.3B 即完成，本团队 PDSA 核心组件仅 ~2.74M 参数 + 冻结 backbone。

### 15.1 训练算力估算

`训练 FLOPs ≈ 6 × N_active × tokens`。下表 bf16 行按 40% MFU（H100 有效 ~4.0e14、A100 ~1.25e14 FLOPS）把操作数换成小时；**默认 Phase B 墙钟是 C1+FP8**，不是联合 bf16。

| 方案 | H100-h | A100-h | 8×H100 天 |
| --- | ---: | ---: | ---: |
| **C1+FP8（Phase B 定稿）** | **761** | **2,434** | **4.0** |
| C1 解冻课程 8+27+15B（bf16 操作数账） | 1,090 | 3,487 | 5.7 |
| 12B 中间档 × 50B tok（emb 只计一次；联合 bf16 对照） | 1,354 | 4,331 | 7.1 |
| 12B 中间档 × 50B tok（enc+dec 激活，emb×2 计划口径） | 1,413 | 4,520 | 7.4 |
| 12B 中间档 × 200B tok | 5,652 | 18,080 | 29.4 |
| 24B × 200B tok（远期；按当时 3B+6B 激活） | 7,583 | 24,038 | 39.5 |
| 24B × 50B tok（远期最小恢复） | 1,896 | 6,010 | 9.9 |
| ~6B × 60B tok | 885 | 2,804 | 4.6 |
| ~3B × 50B tok | 316 | 1,002 | 1.6 |
| ~1B × 20B tok | 51 | 160 | 0.3 |
| ~0.5B × 10B tok（架构验证） | 13 | 40 | 0.1 |

> **C1+FP8 = 761 是 Phase B 发布墙钟。** 1,354 / 1,413 / 1,090 是 bf16 操作数对照。FP8 **不改** 6NT（[`FP8_THEORY.md`](FP8_THEORY.md)）。发布加速比 **1.5×**（MoE 活跃份额 71.5% 的 Amdahl 为 1.56×）；**2× 只是 H100 峰值上界，不写入配方**。蒸馏/upcycling 显著减少所需 tokens；长上下文阶段占比小、另计。

### 15.2 三条低预算路线（按实际卡数选）

- **Route A — 架构验证（最省，≤ 几张卡，~10–50 H100-h，可租）**：upcycle 小 MiniCPM（1B/2B）→ **0.5–1.5B 小 MoE**，装 YOCO+CSA/HCA(+可选 KDA)，继续训 10–20B tok。目标：证明这套注意力/编解码器能跑、不掉点、长上下文省 KV。**推荐作为默认起点。**
- **Route B — 放大到 12B 目标（~8×A100/H100 两周档或租；Phase B 定稿 C1+FP8 ≈ 761 H100-h）**：MiniCPM-2B → **12B** upcycle（可先经 3–6B 里程碑），继续训 50–60B tok + 短长上下文阶段，得到目标模型。联合 bf16 1,354 只作对照。
- **Route C — PDSA 扩展（几乎不花训练算力，契合已有工作）**：冻结 backbone，仅训小组件（写入器/reranker/阈值）+ 落地"校准回退 / 可训练 editable memory"（§14）。**零预算最优**，直接产出 PDSA 的可训练生命周期后续。
- **Route D — 24B（远期，暂不作为目标）**：仅在拿到真集群/算力资助后再考虑放大。

### 15.3 省算力杠杆（优先级从高到低）

1. **upcycling**（复用 MiniCPM 权重，绝不 from-scratch）；2. **蒸馏**（teacher=MiniCPM，减 tokens）；
3. **高稀疏 MoE**（减激活参数=减 FLOPs）；4. **4K 上下文占训练大头**，长上下文只短暂一段；
5. **解冻课程**（§4.0 / [`CURRICULUM_THEORY.md`](CURRICULUM_THEORY.md)：**C1 定稿**——两栈先 MoE、B0/B1 冻 Encoder，约省 19% Phase B FLOPs、B1 Adam 状态 60%、Encoder 激活 ~40%；结尾必须短联合 B2≥10B）；
6. **FP8**（[`FP8_THEORY.md`](FP8_THEORY.md)：**与 C1 敲死为 C1+FP8**——B1/B2 MoE 专家 GEMM + 冻结 Encoder 前向；B0 student / L0 / indexer / 白名单保持高精度；发布 1.5×。Phase B 墙钟 **761 H100-h，联合 bf16 的 56%**；2× 只作峰值上界）；7. **Muon**（减步数；Newton-Schulz 仍 fp32，与 FP8 GEMM 正交）；8. **新模块全训 + 其余 LoRA**（减优化器显存，能上更小/更少卡）；
9. **关键短跑租 spot GPU**（不必自购）；10. seq packing + 激活重计算（塞进更少卡）。

### 15.4 修订后的默认路径

**L0 tiny 正确性 → Route A（0.5–1.5B 架构验证，出架构论文）→ 有预算再 Route B（放大到 12B 目标）→ 远期（可选）Route D（24B）。**
§1.2 的 **12B 为目标规格**；预算紧时**当前默认先执行 Route A**，§3 的预算脚本可直接把 `Nr_e/Nr_d` 调小到 0.5–1.5B 档做验证。

---

## 16. 长上下文可行性（主目标 128K–256K；1M 为 stretch）

> 先定调：**主交付目标是"前 128K–256K 可用"**，这在 12B 上是现实的；下面对 1M 的讨论是"能不能支持得起"的 stretch 分析，**不是**要在 1M 上对标 2T 前沿质量。

结论分三层，别混为一谈：

### 16.1 推理侧：非常有戏（这套架构就是为 1M 设计的）

关键是 KV cache。**YOCO 只缓存一次（单一全局 cache）+ CSA/HCA 序列压缩**，把 1M 的 KV 从"放不下"压到"零头"：

| 12B 模型在 1M token 的 KV cache | 大小 |
| --- | --- |
| decoder-only + MHA（全部 40 层缓存） | ≈369 GB（放不下） |
| decoder-only + GQA4 / MLA | ≈41 / 46 GB |
| **YOCO + MLA（单一全局 cache）** | **≈1.15 GB** |
| **YOCO + MLA + CSA \(m=4\)（全局 cache 沿序列 ÷4）** | **≈0.29 GB** |
| YOCO + MLA + 额外序列÷8（stretch，非 CSA 默认） | ≈0.14 GB |
| （加 24 层 8K 滑窗分支，与 N 无关） | +≈0.23 GB |

再叠加 **encoder(self-decoder) 的 prefill early-exit**——超长输入只需跑完 encoder 产出全局 cache，不必跑满全部层——1M **prefill 也便宜**。这正是"输入侧轻(≈2.3B)"的意义。中间档 prefill 激活份额约 34%。**Decode 侧**：若全局 cache 不压缩，128K/256K 上 24 层 cross-attn 分别约为 decoder MLP 的 3.2× / 6.5×，必须走 CSA \(m=4\) 或 top-\(k\) 选择，否则 encoder 省下的算力会在 cross-attn 被吐回（见理论验证 §6）。YOCO 原论文在 1M 报告近满分 needle 检索。**所以 1M 推理在中等硬件上都可行。**

### 16.2 训练出"支持 1M（needle/RULER 通过）"：现实，但要花心思

不 pretrain 在 1M；主训练在 4–8K，末尾加一个**渐进长上下文扩展阶段**（8K→32K→128K→256K→1M）。要点：
1. **RoPE/YaRN 缩放**到目标长度；2. **长数据**：书/代码仓库级拼接 + 合成长依赖 + IN2 中段样本；3. **渐进长度课程**；
4. **训练期显存瓶颈是 1M 序列的激活**（不是 KV）→ 用 **context/sequence parallelism**；YOCO early-exit + CSA/HCA 压缩显著降激活；**非对称设计天然契合**（encoder 处理长输入、decoder 生成短 → 长 prefill 便宜）；
5. **验证**：RULER-1M / needle-in-haystack。这一阶段 token 量不大（几 B～十几 B），成本相对主训练小，可租卡短跑。

### 16.3 对标前沿 1M 质量：小预算达不到

DeepSeek/Kimi 的 1M 是 32T 级数据 + 大算力喂出来的"充分利用"。小预算能拿到"**支持 1M + needle/RULER 不错**"，但"1M 上多跳深推理达到前沿"不现实——如实说明。

### 16.4 低预算实操建议

- **分阶段**：先稳 **128K–256K**（便宜、够用），再单独冲 **1M capability** 并用 needle/RULER 验证；别一上来就 1M。
- **PDSA 路线是"有效 1M"的省钱替代**：bounded editable memory + 校准稀疏回退 + 检索（§14），**不必训练原生 1M 注意力**就能拿到长程召回——你自己的工作，且 §5 实测显示在 8.2k 上 bounded 选择已优于读全文。对极限长上下文，这可能比硬训 1M 注意力更划算。
- 里程碑上把 1M 归入 **M4**（长上下文），作为 capability 目标而非质量目标。
- **后期用 RL 提升"可用度"**：预训练/扩展只解决"能吞下 128K–256K"，**能不能真用好**很大程度靠后期 **长上下文 RLVR**（见 §4 Phase G）——用可验证长文任务 + grounding 奖励把中段召回/长指令遵循直接优化上来，且用 PS-PPO/PDSA 选证据把长 rollout 成本压下来。这是小预算下把"可用度"再抬一档的关键杠杆。

> **定位（重要，避免误解）**：本项目**不是**用 12B 去对标 2T 级前沿模型的质量——那不可能。
> **主交付目标是"前 128K–256K 长上下文可用"**（在 12B 规模上，这理论与工程都有戏）；
> 1M 只是"架构支持得起 + needle/RULER 能过"的加分项，不是质量目标。
> 想要更极限的长上下文又省钱，优先走 PDSA 记忆 + 检索回退（§14），而非硬训原生 1M 注意力。

> 一句话：**主战场 = 128K–256K 可用；1M 推理/needle 是 stretch；2T 级前沿质量不在目标内。**

---

## 参考（本计划的架构依据）

- **DeepSeek-V4**（CSA/HCA、mHC、Muon、MTP、Hash-MoE bootstrap；V4-Flash 284B/13B、1M ctx、32T tokens）：arXiv `2606.19348`；HuggingFace `transformers` `deepseek_v4` 模型文档（`layer_types`、`compress_rates`、`sliding_window`、`index_topk`、`mlp_layer_types` 等配置）。
- **DeepSeek Sparse Attention (DSA)** 与 **FlashMLA** 稀疏 kernel（Lightning Indexer + top-k + FlashMLA）：DeepSeek-V3.2 报告；`deepseek-ai/FlashMLA`。
- **Native Sparse Attention (NSA)**（压缩 + 选择 + 滑窗三分支、硬件对齐、可原生训练）：arXiv `2502.11089`。
- **Upcycling LLMs into MoE**（virtual-group 初始化、权重缩放、softmax-then-topK；Megatron `upcycling_utils.py`）：arXiv `2410.07524`。
- **DeepSeekMoE**（细粒度专家 + 共享专家）：arXiv `2401.06066`。
- **MiniCPM**（2.4B 非嵌入参数、WSD 调度、μP 缩放；MiniCPM-2B-128k 长上下文变体）：OpenBMB 官方 config / 博客。
- **Gemma 2 / Qwen3-Next**（局部滑窗 × 全局注意力交错、混合注意力层比例）：作为层调度与滑窗设计参考。
- **YOCO — You Only Cache Once**（decoder-decoder：self-decoder 产出单一全局 KV cache，cross-decoder 复用；prefill early-exit；1M ctx 近满分 needle）：arXiv `2405.05254`；`microsoft/unilm` YOCO。
- **Kimi Linear / KDA**（Kimi Delta Attention：细粒度门控 Gated-DeltaNet + DPLR chunk kernel；3:1 KDA:MLA 混合，MLA 用 NoPE；1M KV cache ↓~75%、解码 ↑~6×）：arXiv `2510.26692`；`MoonshotAI/Kimi-Linear`。
- **Hybrid Linear Attention 系统分析**（线性注意力召回弱、需 full 层补偿；gated-delta 在 3:1~6:1 达 Transformer 级召回）：arXiv `2507.06457`。
- **Lost in the Middle**（中段位置偏置，softmax 亦有，靠位置编码校准缓解）：arXiv `2307.03172`。
- **FILM / IN2 训练**（信息密集型长上下文训练，合成"关键信息在中段"样本以修 lost-in-the-middle）：`Make Your LLM Fully Utilize the Context`，arXiv `2404.16811`。
- **OLMo 2 / Gemma 2**（QK-Norm、双 RMSNorm、logit soft-capping、z-loss 等稳定性技巧）：arXiv `2501.00656` / `2408.00118`。
- **EAGLE / 投机解码**（复用 MTP 头做自投机加速）：arXiv `2401.15077`。
- **YaRN**（RoPE 长上下文外推缩放）：arXiv `2309.00071`。
- **PS-PPO — Prefix-Sampling PPO**（critic-free RLHF 只回传采样前缀、无偏截断，降长 trace RL 算力/显存）：arXiv `2606.29758`。
- **PDSA / Memory-Managed Long-Context Attention**（有界可编辑记忆 + 硬生命周期 overwrite/protection/eviction + query-independent 写入器 + query-aware 读取 + 校准稀疏回退；实测"无写入时信号"边界、bounded 选择在长文优于读全文）：Zou & Donz，arXiv `2606.28876`（本团队工作；其"下一步"为可训练生命周期，本计划 §14 承接）。
- **MSA — Memory Sparse Attention**（静态文档稀疏记忆，PDSA 的最近邻）：arXiv `2603.23516`。
- **Gated DeltaNet / Gated DeltaNet-2**（KDA 的前身；解耦擦除与写入）：arXiv `2412.06464` / `2605.22791`。
