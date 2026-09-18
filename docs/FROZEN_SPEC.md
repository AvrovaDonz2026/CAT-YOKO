# CAT-YOKO 发布规格（已敲死）

> 这份表是训练代码的输入。旋钮不再开放；要改就改版本号，不要在实现里重新讨论。
> 中间档账本 / 因果 / C1 / C1+FP8 仍以四篇理论为准。这里只钉**实现默认**。

机器可读副本：`cat_yoko/config.py` 的 `CATYokoConfig.middle_12b()`。

---

## 0. 一句话

**CAT-YOKO-12B**：MiniCPM-2B 上采样的因果 YOCO MoE；Phase B 按 **C1+FP8** 训；注意力 Phase B 只跑滑窗 + 门控 cross-attn；M2 关；KDA / mHC / MTP / Muon / PDSA 默认关。

仓库训练代码是这份 12B 图的 **PyTorch 参考实现**；规模化并行预留 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 适配面（`cat_yoko.megatron`，双栈 TransformerConfig + model_provider 钩子）。**不要**把 YOCO 塞进现成 `GPTModel`。tiny 配置只给单测。

---

## 1. 架构

| 项 | 定稿 |
| --- | --- |
| Encoder | **因果** self-decoder，不是双向 |
| 切分 | MiniCPM **前 16 层 → Encoder，后 24 层 → Decoder** |
| 隐藏维 / 词表 / 头 | \(d=2304\)，\(V=122753\)，36 头，\(d_h=64\)，**tied** \(E\) |
| μP | `scale_emb=12`，`dim_model_base=256`（logits /9），残差 `1.4/√40`（**不要** √16/√24） |
| MoE | DeepSeek 细粒度，`moe_intermediate_size=2048`，SwiGLU |
| Enc 专家 | 1 shared + **17** routed，top-\(k=6\) |
| Dec 专家 | 1 shared + **17** routed，top-\(k=8\) |
| 首层 dense | **关**（C1：token 0 两栈都是 MoE） |
| 路由 | softmax-then-topK；打分 `sqrt(softplus(·))`；aux-loss-free 偏置 + 轻 seq-balance \(10^{-3}\) |
| Hash-MoE | Decoder **前 2 层**；Encoder 只在 **B2** 才考虑，默认仍学路由 |
| 残差 | 普通残差。**mHC 关** |
| MTP / KDA / PDSA | **关** |

注意力（实现，不是占位账本）：

| 项 | 定稿 |
| --- | --- |
| Phase B | **滑窗因果 MHA**（\(4d^2\)）+ **门控 cross-attn**（\(4d^2\)）。不开 top-k，不开 HCA 压缩。 |
| \(n_{\mathrm{win}}\) | **8192**（4K 主训练上等于全因果） |
| Encoder 层类型标签 | **2 sliding + 7 CSA + 7 HCA**（Phase B 计算仍当滑窗；CSA/HCA 只在 Phase C 点亮） |
| Decoder self-attn | **全部滑窗**。不铺 CSA |
| CSA \(m\) / HCA \(m'\) / `index_topk` | 4 / 128 / **256**（Phase C 才用） |
| M2 全局 cache 池化 | **关**。\(\hat K,\hat V = X^{16}W_K,X^{16}W_V\)，\(d_{\mathrm{kv}}=d\)，槽数 = 序列长 |
| M3 | Phase C 之后。Phase B decoder cross-attn **dense 因果**（位置 \(t\) 读 cache \(0..t\)） |
| QK-Norm | **开** |
| 双 RMSNorm | **关**（沿用 MiniCPM pre-norm） |

发布账本仍是占位注意力 1.25×MHA、12.05B / 2.29 / 4.49。训练代码 Phase B 用真 MHA、全 MoE 1+17/1+17；YOCO 的 \(W_K/W_V\) **只在 Encoder 顶投影一次**（每层 cross-attn 只有 \(W_Q,W_O\)），存储约 **11.59B**。把 decoder cross 按每层 \(4d^2\) 计才会看到 ~11.83B。CSA 投影在 Phase C 进图后再动 routed 对齐 12.05B。

---

## 2. Phase B 课程（C1+FP8）

| 子阶段 | token | 可训练 | detach | gate | dtype |
| --- | ---: | --- | --- | --- | --- |
| B0 | **8B** | cross-attn、\(W_K/W_V\)、gate、新 LN | 是 | 0→0.3 | student **bf16**；冻结 Encoder 前向可 FP8 |
| B1 | **27B** | 整个 Decoder | 是 | →1 | **fp8_moe** |
| B2 | **15B** | 全模型 + tied \(E\) | 否 | 1 | **fp8_moe** |

- tied \(E\)：B0/B1 **freeze_tied**（不解绑）。禁止冻 Encoder 时训 \(E\)。
- 总信封 **50B**。质量不稳 **加长 B2**，不改回联合 bf16，不做两个 LM。
- 墙钟 **761 H100-h**。联合 bf16 1,354 是对照；C1 bf16 1,090 是操作数账。
- Phase C indexer **bf16**，叠在 B2 后。L0/单测 **bf16**，不开 FP8。
- 高精度白名单：tied \(E\)、RMSNorm、router、gate、indexer、attn softmax。
- FP8 加速比发布 **1.5×**。无 CUDA 时自动 bf16。

---

## 3. 优化器 / 超参

| 项 | 定稿 |
| --- | --- |
| 优化器 | **AdamW**（\(\beta=(0.9,0.95)\)，wd=0.1）。**Muon 关**（开关留着，默认 false） |
| LR | WSD。warmup **0.5B** tok。stable **1e-4**。B2 **3e-5**。Decay 留给 Phase E |
| Grad clip | 1.0 |
| 序列 | Phase B **4096** |
| 全局 batch | **4M tok/step**（\(1024\times 4096\)）。micro-batch 按卡切 |
| z-loss | router-z **1e-4** |
| KD | 有 MiniCPM teacher 路径则 **开**（logit KL，T=2，权重 0.5→0）；没有则关 |
| 文档 mask | packing 时 **开** |

---

## 4. 上采样

- 权重：Enc ← MiniCPM 0..15，Dec ← 16..39，emb 共享。
- 新模块（cross-attn、\(W_K/W_V\)、router）：小尺度随机。
- MoE：dense SwiGLU `5760` **不能整除** `2048`。virtual-group **不是**精确恒等：每个专家复制 dense 的前 2048 行并按 \((E G^2 / T)^{1/3}\) 缩放。恢复靠 Phase B，不靠手术瞬间。
- gate 初值 **0**（定理 A：旁路 cross-attn）。

---

## 5. 代码范围（这份仓库）

做：

- 12B 配置的 YOCO MoE 图、C1 冻结 API、C1+FP8 策略对象、WSD、上采样、单卡/DDP/FSDP 入口。
- **C1 训练循环**：packing + 文档 mask、**DDP 数据分片**、梯度累积、激活重计算 `--grad-ckpt`、AdamW 分组、checkpoint / resume（含 packed 游标与 RNG）、jsonl 日志（nll/aux/grad_norm/tok/s/mem）、可选 MiniCPM logit KD。tiny 单测（gate=0、detach 无 Encoder 梯度、freeze_tied、ckpt）。
- **OpenBMB 数据路径**：`cat_yoko.prepare` 把 Ultra-FineWeb en/zh + UltraData-Math（默认 0.60/0.30/0.10）打成 seq_len 对齐的 int32 mmap `.bin`；tokenizer 默认 `openbmb/MiniCPM-2B-sft-bf16`。Trainer 对 `.bin` 走 `PackedBinStream`，并从 `*.bin.meta.json` 读 `eos_id`。`--upcycle-hf` / `--teacher-hf` 拉 MiniCPM-2B。
- **CUDA 烟测**：`python3 -m cat_yoko.gpu_smoke` 跑 tiny（含 encoder offload / CPU Adam / B0→B1→B2 链）。`--middle` 在 ≥28GiB GPU 上对 **12B 图**做一步 C1 B0（bf16 直接建在 CUDA 上，禁止 CPU fp32 再 cast）。`--middle --phase B1` 冻 Encoder 卸载到 CPU + Adam 动量在 host。`--c1` 同一张 12B 图上 B0→B1→B2 各一步；B2 走逐层 offload，不在 GPU 上放全参 Adam。无卡 skip。4M global batch / 全参 GPU Adam 仍要多卡或 ZeRO。
- Megatron-LM 适配面：`ParallelPlan`（TP/PP/EP/CP）、双栈 `TransformerConfig` 映射、`model_provider` / `forward_step` 钩子。不 vendoring Megatron。

不做（本步）：

- 把 `GPTModel` / 双向 T5 encoder 当 YOCO；实现 EP/TP 训练循环（等装上 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 再填 provider）。
- 自研 CSA kernel / FP4 / **把 50B 语料检进 git** / 评测套件 / Phase C indexer 训练循环。数据接口吃 prepare 产出的 mmap `.bin`，或 jsonl `tokens`。
- 在本机 CPU 上分配 12B 权重（约 48GB fp32 / 24GB bf16），或在 CI / 小 VM 上下载 Ultra-FineWeb / MiniCPM 权重。`--config 12b --meta` 只建 meta 图。12B 开训必须 `--device cuda --dtype bf16`，并显式 `--steps` 或 `--tokens`。单卡 32GB：B0 一步可直接跑；B1 靠冻结 Encoder 卸载；B2 靠逐层 offload + CPU Adam（host 需要约 90GiB 动量）。4M global batch / 全参 GPU Adam 仍要多卡或 ZeRO。

Megatron 约束（已写进 `cat_yoko.parallel`）：

- TP 必须整除 36 头与 \(d=2304\)：`{1,2,3,4,6,9,12,18,36}`。
- 路由专家 17 是质数，EP 只能是 **1 或 17**。
- Encoder CSA 比例 `{0,4,128}` 与 Megatron Core `csa_compress_ratios` 合法值对齐；Phase C 再点亮。

---

## 6. 明确关掉的开放问题

| 曾开放 | 定稿 |
| --- | --- |
| Encoder 是否双向 | **否。因果 YOCO** |
| 16/24 vs 20/20 | **16/24** |
| 独立训两栈再焊 | **禁止** |
| M2 默认 | **关** |
| KDA / mHC / MTP / Muon | **关** |
| 首层 dense | **关**（C1 全 MoE） |
| Phase B 是否 CSA 稀疏 | **否** |
| 解冻课程 | **C1** |
| 墙钟 | **C1+FP8 = 761 H100-h** |
