# CAT-YOKO 发布规格（已敲死）

> 这份表是训练代码的输入。旋钮不再开放；要改就改版本号，不要在实现里重新讨论。
> 中间档账本 / 因果 / C1 / C1+NVFP4 仍以理论文档为准。这里只钉**实现默认**。

机器可读副本：`cat_yoko/config.py` 的 `CATYokoConfig.middle_12b()`。

---

## 0. 一句话

**CAT-YOKO-12B**：MiniCPM5-2B 上采样的因果 YOCO MoE；Phase B 按 **C1+NVFP4** 训。发布档 B0 实测在 **NVIDIA B200 / SM 10.0**（冻 encoder 硬件 NVFP4 FPROP）；sm_120（6000D）走仿真。注意力 Phase B 只跑滑窗 GQA + 门控 cross-attn；M2 关；KDA / mHC / MTP / Muon / PDSA 默认关。

**底座许可**：`MiniCPM-2B-sft-bf16` 走 OpenBMB GML / 需商业授权；`MiniCPM5-2B` 是 **Apache-2.0**，所以发布底座是 MiniCPM5。上采样 / teacher 用 `openbmb/MiniCPM5-2B-Base`（`LlamaForCausalLM` GQA），tokenizer 用 `openbmb/MiniCPM5-2B`。

**仓库许可**：CAT-YOKO 代码与派生权重 **Apache-2.0**（[`LICENSE`](../LICENSE)）。

仓库训练代码是这份 12B 图的 **PyTorch 参考实现**；规模化并行预留 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 适配面（`cat_yoko.megatron`，双栈 TransformerConfig + model_provider 钩子）。**不要**把 YOCO 塞进现成 `GPTModel`。tiny 配置只给单测。

---

## 1. 架构

| 项 | 定稿 |
| --- | --- |
| Encoder | **因果** self-decoder，不是双向 |
| 切分 | MiniCPM5 **前 16 层 → Encoder，后 26 层 → Decoder**（共 42；多出的 2 层进 decoder） |
| 隐藏维 / 词表 / 头 | \(d=2048\)，\(V=130560\)，**16 Q / 2 KV** GQA，\(d_h=128\)，\(d_{\mathrm{kv}}=256\)，**untied** \(E\) / `lm_head` |
| μP | **无 MiniCPM-2B μP**（MiniCPM5 是 Llama 恒等尺度）：`use_mup=False` 时 `embed_scale=1`，`residual_scale=1`，`logit_scale=1`，即使误填 `dim_model_base` 也不除 logits。`rms_eps=1e-6`，`rope_theta=5e6` |
| MoE | DeepSeek 细粒度，`moe_intermediate_size=2048`，dense SwiGLU **6144**（\(6144/2048=3\) 整除，比 MiniCPM-2B 的 5760 干净） |
| Enc 专家 | 1 shared + **20** routed，top-\(k=7\) |
| Dec 专家 | 1 shared + **20** routed，top-\(k=10\) |
| 首层 dense | **关**（C1：token 0 两栈都是 MoE） |
| 路由 | softmax-then-topK；打分 `sqrt(softplus(·))`；aux-loss-free 偏置 + 轻 seq-balance \(10^{-3}\) |
| Hash-MoE | Decoder **前 2 层**；Encoder 只在 **B2** 才考虑，默认仍学路由 |
| 残差 | 普通残差。**mHC 关** |
| MTP / KDA / PDSA | **关**（KDA 参考实现在 `cat_yoko.kda`，`use_kda=False`；开了才是 3:1 换层，不改 YOCO 全局 cache） |

档位只改 top-\(k\)（同一套 1+20 / **882** 专家槽）：

| 档 | Enc top-\(k\) | Dec top-\(k\) |
| --- | ---: | ---: |
| low | 4 | 6 |
| **middle（发布）** | **7** | **10** |
| near_dense | 12 | 16 |

注意力（实现，**真 GQA**，不是 1.25×MHA 占位账本）：

| 项 | 定稿 |
| --- | --- |
| Phase B | **滑窗因果 GQA**（\(2d^2+2\cdot d\cdot d_{\mathrm{kv}}\approx 9.44\mathrm{M}\)/层）+ **门控 cross-attn**（\(2d^2\)，只有 \(W_Q,W_O\)）。不开 top-k，不开 HCA 压缩。 |
| \(n_{\mathrm{win}}\) | **8192**（4K 主训练上等于全因果） |
| Encoder 层类型标签 | **2 sliding + 7 CSA + 7 HCA**（仍钉在 16 层上；Phase B 计算仍当滑窗 GQA，**不是 CSA kernel**；CSA/HCA 只在 Phase C 点亮）。`use_kda=True` 时变成 **2 sliding + 11 KDA + 2 CSA + 1 HCA**；点亮顺序 **KDA → indexer/CSA → HCA**（HCA 层最少、write-first，放最后） |
| Decoder self-attn | **全部滑窗 GQA**。不铺 CSA。多出的 2 层 MiniCPM5 进 decoder（26）。`use_kda` 时 3:1 KDA:sliding（decode KV 的省显存来源） |
| CSA \(m\) / HCA \(m'\) / `index_topk` | 4 / 128 / **256**（Phase C 才用） |
| M2 全局 cache 池化 | **关**。\(\hat K,\hat V = X^{16}W_K,X^{16}W_V\)，\(W_K/W_V=\mathrm{Linear}(d,d_{\mathrm{kv}}=256)\)，槽数 = 序列长；**只在 Encoder 顶投影一次** |
| M3 | Phase C 之后。Phase B decoder cross-attn **dense 因果**（位置 \(t\) 读 cache \(0..t\)） |
| QK-Norm | **开** |
| 双 RMSNorm | **关**（沿用 MiniCPM5 pre-norm） |

发布账本按真 GQA 计，不是 1.25×MHA 占位：存储约 **12.25B**；Encoder 活跃约 **2.03B** / 输入 token；Decoder 活跃约 **4.33B** / 输出 token；专家槽 **882**。YOCO cache 的 \(W_K/W_V\) 不按每层 \(4d^2\) 重复计入。

---

## 2. Phase B 课程（C1+NVFP4）

| 子阶段 | token | 可训练 | detach | gate | dtype |
| --- | ---: | --- | --- | --- | --- |
| B0 | **8B** | cross-attn、\(W_K/W_V\)、gate、新 LN | 是 | 0→0.3 | student **bf16**；冻结 Encoder 前向 **nvfp4** |
| B1 | **27B** | 整个 Decoder + **untied `lm_head`** | 是 | →1 | **nvfp4**（允许的线性 GEMM，含 lm_head 与 attn QKV/O） |
| B2 | **15B** | 全模型（含 input embed + `lm_head`） | 否 | 1 | **nvfp4** |

- MiniCPM5 **untied**：B0/B1 **冻 Encoder + 输入 embed**；B0 额外冻 `lm_head` **和最终 RMSNorm**（只训 cross-attn / \(W_K/W_V\) / `ln_cross`）；B1 训 `lm_head` 与最终 RMSNorm；B2 全开。禁止冻 Encoder 时训输入表。
- 总信封 **50B**。质量不稳 **加长 B2**，不改回联合 bf16，不做两个 LM。
- 墙钟 **571 H100-h**（联合 bf16 1,325 的 **43%**；≈ RTX PRO 6000-h）。联合 bf16 1,325 是对照；C1 bf16 1,046 是操作数账；C1+FP8 729 是 Hopper/Ada 回退。
- Phase C indexer **bf16**，叠在 B2 后。L0/单测 **bf16**，不开 NVFP4。
- **必须高精度**：input embed、RMSNorm / QK-Norm、router、gate、indexer、attn softmax / SDPA score。**lm_head 与 attn QKV/O 投影不是必须 bf16，走 NVFP4。** 实现上 RMSNorm、router logits/softmax、attn softmax 在 autocast 下走 fp32。
- NVFP4 加速比发布 **2.0× vs bf16**（相对旧 FP8 1.5× 再 ×1.33）。B200 / SM 10.0 / 10.3 走 ``TeNvfp4Linear``（TE 默认 ``NVFP4BlockScaling``：2D + RHT + SR）。无 Blackwell 或 sm_120 时 ``Nvfp4Linear`` E2M1/16 仿真，再退 FP8 placeholder / bf16 autocast。注意力保持因果 YOCO window，softmax / SDPA 高精度。
- `use_muon=True` 在 `build_optimizer` 抛 `NotImplementedError`（发布默认关）。

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
| KD | 有 MiniCPM5 teacher 路径则 **开**（logit KL，T=2，权重 0.5→0）；没有则关 |
| 文档 mask | packing 时 **开** |

---

## 4. 上采样

- 权重：Enc ← MiniCPM5 0..15，Dec ← 16..41，**untied** 分别拷 `embed_tokens` 与 `lm_head`。
- 新模块（cross-attn、\(W_K/W_V\)、router）：小尺度随机（`init_new_modules`，std=0.02；meta 设备跳过）。
- MoE：dense SwiGLU `6144` **整除** `2048`（精确 3 组）。每个专家仍复制 dense 的前 2048 行并按 \((E G^2 / T)^{1/3}\) 缩放；比 MiniCPM-2B 的 `5760` 干净。恢复靠 Phase B，不靠手术瞬间。
- gate 初值 **0**（定理 A：旁路 cross-attn）。

---

## 5. 代码范围（这份仓库）

做：

- 12B 配置的 YOCO MoE 图、C1 冻结 API、C1+NVFP4 策略对象、WSD、上采样、单卡/DDP/FSDP 入口。
- **C1 训练循环**：packing + 文档 mask（`labels_with_doc_boundaries` 打 **最后一维**，`PackedBinStream` 的 `[B,S]` 不会误用 batch 维）、**DDP 数据分片**（C1 每阶段 `apply_freeze` 后重新 wrap，累积步 `no_sync`）、梯度累积、激活重计算 `--grad-ckpt`、AdamW 分组（router 不 decay，SwiGLU `gate_proj` 仍 decay）、checkpoint / resume（含 packed 游标与 RNG；stream `kind` / DDP stride 不匹配则跳过）、DummyStream 按 rank 偏移 seed（trainer 不再把 rank 加进 seed）、训练/eval nll 按 `n_valid` 加权（DDP 对 `(nll·n_valid, n_valid)` 做 sum，空 rank 贡献 0 而不是 nan）、jsonl 日志（nll/ppl/aux/moe_cv/grad_norm/tok/s/mem/eval/n_valid/kd_w；非有限浮点写成 JSON null）、`--eval-batches`、可选 MiniCPM5 logit KD（CUDA 上 teacher bf16；KL 按 token 均值，忽略 `-100`）。冻住的 MoE 不进 aux、不改 router bias。DDP 在 opt.step 前 allreduce 专家 load，aux-loss-free bias 每优化步更新一次。无窗口截断且行内无跨文档时注意力走因果 SDPA，不物化 S×S mask。`init_process_group` 在 NCCL 上传 `device_id`；FSDP world=1 用 `NO_SHARD`。tiny 单测（gate=0、detach 无 Encoder 梯度、冻 embed / B0 冻 lm_head 与最终 RMSNorm、ckpt、2-rank gloo DDP / C1 链）。**跨阶段 resume**：`--phase B1 --resume b0/latest.pt` 只接手权重和数据游标，不恢复 B0 的 optimizer / step；phase 以 CLI 为准。`--resume` 可以是文件或目录（目录优先 `latest.pt`，否则最新 `step_*.pt`）。`--c1` / `--c1-smoke` 在同一张图上跑 B0→B1→B2，packed 游标连续，`--save-dir/{B0,B1,B2}/latest.pt`。ckpt 先落到 CPU（非 FSDP 也 `.cpu()` 拷再 save），避免 12B 双份占 GPU。12B 默认 weights-only（`--no-save-optim`）；`--keep-last N` 修剪 `step_*.pt`（不删与 `latest.pt` 同 inode 的 step 文件）。**`latest.pt` 在已有 `step_{last}.pt` 时 hardlink（或同卷 copy），禁止再 `torch.save` 一份 23GiB**——RTX 4080 SUPER 上 `/tmp` 写完 `step_1.pt` 再写 `latest.pt` 会把 overlay 写满（`PytorchStreamWriter` / `latest.pt.tmp` 残留）。12B ckpt 放到大盘（如 `/root/autodl-tmp`），不要放 `/tmp`。开训前清 `*.pt.tmp`；卷空间不足（payload+1GiB）直接报错。B2 `--offload-blocks` 只能 `--accum 1`（`--tokens` 也不会再自动扩成 4M global batch）。DDP/FSDP 与 CPU offload 互斥。
- **B0 / B1 / B2 入口**：`python3 -m cat_yoko.b0|b1|b2`（或 `scripts/train_b{0,1,2}.py` / `cat-yoko-b0`）。无 `--try` 时注入发布 token 包络（8e9 / 27e9 / 15e9）以及 `--no-save-full --save-trainable --no-save-optim`；B0 `--offload-encoder`；B1 再加 `--optim-cpu`；B2 `--offload-blocks --optim-cpu --accum 1`。`<40GiB` 卡拒绝默认包络（32GB 跑不完 8B token），必须 `--try` 或显式 `--steps` / `--tokens`。`--try`：32 步、seq=64、无 `--upcycle*` 时 `--dummy-upcycle`，产物 `trainable.pt`（B0 ≈0.44GiB bf16）。权重**不进 GitHub**（`.gitignore` 掉 `*.pt`）；overlay / 全图 / shard 发 [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)。`--resume DIR` 优先 `latest.pt`，其次 `trainable.pt`，再最新 `step_*.pt` / `trainable_step_*.pt`。B1 接手 B0 overlay = MiniCPM5 上采样 + `load_trainable_state`。B2 接手 B1 overlay = MiniCPM5 上采样（encoder + 输入 embed）+ `load_trainable_state`（decoder + `lm_head` + 最终 RMSNorm）；`--offload-blocks` 拒绝 `accum>1`。
- **未知卡 B0 配方**：`python3 -m cat_yoko.hw_recipe`（`--json` / `--argv` / `--shell`）按 SM family + VRAM 选出 profile，不接 Megatron 训练循环。`scripts/run_b0_next.sh` 拉 MiniCPM5 + Hub `b0-full` overlay 后把 flags 交给 `cat_yoko.b0`。SM100/103 ≥160GiB → mb=2、无 offload、无 grad-ckpt、TE NVFP4；sm_120 ≥90GiB → mb=1、仿真 NVFP4；Hopper ≥40GiB → 发布信封 + 仿真（FP8 只是文档回退）；`<40GiB` → `--try`；CPU 只打 JSON。卡名脚本 `run_b0_full_b200.sh` / `run_b0_full_autodl.sh` 仍是已知 SKU 快捷方式。
- **C–G 入口**：`python3 -m cat_yoko.c|d|e|f|g`（`scripts/train_{c-g}.py` / `cat-yoko-c`）。复用同一套 trainer + overlay。`--try` 仍是 32 步、seq=64。**C** `--stage indexer|topk|hca|win`（默认 indexer）或 `--chain`：冻主干、bf16 Lightning Indexer **层内 KL** 对齐稠密滑窗注意力（只挂 Encoder CSA，不复用到 decoder query）；随后 CSA 是 **滑窗 ∪ 压缩块 top-k**（自身块排除，定理 B），HCA 是 **滑窗 concat 均值池化槽**，都不是 CSA CUDA kernel。**KDA**（`cat_yoko.kda`）默认关。**先实现、后点亮**：B `--use-kda` 把 3:1 图和 `KDAGates` 建进图（计算仍滑窗，KDA 参数冻结）；C 从 overlay 继承后 **`C-kda` → indexer → topk → hca → win**。禁止在 C 才把 KDA 焊到 `use_kda=False` 的 B overlay。不缩小 YOCO 全局 cache，不是 DPLR CUDA kernel。默认 `use_kda=False` 仍是 indexer→topk→hca→win，发布 B0 overlay 132 张量不变。**D** `--stage 8k|32k|128k` 或 `--chain`：4K packed `.bin` **按目标 seq 重切窗**（拼行，不要求 sidecar 相等）；DummyStream 在中段种 needle。D/E/F **钉 `sparse=hca`**（无 KDA 也不回到 window）；`--use-kda` 从 resume extra 继承。prepare `--mix phase-d`（web+code）。**E** WSD 1-sqrt decay 到峰值 ~1/100；prepare `--mix phase-e`（HQ web + math + code + UltraChat 当正文）。**F** SFT seq=8192，prompt 段 `labels=-100`；prepare `--mix phase-f` 写 jsonl（UltraChat `data` / `messages` / alpaca，多段拼到 seq）；Trainer 吃 `tokens`+`labels` 或 `prompt_ids`/`response_ids`。**G** `--algo grpo|dpo`（步数信封，默认 10000；`--try` 用 dummy RLVR）。C–G 默认不下载 Ultra-FineWeb / UltraChat。
- **OpenBMB 数据路径**：`cat_yoko.prepare` 把 Ultra-FineWeb en/zh + UltraData-Math（默认 0.60/0.30/0.10）打成 seq_len 对齐的 int32 mmap `.bin`；tokenizer 默认 `openbmb/MiniCPM5-2B`（\(V=130560\)）。Trainer 对 `.bin` 走 `PackedBinStream`，并从 `*.bin.meta.json` 读 `eos_id`。`--upcycle-hf` / `--teacher-hf` 拉 `openbmb/MiniCPM5-2B-Base`。
- **底座加载**：只接受 **MiniCPM5-2B**（`openbmb/MiniCPM5-2B-Base` / `openbmb/MiniCPM5-2B`）。MiniCPM-2B / MiniCPM3 / MiniCPM4 Hub id 在 CLI 和 `load_minicpm_state` 直接拒绝；HF 加载走 `LlamaForCausalLM`，**无** `trust_remote_code`，**`device_map="cpu"`**（拷完 state_dict 立刻 `empty_cache`），避免 MiniCPM5 占 4–5GiB VRAM 再叠 12B 图。上采样对 embed / GQA K·V / dense SwiGLU 做精确形状检查，MiniCPM-2B MHA（\(d=2304\)）或 `5760` FFN 不会被截断拷进 12B 图。`use_mup=False` 时 embed/residual/logit 尺度恒等 1，即使误填 MiniCPM-2B 的 `dim_model_base` 也不除 logits。
- **CUDA 烟测**：入口 `setdefault PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（B1 约 28.4/32GiB，encoder offload 会碎 caching allocator）。`<40GiB` 卡上 `--config 12b` 必须 `--seq-len`（默认 4096 会 OOM），并拒绝 `--teacher-hf`（student 23–28GiB + MiniCPM5 ~5GiB）。12B save 在 CPU copy 前检查 cgroup（~23GiB host tensors；不要和 B1 常驻 CPU Adam 叠）。`python3 -m cat_yoko.gpu_smoke` 跑 tiny（含 encoder offload / CPU Adam / B0→B1→B2 链 / KD / dummy upcycle / eval / 跨阶段 resume / 2-rank gloo DDP + **CUDA DDP** / **CUDA C1 DDP** / **1-rank FSDP**）。`--middle` 在 ≥28GiB GPU 上对 **MiniCPM5 GQA 12.25B 图**做 C1 步（默认 1；`--steps N` 生效；bf16 直接建在 CUDA 上，禁止 CPU fp32 再 cast）。`--middle --phase B1` 冻 Encoder 卸载到 CPU + CPU Adam。`--middle --save-dir DIR` 写 `step_N.pt` 并 hardlink `latest.pt`。`--c1` 同一张 12B 图上 B0→B1→B2（默认各 1 步）。B2 逐层 offload，backward 完一层就 clip+Adam 掉梯度。无卡 skip。4M global batch / 全参 GPU Adam 仍要多卡或 ZeRO。FSDP wrap 传 `device_id` 与 `use_orig_params=True`。ckpt extra 含 `cfg`/`seed`。在 RTX 4080 SUPER 32GB + 62GiB cgroup 上实测 **MiniCPM5-2B GQA 12.25B 图**（params=12,250,381,312，`--seq-len 64`；峰值在 freeze/offload 之后起算，不含 `build_model` 高水位）：独立进程 `--middle` B0 一步 peak **22.83GiB**（23378MiB）、B1 **28.6GiB**（29283MiB，Encoder offload + ephemeral CPU Adam + `expandable_segments:True`；此前 28.43/29115 同路径）、B2 **2.6GiB**（2662MiB，逐层 offload 工作集）；`--c1` 同图 B0 **22.83GiB**、B1 **28.34GiB**（29019MiB）、B2 **2.6GiB**。CPU `step_1.pt` resume 到 step 2 成功（nll=11.9425，mem=23378MiB）。`/root/autodl-tmp/b0ckpt` weights-only **hardlink**：`latest.pt` 与 `step_1.pt` 同 inode、nlink=2、24,501,855,029 bytes，**没有**第二份 23GiB；从该目录 `--resume` 再跑 step 2 peak 仍 22.83GiB。此前 `/tmp` 二次 `torch.save` 把 overlay 写满（`latest.pt.tmp` 7G 残留）已修。旧 MiniCPM-2B 占位图 21.65 / 26.3 / 13.38 / 22.73 GiB **作废**；此前把 B2 写成 22.82 / 14.68 是建图或 B1 decoder 残留，不是 B2 工作集。12B unittest 每个 case **单独子进程**（`python -m cat_yoko.gpu_smoke`），禁止同一进程叠两份 12.25B。tiny CUDA overall ok（peak_mib 17.5；含 2-rank CUDA DDP / C1 DDP / 1-rank FSDP）。跑完 GPU idle 1 MiB。`expandable_segments` 复核：import 后 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；`--config 12b --device cuda` 无 `--seq-len` 与 `--teacher-hf` 均 argparse exit 2（未建 12B 图）；独立 B0 nll=11.9544 peak 22.83GiB；独立 B1 nll=11.9345 peak **28.6GiB**，idle 1 MiB。
- Megatron-LM 适配面：`ParallelPlan`（TP/PP/EP/CP）、双栈 `TransformerConfig` 映射、`model_provider` / `forward_step` 钩子。不 vendoring Megatron。

不做（本步）：

- 把 `GPTModel` / 双向 T5 encoder 当 YOCO；实现 EP/TP 训练循环（等装上 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 再填 provider）。
- 自研 CSA CUDA kernel / V4 式 FP4 专家存储 / **把 50B 语料检进 git** / 评测套件。数据接口吃 prepare 产出的 mmap `.bin`，或 jsonl `tokens` / SFT `labels` / `prompt_ids`+`response_ids`。Phase C 用 `cat_yoko.indexer.LightningIndexer` + 层内 KL，以及定理 B 的滑窗∪压缩 SDPA 掩码 / HCA concat；**不是** CSA kernel。
- 在本机 CPU 上分配 12B 权重（约 49GB fp32 / 24.5GB bf16），或在 CI / 小 VM 上下载 Ultra-FineWeb / MiniCPM5 权重。`--config 12b --meta` 只建 meta 图。12B 开训必须 `--device cuda --dtype bf16`，并显式 `--steps` 或 `--tokens`。单卡 32GB：B0 一步可直接跑；B1 靠冻结 Encoder 卸载 + CPU Adam（host cgroup 不够 fp32 动量时自动改 fp16 动量）；B2 靠逐层 offload，host 不够存动量时一步 smoke 走 ephemeral AdamW。4M global batch / 全参 GPU Adam 仍要多卡或 ZeRO。

Megatron 约束（已写进 `cat_yoko.parallel`）：

- TP 必须整除 16 头与 \(d=2048\)：`{1,2,4,8,16}`。
- 路由专家 20，EP 必须整除 20：`{1,2,4,5,10,20}`。
- Encoder CSA 比例 `{0,4,128}` 与 Megatron Core `csa_compress_ratios` 合法值对齐；Phase C 再点亮。

---

## 6. 明确关掉的开放问题

| 曾开放 | 定稿 |
| --- | --- |
| Encoder 是否双向 | **否。因果 YOCO** |
| 16/24 vs 20/20 | **16/26**（MiniCPM5 42 层；多出 2 层进 decoder） |
| 独立训两栈再焊 | **禁止** |
| M2 默认 | **关** |
| KDA / mHC / MTP / Muon | **关**（KDA 代码 opt-in `use_kda`；默认不算进发布图） |
| 首层 dense | **关**（C1 全 MoE） |
| Phase B 是否 CSA 稀疏 | **否**（滑窗 GQA，不是 CSA kernel） |
| 注意力账本 | **真 GQA 16/2**，不是 1.25×MHA 占位 |
| 解冻课程 | **C1**（B0 8B / B1 27B / B2 15B） |
| 底座 | **MiniCPM5-2B Apache-2.0**（不用 MiniCPM-2B-sft-bf16 GML） |
| 墙钟 | **C1+NVFP4 = 571 H100-h**（C1+FP8 729 为 Hopper/Ada 回退） |
