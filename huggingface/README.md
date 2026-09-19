---
library_name: transformers
pipeline_tag: text-generation
license: bsd-3-clause
language:
- en
- zh
tags:
- moe
- yoco
- minicpm
- causal-lm
base_model: openbmb/MiniCPM5-2B-Base
---

# CAT-YOKO-12B

YOCO 式因果 encoder-decoder MoE。从 MiniCPM5-2B 上采样：[`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base)（Apache-2.0，Llama GQA）。

本卡对应 [`AvrovaDonz/CAT-YOKO`](https://huggingface.co/AvrovaDonz/CAT-YOKO)。训练代码：[`AvrovaDonz2026/CAT-YOKO`](https://github.com/AvrovaDonz2026/CAT-YOKO)。

`library_name: transformers` 只表示 tokenizer / 分片约定走 HuggingFace 生态。当前图是仓库里的 `cat_yoko` PyTorch 实现，不是 Hub 上可 `AutoModelForCausalLM` 直接加载的架构。

## 规格

| 项 | 值 |
| --- | --- |
| 名称 | CAT-YOKO-12B |
| 架构 | YOCO 式因果 encoder-decoder MoE |
| 底座 | MiniCPM5-2B（Llama GQA，untied） |
| \(d\) | 2048 |
| \(V\) | 130560 |
| \(L\) | 42（16 encoder + 26 decoder） |
| 注意力 | 16 Q / 2 KV，`head_dim=128` |
| FFN | SwiGLU 中间维 6144 |
| MoE | 1 shared + 20 routed |
| top-\(k\) | encoder 7 / decoder 10 |
| 存储参数 | 12,250,381,312（≈12.25B） |
| Encoder 活跃 | ≈2.03B / 输入 token |
| Decoder 活跃 | ≈4.33B / 输出 token |

## 课程 C1

| 子阶段 | token | Encoder | 可训练 |
| --- | ---: | --- | --- |
| B0 | 8B | 冻结 | 仅新模块 |
| B1 | 27B | 冻结 | decoder + `lm_head` + 最终 RMSNorm |
| B2 | 15B | 可训练 | 全部 |

## 墙钟（发布账本）

50B token 信封上的理论墙钟，**不是**实测。B200 / SM100 允许的线性 GEMM 走 ``TeNvfp4Linear``（TE ``NVFP4BlockScaling``；B0 冻 encoder 走 FPROP，WGRAD 留给 B1/B2）。无 TE / sm_120 时 ``Nvfp4Linear`` E2M1/16 仿真。attn softmax / SDPA 仍 fp32。

| 配方 | H100-h | 角色 |
| --- | ---: | --- |
| C1+NVFP4 | 571 | 发布墙钟。B0 student bf16；B1/B2 允许的 GEMM 走 NVFP4 |
| C1+FP8 | 729 | Hopper / Ada 回退 |
| 联合 bf16 | 1325 | 100% 对照基线 |

## 数据与分词

| 项 | 值 |
| --- | --- |
| 语料 | Ultra-FineWeb（en / zh）+ UltraData-Math |
| Tokenizer | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |

## 当前 B0 快照

发布信封 **进行中**（DummyStream，尚未跑完 8e9）。训练还在跑时约每 10 分钟覆盖同路径。不是终局。

| 项 | 值 |
| --- | --- |
| 文件 | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| 文件夹说明 | [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md) |
| step | **25460** |
| tokens_in_phase | 117,917,696（信封 8e9 的 ≈1.47%） |
| sha256 | `efef3464730eaaee6b62a0e199437b3a06049812bfbc25e9a98f39467fdf0a2c` |
| 张量 | 132，无 Adam |
| 机器 | Vast NVIDIA B200 SM 10.0 |
| 运行时 | torch 2.11+cu128 + TE nvcc 12.9 SM100 |
| 吞吐 / 显存 | **micro-batch=2**，~15.7k tok/s，Trainer ~138GiB |
| 续训 | MiniCPM5 上采样后 overlay 本文件；同阶段 resume 保留 `tokens_in_phase` |

代码与指针：[GitHub AvrovaDonz2026/CAT-YOKO](https://github.com/AvrovaDonz2026/CAT-YOKO)（不用 LFS）。

## 当前权重

| 路径 | 来源 | 说明 |
| --- | --- | --- |
| `checkpoints/b0/trainable.pt` | 6000D `--try` **32** 步，MiniCPM5 上采样 | gate 0.301；peak 24244 MiB；sha256 `9012e5ac55c2f59ef7cacc34d5769444413d070116dbff0696c7b258b9aa0636` |
| `checkpoints/b0-nvfp4-try/trainable.pt` | 6000D NVFP4 wrap `--try` **2** 步 | `nvfp4_n=2815`；gate 0.301；peak 34442 MiB；sha256 `461b4ffc05fd46e2668448393789764ccf9dd673644040fe4527259b176a510e` |
| `checkpoints/b0-full/trainable.pt` | Vast B200 发布档 B0 进行中（8e9 信封，seq=4096） | 见上一节「当前 B0 快照」。step **25460**，sha256 `efef3464730eaaee6b62a0e199437b3a06049812bfbc25e9a98f39467fdf0a2c`。 |
| `checkpoints/b1/trainable.pt` | 6000D B1 `--try`（等 GPU） | decoder + `lm_head` + 最终 RMSNorm；resume B0 overlay + MiniCPM5。尚未上传 |
| `checkpoints/b2/` | 6000D B2 `--try`（待 GPU） | 全模型 overlay；resume B1 + MiniCPM5 encoder/embed。指针 [`checkpoints/b2/README.md`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/checkpoints/b2) |

这些 overlay 里，`checkpoints/b0/` 与 `checkpoints/b0-nvfp4-try/` **不是** 8B token 信封（只是 `--try`）。`checkpoints/b0-full/trainable.pt` 是发布信封 **进行中** 的 B0 overlay（尚未跑完 8e9）。尚未上传 23GiB 全图。GitHub 不存权重、不用 Git LFS。日志在 GitHub [`artifacts/autodl-rtx6000d/`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/autodl-rtx6000d)。

无公开评测分数。

## 许可

| 产物 | 许可 |
| --- | --- |
| 本仓库代码与派生权重 | BSD-3-Clause |
| 底座 MiniCPM5-2B | Apache-2.0 |

---

## English (short)

**CAT-YOKO-12B** is a YOCO-style causal encoder-decoder MoE upcycled from [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base) (Apache-2.0, Llama GQA). Code: [GitHub](https://github.com/AvrovaDonz2026/CAT-YOKO). The `transformers` tag is for tokenizer / shard convention; the graph lives in `cat_yoko`, not `AutoModelForCausalLM`.

| Item | Value |
| --- | --- |
| \(d\) / \(V\) / \(L\) | 2048 / 130560 / 42 (16 encoder + 26 decoder) |
| Attention | 16 Q / 2 KV, `head_dim=128` |
| FFN / MoE | SwiGLU 6144; 1+20 routed; top-\(k\) 7/10 (enc/dec) |
| Stored params | 12,250,381,312 (≈12.25B) |
| Active | encoder ≈2.03B/in, decoder ≈4.33B/out |

**C1:** B0/B1 freeze the encoder. B0 trains new modules only. B1 trains decoder + `lm_head` + final RMSNorm. B2 trains all. Tokens B0/B1/B2 = 8/27/15B.

**Published wall-clock** (50B-token envelope, not measured). B200 / SM100 allowed linear GEMMs use ``TeNvfp4Linear`` (TE ``NVFP4BlockScaling``; B0 frozen encoder is FPROP, WGRAD is B1/B2). Without TE or on sm_120, ``Nvfp4Linear`` E2M1/16 emulation. Attn softmax / SDPA stay fp32.

| Recipe | H100-h | Role |
| --- | ---: | --- |
| C1+NVFP4 | 571 | published; B0 student bf16; B1/B2 allowed GEMMs NVFP4 |
| C1+FP8 | 729 | Hopper / Ada fallback |
| joint bf16 | 1325 | 100% baseline |

Data: Ultra-FineWeb en/zh + UltraData-Math. Tokenizer: [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B).

This Hub has RTX 6000D MiniCPM5-upcycle overlays (not the 8B/27B/15B-token envelopes): `checkpoints/b0/trainable.pt` (32 steps), `checkpoints/b0-nvfp4-try/trainable.pt` (2 steps, NVFP4 wrap), `checkpoints/b1/` (B1 `--try`, pending GPU), and `checkpoints/b2/` (B2 `--try`, pending GPU).

**Current published B0 overlay** (in progress, DummyStream, not finished 8e9): [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt). Step **25460**, `tokens_in_phase=117,917,696` (≈1.47% of 8e9), sha256 `efef3464730eaaee6b62a0e199437b3a06049812bfbc25e9a98f39467fdf0a2c`. Vast B200, micro-batch=2, ~15.7k tok/s. Same path overwritten about every 10 minutes. Folder card: [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md). Weights do not live on GitHub. Logs: GitHub `artifacts/autodl-rtx6000d/`.

License: this repo BSD-3-Clause; MiniCPM5 base Apache-2.0. No eval scores.
