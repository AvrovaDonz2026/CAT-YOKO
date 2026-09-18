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

50B token 信封上的理论墙钟，**不是**实测。NVFP4 kernel **未实现**；trainer 仍是 bf16 autocast placeholder。

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

## 当前权重

本 Hub `checkpoints/b0/trainable.pt`：AutoDL RTX 6000D sm_120 上 `--try` 32 步、真实 MiniCPM5-2B-Base 上采样；gate 0.301；peak 24244 MiB。**不是** 8B token 信封。尚未上传全图 / 8B 信封权重。

GitHub 仓库不再存权重、也不再用 Git LFS。

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

**Published wall-clock** (50B-token envelope, not measured). NVFP4 kernels are **not** implemented; the trainer is still a bf16 autocast placeholder.

| Recipe | H100-h | Role |
| --- | ---: | --- |
| C1+NVFP4 | 571 | published; B0 student bf16; B1/B2 allowed GEMMs NVFP4 |
| C1+FP8 | 729 | Hopper / Ada fallback |
| joint bf16 | 1325 | 100% baseline |

Data: Ultra-FineWeb en/zh + UltraData-Math. Tokenizer: [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B).

This Hub copy is the RTX 6000D MiniCPM5-upcycle `--try` (32 steps, gate 0.301). It is not the 8B-token envelope. Weights do not live on GitHub.

License: this repo BSD-3-Clause; MiniCPM5 base Apache-2.0. No eval scores.
