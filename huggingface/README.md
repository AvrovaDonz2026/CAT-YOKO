---
library_name: transformers
pipeline_tag: text-generation
license: apache-2.0
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

YOCO-style causal encoder-decoder MoE, upcycled from [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base) (Apache-2.0, Llama GQA).

This card is [`AvrovaDonz/CAT-YOKO`](https://huggingface.co/AvrovaDonz/CAT-YOKO). Training code: [`AvrovaDonz2026/CAT-YOKO`](https://github.com/AvrovaDonz2026/CAT-YOKO).

`library_name: transformers` only means tokenizer / shard convention follows the Hugging Face ecosystem. The graph is the in-repo `cat_yoko` PyTorch implementation, not an `AutoModelForCausalLM` architecture on Hub.

## Spec

| | |
| --- | --- |
| Name | CAT-YOKO-12B |
| Architecture | YOCO-style causal encoder-decoder MoE |
| Base | MiniCPM5-2B (Llama GQA, untied) |
| \(d\) | 2048 |
| \(V\) | 130560 |
| \(L\) | 42 (16 encoder + 26 decoder) |
| Attention | 16 Q / 2 KV, `head_dim=128` |
| FFN | SwiGLU intermediate 6144 |
| MoE | 1 shared + 20 routed |
| top-\(k\) | encoder 7 / decoder 10 |
| Stored params | 12,250,381,312 (≈12.25B) |
| Encoder active | ≈2.03B / input token |
| Decoder active | ≈4.33B / output token |

## Curriculum C1

| Stage | tokens | Encoder | Trainable |
| --- | ---: | --- | --- |
| B0 | 8B | frozen | new modules only |
| B1 | 27B | frozen | decoder + `lm_head` + final RMSNorm |
| B2 | 15B | trainable | all |

## Wall-clock (published ledger)

Theoretical hours on a 50B-token envelope, **not** a measured run. On B200 / SM100, allowed linear GEMMs use `TeNvfp4Linear` (TE `NVFP4BlockScaling`; B0 frozen encoder is FPROP, WGRAD is B1/B2). Without TE or on sm_120, `Nvfp4Linear` E2M1/16 emulation. Attn softmax / SDPA stay fp32.

| Recipe | H100-h | Role |
| --- | ---: | --- |
| C1+NVFP4 | 571 | published wall-clock. B0 student bf16; B1/B2 allowed GEMMs NVFP4 |
| C1+FP8 | 729 | Hopper / Ada fallback |
| joint bf16 | 1325 | 100% baseline |

## Data and tokenizer

| | |
| --- | --- |
| Mix | Ultra-FineWeb en 55% / zh 30% + UltraData-Math 10% + StarCoder **5%** (thinking mix; no 50B download) |
| Tokenizer | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |

## Current B0 snapshot

Published envelope **in progress** (DummyStream, 8e9 not finished). Vast B200 **recycled** (2026-09-19). This file is a pre-recycle snapshot, not a finished run. GitHub pin: [`docs/STATUS.md`](https://github.com/AvrovaDonz2026/CAT-YOKO/blob/main/docs/STATUS.md).

| | |
| --- | --- |
| File | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| Folder card | [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of the 8e9 envelope) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Tensors | 132, no Adam |
| Machine | Vast NVIDIA B200 SM 10.0 (recycled) |
| Runtime | torch 2.11+cu128 + TE nvcc 12.9 SM100 |
| Throughput / memory | **micro-batch=2**, ~15.7k tok/s, trainer ~138GiB |
| Resume | MiniCPM5 upcycle, then overlay this file; same-phase resume keeps `tokens_in_phase`. Unknown next GPU: GitHub `python3 -m cat_yoko.hw_recipe` + `scripts/run_b0_next.sh` |

Code and pointers: [GitHub AvrovaDonz2026/CAT-YOKO](https://github.com/AvrovaDonz2026/CAT-YOKO) (no LFS).

## Current weights

| Path | Source | Notes |
| --- | --- | --- |
| `checkpoints/b0/trainable.pt` | 6000D `--try` **32** steps, MiniCPM5 upcycle | gate 0.301; peak 24244 MiB; sha256 `9012e5ac55c2f59ef7cacc34d5769444413d070116dbff0696c7b258b9aa0636` |
| `checkpoints/b0-nvfp4-try/trainable.pt` | 6000D NVFP4 wrap `--try` **2** steps | `nvfp4_n=2815`; gate 0.301; peak 34442 MiB; sha256 `461b4ffc05fd46e2668448393789764ccf9dd673644040fe4527259b176a510e` |
| `checkpoints/b0-full/trainable.pt` | Vast B200 published B0 in progress (8e9 envelope, seq=4096) | See snapshot above. Step **26940**, sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`. |
| `checkpoints/b0-3090-bf16/trainable.pt` | RTX 3090 BF16 sibling (same-phase resume, `--no-nvfp4`) | step **33800**, sha256 `2dc31406…`. **Not** the published pin; does not overwrite `b0-full`. Machine pending release. |
| `checkpoints/b1/trainable.pt` | 6000D B1 `--try` (waiting on GPU) | decoder + `lm_head` + final RMSNorm; resume B0 overlay + MiniCPM5. Not uploaded yet |
| `checkpoints/b2/` | 6000D B2 `--try` (waiting on GPU) | full-model overlay; resume B1 + MiniCPM5 encoder/embed. Pointer [`checkpoints/b2/README.md`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/checkpoints/b2) |

`checkpoints/b0/` and `checkpoints/b0-nvfp4-try/` are **not** the 8B-token envelope (they are `--try` only). `checkpoints/b0-full/trainable.pt` is the published envelope **in progress** (8e9 not finished). The 23GiB full graph has not been uploaded. Weights do not live on GitHub; GitHub does not use Git LFS. Logs: GitHub [`artifacts/vast-b200/`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/vast-b200), [`artifacts/autodl-rtx6000d/`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/autodl-rtx6000d), [`artifacts/autodl-rtx3090/`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/autodl-rtx3090).

No public eval scores.

## License

| Artifact | License |
| --- | --- |
| This repo's code and derived weights | Apache-2.0 |
| MiniCPM5-2B base | Apache-2.0 |
