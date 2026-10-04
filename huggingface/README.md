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
| Published recipe | Ultra-FineWeb en 55% / zh 30% + UltraData-Math 10% + StarCoder 5%; planned mixture |
| Current real-text pilot | Ultra-FineWeb en 60% / zh 30% + UltraData-Math L2 10%; no code slice |
| Prepared pilot | 79,998,976 training tokens / 999,424 validation tokens, packed at 4096 |
| Tokenizer | MiniCPM5-2B-Base tokenizer files; identities recorded in the release manifest |

## Current B0 snapshot

The recommended snapshot is **step 52616**, saved on **2026-10-04T03:44:25.977204Z** during real-text B0 training on an AMD RX 7900 XTX. B0's 8B-token budget is unfinished; this snapshot does not enter B1. The earlier B200 and RTX 3090 releases remain available as historical snapshots.

| | |
| --- | --- |
| Recovery file | [`checkpoints/b0-rocm-realtext/step-52616/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/trainable.pt) |
| Recovery file bytes | 2,192,257,315 |
| Recovery SHA256 | `d4c4898be1cd248b2742bd9705a11de8af37003a0d209a91347498d47ec181df` |
| Weights-only file | [`weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/weights-only.pt), 438,477,743 bytes |
| Weights-only SHA256 | `a627dfa07a402fe6c3ed9cd5524c7b8cda518dffdfcd03ff3f32759d8a9a7570` |
| Saved state | 132 BF16 trainable tensors; recovery file also has 264 CPU FP32 Adam moments, counters, RNG and packed cursor |
| Global step / Adam counter / cursor | 52616 / 17814 / 17814 |
| `tokens_in_phase` / `tokens_seen` | 235,210,752, including earlier DummyStream history |
| Real packed training tokens | **72,966,144** since real-text training began; counted separately from the cumulative clock |
| Runtime | RX 7900 XTX `gfx1100`; PyTorch 2.9.1 + ROCm 6.4, BF16 |
| Operators | Packed FP32 MATH attention, shared-storage frozen MoE, original CPU FP32 Adam; round-three candidates are not installed |
| Code and file identities | [Code pin `341e0ed`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/341e0ed); [release manifest](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/release.json) records deployed source and artifact hashes |
| Snapshot card | [Recovery instructions and validation scope](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/README.md) |

These files contain the B0 **trainable overlay**, not the complete 12B model. Reconstruct the frozen graph by upcycling MiniCPM5-2B-Base, then load the overlay. Both files have byte-identical trainable weights. The smaller file omits Adam and cannot provide the same optimizer recovery as `trainable.pt`.

The default downloader selects the recovery file:

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir /workspace/hub/b0-rocm-realtext-step-52616
# Optional smaller weights-only overlay:
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir /workspace/hub/b0-rocm-realtext-step-52616-weights
```

For training recovery, follow the [snapshot card](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/README.md) and the [real-text guide](https://github.com/AvrovaDonz2026/CAT-YOKO/blob/main/operators/rocm/REAL_TRAINING.md). They specify the original packed train/evaluation files and CPU Adam path. Historical DummyStream launchers are not the recovery recipe for this real-text snapshot. Weights remain on Hub; GitHub stores code, docs and logs without LFS.

## Current weights

| Path | Source | Notes |
| --- | --- | --- |
| `checkpoints/b0-rocm-realtext/step-52616/trainable.pt` | Current recommended B0 recovery snapshot | Step 52616; overlay plus CPU Adam, RNG and cursor; identities above |
| `checkpoints/b0-rocm-realtext/step-52616/weights-only.pt` | Same snapshot, smaller overlay | Same 132 weights byte-for-byte; no Adam |
| `checkpoints/b0-full/trainable.pt` | Historical Vast B200 DummyStream B0 | Step 26940, 130,041,856 cumulative phase tokens; sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`; weights only |
| `checkpoints/b0-3090-bf16/trainable.pt` | Historical RTX 3090 BF16 DummyStream B0 | Step 33800; sha256 `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690`; weights only |
| `checkpoints/b0/trainable.pt` | Historical 6000D `--try`, 32 steps | sha256 `9012e5ac55c2f59ef7cacc34d5769444413d070116dbff0696c7b258b9aa0636`; smoke snapshot |
| `checkpoints/b0-nvfp4-try/trainable.pt` | Historical 6000D NVFP4 `--try`, 2 steps | sha256 `461b4ffc05fd46e2668448393789764ccf9dd673644040fe4527259b176a510e`; smoke snapshot |
| `checkpoints/b1/`, `checkpoints/b2/` | Future stages | No B1/B2 trained snapshot is included in this release |

The historical paths are preserved. The complete 23 GiB graph is not included. Historical logs: [B200](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/vast-b200), [6000D](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/autodl-rtx6000d), [3090](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/autodl-rtx3090). ROCm reports: [real-text training](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/artifacts/rocm-rx7900xtx), [operator checks](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/main/operators/rocm).

## Validation observation

The latest fixed held-out evaluation preceding this snapshot was at **step 52500**: NLL **7.900367058954696** across **32 batches and 130,877 valid next-token loss tokens**. This is a sample from the pilot's separate validation bin, not an evaluation of the full validation corpus or a general capability benchmark. The weights in this release were saved 116 updates later, at step 52616. No public generation, reasoning or base-model comparison benchmark is claimed here.

## License

| Artifact | License |
| --- | --- |
| This repo's code and derived weights | Apache-2.0 |
| MiniCPM5-2B base | Apache-2.0 |
