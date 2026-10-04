# CAT-YOKO

Causal encoder-decoder MoE in the YOCO style, upcycled from MiniCPM5-2B. License: **Apache-2.0**.

Middle tier: ≈12.25B stored parameters. Encoder ≈2.03B active / input token; decoder ≈4.33B active / output token. Phase B published wall-clock is **C1+NVFP4 = 571 H100-h** (ledger, not a measured run). Phase B attention is sliding-window GQA plus gated cross-attention. This repo does **not** implement a CSA CUDA kernel.

## Status

Published **B0 is in progress**. The latest recommended snapshot is real-text
BF16 training on an RX 7900 XTX; the 8B-token envelope is unfinished.

| Item | Value |
| --- | --- |
| Hub | [`ROCm real-text step 53307`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-rocm-realtext/step-53307) |
| Full resume file | `trainable.pt`: 132 BF16 weights, 264 CPU FP32 Adam moments, RNG and packed cursor |
| Training operators | Document-split FP32 attention, shared frozen expert storage and cached BF16 CPU-shadow Adam; native CPU FP32 moments |
| Total phase tokens | **238,041,088**, including earlier DummyStream history |
| Real-text tokens | **75,796,480**, packed cursor **18505 / 19531** |
| SHA256 | `3dd9a62f7acfb4ae025ff44b0589b017104f07abc8b177e0098f5c4a910bd2b2` |
| Fixed pilot validation | NLL **7.885218** at step **53250**, 57 updates before the snapshot; 32 batches / 130877 valid loss tokens, not a standard benchmark |

Download the complete state with:

```bash
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir runs/b0-rocm-realtext
```

Rebuild frozen weights from MiniCPM5-2B-Base, then resume with the original
packed data and native CPU Adam. See the
[snapshot card](checkpoints/b0-rocm-realtext/step-53307/README.md) for recovery
commands and the lightweight weights-only option. Training progress and
historical snapshots: [`docs/STATUS.md`](docs/STATUS.md).

The [operator repository](https://huggingface.co/AvrovaDonz/CAT-YOKO-KERNEL)
and [switch report](operators/rocm/OPERATOR_SWITCH_20261004.md) record the new
backend. Two deterministic real updates preserved all weights and moments
byte-for-byte; production keeps its ordinary `deterministic_algorithms=False`
policy. Short-window whole-update gains were **2.55–3.03%**, below the 5%
automatic gate; adoption followed the user request and numerical/recovery checks.

Historical [ROCm step **52616**](checkpoints/b0-rocm-realtext/step-52616/README.md),
B200 step **26940** and RTX 3090 step **33800** retain their original Hub paths
and provenance; the recommended snapshot above is the newer release.

## Spec

| | |
| --- | --- |
| Base | [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base) (Llama GQA, Apache-2.0) |
| \(d\) / \(V\) / \(L\) | 2048 / 130560 / 42 (16 encoder + 26 decoder) |
| Attention | 16 Q / 2 KV, `head_dim=128` |
| FFN / MoE | SwiGLU 6144; 1 shared + 20 routed; top-\(k\) 7/10 (enc/dec) |
| Tokenizer | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |

Published training entry points are **B0 / B1 / B2**, plus **C–G** which are implemented but not enabled (`python3 -m cat_yoko.c|d|e|f|g`; C/D accept `--chain`). B0 writes a ~419MiB `trainable.pt` overlay by default. The 23GiB full graph does not go on GitHub. Attention **wires the graph first, then enables it**: Phase B is sliding-window GQA (`--use-kda` only inserts the 3:1 graph); Phase C enables KDA → CSA → HCA. Not a CSA kernel. See `cat_yoko.kda`.

## Curriculum C1

| Stage | tokens | Encoder | Trainable |
| --- | ---: | --- | --- |
| B0 | 8B | frozen | new modules only |
| B1 | 27B | frozen | decoder + `lm_head` + final RMSNorm |
| B2 | 15B | trainable | all |

B1/B2 have not started. `--try` is a 32-step, seq=64 smoke; it cannot finish the envelope.

## Docs

**Spec and theory**

- [`docs/FROZEN_SPEC.md`](docs/FROZEN_SPEC.md) — published recipe (training code implements this)
- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, upcycle, data, optimizer
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — param / FLOP / KV / μP ledger
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — causality, residual cut, M1/M2/M3
- [`docs/PLAN_VERIFY.md`](docs/PLAN_VERIFY.md) — dedicated-dir mini-train for attention / YOCO / PDSA / C1
- [`docs/AMPERE_OPS_MFU.md`](docs/AMPERE_OPS_MFU.md) — RTX 3090 operator roofline and tuning
- [`docs/CURRICULUM_THEORY.md`](docs/CURRICULUM_THEORY.md) — freeze curriculum C1
- [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) — C1+NVFP4 wall-clock (571 H100-h)
- [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) — C1+FP8 fallback (729 H100-h)
- [`docs/DEEPSPEED_ZERO.md`](docs/DEEPSPEED_ZERO.md) — optional DeepSpeed ZeRO (3090 48GiB uses ZeRO-3 + CPU offload)

**Training and artifacts**

- [`docs/STATUS.md`](docs/STATUS.md) — **current progress**
- [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md) — B200 / SM100 operators; unknown SKU uses `hw_recipe` / `run_b0_next.sh`
- [`docs/ROCM_TRAIN.md`](docs/ROCM_TRAIN.md) — AMD BF16 B0 continuation and CPU block offload
- [`operators/rocm/`](operators/rocm/README.md) — measured AMD operator/layout experiments and numerical checks
- [`docs/HF_HUB.md`](docs/HF_HUB.md) — weights on Hub only; `scripts/push_to_hf.sh`
- [`huggingface/README.md`](huggingface/README.md) — Hub model-card source
- [`checkpoints/b0-full/README.md`](checkpoints/b0-full/README.md) — published B0 overlay pointer
- [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) — B200 pre-recycle logs
- [`artifacts/autodl-rtx6000d/`](artifacts/autodl-rtx6000d/README.md) — 6000D logs (released)
- [`artifacts/autodl-rtx4080-super/`](artifacts/autodl-rtx4080-super/README.md) — 4080 SUPER smoke (released)
- [`artifacts/autodl-rtx3090/plan-verify/`](artifacts/autodl-rtx3090/plan-verify/README.md) — older 3090 plan-probe A→E (128 claims)
- [`artifacts/autodl-rtx3090/bf16-verify/`](artifacts/autodl-rtx3090/bf16-verify/README.md) — 3090 BF16 Flash-shaped probe A→E (142 claims)
- [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](artifacts/autodl-rtx3090/bf16-verify/mfu/README.md) — theoretical vs measured operator MFU

## Local tests

This repo does not ingest 50B tokens. The tiny config is for unit tests only.

```bash
python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
python3 -m unittest tests.test_train tests.test_trainer tests.test_phases tests.test_checkpoint \
  tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload tests.test_b1 tests.test_b2 \
  tests.test_nvfp4_linear tests.test_nvfp4_hw tests.test_moe_ops tests.test_b0_full tests.test_phase_cg \
  tests.test_hw_recipe tests.test_kda tests.test_deepspeed_zero
python3 scripts/param_budget.py --verify
python3 scripts/arch_verify.py --verify
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1 --graph plan
python3 -m unittest tests.test_plan_verify tests.test_attention_plan tests.test_ampere_mfu
```

With CUDA:

```bash
python3 -m cat_yoko.gpu_smoke --middle
python3 -m cat_yoko.gpu_smoke --c1
```

## License

Apache-2.0. See [`LICENSE`](LICENSE). The MiniCPM5-2B base is also Apache-2.0.
