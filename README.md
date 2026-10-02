# CAT-YOKO

Causal encoder-decoder MoE in the YOCO style, upcycled from MiniCPM5-2B. License: **Apache-2.0**.

Middle tier: ≈12.25B stored parameters. Encoder ≈2.03B active / input token; decoder ≈4.33B active / output token. Phase B published wall-clock is **C1+NVFP4 = 571 H100-h** (ledger, not a measured run). Phase B attention is sliding-window GQA plus gated cross-attention. This repo does **not** implement a CSA CUDA kernel.

## Status

Published **B0 is in progress** and has not finished 8e9 tokens. The Vast B200 was recycled on 2026-09-19. The latest overlay lives on Hugging Face, not GitHub.

| | |
| --- | --- |
| Hub | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| last machine | Vast B200, micro-batch=2, ~15.7k tok/s |
| logs | [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) |

Full pin, machine history, and do-nots: [`docs/STATUS.md`](docs/STATUS.md).

If the next GPU is unknown, probe first, then dispatch. Do not default to `run_b0_full_b200.sh` (non-SM100 exits 4). Do not take that as a reason to implement Megatron:

```bash
python3 -m cat_yoko.hw_recipe --json
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_next.sh
```

Known B200 / SM100 still uses `run_b0_full_b200.sh` in [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md). `MICRO_BATCH=1` is a fallback. Same-phase resume keeps `tokens_in_phase`. **Do not** resume the 32-step `--try` at `checkpoints/b0/`. Do not `--save-full`.

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
