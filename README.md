# CAT-YOKO

Causal encoder-decoder (YOCO-style) MoE, upcycled from MiniCPM5-2B. License **Apache-2.0**.

Middle tier: ≈12.25B stored; encoder ≈2.03B active / input token, decoder ≈4.33B active / output token. Phase B wall-clock **C1+NVFP4 = 571 H100-h** (theory envelope, not measured). Phase B attention is sliding-window GQA + gated cross-attn; **CSA is not implemented**.

## Status

Published **B0 is in progress** and has not finished 8e9 tokens. The Vast B200 was recycled on 2026-09-19. The latest overlay lives on Hugging Face, not GitHub.

| Item | Value |
| --- | --- |
| Hub | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Last machine | Vast B200, micro-batch=2, ~15.7k tok/s |
| Logs | [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) |

Full accounting, machine history, and forbidden actions: [`docs/STATUS.md`](docs/STATUS.md).

When the next GPU is unknown, probe then dispatch (do not default to `run_b0_full_b200.sh`; non-SM100 exits 4; do not wire Megatron for this):

```bash
python3 -m cat_yoko.hw_recipe --json
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_next.sh
```

Known B200 / SM100 can still use `run_b0_full_b200.sh` in [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md). `MICRO_BATCH=1` is the fallback. Same-phase resume continues `tokens_in_phase`. **Do not** resume the 32-step `--try` under `checkpoints/b0/`, and do not `--save-full`.

## Spec

| Item | Value |
| --- | --- |
| Base | [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base) (Llama GQA, Apache-2.0) |
| \(d\) / \(V\) / \(L\) | 2048 / 130560 / 42 (16 encoder + 26 decoder) |
| Attention | 16 Q / 2 KV, `head_dim=128` |
| FFN / MoE | SwiGLU 6144; 1 shared + 20 routed; top-\(k\) 7/10 (enc/dec) |
| Tokenizer | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |

Published entry points are **B0 / B1 / B2**, plus **C–G** which have not started (`python3 -m cat_yoko.c|d|e|f|g`; C/D support `--chain`). B0 writes a ~419MiB `trainable.pt` overlay by default. The 23GiB full graph does not go on GitHub. Attention is **implement-then-light**: Phase B is sliding-window GQA (`--use-kda` only builds the 3:1 graph); Phase C then lights KDA→CSA→HCA. **Not a CSA kernel**. See `cat_yoko.kda`.

## Curriculum C1

| Sub-phase | tokens | Encoder | Trainable |
| --- | ---: | --- | --- |
| B0 | 8B | frozen | new modules only |
| B1 | 27B | frozen | decoder + `lm_head` + final RMSNorm |
| B2 | 15B | trainable | all |

B1/B2 have not started. Smoke with `--try` (32 steps, seq=64); that does not finish the envelope.

## Docs

**Spec and theory**

- [`docs/FROZEN_SPEC.md`](docs/FROZEN_SPEC.md) — published recipe (training code implements this)
- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, upcycling, data, optimizer
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — parameter / FLOPs / KV / μP ledger
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — causality, residual cut, M1/M2/M3
- [`docs/PLAN_VERIFY.md`](docs/PLAN_VERIFY.md) — minimized training in a separate directory; proves attention / YOCO / PDSA-in-graph and the C1 plan
- [`docs/AMPERE_OPS_MFU.md`](docs/AMPERE_OPS_MFU.md) — per-op theoretical MFU and kernel tuning on RTX 3090
- [`docs/CURRICULUM_THEORY.md`](docs/CURRICULUM_THEORY.md) — freeze curriculum C1
- [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) — C1+NVFP4 wall-clock (571 H100-h)
- [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) — C1+FP8 fallback (729 H100-h)
- [`docs/DEEPSPEED_ZERO.md`](docs/DEEPSPEED_ZERO.md) — DeepSpeed ZeRO (optional; 3090 48GiB uses ZeRO-3 + CPU offload)

**Training and artifacts**

- [`docs/STATUS.md`](docs/STATUS.md) — **current progress**
- [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md) — B200 / SM100 operators; unknown SKU uses `hw_recipe` / `run_b0_next.sh`
- [`docs/HF_HUB.md`](docs/HF_HUB.md) — weights live only on the Hub; `scripts/push_to_hf.sh`
- [`huggingface/README.md`](huggingface/README.md) — Hub model-card source
- [`checkpoints/b0-full/README.md`](checkpoints/b0-full/README.md) — published B0 overlay pointer
- [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) — B200 logs before recycle
- [`artifacts/autodl-rtx6000d/`](artifacts/autodl-rtx6000d/README.md) — 6000D logs (released)
- [`artifacts/autodl-rtx4080-super/`](artifacts/autodl-rtx4080-super/README.md) — 4080 SUPER smokes (released)
- [`artifacts/autodl-rtx3090/plan-verify/`](artifacts/autodl-rtx3090/plan-verify/README.md) — older plan-probe A→E on 3090 (128 claims)
- [`artifacts/autodl-rtx3090/bf16-verify/`](artifacts/autodl-rtx3090/bf16-verify/README.md) — BF16 Flash-shaped probe A→E on 3090 (142 claims)
- [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](artifacts/autodl-rtx3090/bf16-verify/mfu/README.md) — per-op theoretical vs measured MFU

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
