# CAT-YOKO

Causal Encoder-Decoder (YOCO-style) hybrid-attention MoE, upcycled from MiniCPM-2B.

Default spec (**middle compute tier**): ≈12B total, Encoder ≈2.3B active / input token, Decoder ≈4.5B active / output token. CSA/HCA + 8K sliding window; primary long-context target 128K–256K.

## Docs

- [`docs/FROZEN_SPEC.md`](docs/FROZEN_SPEC.md) — **published recipe** (what the 12B trainer implements)
- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, staged upcycling recipe, data, optimizer, eval
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — middle-tier parameter / FLOPs / KV / μP ledger
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — causality, residual-cut equivalence, M1/M2/M3 cache interface
- [`docs/CURRICULUM_THEORY.md`](docs/CURRICULUM_THEORY.md) — freeze-curriculum **C1** (MoE both stacks, freeze encoder in B0/B1)
- [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) — **C1+FP8** frozen Phase B wall-clock (761 H100-h)

## Train (12B graph; tiny for tests)

```bash
python3 -m cat_yoko.train --config tiny --phase B0 --steps 3
python3 -m cat_yoko.train --config 12b --meta          # count params, no 24GB alloc
python3 -m cat_yoko.train --config 12b --dump-megatron # Megatron-LM mapping JSON
python3 -m unittest tests.test_train tests.test_megatron
```

## Recalculate / verify

```bash
python3 scripts/param_budget.py --verify     # middle-tier + freeze-curriculum + FP8 ledger
python3 scripts/param_budget.py --staged --curriculum --fp8
python3 scripts/arch_verify.py --verify      # architecture invariants
python3 -m unittest tests.test_param_budget tests.test_arch_verify tests.test_train tests.test_megatron
```
