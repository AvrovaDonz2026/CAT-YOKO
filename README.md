# CAT-YOKO

Causal Encoder-Decoder (YOCO-style) hybrid-attention MoE, upcycled from MiniCPM-2B.

Default spec (**middle compute tier**): ≈12B total, Encoder ≈2.3B active / input token, Decoder ≈4.5B active / output token. CSA/HCA + 8K sliding window; primary long-context target 128K–256K.

## Docs

- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, staged upcycling recipe, data, optimizer, eval
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — middle-tier parameter / FLOPs / KV / μP ledger
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — causality, residual-cut equivalence, M1/M2/M3 cache interface

## Recalculate / verify

```bash
python3 scripts/param_budget.py --verify     # middle-tier budget ledger
python3 scripts/arch_verify.py --verify      # architecture invariants
python3 -m unittest tests.test_param_budget tests.test_arch_verify
```
