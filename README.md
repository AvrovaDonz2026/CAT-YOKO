# CAT-YOKO

Causal Encoder-Decoder (YOCO-style) hybrid-attention MoE, upcycled from MiniCPM-2B.

Default spec (**middle compute tier**): ≈12B total, Encoder ≈2.3B active / input token, Decoder ≈4.5B active / output token. CSA/HCA + 8K sliding window; primary long-context target 128K–256K.

## Docs

- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, staged upcycling recipe, data, optimizer, eval
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — middle-tier parameter / FLOPs / KV / μP ledger (this is the source of the published numbers)

## Recalculate / verify the middle-tier budget

```bash
python3 scripts/param_budget.py              # middle-tier summary
python3 scripts/param_budget.py --tier all
python3 scripts/param_budget.py --full
python3 scripts/param_budget.py --verify     # must exit 0
python3 -m unittest tests.test_param_budget
```
