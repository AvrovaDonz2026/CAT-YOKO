# RTX 3090 plan-probe (Phase A→E)

DummyStream short training in a separate directory `/root/autodl-tmp/plan-verify/`. This is not 12B B0.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, sm_86, 48 GiB |
| torch | 2.8.0+cu128 |
| Graph | `CATYokoConfig.plan_probe()` (3+2, seq=32, n_win=8) |
| Chain | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| Steps | A is a 0-token surgery; the rest are 2 steps each |
| ledger | **ok=True, 128 claims, 0 fail, 2 deferred** (PDSA Tier 1/3) |
| Wall-clock | 3.697 s |
| Peak | ~20 MiB |

Not F/G. Do not pull Ultra-FineWeb. Do not `--save-full`.

Files: [`plan_verify.log`](plan_verify.log), [`ledger.summary.json`](ledger.summary.json), [`ledger.json`](ledger.json).
