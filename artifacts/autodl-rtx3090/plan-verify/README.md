# RTX 3090 plan-probe (Phase A→E)

DummyStream short train in a dedicated directory `/root/autodl-tmp/plan-verify/`. This is not 12B B0.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, sm_86, 48 GiB |
| torch | 2.8.0+cu128 |
| graph | `CATYokoConfig.plan_probe()` (3+2, seq=32, n_win=8) |
| chain | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| steps | A is a 0-token surgery; every other phase is 2 steps |
| ledger | **ok=True, 128 claims, 0 fail, 2 deferred** (PDSA Tier 1/3) |
| wall-clock | 3.697 s |
| peak | ~20 MiB |

Does not include F/G. Does not pull Ultra-FineWeb. Does not `--save-full`.

Files: [`plan_verify.log`](plan_verify.log), [`ledger.summary.json`](ledger.summary.json), [`ledger.json`](ledger.json).
