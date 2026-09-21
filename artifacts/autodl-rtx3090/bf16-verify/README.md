# RTX 3090 BF16 mini-verify (Phase A→E)

DummyStream short train in a dedicated directory `/root/autodl-tmp/bf16-verify/`. This is not 12B B0, and it is not the old `plan-verify` directory.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, sm_86, 48 GiB |
| torch | 2.8.0+cu128 |
| precision | **BF16** (NVFP4 / FP8 / INT8 all off) |
| graph | `CATYokoConfig.bf16_probe()` (3+2, hidden=128, hd=32, seq=128, n_win=32) |
| chain | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| steps | A is a 0-token surgery; every other phase is 2 steps |
| ledger | **ok=True, 142 claims, 0 fail, 2 deferred** (PDSA Tier 1/3) |
| wall-clock | 3.1 s (2026-09-19T17:05:23Z exit 0) |
| peak | ~38 MiB |
| operators | dense GQA **Flash**; masked / HCA concat **cuDNN bf16**; `math_fp32=0` during training; TF32 `high`; `grouped_mm=False` (SM90 gate, uses bmm); fused QKV |
| MFU cross-check | [`mfu/`](mfu/): 12B-shape Flash **85%** of roofline, MoE bmm **81%**, fused QKV frozen cache saturates tensor cores |

Does not include F/G. Does not pull Ultra-FineWeb. Does not write a full-graph checkpoint.

Files: [`plan_verify.log`](plan_verify.log), [`ledger.summary.json`](ledger.summary.json), [`ledger.json`](ledger.json).
