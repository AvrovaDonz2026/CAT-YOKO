# RTX 3090 BF16 最小化验证（Phase A→E）

独立目录 `/root/autodl-tmp/bf16-verify/` 上的 DummyStream 短训，不是 12B B0，也不是旧的 `plan-verify` 目录。

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，sm_86，48 GiB |
| torch | 2.8.0+cu128 |
| 精度 | **BF16**（NVFP4 / FP8 / INT8 都关） |
| 图 | `CATYokoConfig.bf16_probe()`（3+2，hidden=128，hd=32，seq=128，n_win=32） |
| 链 | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| 步数 | A 为 0 token 手术；其余各 2 步 |
| ledger | **ok=True，142 claims，0 fail，2 deferred**（PDSA Tier 1/3） |
| 墙钟 | 3.1 s（2026-09-19T17:05:23Z exit 0） |
| 峰值 | ~38 MiB |
| 算子 | dense GQA **Flash**；masked / HCA concat **cuDNN bf16**；训练中 `math_fp32=0`；TF32 `high`；`grouped_mm=True`；fused QKV |

不到 F/G。不拉 Ultra-FineWeb。不写完整图 checkpoint。

文件：[`plan_verify.log`](plan_verify.log)、[`ledger.summary.json`](ledger.summary.json)、[`ledger.json`](ledger.json)。
