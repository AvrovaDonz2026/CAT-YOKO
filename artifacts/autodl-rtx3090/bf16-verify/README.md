# RTX 3090 BF16 最小化验证（Phase A→E）

独立目录 `/root/autodl-tmp/bf16-verify/` 上的 DummyStream 短训，不是 12B B0，也不是旧的 `plan-verify` 目录。

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，sm_86，48 GiB |
| 精度 | **BF16**（NVFP4 / FP8 / INT8 都关） |
| 图 | `CATYokoConfig.bf16_probe()`（3+2，hidden=128，hd=32，seq=128，n_win=32） |
| 链 | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| 算子 | dense Flash/cuDNN GQA；masked cuDNN/efficient bf16；fused QKV；TF32 |

跑完后这里放 `plan_verify.log`、`ledger.summary.json`、`ledger.json`。不到 F/G。不拉 Ultra-FineWeb。不写完整图 checkpoint。
