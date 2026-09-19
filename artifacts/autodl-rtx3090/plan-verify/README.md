# RTX 3090 plan-probe（Phase A→E）

独立目录 `/root/autodl-tmp/plan-verify/` 上的 DummyStream 短训，不是 12B B0。

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，sm_86，48 GiB |
| torch | 2.8.0+cu128 |
| 图 | `CATYokoConfig.plan_probe()`（3+2，seq=32，n_win=8） |
| 链 | **A → B0 → B1 → B2 → C-index → C-topk → C-hca → C-win → D-8k → E** |
| 步数 | A 为 0 token 手术；其余各 2 步 |
| ledger | **ok=True，128 claims，0 fail，2 deferred**（PDSA Tier 1/3） |
| 墙钟 | 3.697 s |
| 峰值 | ~20 MiB |

不到 F/G。不拉 Ultra-FineWeb。不 `--save-full`。

文件：[`plan_verify.log`](plan_verify.log)、[`ledger.summary.json`](ledger.summary.json)、[`ledger.json`](ledger.json)。
