# RTX 3090 B0 BF16 同阶段 resume（不覆盖 Hub）

发布口径仍是 Hub `checkpoints/b0-full` step **26940** / sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`。这台 3090 只写 sibling 目录 `/root/autodl-tmp/b0-3090-bf16`。**没有** `push_to_hf`，**没有** `--save-full`。

2026-09-20T01:22Z–01:24Z，`--more-steps 8`，ZeRO-3 + param CPU offload，`--no-nvfp4`，DummyStream seq=4096。

| 项 | Hub 只读副本 | 3090 SAVE（未发布） |
| --- | --- | --- |
| 路径 | `/root/autodl-tmp/hub-b0-full/trainable.pt` | `/root/autodl-tmp/b0-3090-bf16/trainable.pt` |
| step | **26940** | 26948 |
| tokens_in_phase | 130,041,856 | 130,074,624（+8×4096） |
| gate | 0.0048828125 | 0.0048828125 |
| n_tensors | 132 | 132 |
| sha256 | `7eebc9a4…`（校验通过，mode 0444） | 本地 sibling，未上传 |

吞吐约 **730 tok/s**，显存约 41GiB。C1+NVFP4 墙钟结论不变；一张 3090 不能把剩余 ~7.87e9 token 跑完。

Git 只收日志：[`b0_3090.log`](b0_3090.log)、[`metrics.jsonl`](metrics.jsonl)。overlay 不进 GitHub。
