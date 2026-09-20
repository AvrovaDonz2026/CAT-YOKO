# RTX 3090 B0 BF16 同阶段 resume（不覆盖 Hub 发布档）

发布口径仍是 Hub `checkpoints/b0-full` step **26940** / sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`。

3090 只写 sibling `/root/autodl-tmp/b0-3090-bf16`。2026-09-20T05:42Z 把已写完的 `trainable_step_28800.pt`（mtime 年龄 ≫5s，nlink=2，**没有** SIGINT / 杀进程）备份并上传到 Hub **新路径** [`checkpoints/b0-3090-bf16/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16)（sha256 `17a2c495…`）。**没有**覆盖 `b0-full`，**没有** `--save-full`。此前 Hub sibling 针脚是 step 28600 / `46ef6f99…`，本档替换为 28800。

| 项 | Hub 发布档 | 3090 sibling / Hub `b0-3090-bf16` |
| --- | --- | --- |
| 路径 | `checkpoints/b0-full/trainable.pt` | `checkpoints/b0-3090-bf16/trainable.pt` |
| step | **26940** | **28800** |
| tokens_in_phase | 130,041,856 | 137,660,416（≈1.72% of 8e9） |
| n_tensors | 132 | 132 |
| sha256 | `7eebc9a4…` | `17a2c495b31033d149ed2b95a42ecaa49dcd3d81acb25b0b19b64b33386d3393` |
| dtype | 发布 C1+NVFP4（B200） | BF16 `--no-nvfp4` |
| 吞吐 | ~15.7k tok/s（mb=2） | occupancy v2 ~720 tok/s（`adam=ds-cpuadam`，mb=1） |

重启后 `SAVE_EVERY=200`（约 18 分钟一次 gather）。思考 DummyStream 掺 **5%** 仓库内短代码 snippet（不拉 StarCoder / 50B）。overlay 不进 GitHub。日志：[`metrics.jsonl`](metrics.jsonl)。
