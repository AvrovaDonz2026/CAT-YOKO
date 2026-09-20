# RTX 3090 B0 BF16 同阶段 resume（不覆盖 Hub 发布档）

发布口径仍是 Hub `checkpoints/b0-full` step **26940** / sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`。

3090 只写 sibling `/root/autodl-tmp/b0-3090-bf16`。2026-09-20T02:30Z 把已写完的 `trainable_step_27400.pt` 备份并上传到 Hub **新路径** [`checkpoints/b0-3090-bf16/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16)（sha256 `4951c637…`）。**没有**覆盖 `b0-full`，**没有** `--save-full`。

| 项 | Hub 发布档 | 3090 sibling / Hub `b0-3090-bf16` |
| --- | --- | --- |
| 路径 | `checkpoints/b0-full/trainable.pt` | `checkpoints/b0-3090-bf16/trainable.pt` |
| step | **26940** | **27400** |
| tokens_in_phase | 130,041,856 | 131,926,016 |
| n_tensors | 132 | 132 |
| sha256 | `7eebc9a4…` | `4951c6373f3438cc29a735c2ffcbfb57dd4c315c9cc3f8a515abe33ec7b14143` |
| dtype | 发布 C1+NVFP4（B200） | BF16 `--no-nvfp4` |

重启后 `SAVE_EVERY=200`（约 18 分钟一次 gather，不再每 50 步掉占用）。思考 DummyStream 掺 **5%** 仓库内短代码 snippet（不拉 StarCoder / 50B）。overlay 不进 GitHub。日志：[`metrics.jsonl`](metrics.jsonl)。
