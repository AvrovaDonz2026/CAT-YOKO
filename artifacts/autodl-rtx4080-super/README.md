# AutoDL RTX 4080 SUPER 销毁前迁移

实例 `autodl-container-6d164c9f44-4dccdd73`（torch 2.8.0+cu128，驱动 595.71.05）。
清单时刻 UTC `2026-09-18T14:39:15Z`。权重已在 Git LFS：[`checkpoints/b0/trainable.pt`](../../checkpoints/b0/trainable.pt)。

## 已迁入本目录

| 路径 | 内容 |
| --- | --- |
| `minicpm5_peak/*.json` | 12B C1 峰值：B0 22.83GiB / B1 28.6GiB / B2 2.6GiB |
| `minicpm5_cuda/` | 同卡 unittest + 独立 B0/B1/B2/C1 烟测 |
| `p0_gpu/` | 更早一轮 MiniCPM5 图烟测（含 tiny） |
| `gpu_run*.log` | 远程套件 stdout |
| `runs/b0_train.log` | B0 `--try` 32 步完整日志 |
| `runs/b0/metrics.jsonl` | 同跑 jsonl（与 `checkpoints/b0/metrics.jsonl` 相同） |
| `MIGRATE_MANIFEST.txt` | 销毁前文件树 |

## 未迁（有意丢掉）

- 两份 23GiB 全图 `b0ckpt` / `old_b0ckpt`：迁出前已删；GitHub LFS 单文件上限 5GiB。
- `trainable_step_24.pt`：与最终 overlay 同结构的中间步；曲线在 `metrics.jsonl`，最终权重是 step 32。
- MiniCPM5-2B-Base `model.safetensors`：hf-mirror 卡在 20MiB incomplete；完整权重回 HuggingFace。
- `miniconda3`、AutoDL `autopanel*.db`、gzip 分片。
