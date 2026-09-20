# RTX 3090 B0 BF16 同阶段 resume（不覆盖 Hub 发布档）

发布口径仍是 Hub `checkpoints/b0-full` step **26940** / sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`。

3090 只写 sibling `/root/autodl-tmp/b0-3090-bf16`。2026-09-20T13:53Z **释放前备份**：已写完的 `trainable_step_33800.pt`（mtime 年龄 669s，nlink=2，**没有** SIGINT / 杀进程）拷到本机并上传 Hub [`checkpoints/b0-3090-bf16/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16)（sha256 `2dc31406…`）。**没有**覆盖 `b0-full`，**没有** `--save-full`。此前 Hub sibling 针脚是 step 28800 / `17a2c495…`。

未带走（可再下，或已在 Hub）：

- `/root/autodl-tmp/hf/MiniCPM5-2B-Base`（~4.7GiB，`openbmb/MiniCPM5-2B-Base`）
- `/root/autodl-tmp/hub-b0-full/trainable.pt`（mode 444，sha256 `7eebc9a4…`，发布档已在 Hub）
- miniconda / 系统环境

带走：本目录日志、[`occupancy/`](../occupancy/README.md) 占用样例、Hub sibling overlay。

| 项 | Hub 发布档 | 3090 sibling / Hub `b0-3090-bf16` |
| --- | --- | --- |
| 路径 | `checkpoints/b0-full/trainable.pt` | `checkpoints/b0-3090-bf16/trainable.pt` |
| step | **26940** | **33800** |
| tokens_in_phase | 130,041,856 | 158,140,416（≈1.98% of 8e9） |
| n_tensors | 132 | 132 |
| sha256 | `7eebc9a4…` | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| dtype | 发布 C1+NVFP4（B200） | BF16 `--no-nvfp4` |
| 吞吐 | ~15.7k tok/s（mb=2） | ~760 tok/s（`adam=ds-cpuadam`，mb=1） |

拷盘时 live 已到 step ~33960；针脚是年龄够的 **33800**（SAVE_EVERY=200）。思考 DummyStream 掺 **5%** 仓库内短代码 snippet（不拉 StarCoder / 50B）。overlay 不进 GitHub。日志：[`metrics.jsonl`](metrics.jsonl)、[`b0_3090.log`](b0_3090.log)。
