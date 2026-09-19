# Vast B200 释放前快照

实例 SSH Host `vast-b200`（不要把机器 IP 写进脚本）。**卡要回收 / destroy 会清盘**；stop/start 才保容器盘。本目录只留日志和探针，**权重不进 GitHub**。

权重：https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

| 项 | 值 |
| --- | --- |
| 时间 | 2026-09-19T06:44Z |
| 机器 | Vast NVIDIA B200 SM 10.0，183359 MiB |
| 运行时 | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 |
| 训练 | tmux `b0-full`，pid 31930，`--seq-len 4096`，`--micro-batch 2` |
| overlay | step **26940**，`tokens_in_phase=130,041,856`（8e9 的 ≈1.63%） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 吞吐 / 显存 | ~15.6k tok/s，Trainer 137952 MiB，nvidia-smi ~141/183 GiB |
| 许可 | Apache-2.0 |

不要 resume `checkpoints/b0/` 那份 32 步 `--try`。下一台：

```bash
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

日志：[`b0-full/`](b0-full/)（`metrics.jsonl`、tmux 尾）。TE 探针：[`te/`](te/)。
