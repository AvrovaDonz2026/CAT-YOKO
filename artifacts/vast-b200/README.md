# Vast B200 snapshot before release

SSH Host was `vast-b200` (do not write the machine IP into scripts). **SSH refused from 2026-09-19T06:47Z**; treat as recycled. destroy will wipe the disk. This directory keeps only logs and probes; **weights do not go into GitHub**.

Weights: https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

Progress: [`docs/STATUS.md`](../../docs/STATUS.md).

| Item | Value |
| --- | --- |
| Time | 2026-09-19T06:44Z disk copy |
| Machine | Vast NVIDIA B200 SM 10.0, 183359 MiB (recycled) |
| Runtime | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 |
| Training at the time | tmux `b0-full`, `--seq-len 4096`, `--micro-batch 2` |
| overlay | step **26940**, `tokens_in_phase=130,041,856` (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Throughput / VRAM | ~15.6k tok/s, Trainer 137952 MiB, nvidia-smi ~141/183 GiB |
| License | Apache-2.0 |

Do not resume the 32-step `--try` under `checkpoints/b0/`. Next machine:

```bash
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

Logs: [`b0-full/`](b0-full/) (`metrics.jsonl`, tmux tail). TE probes: [`te/`](te/).
