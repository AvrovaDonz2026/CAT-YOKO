# B0 published overlay (B200, 8e9 tokens)

Weights do not live on GitHub. The latest overlay is on Hugging Face:

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full

This is the C1 B0 **published envelope** (`--tokens 8e9`, `seq=4096`) **in progress**: a trainable overlay, not `--try`, and not a finished 8e9 run.
Do not overwrite the 32-step MiniCPM5 overlay under `checkpoints/b0/`.

The same card is pushed to the Hub folder: https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md

Vast B200 **was recycled** (2026-09-19). The table is the pre-recycle snapshot. Progress: [`docs/STATUS.md`](../../docs/STATUS.md).

| | |
| --- | --- |
| File | `trainable.pt` (weights-only overlay, ~419MiB) |
| Stage | B0 |
| step | 26940 |
| tokens_in_phase | 130,041,856 (≈1.63% of the 8e9 envelope) |
| seq | 4096 |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Trainable tensors | 132 (no Adam) |
| Machine | Vast NVIDIA B200 SM 10.0 (recycled) |
| Runtime | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 cubin |
| Throughput / memory | ~15.7k tok/s, trainer ~138GiB (micro-batch=2) |
| Notes | DummyStream; student bf16; frozen encoder GEMM is hardware NVFP4 FPROP; same-phase resume can continue. Not finished. `MICRO_BATCH=1` is a fallback. |

The previous 6000D snapshot was step **16020** (`b5763b98…`). This file replaced that Hub path.

Same-phase B0: **do not** resume the 32-step `--try` at `checkpoints/b0/`.

```bash
bash scripts/run_b0_next.sh
# known B200 / SM100:
bash scripts/run_b200.sh
# or
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

Trainer: MiniCPM5 upcycle → overlay `trainable.pt` → restore `step` / `tokens_in_phase` / stream / RNG.
