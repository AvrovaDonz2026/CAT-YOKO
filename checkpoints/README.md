# Phase B checkpoints

Weights **do not enter GitHub**. Overlay / full graph / shards go to [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO). This directory keeps pointers and small `--try` logs only. Progress: [`docs/STATUS.md`](../docs/STATUS.md).

| Path | Contents | Approx size | Destination |
| --- | --- | --- | --- |
| [`b0/`](b0/) | 6000D `--try` 32-step new-module overlay | ~419MiB | Hub `checkpoints/b0/` |
| [`b0-nvfp4-try/`](b0-nvfp4-try/README.md) | 6000D NVFP4 wrap `--try` 2 steps | ~419MiB | Hub |
| [`b0-full/`](b0-full/README.md) | **published B0 envelope** (8e9, in progress) | ~419MiB | Hub `checkpoints/b0-full/` |
| [`b1/`](b1/README.md) | decoder + `lm_head` + final RMSNorm | Hub after `--try` | Hub `checkpoints/b1/` |
| [`b2/`](b2/README.md) | full-model overlay | Hub after `--try` | Hub `checkpoints/b2/` |

12B full-graph `latest.pt` ≈ 23GiB: write only to a large disk, never git.

Hub `checkpoints/b0/` **is not** the 8B envelope. The in-progress published overlay is [`b0-full/`](b0-full/README.md) (step **26940** before B200 recycle, ≈1.63% of 8e9; micro-batch=2).

Resume published B0 (do not use the 32-step `--try`):

```bash
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
python3 -m cat_yoko.b0 --resume /workspace/runs/b0-full \
  --upcycle-hf openbmb/MiniCPM5-2B-Base
# or SM100: bash scripts/run_b0_full_b200.sh
```

The trainer MiniCPM5-upcycles then overlays `trainable.pt`. B1: same MiniCPM5 + **B0-full** overlay, then train the decoder. B2: MiniCPM5 fills encoder+embed, then stack the B1 overlay.

Logs: [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md), [`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md), [`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md).

32GB try run (cannot finish the envelope; artifacts stay local / Hub):

```bash
python3 -m cat_yoko.b0 --try --save-dir /tmp/runs/b0
python3 -m cat_yoko.b1 --try --resume /tmp/runs/b0 --save-dir /tmp/runs/b1
python3 -m cat_yoko.b2 --try --resume /tmp/runs/b1 --save-dir /tmp/runs/b2
```
