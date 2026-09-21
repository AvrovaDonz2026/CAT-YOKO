# B0 Ampere BF16 sibling (3090; does not overwrite the published pin)

Weights do not live on GitHub. Overlay on Hugging Face:

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16

This is an RTX 3090 **same-phase resume** of published B0 in BF16 (`--no-nvfp4`, ZeRO-3 + param CPU offload, DummyStream). It is **not** the published pin. Do **not** overwrite [`b0-full/`](../b0-full/README.md) step **26940** / sha256 `7eebc9a4…`.

AutoDL westd 3090 is **pending release**. This file is the 2026-09-20T13:53Z copy (`trainable_step_33800.pt`, mtime age ≫8s, **no** SIGINT). The previous Hub sibling was step **28800** / `17a2c495…`.

| | |
| --- | --- |
| File | `trainable.pt` (weights-only overlay, ~419MiB) |
| Stage | B0 |
| step | 33800 |
| tokens_in_phase | 158,140,416 (≈1.98% of the 8e9 envelope) |
| seq | 4096 |
| sha256 | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| Trainable tensors | 132 (no Adam) |
| Machine | AutoDL RTX 3090 sm_86 (pending release) |
| Runtime | BF16, `--backend deepspeed --zero 3 --zero-offload --zero-offload-param` |
| Throughput / memory | ~760 tok/s (`adam=ds-cpuadam`), trainer ~44GiB |
| Notes | DummyStream (thinking mix 5% hashed code lines); student bf16. Resumed Hub `b0-full` 26940 into this sibling. Same-phase resume keeps `tokens_in_phase`. |

```bash
python scripts/download_hub_overlay.py --name b0-3090-bf16 --out-dir /workspace/runs/b0-3090-bf16
# next Ampere continue (SAVE must be the sibling, never hub-b0-full):
STEPS=0 SAVE_EVERY=200 \
  RESUME=/path/to/b0-3090-bf16 \
  SAVE=/path/to/b0-3090-bf16 \
  bash scripts/run_b0_ampere_3090.sh
```
