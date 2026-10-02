# B2 overlay (Hugging Face pointer)

Weights **do not enter GitHub**. Files on Hugging Face:

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2

This is the C1 **B2** landing (published envelope `--tokens 15e9`, `seq=4096`; 6000D smoke uses `--try` 32 steps, `seq=64`).
Do not overwrite `checkpoints/b0/` / `checkpoints/b1/`.

| Item | Value |
| --- | --- |
| Published envelope | 15e9 tokens, student **nvfp4**, full model trainable |
| `detach` | `False` (gate held at 1.0) |
| Memory | `--offload-blocks` + `--optim-cpu`, **`accum=1`** (per-layer Adam cannot accumulate) |
| `--try` handoff | MiniCPM5 upcycle encoder+embed, then overlay B1 `trainable.pt` |
| Artifacts | Hub overlay / `shard-*.pt`; **no** 23GiB `latest.pt` / `--save-full` |

Launch (GPU free and B1 overlay already at `/root/autodl-tmp/runs/b1`):

```bash
bash scripts/run_b2_try_autodl.sh
```

Logs: [`artifacts/autodl-rtx6000d/b2/`](../../artifacts/autodl-rtx6000d/b2/).
