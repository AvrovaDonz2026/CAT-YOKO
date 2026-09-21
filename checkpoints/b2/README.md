# B2 overlay (Hub pointer)

Weights do not live on GitHub. Files are on Hugging Face:

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2

This is the C1 B2 overlay (published envelope `--tokens 15e9`, `seq=4096`). The 6000D smoke uses `--try` for 32 steps at `seq=64`.
Do not overwrite `checkpoints/b0/` or `checkpoints/b1/`.

| | |
| --- | --- |
| Envelope | 15e9 tokens; student **nvfp4**; full graph trainable |
| `detach` | `False` (gate stays 1.0) |
| Memory | `--offload-blocks` + `--optim-cpu`, **`accum=1`** (per-layer Adam cannot accumulate) |
| `--try` resume | MiniCPM5 upcycle of encoder + embed, then overlay B1 `trainable.pt` |
| Artifacts | Hub overlay / `shard-*.pt`; **do not** write a 23GiB `latest.pt` or `--save-full` |

Launch when the GPU is free and the B1 overlay is at `/root/autodl-tmp/runs/b1`:

```bash
bash scripts/run_b2_try_autodl.sh
```

Logs: [`artifacts/autodl-rtx6000d/b2/`](../../artifacts/autodl-rtx6000d/b2/).
