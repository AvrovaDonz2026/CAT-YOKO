# 6000D C1 B2 `--try`

`python3 -m cat_yoko.b2 --try --resume /root/autodl-tmp/runs/b1`: 32 steps, `seq=64`, NVFP4 wrap of every allowed GEMM, unfreeze the full graph, `detach=False`, gate=1.0.
Per-layer `--offload-blocks` + CPU Adam, **`accum=1`**. DummyStream; do not load 50B Ultra-FineWeb.
The overlay writes only `trainable.pt` (Hub `checkpoints/b2/`), not a 23GiB `latest.pt`.

Remote directory: `/root/autodl-tmp/runs/b2`. Script: [`scripts/run_b2_try_autodl.sh`](../../../scripts/run_b2_try_autodl.sh).
B1 overlay: `/root/autodl-tmp/runs/b1`. encoder+embed come from a local MiniCPM5-2B-Base upcycle, then the B1 decoder/`lm_head`/RMSNorm are stacked on top.
Do not `git fetch` (GitHub fetch hangs on this machine). Do not touch the running published B0 tmux.

Run only when the GPU is free and B1 `trainable.pt` is complete. After the run, logs land in this directory; large weight files go only to HuggingFace [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2).
