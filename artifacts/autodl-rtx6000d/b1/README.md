# 6000D C1 B1 `--try`

`bash scripts/run_b1_try_autodl.sh` → `python3 -m cat_yoko.b1 --try` (32 steps, seq=64).

| Item | Value |
| --- | --- |
| envelope (published) | 27e9 tokens; this script is a GPU smoke, not a finished 27B run |
| student | nvfp4 (cache / cross / `lm_head` / decoder GEMM; router high precision) |
| trainable | decoder stack + untied `lm_head` + final RMSNorm |
| frozen | encoder + input embed |
| detach | True |
| gate | 0.3 → 1.0 |
| offload | `--offload-encoder --optim-cpu` (no `--offload-blocks`) |
| upcycle | MiniCPM5-2B-Base (`/root/autodl-tmp/hf/MiniCPM5-2B-Base`) |
| resume | `/root/autodl-tmp/runs/b0-full` (if an overlay exists) else `/root/autodl-tmp/runs/b0` |
| artifacts | write only `trainable.pt`; **do not write** 23GiB `latest.pt` |
| Hub | [`checkpoints/b1/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b1) |
| Hub download | `HF_ENDPOINT=https://hf-mirror.com` |

DummyStream; do not load a 50B corpus. Do not pull GitHub. Do not write passwords / deploy keys into this directory. Weights go to HuggingFace only, not GitHub.

Do not steal the GPU while 6000D is running published B0; run this script after B0 finishes.
