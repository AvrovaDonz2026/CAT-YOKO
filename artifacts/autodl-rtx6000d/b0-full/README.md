# 6000D published B0

`python3 -m cat_yoko.b0` (no `--try`) → `--tokens 8e9 --seq-len 4096 --no-offload-encoder`.
student bf16; frozen encoder GEMM goes through the NVFP4 wrap. DummyStream; do not load 50B Ultra-FineWeb.
The overlay writes only `trainable.pt` (Hub `checkpoints/b0-full/`), not the 23GiB `latest.pt`.

The remote directory was `/root/autodl-tmp/runs/b0-full`. tmux session: `b0-full`.
The 32-step `--try` lives in `/root/autodl-tmp/runs/b0`; do not resume that copy again.

**This machine is about to be released.** Weights are on the Hub [`checkpoints/b0-full/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full); logs are in this directory. B200 has already continued training; the current nail is [`checkpoints/b0-full/README.md`](../../../checkpoints/b0-full/README.md) (step **26940**).

Same-phase restart from `trainable_step_1400.pt` at 2026-09-18T18:06Z:

| Item | Before restart (miniconda 2.8) | After restart (venv-nightly) |
| --- | --- | --- |
| Python | `/root/miniconda3/bin/python3` torch 2.8.0+cu128 | `/root/autodl-tmp/venv-nightly/bin/python` torch 2.15.0.dev20260918+cu130 |
| Operators | old graph at launch | permute MoE + padded SwiGLU + frozen NVFP4 cache + GQA flash SDPA |
| tok/s | ~1350 | ~1670 |
| mem | 30789 MiB | 42092 MiB (frozen encoder weight quantization cache) |
| NVFP4 GEMM | E2M1/16 emulation | still emulation (nightly `float4` `copy_` is still NotImplemented; TE pytorch can already import) |

Same-phase restart again from `trainable_step_10560.pt` at 2026-09-19T00:27Z:

| Item | Before restart | After restart |
| --- | --- | --- |
| Operators | padded batched MoE | jagged `grouped_mm` MoE (falls back to padded on failure); fused QKV; chunked lm_head+CE |
| tok/s | ~1670 | ~2820 |
| mem | 42092 MiB | 56090 MiB |
| banner | — | `grouped_mm=True te=True te_nvfp4=False return_logits=False nvfp4_n=1072` |

**Snapshot before release** step **16020** (`tokens_in_phase=65,488,896`, ≈0.82% of the 8e9 envelope, sha256 `b5763b98…`). B0 **has not finished training**.
