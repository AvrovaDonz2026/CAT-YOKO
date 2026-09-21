# NVFP4 wrap smoke (RTX 6000D sm_120)

2026-09-18. Attention remains causal YOCO `WindowAttention` + fp32 SDPA; only allowed Linears are wrapped.

| Item | Value |
| --- | --- |
| tiny CUDA wrap | ok; B0 77 frozen GEMMs; B1 `nvfp4_n=83` |
| causal window | after wrap, changing a future token does not leak |
| 12B B0 `--try` | 2 steps; `nvfp4=True`; wrapped **2815** Linears |
| gate / nll_last | 0.301 / 18.06 (DummyStream, not an eval) |
| peak | 34442 MiB |
| tok/s | ≈7–8 (E2M1/16 emulation, not a TE kernel) |
| TE (2.8 miniconda) | 2.19 cu12 installed; `transformer_engine.pytorch` cannot import because of `ncclCommWindowRegister`. Uses emulation. |
| TE (nightly cu130) | `transformer_engine.pytorch` **imports** on torch `2.15.0.dev20260918+cu130`. `float4` `copy_` still fails; TE NVFP4 Linear needs leading dim % 16 == 0. Probe: [`NVFP4_PROBE_nightly.json`](NVFP4_PROBE_nightly.json). |

**Large files do not go on GitHub.** Overlay `trainable.pt` (419MiB, sha256 `461b4ffc05fd46e2668448393789764ccf9dd673644040fe4527259b176a510e`) is on HuggingFace [`checkpoints/b0-nvfp4-try/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-nvfp4-try). It does not overwrite the original 32-step `checkpoints/b0/trainable.pt`.
