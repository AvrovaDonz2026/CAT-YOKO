# NVFP4 wrap 烟测（RTX 6000D sm_120）

2026-09-18。注意力仍是因果 YOCO `WindowAttention` + fp32 SDPA；只 wrap 允许的 Linear。

| 项 | 值 |
| --- | --- |
| tiny CUDA wrap | ok；B0 77 个冻结 GEMM；B1 `nvfp4_n=83` |
| 因果 window | wrap 后改未来 token 不泄漏 |
| 12B B0 `--try` | 2 步；`nvfp4=True`；wrap **2815** 个 Linear |
| gate / nll_last | 0.301 / 18.06（DummyStream，不是评估） |
| peak | 34442 MiB |
| tok/s | ≈7–8（E2M1/16 仿真，不是 TE kernel） |
| TE | 2.19 cu12 装上了；`transformer_engine.pytorch` 因 `ncclCommWindowRegister` 导不进。走仿真。 |

**大文件不进 GitHub。** overlay `trainable.pt`（419MiB，sha256 `461b4ffc05fd46e2668448393789764ccf9dd673644040fe4527259b176a510e`）在 HuggingFace [`checkpoints/b0-nvfp4-try/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-nvfp4-try)。不覆盖原来的 32 步 `checkpoints/b0/trainable.pt`。
