# 6000D 发布档 B0

`python3 -m cat_yoko.b0`（无 `--try`）→ `--tokens 8e9 --seq-len 4096 --no-offload-encoder`。
student bf16；冻结 encoder GEMM 走 NVFP4 wrap。DummyStream，不上 50B Ultra-FineWeb。
overlay 只写 `trainable.pt`（Hub `checkpoints/b0-full/`），不写 23GiB `latest.pt`。

远端目录：`/root/autodl-tmp/runs/b0-full`。tmux session：`b0-full`。
32 步 `--try` 仍在 `/root/autodl-tmp/runs/b0`，不要再 resume 那份。

2026-09-18T18:06Z 从 `trainable_step_1400.pt` 同阶段重启：

| 项 | 重启前（miniconda 2.8） | 重启后（venv-nightly） |
| --- | --- | --- |
| Python | `/root/miniconda3/bin/python3` torch 2.8.0+cu128 | `/root/autodl-tmp/venv-nightly/bin/python` torch 2.15.0.dev20260918+cu130 |
| 算子 | 启动时旧图 | permute MoE + padded SwiGLU + frozen NVFP4 cache + GQA flash SDPA |
| tok/s | ~1350 | ~1670 |
| mem | 30789 MiB | 42092 MiB（冻结 encoder 权重量化缓存） |
| NVFP4 GEMM | E2M1/16 仿真 | 仍仿真（nightly `float4` `copy_` 仍 NotImplemented；TE pytorch 已能 import） |

Hub overlay step **9720**（`tokens_in_phase=39,684,096`，信封 8e9 的 ≈0.50%）：[`checkpoints/b0-full/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full)。实例随时可能没，这是快照，B0 **还没训完**。
