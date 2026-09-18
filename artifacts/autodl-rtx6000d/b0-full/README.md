# 6000D 发布档 B0

`python3 -m cat_yoko.b0`（无 `--try`）→ `--tokens 8e9 --seq-len 4096 --no-offload-encoder`。
student bf16；冻结 encoder GEMM 走 NVFP4 wrap。DummyStream，不上 50B Ultra-FineWeb。
overlay 只写 `trainable.pt`（Hub `checkpoints/b0-full/`），不写 23GiB `latest.pt`。

远端目录：`/root/autodl-tmp/runs/b0-full`。tmux session：`b0-full`。
32 步 `--try` 仍在 `/root/autodl-tmp/runs/b0`，本跑 resume 那份 overlay 再上采样 MiniCPM5。
