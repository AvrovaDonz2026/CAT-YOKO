# 6000D C1 B2 `--try`

`python3 -m cat_yoko.b2 --try --resume /root/autodl-tmp/runs/b1`：32 步、`seq=64`、NVFP4 wrap 全部允许的 GEMM、解冻全图、`detach=False`、gate=1.0。
逐层 `--offload-blocks` + CPU Adam，**`accum=1`**。DummyStream，不上 50B Ultra-FineWeb。
overlay 只写 `trainable.pt`（Hub `checkpoints/b2/`），不写 23GiB `latest.pt`。

远端目录：`/root/autodl-tmp/runs/b2`。脚本：[`scripts/run_b2_try_autodl.sh`](../../../scripts/run_b2_try_autodl.sh)。
B1 overlay：`/root/autodl-tmp/runs/b1`。encoder+embed 走本机 MiniCPM5-2B-Base 上采样，再叠 B1 decoder/`lm_head`/RMSNorm。
不要 `git fetch`（这台 GitHub fetch 会挂）。不要动正在跑的发布档 B0 tmux。

GPU 空闲且 B1 `trainable.pt` 齐了再跑。跑完日志落到本目录；权重大文件只上 HuggingFace [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2)。
