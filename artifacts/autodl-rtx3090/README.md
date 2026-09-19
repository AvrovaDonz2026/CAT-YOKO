# AutoDL RTX 3090 INT8 mini-verify

Ampere **sm_86**，约 48GiB，torch `2.8.0+cu128`，Triton `3.4.0`。无 Transformer Engine。
独立目录跑 [`cat_yoko.mini_verify`](../../cat_yoko/mini_verify.py)（DummyStream，不拉 50B，不 `--save-full`）。

SageBwd INT8 `QK^T` 来自 [luyanaa/flash-attn-triton](https://github.com/luyanaa/flash-attn-triton/blob/main/flash_attn_triton/triton_kernel/attention_kernel_int8.py)（BSD-3-Clause, Copyright 2025 Alyssa Vance）。**不是 CSA CUDA kernel**。softmax 仍 fp32；PV / backward 保持高精度。上游核标成 Turing sm_75；本卡试跑后相对 SDPA cosine ≈ **0.99998**，训练走 `int8_backend=triton`。

## 结果

| 跑 | 图 | seq | 步 | 后端 | 墙钟 | 峰值 | 口径 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| `bf16` | tiny | 16 | 2×B0→F | none | 5.6s | ~19MiB | 全流程 ok |
| `int8` | `int8_probe` hd=32 | 64 | 2×B0→F | **triton** | 5.4s | ~26MiB | 全流程 ok |
| `int8` 12B B0 | dummy-upcycle | 64 | 1 | **triton** | nll=12.14 有限 | ~36GiB | 可训练 219.21M；overlay 不进 GitHub |

`int8-12b-b0/trainable.pt`（~419MiB）留在机器数据盘，**不进 GitHub / 不用 LFS**。发布档 B0 overlay 仍是 Hub step **26940**，本跑不覆盖。

## 本目录

| 路径 | 内容 |
| --- | --- |
| `mini-verify/summary.json` | 两路 recipe 总表 |
| `mini-verify/console.log` | bf16 + int8_probe stdout |
| `mini-verify/bf16/` `int8/` | 分 recipe `summary.json` + `metrics.jsonl` |
| `mini-verify/int8-12b-b0/` | 12B B0 一步日志 |

不要把 SSH 主机名、端口、密码写进 git。
