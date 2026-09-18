# 6000D C1 B1 `--try`

`bash scripts/run_b1_try_autodl.sh` → `python3 -m cat_yoko.b1 --try`（32 步、seq=64）。

| 项 | 值 |
| --- | --- |
| 信封（发布） | 27e9 tokens；本脚本是 GPU 烟测，不是 27B 跑完 |
| student | nvfp4（cache / cross / `lm_head` / decoder GEMM；router 高精度） |
| 可训练 | decoder 栈 + untied `lm_head` + 最终 RMSNorm |
| 冻结 | encoder + 输入 embed |
| detach | True |
| gate | 0.3 → 1.0 |
| 卸载 | `--offload-encoder --optim-cpu`（无 `--offload-blocks`） |
| 上采样 | MiniCPM5-2B-Base（`/root/autodl-tmp/hf/MiniCPM5-2B-Base`） |
| resume | `/root/autodl-tmp/runs/b0-full`（若有 overlay）否则 `/root/autodl-tmp/runs/b0` |
| 产物 | 只写 `trainable.pt`；**不写** 23GiB `latest.pt` |
| Hub | [`checkpoints/b1/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b1) |
| Hub 下载 | `HF_ENDPOINT=https://hf-mirror.com` |

DummyStream，不上 50B 语料。不拉 GitHub。不把密码 / deploy key 写进本目录。权重只上 HuggingFace，不进 GitHub。

6000D 正在跑发布档 B0 时不要抢 GPU；B0 结束后再跑本脚本。
