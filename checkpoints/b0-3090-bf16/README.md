# B0 Ampere BF16 sibling（3090，不覆盖发布档）

权重不进 GitHub。overlay 在 HuggingFace：

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16

这是 RTX 3090 上 **同阶段 resume** 发布档 B0 的 BF16 快照（`--no-nvfp4`，ZeRO-3 + param CPU offload，DummyStream）。**不是** 发布口径，也**不要**覆盖 [`b0-full/`](../b0-full/README.md) step **26940** / sha256 `7eebc9a4…`。

AutoDL westd 3090 **待释放**。本文件是 2026-09-20T13:53Z 拷盘针脚（`trainable_step_33800.pt`，mtime 年龄 ≫8s，**没有** SIGINT）。此前 Hub sibling 是 step **28800** / `17a2c495…`。

| 项 | 值 |
| --- | --- |
| 文件 | `trainable.pt`（weights-only overlay，约 419MiB） |
| 阶段 | B0 |
| step | 33800 |
| tokens_in_phase | 158,140,416（信封 8e9 的 ≈1.98%） |
| seq | 4096 |
| sha256 | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| 可训练张量 | 132（无 Adam） |
| 机器 | AutoDL RTX 3090 sm_86（待释放） |
| 运行时 | BF16，`--backend deepspeed --zero 3 --zero-offload --zero-offload-param` |
| 吞吐 / 显存 | ~760 tok/s（`adam=ds-cpuadam`），Trainer ~44GiB |
| 说明 | DummyStream（思考 5% 哈希代码行）；student bf16。从 Hub `b0-full` 26940 接到 sibling。同阶段 resume 保留 `tokens_in_phase`。 |

```bash
python scripts/download_hub_overlay.py --name b0-3090-bf16 --out-dir /workspace/runs/b0-3090-bf16
# 下一张 Ampere 续训（SAVE 必须是 sibling，不能是 hub-b0-full）：
STEPS=0 SAVE_EVERY=200 \
  RESUME=/path/to/b0-3090-bf16 \
  SAVE=/path/to/b0-3090-bf16 \
  bash scripts/run_b0_ampere_3090.sh
```
