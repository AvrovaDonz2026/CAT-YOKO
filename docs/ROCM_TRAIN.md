# ROCm B0 续训

入口是 `scripts/run_b0_rocm.py`，用于小显存 AMD GPU 的 BF16 B0
续训。它在 CPU 直接构造 BF16 12B 图，从 MiniCPM5 底座重建冻结参数，
加载 B0 trainable overlay，随后逐层搬到 GPU 做前向和反向。
Adam 状态留在 CPU。不要把 AMD 的 `gfx` capability 当成 NVIDIA SM。

目前可接的 BF16 Hub 分支：

| 项 | 值 |
| --- | --- |
| 路径 | `checkpoints/b0-3090-bf16/trainable.pt` |
| step | 33800 |
| tokens_in_phase | 158,140,416 |
| SHA256 | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| 来源 | 从 B0-full step 26940 同阶段续训的 BF16 分支 |

新产物写到独立目录。B0-full 发布指针仍见 [STATUS.md](STATUS.md)。

```bash
YOKO_WORK=/path/to/large/workspace
HF_ENDPOINT=https://hf-mirror.com python scripts/download_minicpm5.py \
  --local-dir "$YOKO_WORK/hf/MiniCPM5-2B-Base" --check-hash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_hub_overlay.py \
  --name b0-3090-bf16 --out-dir "$YOKO_WORK/hf/b0-3090-bf16"

python scripts/run_b0_rocm.py \
  --base "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --resume "$YOKO_WORK/hf/b0-3090-bf16/trainable.pt" \
  --save-dir "$YOKO_WORK/runs/b0-rocm" \
  --run-steps 1000 --seq-len 4096 --save-every 10
```

需要已安装 ROCm 版 PyTorch、transformers、safetensors 和 huggingface_hub，
并有足够 CPU 内存容纳约 23GiB BF16 参数及加载、优化器开销。
4096 是起点；显存不足时可缩短 `--seq-len`，但须记录为不同运行配置。

`--run-steps N` 在恢复 checkpoint 之后计数，最多新增 N 次更新；
它保留 `--tokens 8e9` 的 gate/LR 进度。已有 `--steps` 仍表示绝对
终止步数。不要把 `--try` 的绝对 32 步上限用于 step 33800 的续训。

每次保存只写 B0 的 132 个可训练张量。日志是 `metrics.jsonl`，
结束时写 `result.json`。`trainable.pt` 指向最新已完成保存的 overlay；
原始 Hub 输入保留。没有完整模型 checkpoint 或自动上传。

这仍使用原来的 DummyStream，不是语言质量实验。原 overlay 不带 Adam
状态，因此 Adam moments 重新开始。逐层卸载使用既有的分块梯度裁剪。
CPU 初始化也不保证复现旧 CUDA 的冻结 router 随机值；当前冻结专家由
相同底座复制，routing 的差别主要体现为低精度求和顺序。ROCm 上不支持
的融合注意力会使用 FP32 math 回退，没有硬件 NVFP4 加速。

已验证运行记录见 [artifacts/rocm-rx7900xtx](../artifacts/rocm-rx7900xtx/README.md)。
独立算子优化实验见 [operators/rocm](../operators/rocm/README.md)。
