# ROCm 算子实验

独立于训练代码的算子正确性与性能目录。2026-10-02 在 RX 7900 XTX
24GiB / PyTorch 2.9.1+ROCm6.4 实测，使用 CAT-YOKO 12B 形状。
测量时短暂 SIGSTOP 本次 B0 任务，运行器 finally 恢复同一进程；
训练权重仍占用显存，但没有其他 GPU 计算竞争。

## 实测结论

| 算子候选 | 原实现 → 候选 | 正确性与适用范围 |
| --- | --- | --- |
| 相同冻结 MoE 专家合并，真实 MiniCPM5 layer16、BF16、4096 tokens、top-k10 | 前向 23.29 → 4.17ms（5.58×）；前向+输入反向 46.89 → 7.31ms（6.42×） | 输出相对 L2 0.248%，输入梯度 0.313%；专家必须冻结且权重字节相同；未计 CPU 卸载搬运 |
| query 分块 FP32 causal GQA，4096、chunk512 | 前向约 17 → 12.3ms；前向+反向约 35–36 → 37ms；额外峰值 4208 → 668MiB | 输出/梯度相对 L2 <0.007%；省显存，训练速度收益有限 |
| 冻结 QKV 预拼接+pinned GPU buffer | 卸载搬运+前向+输入梯度 4.31 → 4.13ms（约4%） | 搬运后权重字节相同；输出/梯度与缓存融合投影相同 |
| shared gate/up 预拼接+pinned GPU buffer | 卸载搬运+前向+输入梯度 6.67 → 6.53ms（约2%） | 同上 |
| 实验 AOTriton Flash BF16 causal GQA | 排除 | Q 梯度相对 L2 约55.8%；原始GQA、重复KV、连续梯度hook、两种输入布局都失败 |

MoE 是优先候选，但上述倍率是单个 GPU 算子，不是整模型吞吐。
同专家合并改变 BF16 求和顺序，非 bitwise 等价；校验门槛是预先设定的
相对 L2 1.5%，FP32 1e-4，结果同时给出最大绝对误差。
真实权重测试的 router 是固定随机 router，原 B0 overlay 不含冻结
router；归一化 gates、输入梯度和 native MoE dispatch 均参与对照。
不能用于专家已分化或开始训练 MoE 的 B1/B2。

生产投影已经融合 QKV 和 gate/up，不能把“相对分开 GEMM 的2倍”当成
当前训练的可得收益。测得单向搬运约3.3–3.5GB/s；设备当前报告
PCIe 4.0 x16。预拼接的端到端改善只有2–4%，优先级低于 MoE。

目前所有候选只保存在此目录，持续训练使用已经验证的实现。

## 文件和复现

- `frozen_moe.py`：检查相同冻结权重，保留 router gates，比较 native MoE 输出/输入梯度；支持真实底座切片。
- `attention_bench.py`：真实16Q/2KV/hd128，FP32基准加独立因果公式对照；检查输入/上游梯度布局、Flash/重复KV/分块数学实现。
- `projection_bench.py`：真实底座QKV与shared gate/up，融合前后数值与pageable/pinned传输测量。
- `run_benchmarks.py`：无竞争测量，超时或报错后恢复本次训练；仅允许挂起 `run_b0_rocm.py` 进程。
- [results/rx7900xtx-20261002/](results/rx7900xtx-20261002/)：全部JSON/JSONL结果，含失败候选和运行退出码。

```bash
python operators/rocm/frozen_moe.py \
  --device cuda --base /path/to/MiniCPM5-2B-Base --layer 16 \
  --modes bf16 --topk 10 --tokens 4096 --warmup 2 --iterations 5 \
  --json /path/to/operator-results/frozen_moe_actual.json

TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python operators/rocm/attention_bench.py \
  --seq-lens 128 512 4096 --dtypes bf16 --layouts bhsd bshd \
  --reference-chunk-size 512 --chunk-size 512 --warmup 1 --repeats 2 \
  --output /path/to/operator-results/attention.jsonl

python operators/rocm/projection_bench.py \
  --device cuda --base /path/to/MiniCPM5-2B-Base --tokens 4096 \
  --warmup 2 --repeats 5 --out /path/to/operator-results/projection.json
```

注意力输出标为 `numerical_failure` 的候选不会计时或推荐。
完整模型接入还需要保持路由日志、checkpoint键与冻结边界，并做整模型
前向/梯度对照和独立 checkpoint 续训比较。
