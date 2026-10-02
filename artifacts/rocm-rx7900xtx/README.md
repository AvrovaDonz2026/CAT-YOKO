# RX 7900 XTX B0 续训（2026-10-02）

Ubuntu 24.04，Radeon RX 7900 XTX 24GiB，CPU RAM 125GiB，
PyTorch `2.9.1+rocm6.4` / HIP `6.4.43484`。
工作目录 `/home/donz/cat-yoko-rocm-20261002/`。

从 Hub `checkpoints/b0-3090-bf16/trainable.pt` step **33800**、
158,140,416 tokens 接续 B0。输入 SHA256
`2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690`，
MiniCPM5 底座 SHA256
`d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`。
两者校验通过，输入文件保留。

CPU BF16 构图并 upcycle，逐层 GPU 前向/反向和 CPU Adam；
`seq=4096`、micro-batch=1、accum=1、8e9 token 预算，
`--run-steps` 限制本次新增更新而不改变 gate/LR 调度。
仅保存 132 张量 trainable overlay，不写完整图。

| 已完成验证 | 值 |
| --- | --- |
| step | 33800 → 33802 |
| tokens_in_phase | 158,140,416 → 158,148,608 |
| NLL | 11.8067 / 11.8076 |
| 吞吐 | 90 / 113 tok/s（包含逐层卸载） |
| peak allocated | 8526.8 MiB（约 8.33 GiB） |
| 参数更新 | 132 张量中 54 张量发生实际 BF16 值变化 |
| checkpoint | 远程 `runs/b0-rocm-verify/trainable.pt` |

证据：[验证结果](verify/result.json)、[指标](verify/metrics.jsonl)、
[完整两步日志](experiments/b0-rocm-verify-retry.log)。
续训/注意力/MoE/卸载的必要回归通过，日志在 `experiments/`。

随后从 **33802** 后台启动 `runs/b0-rocm/`，计划新增 **1000** 步，
每 10 步保存、保留最后两个编号 overlay。
进程 PID 初始为 `1018872`，日志 `experiments/b0-rocm.log`；
以远程日志/进程现状判断是否仍在运行。

本次最后核对：已完成并保存 **step 33830**，158,263,296 tokens，
约130 tok/s，进程仍运行，目标step34802。
见 [观察状态](observed_status.json) 与 [持续训练指标](metrics.jsonl)。

实验 gfx11 AOTriton 开关能启用 Flash/efficient SDPA，但初次 causal
GQA 梯度对照不通过：[失败记录](experiments/flash-validation.log)。
持续训练使用已验证的 FP32 math 注意力回退，显式关闭该实验开关。
进一步算子实验放在 `operators/rocm/`，不自动修改正在训练的图。
实测结果与候选说明见 [算子目录](../../operators/rocm/README.md)。

限制：仍是 DummyStream，没有语言质量提升结论；overlay 没有 Adam
状态，moments 从零恢复；分块梯度裁剪及 CPU 初始化 router 的舍入差异
见 [ROCM_TRAIN.md](../../docs/ROCM_TRAIN.md)。Hub B0-full 不覆盖。
