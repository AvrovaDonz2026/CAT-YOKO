# RTX 3090 GPU 占用样例（释放前备份）

从待释放 AutoDL 3090 的 `/tmp/gpu_util_*` 拷出。1Hz / 高采样 nvidia-smi 记录，用来对齐 occupancy v2 与 1s SM 0% 空窗。不含 SSH 主机、密码、密钥。

| 文件 | 内容 |
| --- | --- |
| `gpu_util_sample.csv` | 早期高采样 occupancy |
| `gpu_util_leaf_baseline.*` | leaf 前基线 |
| `gpu_util_moe_leaf.*` | 只 leaf MoE |
| `gpu_util_block_leaf_cap8.log` | Block leaf + inflight cap=8（已弃用） |
| `gpu_util_cap2.*` | inflight=2 对照 |
