# RTX 3090 GPU occupancy samples (pre-release backup)

Copied from `/tmp/gpu_util_*` on the AutoDL 3090 that is pending release. 1 Hz / high-rate `nvidia-smi` traces used to align occupancy v2 with the 1s SM 0% gaps. No SSH hosts, passwords, or keys.

| File | Contents |
| --- | --- |
| `gpu_util_sample.csv` | Early high-rate occupancy |
| `gpu_util_leaf_baseline.*` | Baseline before leaf partitioning |
| `gpu_util_moe_leaf.*` | MoE-only leaf |
| `gpu_util_block_leaf_cap8.log` | Block leaf + inflight cap=8 (abandoned) |
| `gpu_util_cap2.*` | inflight=2 control |
