# 计划最小化验证（Phase A→E）

单独目录里用 DummyStream 按发布配方跑 **Phase A 到 Phase E**：

| 图 | 用途 |
| --- | --- |
| **plan-probe** | CPU / CI。3 encoder（sliding / CSA / HCA）+ 2 decoder，`seq=32`，`n_win=8`，`hd=16` |
| **bf16-probe** | Ampere GPU。同一拓扑，`hidden=128`，`hd=32`，`seq=128`，`n_win=32 < seq`，`m=m'=8`。给 Flash / cuDNN / mem-efficient 一个够大的形状，同时留下 Theorem B 的压缩洞 |

| 阶段 | 探针做什么 | Ampere BF16 算子 |
| --- | --- | --- |
| **A** | 0 token。dummy MiniCPM5 上采样、两栈 MoE、gate=0、注意力仍是滑窗（先实现后点亮）。不拉 Hub 权重。 | YOCO cross：dense Flash/cuDNN GQA；encoder 窗：masked cuDNN/efficient bf16；fused QKV；TF32 |
| **B0/B1/B2** | C1 冻结：新模块 → 冻 Encoder 训 decoder → 短联合 | 同上；B2 起 MoE padded bmm（`grouped_mm` 仅 SM90+） |
| **C** | indexer → topk → hca → win | indexer 分数 fp32 GEMM；CSA union / HCA concat 走 masked bf16，不升 fp32 |
| **D-8k** | DummyStream 中段针；sparse 保持 hca（探针 seq 仍是图宽，不拉到发布 8K） | 同 C-hca |
| **E** | WSD decay | 同 C-hca |

**不到 F/G**（SFT / RL 是后训练）。不是 12B 质量实验，不拉 Ultra-FineWeb，不写 CSA CUDA kernel，不把完整图 checkpoint 写进这个目录。

GPU 脚本默认写 `/root/autodl-tmp/bf16-verify/`，不要和旧的 plan-probe 目录混用。精度是 **BF16**（不是 INT8，不是 FP8，不是 NVFP4 仿真）。

## 要证明什么

| 组 | 进 ledger 的断言 |
| --- | --- |
| 注意力 | 滑窗 GQA 因果；CSA 是 union mask（不是 class CSA）；HCA concat 窗 KV ∥ 池化槽；softmax 高精度 |
| YOCO | 发布 16/26 与 2+7+7；探针 sliding+csa+hca；encoder 顶投影一次；CrossAttention 只有 \(W_Q,W_O\)；M2 关；gate 0→0.3→1；**A 上 gate=0 时改 \(W_K\) 不改 logits** |
| PDSA 已进图 | query-time 窗回退；indexer 只从 \(S_{\mathrm{comp}}\) **删**不增；M1≠M3；HCA 自己块剔除；C 链 indexer→topk→hca |
| 训练计划 | A 两栈 MoE / 无 μP / 仍 window；C1 8+27+15B；B0 只训新模块；B1 仍冻 encoder；B2 detach 关；C 点亮单调；D 针；E WSD |
| 算子 | dense 优先 Flash；masked 不走 Flash（它拒 `attn_mask`），先 cuDNN/efficient **bf16** 再 fp32 math；fused QKV；TF32；bf16-probe 保持 Theorem B 洞 |
| 延期（不算失败） | PDSA Tier 1 校准回退、Tier 3 可编辑记忆（`FROZEN_SPEC` PDSA 关） |

## 怎么跑

```bash
# CPU / CI（plan-probe）
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1 --graph plan
python3 -m unittest tests.test_plan_verify tests.test_attention_plan

# GPU BF16，产物进大盘独立目录
bash scripts/run_plan_verify.sh
# GRAPH=bf16 OUT=/root/autodl-tmp/bf16-verify/runs STEPS=2
```

入口：`python3 -m cat_yoko.plan_verify` 或 `cat-yoko-plan-verify`。写出 `--out/ledger.json`（含每阶段 `ops`）。

各算子相对 3090 dense BF16 峰值的 **理论 MFU** 与调算子记录：[`docs/AMPERE_OPS_MFU.md`](AMPERE_OPS_MFU.md)，`python3 -m cat_yoko.ampere_mfu` / `bash scripts/run_ampere_mfu.sh`。

探针把窗故意做得比序列短，压缩支路才会露出来；发布 12B 仍是 `n_win=8192`、`m'=128`。默认 `use_kda=False`，不改已发布 B0 overlay 的 132 张量。
