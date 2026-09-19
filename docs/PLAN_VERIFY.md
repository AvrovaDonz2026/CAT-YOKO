# 计划最小化验证（Route A 探针）

单独目录里用 **plan-probe** 图（3 encoder：sliding / CSA / HCA + 2 decoder，`seq=32`，`n_win=8 < seq`，`m=m'=4`）跑 DummyStream 短训，证明发布计划里**已经进计算图**的注意力、YOCO 与 PDSA 借鉴成立。不是 12B 质量实验，不拉 Ultra-FineWeb，不写 CSA CUDA kernel，不 `--save-full`。

## 要证明什么

| 组 | 进 ledger 的断言 |
| --- | --- |
| 注意力 | 滑窗 GQA 因果；CSA 是 union mask（不是 class CSA）；HCA concat 窗 KV ∥ 池化槽；softmax 高精度 |
| YOCO | 发布 16/26 与 2+7+7；探针 sliding+csa+hca；encoder 顶投影一次；CrossAttention 只有 \(W_Q,W_O\)；M2 关（cache 槽数 = 序列长）；gate 0→0.3→1；gate=0 ≡ 残差切 |
| PDSA 已进图 | query-time 窗回退；indexer 只从 \(S_{\mathrm{comp}}\) **删**不增；M1≠M3（indexer 不装 decoder）；HCA 自己块剔除；C 链 indexer→topk→hca；KDA 若实现则先点亮 |
| 训练计划 | C1 8+27+15B；B0 只训新模块；B1 仍冻 encoder；B2 detach 关且解冻 encoder；C-index 只训 indexer；C-topk 点亮 CSA、HCA 仍暗；此后 HCA 保持；D DummyStream 中段针；E WSD；F prompt `-100` |
| 延期（不算失败） | PDSA Tier 1 校准回退、Tier 3 可编辑记忆（`FROZEN_SPEC` PDSA 关） |

## 怎么跑

```bash
# CPU / CI
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1
python3 -m unittest tests.test_plan_verify

# GPU，产物进大盘独立目录（不要放 overlay `/tmp`）
bash scripts/run_plan_verify.sh
# OUT=/root/autodl-tmp/plan-verify/runs STEPS=2
```

入口：`python3 -m cat_yoko.plan_verify` 或 `cat-yoko-plan-verify`。写出 `--out/ledger.json`（`ok` / 每条 claim / 各阶段 NLL）。

探针把窗故意做得比序列短，压缩支路才会露出来；发布 12B 仍是 `n_win=8192`、`m'=128`。默认 `use_kda=False`，不改已发布 B0 overlay 的 132 张量。
