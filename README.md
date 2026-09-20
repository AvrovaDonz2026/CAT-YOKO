# CAT-YOKO

Causal Encoder-Decoder（YOCO 式）MoE，从 MiniCPM5-2B 上采样。许可 **Apache-2.0**。

中间档：≈12.25B 存储；Encoder ≈2.03B active / 输入 token，Decoder ≈4.33B active / 输出 token。Phase B 墙钟 **C1+NVFP4 = 571 H100-h**（理论信封，不是实测）。注意力 Phase B 是滑窗 GQA + 门控 cross-attn，**不实现 CSA**。

## 现状

发布档 **B0 进行中**，尚未跑完 8e9。Vast B200 已于 2026-09-19 回收。最新 overlay 在 HuggingFace，不是 GitHub。

| 项 | 值 |
| --- | --- |
| Hub | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856（≈1.63% of 8e9） |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| 上次机器 | Vast B200，micro-batch=2，~15.7k tok/s |
| 日志 | [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) |

完整口径、机器沿革、禁止项：[`docs/STATUS.md`](docs/STATUS.md)。

下一张卡未知时，先探测再 dispatch（不要默认 `run_b0_full_b200.sh`，非 SM100 会 exit 4；也不要为此去接 Megatron）：

```bash
python3 -m cat_yoko.hw_recipe --json
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_next.sh
```

已知 B200 / SM100 仍可用 [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md) 里的 `run_b0_full_b200.sh`。`MICRO_BATCH=1` 可回退。同阶段 resume 会接上 `tokens_in_phase`。**不要** resume `checkpoints/b0/` 那份 32 步 `--try`，不要 `--save-full`。

## 规格

| 项 | 值 |
| --- | --- |
| 底座 | [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base)（Llama GQA，Apache-2.0） |
| \(d\) / \(V\) / \(L\) | 2048 / 130560 / 42（16 encoder + 26 decoder） |
| 注意力 | 16 Q / 2 KV，`head_dim=128` |
| FFN / MoE | SwiGLU 6144；1 shared + 20 routed；top-\(k\) 7/10（enc/dec） |
| Tokenizer | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) |

发布入口是 **B0 / B1 / B2**，以及尚未开跑的 **C–G**（`python3 -m cat_yoko.c|d|e|f|g`，C/D 可 `--chain`）。B0 默认写 ~419MiB `trainable.pt` overlay。23GiB 全图不进 GitHub。注意力 **先实现、后点亮**：Phase B 是滑窗 GQA（`--use-kda` 只把 3:1 图建进图）；Phase C 再点亮 KDA→CSA→HCA。**不是 CSA kernel**。见 `cat_yoko.kda`。

## 课程 C1

| 子阶段 | token | Encoder | 可训练 |
| --- | ---: | --- | --- |
| B0 | 8B | 冻结 | 仅新模块 |
| B1 | 27B | 冻结 | decoder + `lm_head` + 最终 RMSNorm |
| B2 | 15B | 可训练 | 全部 |

B1/B2 还没开。烟测用 `--try`（32 步、seq=64），不能跑完信封。

## 文档

**规格与理论**

- [`docs/FROZEN_SPEC.md`](docs/FROZEN_SPEC.md) — 发布配方（训练代码按这个实现）
- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — 架构、上采样、数据、优化器
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — 参数 / FLOPs / KV / μP 账本
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — 因果、残差切、M1/M2/M3
- [`docs/PLAN_VERIFY.md`](docs/PLAN_VERIFY.md) — 单独目录最小化训练，证明注意力 / YOCO / PDSA 进图与 C1 计划
- [`docs/AMPERE_OPS_MFU.md`](docs/AMPERE_OPS_MFU.md) — RTX 3090 上各算子理论 MFU 与调算子
- [`docs/CURRICULUM_THEORY.md`](docs/CURRICULUM_THEORY.md) — 冻课程 C1
- [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) — C1+NVFP4 墙钟（571 H100-h）
- [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) — C1+FP8 回退（729 H100-h）
- [`docs/DEEPSPEED_ZERO.md`](docs/DEEPSPEED_ZERO.md) — DeepSpeed ZeRO（可选；3090 48GiB 走 ZeRO-3 + CPU offload）

**训练与产物**

- [`docs/STATUS.md`](docs/STATUS.md) — **当前进度**
- [`docs/B200_TRAIN.md`](docs/B200_TRAIN.md) — B200 / SM100 算子；未知卡走 `hw_recipe` / `run_b0_next.sh`
- [`docs/HF_HUB.md`](docs/HF_HUB.md) — 权重只走 Hub；`scripts/push_to_hf.sh`
- [`huggingface/README.md`](huggingface/README.md) — Hub 模型卡源
- [`checkpoints/b0-full/README.md`](checkpoints/b0-full/README.md) — 发布档 B0 overlay 指针
- [`artifacts/vast-b200/`](artifacts/vast-b200/README.md) — B200 释放前日志
- [`artifacts/autodl-rtx6000d/`](artifacts/autodl-rtx6000d/README.md) — 6000D 日志（已释放）
- [`artifacts/autodl-rtx4080-super/`](artifacts/autodl-rtx4080-super/README.md) — 4080 SUPER 烟测（已释放）
- [`artifacts/autodl-rtx3090/plan-verify/`](artifacts/autodl-rtx3090/plan-verify/README.md) — 3090 上旧 plan-probe A→E（128 claims）
- [`artifacts/autodl-rtx3090/bf16-verify/`](artifacts/autodl-rtx3090/bf16-verify/README.md) — 3090 上 BF16 Flash 形探针 A→E（142 claims）
- [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](artifacts/autodl-rtx3090/bf16-verify/mfu/README.md) — 各算子理论 vs 实测 MFU

## 本地测试

仓库不进 50B token。tiny 配置只给单测。

```bash
python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
python3 -m unittest tests.test_train tests.test_trainer tests.test_phases tests.test_checkpoint \
  tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload tests.test_b1 tests.test_b2 \
  tests.test_nvfp4_linear tests.test_nvfp4_hw tests.test_moe_ops tests.test_b0_full tests.test_phase_cg \
  tests.test_hw_recipe tests.test_kda tests.test_deepspeed_zero
python3 scripts/param_budget.py --verify
python3 scripts/arch_verify.py --verify
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1 --graph plan
python3 -m unittest tests.test_plan_verify tests.test_attention_plan tests.test_ampere_mfu
```

有 CUDA：

```bash
python3 -m cat_yoko.gpu_smoke --middle
python3 -m cat_yoko.gpu_smoke --c1
```

## License

Apache-2.0。见 [`LICENSE`](LICENSE)。MiniCPM5-2B 底座同样是 Apache-2.0。
