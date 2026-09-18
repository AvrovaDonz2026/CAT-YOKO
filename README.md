# CAT-YOKO

Causal Encoder-Decoder (YOCO-style) hybrid-attention MoE, upcycled from MiniCPM5-2B (Apache-2.0).

Default spec (**middle compute tier**): ≈12.25B total, Encoder ≈2.03B active / input token, Decoder ≈4.33B active / output token. CSA/HCA + 8K sliding window; primary long-context target 128K–256K. Phase B wall-clock **C1+NVFP4 = 571 H100-h**.

## Docs

- [`docs/FROZEN_SPEC.md`](docs/FROZEN_SPEC.md) — **published recipe** (what the 12B trainer implements)
- [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md) — architecture, staged upcycling recipe, data, optimizer, eval
- [`docs/THEORY_VERIFICATION.md`](docs/THEORY_VERIFICATION.md) — middle-tier parameter / FLOPs / KV / μP ledger
- [`docs/ARCHITECTURE_THEORY.md`](docs/ARCHITECTURE_THEORY.md) — causality, residual-cut equivalence, M1/M2/M3 cache interface
- [`docs/CURRICULUM_THEORY.md`](docs/CURRICULUM_THEORY.md) — freeze-curriculum **C1** (MoE both stacks, freeze encoder in B0/B1)
- [`docs/FP8_THEORY.md`](docs/FP8_THEORY.md) — C1+FP8 Hopper/Ada fallback (729 H100-h)
- [`docs/NVFP4_THEORY.md`](docs/NVFP4_THEORY.md) — **C1+NVFP4** frozen Phase B wall-clock (571 H100-h)

## Train (12B graph; tiny for tests)

Phase B 语料走 OpenBMB：**Ultra-FineWeb**（en/zh）+ **UltraData-Math**，用 **MiniCPM5-2B** tokenizer（`openbmb/MiniCPM5-2B`，`V=130560`，不要 MiniCPM-2B / MiniCPM3）。仓库不进 50B token；`prepare` 只切一块 mmap `.bin`。

发布入口是 **B0 / B1 / B2**，不是裸 `--phase`。12B 默认只写 Git LFS 能吃的 `trainable.pt`（B0 ≈0.44GiB bf16）；23GiB 全图 `latest.pt` **不进 git**（GitHub LFS 单文件 5GiB）。32GB 卡跑不完 8B token，用 `--try`（32 步、seq=64）。

```bash
python3 -m cat_yoko.b0 --try --save-dir checkpoints/b0
python3 -m cat_yoko.b1 --try --resume checkpoints/b0 --save-dir checkpoints/b1
python3 -m cat_yoko.b2 --try --resume checkpoints/b1 --save-dir checkpoints/b2
# H100 包络（8/27/15B tokens；<40GiB 卡会拒绝，除非显式 --steps/--tokens）：
# python3 -m cat_yoko.b0 --upcycle-hf openbmb/MiniCPM5-2B-Base --data data/phaseb.bin
```

```bash
# 本地 jsonl 烟测（不下载 HuggingFace）
python3 -m cat_yoko.prepare --mix local --local texts.jsonl --tokenizer dummy \
  --config tiny --out /tmp/t.bin --max-tokens 256
python3 -m cat_yoko.train --config tiny --phase B0 --steps 3 --accum 1 --data /tmp/t.bin
python3 -m cat_yoko.train --config tiny --phase B0 --steps 3 --accum 1 --data /tmp/t.bin \
  --device cuda --dtype bf16 --grad-ckpt

python3 -m cat_yoko.train --config 12b --meta
python3 -m cat_yoko.train --config 12b --dump-megatron

# 生产（需 pip install 'cat-yoko[data]'，会拉 Ultra-FineWeb；不要在 CI / 小 VM 上跑）
# python3 -m cat_yoko.prepare --mix phase-b --tokenizer openbmb/MiniCPM5-2B \
#   --config 12b --out data/phaseb.bin --max-tokens 1e8
# python3 -m cat_yoko.train --config 12b --phase B0 \
#   --upcycle-hf openbmb/MiniCPM5-2B-Base --data data/phaseb.bin \
#   --dtype bf16 --grad-ckpt --device cuda --steps N --save-dir runs/b0
# 12B 图在 ≥28GiB GPU 上跑 C1（bf16 直接建图，不经 CPU fp32）：
# python3 -m cat_yoko.gpu_smoke --middle
# python3 -m cat_yoko.gpu_smoke --middle --steps 2 --seq-len 64
# python3 -m cat_yoko.gpu_smoke --c1 --seq-len 64
# python3 -m cat_yoko.train --config 12b --device cuda --dtype bf16 --grad-ckpt \
#   --steps 1 --accum 1 --micro-batch 1 --seq-len 64
# python3 -m cat_yoko.train --config 12b --device cuda --c1-smoke --steps 1 --seq-len 64
# 12B 默认 micro-batch=1、grad-ckpt、bf16、weights-only ckpt
# python3 -m cat_yoko.train --config 12b --device cuda --c1 --steps 1 --seq-len 64 --save-dir runs/c1
# 12B ckpt 不要放 /tmp（23GiB×2 会写满 overlay）。save_every 命中末步时 latest.pt 是 step_N 的 hardlink：
# python3 -m cat_yoko.gpu_smoke --middle --save-dir /root/autodl-tmp/b0ckpt
# python3 -m cat_yoko.gpu_smoke --middle --steps 2 --resume /root/autodl-tmp/b0ckpt
python3 -m unittest tests.test_train tests.test_trainer tests.test_phases tests.test_checkpoint tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload
# 有 CUDA 的机器：
python3 -m cat_yoko.gpu_smoke
python3 -m cat_yoko.gpu_smoke --middle
python3 -m cat_yoko.gpu_smoke --c1
python3 -m unittest tests.test_gpu
```

## Recalculate / verify

```bash
python3 scripts/param_budget.py --verify     # middle-tier + freeze-curriculum + FP8 fallback + NVFP4 ledger
python3 scripts/param_budget.py --staged --curriculum --fp8 --nvfp4
python3 scripts/arch_verify.py --verify      # architecture invariants
python3 -m unittest tests.test_param_budget tests.test_arch_verify tests.test_train tests.test_trainer tests.test_phases tests.test_checkpoint tests.test_megatron tests.test_prepare tests.test_gpu tests.test_offload
```
