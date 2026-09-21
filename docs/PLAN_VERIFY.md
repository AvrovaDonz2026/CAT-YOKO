# Minimized plan verification (Phase A→E)

Run **Phase A through Phase E** on DummyStream, in a separate directory, under the published recipe:

| Graph | Purpose |
| --- | --- |
| **plan-probe** | CPU / CI. 3 encoder layers (sliding / CSA / HCA) + 2 decoder, `seq=32`, `n_win=8`, `hd=16` |
| **bf16-probe** | Ampere GPU. Same topology, `hidden=128`, `hd=32`, `seq=128`, `n_win=32 < seq`, `m=m'=8`. Large enough for Flash / cuDNN / mem-efficient, while still leaving Theorem B's compression hole |

| Phase | What the probe does | Ampere BF16 operators |
| --- | --- | --- |
| **A** | 0 tokens. Dummy MiniCPM5 upcycle, both stacks MoE, gate=0, attention still sliding (implement-then-light). Does not pull Hub weights. | YOCO cross: dense Flash/cuDNN GQA; encoder window: masked cuDNN/efficient bf16; fused QKV; TF32 |
| **B0/B1/B2** | C1 freeze: new modules → freeze encoder, train decoder → short joint | same; from B2, MoE padded bmm (`grouped_mm` only on SM90+) |
| **C** | indexer → topk → hca → win | indexer scores are fp32 GEMM; CSA union / HCA concat use masked bf16, not upcast to fp32 |
| **D-8k** | DummyStream mid-sequence needle; sparse stays hca (probe seq is still the graph width, not the published 8K) | same as C-hca |
| **E** | WSD decay | same as C-hca |

**Not F/G** (SFT / RL are post-training). This is not a 12B quality experiment, does not pull Ultra-FineWeb, does not write a CSA CUDA kernel, and does not write a full-graph checkpoint into this directory.

The GPU script writes `/root/autodl-tmp/bf16-verify/` by default; do not mix it with the older plan-probe directory. Precision is **BF16** (not INT8, not FP8, not NVFP4 emulation).

## What this is meant to prove

| Group | Assertions that enter the ledger |
| --- | --- |
| Attention | sliding-window GQA is causal; CSA is a union mask (not a CSA class); HCA concats window KV ∥ pooled slots; softmax is high precision |
| YOCO | published 16/26 and 2+7+7; probe sliding+csa+hca; one projection at encoder top; CrossAttention has only \(W_Q,W_O\); M2 off; gate 0→0.3→1; **on A, with gate=0, changing \(W_K\) does not change logits** |
| PDSA already in the graph | query-time window fallback; indexer only **deletes** from \(S_{\mathrm{comp}}\), never adds; M1≠M3; HCA excludes the own block; C chain is indexer→topk→hca |
| Training plan | A: both stacks MoE / no μP / still window; C1 8+27+15B; B0 trains new modules only; B1 still freezes encoder; B2 turns detach off; C lighting is monotonic; D needle; E WSD |
| Operators | dense prefers Flash; masked does not use Flash (it rejects `attn_mask`), tries cuDNN/efficient **bf16** then fp32 math; fused QKV; TF32; bf16-probe keeps the Theorem B hole |
| Deferred (not a fail) | PDSA Tier 1 calibrated fallback, Tier 3 editable memory (`FROZEN_SPEC` PDSA off) |

## How to run

```bash
# CPU / CI (plan-probe)
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1 --graph plan
python3 -m unittest tests.test_plan_verify tests.test_attention_plan

# GPU BF16; artifacts go to a large-disk separate directory
bash scripts/run_plan_verify.sh
# GRAPH=bf16 OUT=/root/autodl-tmp/bf16-verify/runs STEPS=2
```

Entry points: `python3 -m cat_yoko.plan_verify` or `cat-yoko-plan-verify`. Writes `--out/ledger.json` (including per-phase `ops`).

Per-op **theoretical MFU** relative to the 3090 dense BF16 peak, plus kernel-tuning notes: [`docs/AMPERE_OPS_MFU.md`](AMPERE_OPS_MFU.md), `python3 -m cat_yoko.ampere_mfu` / `bash scripts/run_ampere_mfu.sh`.

The probe keeps the window shorter than the sequence on purpose so the compression branch is visible; published 12B is still `n_win=8192`, `m'=128`. Default `use_kda=False`; that does not change the 132 tensors of the published B0 overlay.
