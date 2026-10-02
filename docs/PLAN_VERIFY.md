# Plan mini-verify (Phase A→E)

Run **Phase A through Phase E** on DummyStream with the published recipe, in a dedicated directory:

| Graph | Purpose |
| --- | --- |
| **plan-probe** | CPU / CI. 3 encoder (sliding / CSA / HCA) + 2 decoder, `seq=32`, `n_win=8`, `hd=16` |
| **bf16-probe** | Ampere GPU. Same topology, `hidden=128`, `hd=32`, `seq=128`, `n_win=32 < seq`, `m=m'=8`. Large enough for Flash / cuDNN / mem-efficient, while leaving Theorem B's compression hole |

| Phase | What the probe does | Ampere BF16 operators |
| --- | --- | --- |
| **A** | 0 tokens. Dummy MiniCPM5 upcycle, two-stack MoE, gate=0, attention still sliding window (implement first, enable later). Does not pull Hub weights. | YOCO cross: dense Flash/cuDNN GQA; encoder window: masked cuDNN/efficient bf16; fused QKV; TF32 |
| **B0/B1/B2** | C1 freeze: new modules → freeze Encoder, train decoder → short joint | same; from B2, MoE padded bmm (`grouped_mm` only on SM90+) |
| **C** | indexer → topk → hca → win | indexer scores fp32 GEMM; CSA union / HCA concat use masked bf16, not promoted to fp32 |
| **D-8k** | DummyStream mid-sequence needle; sparse stays hca (probe seq is still graph width, not pulled out to published 8K) | same as C-hca |
| **E** | WSD decay | same as C-hca |

**Does not include F/G** (SFT / RL are post-training). This is not a 12B quality experiment, does not pull Ultra-FineWeb, does not write a CSA CUDA kernel, and does not write a full-graph checkpoint into this directory.

The GPU script defaults to `/root/autodl-tmp/bf16-verify/`. Do not mix it with the old plan-probe directory. Precision is **BF16** (not INT8, not FP8, not NVFP4 emulation).

## What this must prove

| Group | Assertions that enter the ledger |
| --- | --- |
| Attention | sliding-window GQA causal; CSA is a union mask (not class CSA); HCA concat window KV ∥ pooled slots; softmax high precision |
| YOCO | published 16/26 and 2+7+7; probe sliding+csa+hca; encoder top projection once; CrossAttention has only \(W_Q,W_O\); M2 off; gate 0→0.3→1; **on A with gate=0, changing \(W_K\) does not change logits** |
| PDSA already in graph | query-time window fallback; indexer only **deletes** from \(S_{\mathrm{comp}}\), never inserts; M1≠M3; HCA drops its own blocks; C chain indexer→topk→hca |
| Training plan | A two-stack MoE / no μP / still window; C1 8+27+15B; B0 trains new modules only; B1 still freezes encoder; B2 detach off; C enablement is monotonic; D needle; E WSD |
| Operators | dense prefers Flash; masked skips Flash (it rejects `attn_mask`), try cuDNN/efficient **bf16** then fp32 math; fused QKV; TF32; bf16-probe keeps the Theorem B hole |
| Deferred (not a failure) | PDSA Tier 1 calibrated fallback, Tier 3 editable memory (`FROZEN_SPEC` PDSA off) |

## How to run

```bash
# CPU / CI (plan-probe)
python3 -m cat_yoko.plan_verify --out /tmp/plan-verify --device cpu --steps 1 --graph plan
python3 -m unittest tests.test_plan_verify tests.test_attention_plan

# GPU BF16; artifacts go to a dedicated large-disk directory
bash scripts/run_plan_verify.sh
# GRAPH=bf16 OUT=/root/autodl-tmp/bf16-verify/runs STEPS=2
```

Entrypoint: `python3 -m cat_yoko.plan_verify` or `cat-yoko-plan-verify`. Writes `--out/ledger.json` (including per-phase `ops`).

Theoretical MFU of each operator versus the 3090 dense BF16 peak, plus operator-tuning notes: [`docs/AMPERE_OPS_MFU.md`](AMPERE_OPS_MFU.md), `python3 -m cat_yoko.ampere_mfu` / `bash scripts/run_ampere_mfu.sh`.

The probe keeps the window shorter than the sequence on purpose so the compression branch shows; published 12B is still `n_win=8192`, `m'=128`. Default `use_kda=False`; do not change the 132 tensors of the already published B0 overlay.
