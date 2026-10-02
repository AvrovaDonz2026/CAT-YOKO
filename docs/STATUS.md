# Status (2026-09-19)

This is the **training-progress ledger** on GitHub. Spec knobs still follow [`FROZEN_SPEC.md`](FROZEN_SPEC.md); weights live on Hugging Face.

## One sentence

CAT-YOKO-12B is training **B0** under **C1+NVFP4** (new modules, frozen encoder, 8e9 DummyStream). **It has not finished.** The Vast B200 was recycled; the latest overlay is on the Hub. Resume the same phase on the next machine; do not restart, and do not stack a 32-step `--try`.

## Progress

| Item | Value |
| --- | --- |
| Phase | C1 **B0** (encoder frozen; train new modules only ≈219.21M / 132 tensors) |
| Envelope | 8e9 tokens, `seq=4096`, DummyStream (50B Ultra-FineWeb not pulled) |
| Hub overlay | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Machine (recycled) | Vast NVIDIA B200 SM 10.0; copied at 2026-09-19T06:44Z, then SSH refused |
| Runtime | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 cubin |
| Throughput | **micro-batch=2**, ~15.6–15.7k tok/s, 8192 tok/step, Trainer ~138GiB |
| Precision | student / frozen decoder **bf16**; frozen encoder GEMM **NVFP4 FPROP** |
| License | Apache-2.0 (code, derived weights, MiniCPM5 base) |
| B1 / B2 | not started. Wait for the B0 envelope or a free GPU for `--try` |
| C–G | **C–F training paths are filled in** (D re-windows packed bins with **sparse=hca**; E `phase-e` + WSD decay; F UltraChat→SFT jsonl packed to seq; D/E/F inherit `use_kda`). Flow is **implement-then-light**. Default `use_kda=False`. No CSA CUDA kernel |
| Plan probe | **bf16-probe A→E passed on GPU** (RTX 3090, 142/142 claims, 0 fail, 2 deferred, 3.1s). dense GQA=Flash, masked/HCA=cuDNN bf16, `math_fp32=0`. Directory `/root/autodl-tmp/bf16-verify/`. DummyStream: `cat_yoko.plan_verify --graph bf16` / [`docs/PLAN_VERIFY.md`](PLAN_VERIFY.md). Logs [`artifacts/autodl-rtx3090/bf16-verify/`](../artifacts/autodl-rtx3090/bf16-verify/README.md). Earlier plan-probe 128/128: [`plan-verify/`](../artifacts/autodl-rtx3090/plan-verify/README.md). Operator roofline: [`docs/AMPERE_OPS_MFU.md`](AMPERE_OPS_MFU.md). Not F/G. Do not pull 50B. Do not write a full-graph checkpoint |

## Machine history

| Machine | Role | Stopped at |
| --- | --- | --- |
| RTX 4080 SUPER | graph / `--try` smoke | released; logs [`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md) |
| RTX 6000D sm_120 | published B0 start (NVFP4 **emulation**) | step **16020**, `tokens_in_phase=65,488,896`; logs [`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md) |
| Vast B200 SM 10.0 | published B0 continue (hardware NVFP4 FPROP) | step **26940**; logs [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |
| RTX 3090 sm_86 | BF16 `bf16-probe` A→E + operator roofline | **142 claims ok**; Flash **85%** / MoE bmm **81%** of 71.16T; logs [`bf16-verify/`](../artifacts/autodl-rtx3090/bf16-verify/README.md), [`mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/README.md). Older plan-probe 128 claims: [`plan-verify/`](../artifacts/autodl-rtx3090/plan-verify/README.md) |

Same-phase resume: `tokens_in_phase` keeps adding 8192/step. At step **22100** it was 90,392,576.

## How to pick up on the next machine

When the next GPU is unknown, **do not** wire Megatron EP/TP, and **do not** default to the B200-only launch script (non-SM100 exits 4). Emit a B0 recipe from SM / VRAM / TE, then same-phase resume the Hub overlay:

```bash
python3 -m cat_yoko.hw_recipe --json          # also works with --family sm100 --gib 183 when there is no GPU
python3 scripts/probe_nvfp4_hw.py             # optional: TE FPROP / dX / WGRAD
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_next.sh                   # probe then dispatch; <40GiB becomes --try
# TRY=1 bash scripts/run_b0_next.sh           # force 32-step smoke
# MICRO_BATCH=1 bash scripts/run_b0_next.sh
```

Known-SKU shortcuts remain: `bash scripts/run_b0_full_b200.sh` (SM100, default micro-batch=2), `bash scripts/run_b0_full_autodl.sh` (sm_120).

| GPU | B0 recipe |
| --- | --- |
| SM100/103 and ≥160GiB (B200 class) | published envelope seq=4096, mb=2, encoder on GPU, no grad-ckpt, TE NVFP4 FPROP |
| sm_120 and ≥90GiB (6000D class) | published envelope seq=4096, mb=1, encoder on GPU, grad-ckpt, `Nvfp4Linear` emulation |
| Hopper SM90 and ≥40GiB | published envelope; <90GiB offload encoder; emulate NVFP4 (documented FP8 fallback, no new wrap) |
| Ampere/Ada and 40–90GiB | recipe still defaults to torch encoder offload. To shard the frozen 24.5GiB weights: `--backend deepspeed --zero 3 --zero-offload --zero-offload-param` ([`DEEPSPEED_ZERO.md`](DEEPSPEED_ZERO.md)). Fitting ≠ finishing 8e9 |
| `<40GiB` | **refuse** the 8e9 envelope → `--try` seq=64 / 32 steps |
| CPU | JSON only; do not build the 12B graph |

Details: [`B200_TRAIN.md`](B200_TRAIN.md), [`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md), [`HF_HUB.md`](HF_HUB.md). `cat_yoko.hw_recipe` is a pure function; unit tests do not need a GPU.

**Do not**

- resume Hub `checkpoints/b0/` (that 32-step `--try`)
- `--save-full` / 23GiB `latest.pt` (especially on a 32GiB container disk)
- steal the GPU for B1/B2 `--try` while B0 still owns it
- implement a Megatron EP/TP loop or a CSA CUDA kernel
- treat ZeRO as a reason to restart 8e9 on one 3090 (the Hub overlay is already 1.63%; same-phase resume)
- pull 50B Ultra-FineWeb into the repo or a small disk
- reconnect released AutoDL `westc` / `weste`
- commit SSH passwords or deploy keys

## Where artifacts go

| Thing | Destination |
| --- | --- |
| Code, theory, pointers, logs | this GitHub repo (**no LFS**) |
| `trainable.pt` / full graph / shards | [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO) |
| Model-card source | [`huggingface/README.md`](../huggingface/README.md) → Hub root `README.md` |

`scripts/push_to_hf.sh` ships the root card, `checkpoints/b0-full/README.md`, and `LICENSE`.
