# Status (2026-10-06)

GitHub training-progress pin. Spec knobs stay in [`FROZEN_SPEC.md`](FROZEN_SPEC.md).
Weights live on Hugging Face.

## Current published B0 snapshot

The pinned published release is **ROCm real-text B0 step 77864**, saved on
**2026-10-06T13:02:17.286196Z**. The completed 4000-update window ended at
**21:05:21 Asia/Shanghai**, including final evaluation and full-state checks.
The 8B-token B0 envelope remains unfinished; B1 has not started.

| Item | Value |
| --- | --- |
| Complete overlay | [step-77864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/trainable.pt), 2,192,257,315 bytes |
| Weights only | [weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/weights-only.pt), 438,478,163 bytes; 132 byte-identical weights, no Adam |
| Global step / Adam / absolute cursor | **77864 / 43062 / 43062**, next physical row **4000 / 19531** |
| Phase token clocks | **338,626,560**, including earlier DummyStream history; 4.23% of the 8B envelope |
| Real input consumed | **176,381,952**, including repetitions of **79,998,976 unique training tokens** |
| Native recovery state | **132** BF16 weights, **264** CPU FP32 Adam moments, native groups, counters, RNG and packed cursor |
| Full SHA256 | `adf13e2e44a1fcbabbfc1f60cfab2cc61459d0949c95ec9e6e332287691a8676` |
| Light SHA256 | `0ad37d86006fe68d1cf861d4519484ed5ef5bf882af19f803cd3a056db53e84e` |
| Runtime | RX 7900 XTX / gfx1100; PyTorch 2.9.1 + ROCm 6.4; BF16 |
| Backend | Split packed FP32 attention, shared frozen MoE storage, cached BF16 CPU-shadow Adam with native retained FP32 moments |

The release remains a trainable overlay requiring MiniCPM5-2B-Base upcycling,
not a standalone full 12B graph. Complete CPU inspection, native Adam
load/export and all-132 light-weight byte comparisons pass. See the
[snapshot card](../checkpoints/b0-rocm-realtext/step-77864/README.md) and
[release identities](../checkpoints/b0-rocm-realtext/step-77864/release.json).
The Hub model card identifies subsequent completed releases if they are
newer than this checkout's pinned default.

## Current ROCm continuation

The original pilot remains **79,998,976 training tokens** and **999,424
validation tokens**, with 60% English web / 30% Chinese web / 10% L2 math,
no code slice. Versions are in the
[corpus manifest](../artifacts/rocm-rx7900xtx/realtext-20261002/manifest.json).
Repeated consumption does not add unique data.

The second full pass completed at **73864**, with fixed NLL improving
**7.872862032828506 → 7.486454654140886**. The following 4000-update window
completed normally at **77864**, preserving all 4000 consecutive updates,
operator flags and native state. Its held-out observations improved on both
slices of the same validation file:

| Slice, 32 batches | Initial at 73864 | Final at 77864 |
| --- | ---: | ---: |
| Primary rows 0–31, 130877 valid loss tokens | 7.4860189283560965 | 7.3636053664583905 |
| Additional rows 32–63, 130878 valid loss tokens | 7.599186639848141 | 7.503450781393409 |

Periodic primary NLL median declined **7.475844210160067 →
7.412587716432117** from the first eight evaluations to the last eight.
These observations support another bounded window; they do not establish
external generation or reasoning capability. See the
[completed window report](../operators/rocm/B0_WINDOW4000_20261006.md) and
[completion receipts](../operators/rocm/results/rx7900xtx-20261006-window4000/completion/status.json).

The user requested further training and checkpoint upload. A new detached
run, **`runs/window4000-77864-20261006T1410`**, launched at
**2026-10-06T14:35:45.195196Z** from the complete **77864** state. Its
independent source **`source-window4000-77864-20261006T1410`** matches all
preceding frozen training hashes. The new target is exactly **4000** updates:

| Counter | Source → target |
| --- | --- |
| Global step | **77864 → 81864** |
| Absolute packed cursor / native Adam counter | **43062 → 47062** |
| Phase clocks | **338,626,560 → 355,010,560** |
| Real input including repetition | **176,381,952 → 192,765,952** |
| Next physical row | **4000 → 8000** of 19531 |

Adam, RNG, gate/LR clocks and the original B0 schedule are retained.
Complete saves remain every **300 seconds / keep3**, outside the fixed source
and independently verified recovery point. Primary evaluation runs every
250 updates × 32 batches; disjoint rows 32–63 are evaluated at start/end.
At the archived **2026-10-06T14:54:50Z** observation, actual updates reached
**77999**, and **77980** passed the complete CPU checkpoint audit.
Fresh all-132 GPU gradient/output gates have zero error against the accepted
preceding packed-production backend. The earlier dense-native norm failure
remains recorded under its original source and is not relabelled a pass.

Initial primary NLL is **7.363943263994131**; additional NLL is
**7.5028788111342815**. Each will be compared with its own final score.
The whole-corpus audit ceiling **58593** does not increase the actual
requested stop of **47062**.

A detached local upload watcher waits for exact **81864** completion,
full native-state verification and paired final quality, then exports on
remote CPU, verifies the downloaded payload and publishes using local HF
login. The local machine must remain running and connected. Failed or
incomplete training is not published. See the
[new continuation/upload report](../operators/rocm/B0_WINDOW4000_77864_20261006.md)
and [launch receipts](../operators/rocm/results/rx7900xtx-20261006-window81864/README.md).

This is still **B0**. Use the [real-text recovery guide](../operators/rocm/REAL_TRAINING.md)
with original base, tokenizer and packed corpora; default DummyStream
launchers do not restore this trajectory.

## Previous real-text release

[ROCm step **52616**](../checkpoints/b0-rocm-realtext/step-52616/README.md)
remains downloadable with its original hashes and recovery metadata. It used
the preceding packed-attention/native CPU Adam backend. Its code, data and
operator provenance are not rewritten by this newer release.

## Historical published progress (2026-09-20)

| | |
| --- | --- |
| Stage | C1 **B0** (encoder frozen; new modules ≈219.21M / 132 tensors) |
| Envelope | 8e9 tokens, `seq=4096`, DummyStream (thinking mix: 5% short hashed code snippets; no 50B Ultra-FineWeb / StarCoder download) |
| Hub overlay | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| Machine (recycled) | Vast NVIDIA B200 SM 10.0; disk copied 2026-09-19T06:44Z, SSH refused after |
| Runtime | torch `2.11.0+cu128` + TE `@stable` nvcc 12.9 SM100 cubin |
| Throughput | **micro-batch=2**, ~15.6–15.7k tok/s, 8192 tok/step, trainer ~138GiB |
| Precision | student / frozen decoder **bf16**; frozen encoder GEMM **NVFP4 FPROP** |
| License | Apache-2.0 (code, derived weights, MiniCPM5 base) |
| B1 / B2 | Not started. Wait for the B0 envelope or a free GPU before `--try` |
| C–G | **C–F training paths are implemented** (D rewindows packed bins with **sparse=hca**, E `phase-e` + WSD decay, F UltraChat→SFT jsonl packed to seq, D/E/F inherit `use_kda`). **Wire the graph first, then enable it.** Default `use_kda=False`. No CSA CUDA kernel |
| Plan probe | **bf16-probe A→E passed on GPU** (RTX 3090, 142/142 claims, 0 fail, 2 deferred, 3.1s). Dense GQA=Flash; masked/HCA=cuDNN bf16; `math_fp32=0`. DummyStream: `cat_yoko.plan_verify --graph bf16` / [`PLAN_VERIFY.md`](PLAN_VERIFY.md). Logs: [`artifacts/autodl-rtx3090/bf16-verify/`](../artifacts/autodl-rtx3090/bf16-verify/README.md). Older plan-probe 128/128: [`plan-verify/`](../artifacts/autodl-rtx3090/plan-verify/README.md). Operator roofline: [`AMPERE_OPS_MFU.md`](AMPERE_OPS_MFU.md). Not F/G. No 50B download. No full-graph checkpoint |

## Machines

| Machine | Role | Stopped at |
| --- | --- | --- |
| RTX 4080 SUPER | graph / `--try` smoke | Released. Logs: [`artifacts/autodl-rtx4080-super/`](../artifacts/autodl-rtx4080-super/README.md) |
| RTX 6000D sm_120 | published B0 start (NVFP4 **emu**) | step **16020**, `tokens_in_phase=65,488,896`. Logs: [`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md) |
| Vast B200 SM 10.0 | published B0 continue (hardware NVFP4 FPROP) | step **26940**. Logs: [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |
| RTX 3090 sm_86 | BF16 `bf16-probe` A→E + operator roofline; B0 BF16 **sibling** continue | **142 claims ok**; Flash **85%** / MoE bmm **81%**. Sliding window `BANDED_SEQ_MIN=2048`. Historical B200 pin: Hub `b0-full` step **26940** / `7eebc9a4…`; the current recommended ROCm snapshot is above. 3090 snapshot is Hub [`checkpoints/b0-3090-bf16/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) step **33800** / `2dc31406…` (≈1.98% of 8e9). `SAVE_EVERY=200`. ~760 tok/s (`adam=ds-cpuadam`). **Pending release** (disk copied 2026-09-20T13:53Z). Logs: [`bf16-verify/`](../artifacts/autodl-rtx3090/bf16-verify/README.md), [`mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/README.md), [`b0-3090-bf16/`](../artifacts/autodl-rtx3090/b0-3090-bf16/README.md), [`occupancy/`](../artifacts/autodl-rtx3090/occupancy/README.md) |

Same-phase resume adds `tokens_in_phase` at 8192 tokens/step. At step **22100** it was 90,392,576.

## Historical next-GPU recipes

If the next card is unknown, **do not** implement a Megatron EP/TP loop, and **do not** default to the B200-only launch script (non-SM100 exits 4). Pick a B0 recipe from SM / VRAM / TE, then same-phase resume the Hub overlay:

```bash
python3 -m cat_yoko.hw_recipe --json          # also works with --family sm100 --gib 183
python3 scripts/probe_nvfp4_hw.py             # optional: TE FPROP / dX / WGRAD
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_next.sh                   # dispatch after probe; <40GiB forces --try
# TRY=1 bash scripts/run_b0_next.sh           # force 32-step smoke
# MICRO_BATCH=1 bash scripts/run_b0_next.sh
```

Known-SKU shortcuts: `bash scripts/run_b0_full_b200.sh` (SM100, default micro-batch=2), `bash scripts/run_b0_full_autodl.sh` (sm_120).

| Card | B0 recipe |
| --- | --- |
| SM100/103 and ≥160GiB (B200-class) | published envelope seq=4096, mb=2, encoder on GPU, no grad-ckpt, TE NVFP4 FPROP |
| sm_120 and ≥90GiB (6000D-class) | published envelope seq=4096, mb=1, encoder on GPU, grad-ckpt, `Nvfp4Linear` emu |
| Hopper SM90 and ≥40GiB | published envelope; <90GiB offload encoder; emu NVFP4 (FP8 is a doc fallback, do not write a new wrap) |
| Ampere/Ada and 40–90GiB | recipe still defaults to torch encoder offload. To shard the frozen 24.5GiB: `--backend deepspeed --zero 3 --zero-offload --zero-offload-param` ([`DEEPSPEED_ZERO.md`](DEEPSPEED_ZERO.md)). Fitting in VRAM ≠ finishing 8e9 |
| `<40GiB` | **refuse** the 8e9 envelope → `--try` seq=64 / 32 steps |
| CPU | JSON only; do not build the 12B graph |

Details: [`B200_TRAIN.md`](B200_TRAIN.md), [`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md), [`HF_HUB.md`](HF_HUB.md). `cat_yoko.hw_recipe` is a pure function; unit tests do not need a GPU.

**Do not**

- resume Hub `checkpoints/b0/` (that 32-step `--try`)
- `--save-full` / 23GiB `latest.pt` (especially on a 32GiB container disk)
- steal the GPU for B1/B2 `--try` while B0 still holds it
- implement a Megatron EP/TP loop or a CSA CUDA kernel
- treat ZeRO as a reason to restart 8e9 on one 3090 (the Hub overlay is already 1.63%; same-phase resume)
- overwrite Hub `checkpoints/b0-full` (step **26940**, sha256 `7eebc9a4…`). 3090 BF16 continue writes a sibling directory; see [`scripts/run_b0_ampere_3090.sh`](../scripts/run_b0_ampere_3090.sh)
- pull 50B Ultra-FineWeb into the repo or a small disk
- reconnect retired AutoDL `westc` / `weste`, or the 3090 `westd` host once it is released
- commit SSH passwords or deploy keys

## Where artifacts go

| What | Where |
| --- | --- |
| Code, theory, pointers, logs | this GitHub repo (**no LFS**) |
| `trainable.pt` / full graph / shards | [Hugging Face AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO) |
| Model-card source | [`huggingface/README.md`](../huggingface/README.md) → Hub root `README.md` |

`scripts/push_to_hf.sh` always ships the root card, `checkpoints/b0-full/README.md`, and `LICENSE`.
