# Status (2026-10-08)

GitHub training-progress pin. Spec knobs stay in [`FROZEN_SPEC.md`](FROZEN_SPEC.md).
Weights live on Hugging Face.

## Current published B0 snapshot

The pinned published release is **ROCm real-text B0 step 81864**, saved on
**2026-10-06T20:58:48.075579Z**. The completed 4000-update window ended at
**2026-10-07 05:01:53 Asia/Shanghai**, including final evaluation and full-state checks.
The 8B-token B0 envelope remains unfinished; B1 has not started.

| Item | Value |
| --- | --- |
| Complete overlay | [step-81864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/trainable.pt), 2,192,257,315 bytes |
| Weights only | [weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/weights-only.pt), 438,478,163 bytes; 132 byte-identical weights, no Adam |
| Global step / Adam / absolute cursor | **81864 / 47062 / 47062**, next physical row **8000 / 19531** |
| Phase token clocks | **355,010,560**, including earlier DummyStream history; 4.44% of the 8B envelope |
| Real input consumed | **192,765,952**, including repetitions of **79,998,976 unique training tokens** |
| Native recovery state | **132** BF16 weights, **264** CPU FP32 Adam moments, native groups, counters, RNG and packed cursor |
| Full SHA256 | `20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc` |
| Light SHA256 | `74d8a60b9e43613e917a5debd52d27776604ea69e1564e4a891b4f99dd906085` |
| Runtime | RX 7900 XTX / gfx1100; PyTorch 2.9.1 + ROCm 6.4; BF16 |
| Backend | Split packed FP32 attention, shared frozen MoE storage, cached BF16 CPU-shadow Adam with native retained FP32 moments |

The release remains a trainable overlay requiring MiniCPM5-2B-Base upcycling,
not a standalone full 12B graph. Complete CPU inspection, native Adam
load/export and all-132 light-weight byte comparisons pass. Python and CPU
Torch RNG restore checks pass; saved HIP RNG is schema-checked without GPU
allocation by the publisher. See the
[snapshot card](../checkpoints/b0-rocm-realtext/step-81864/README.md) and
[release identities](../checkpoints/b0-rocm-realtext/step-81864/release.json).
The default and explicit `b0-rocm-realtext-81864` download aliases select the
same complete recovery file; a weights-only alias selects the smaller overlay.

## Current ROCm continuation

The original pilot remains **79,998,976 training tokens** and **999,424
validation tokens**, with 60% English web / 30% Chinese web / 10% L2 math,
no code slice. Versions are in the
[corpus manifest](../artifacts/rocm-rx7900xtx/realtext-20261002/manifest.json).
Repeated consumption does not add unique data.

The second full pass completed at **73864**. The following 4000-update
window completed at **77864**, and the latest **77864 → 81864** window
completed all **4000** consecutive updates normally. Its run is
**`runs/window4000-77864-20261006T1410`**, with independent frozen source
**`source-window4000-77864-20261006T1410`**. The final supervisor receipt is
**2026-10-06T21:01:53.537453Z**. A fresh-data continuation has now started,
as recorded below.

| Counter | Source → completed endpoint |
| --- | --- |
| Global step | **77864 → 81864** |
| Absolute packed cursor / native Adam counter | **43062 → 47062** |
| Phase clocks | **338,626,560 → 355,010,560** |
| Real input including repetition | **176,381,952 → 192,765,952** |
| Next physical row | **4000 → 8000** of 19531 |

Adam, RNG, gate/LR clocks and the original B0 schedule were retained.
Complete saves remained every **300 seconds / keep3**, outside the fixed
source and independently verified recovery point. Primary evaluation ran
every 250 updates × 32 batches; additional rows 32–63 were evaluated at
start/end. All 4000 metrics retain both selected operators and ordinary
`deterministic_algorithms=False`. All-132 pre-update GPU gradient/output gates
have zero error against the preceding accepted packed-production backend.
The earlier dense-native norm failure remains under its original source and
is not relabelled a pass. The whole-corpus audit ceiling **58593** did not
increase the requested stop of **47062**.

| Slice, 32 batches | Initial at 77864 | Final at 81864 |
| --- | ---: | ---: |
| Primary rows 0–31, 130877 valid loss tokens | 7.363943263994131 | 7.29112438273621 |
| Additional rows 32–63, 130878 valid loss tokens | 7.5028788111342815 | 7.387959246989646 |

The first-eight / last-eight periodic primary NLL medians declined
**7.350183368378623 → 7.312580625589918**. Both end-to-end slices and this
periodic comparison improve; these are next-token scores from the same pilot
validation corpus. Both slices have now been observed across earlier
windows. They are not fresh external benchmarks, and no matched base-model,
gate-off, generation or reasoning comparison is claimed. Repeated input
remains **192,765,952** tokens over the same **79,998,976** unique tokens.
See the [completed window report](../operators/rocm/B0_WINDOW4000_77864_20261006.md),
[completion receipts](../operators/rocm/results/rx7900xtx-20261006-window81864/completion/status.json)
and [independent quality review](../operators/rocm/results/rx7900xtx-20261006-window81864/completion/quality-review.json).

The completed snapshot is published only after complete native-state,
paired-quality, allowlisted-payload and commit-pinned Hub verification.
Remote export used CPU only; local HF login remained on the local machine.
The current default is the verified 81864 release, while prior releases retain
immutable paths and explicit aliases.

This is still **B0**. Use the [real-text recovery guide](../operators/rocm/REAL_TRAINING.md)
with original base, tokenizer and packed corpora; default DummyStream
launchers do not restore this trajectory.

## Active fresh-data window

The independent **81864 → 85864** run started with the complete source state,
retaining native Adam, RNG, phase/global clocks and the accepted backend.
Fresh preparation completed **79,998,976 training tokens / 92,264 documents**
and **999,424 validation tokens / 1,171 documents**, retaining the 60:30:10
English/Chinese/math mix. All five exact-document hash intersections passed
independent rechecks: new train/new validation and both new splits against both
old splits. See the [data audit](../artifacts/rocm-rx7900xtx/freshtext-20261007/README.md).

The absolute cursor **47062** now explicitly maps to fresh row **0**, with
no modulo or wrapping. The window consumes **4000** new rows / **16,384,000**
input tokens; the prepared 80M-token pool is larger than this window.
The **2026-10-08 00:04:25 Asia/Shanghai** startup snapshot records **81945 /
81 updates**. All-132 output/gradient gates passed with zero reported error.

| Initial slice at 81864, 32 batches | NLL |
| --- | ---: |
| Old primary rows 0–31 | 7.291744287131125 |
| Old additional rows 32–63 | 7.388477264365619 |
| Fresh validation rows 0–31 | 7.371492111944588 |

Complete saves remain **300 seconds / keep3**. The first save, **81921**, passed
native CPU recovery checks: **132 weights / 264 FP32 moments**, Adam/cursor
**47119**, next fresh row **57**, with intact clocks and RNG schema. Its recorded
SHA256 is `4f5052e4bc3f633888e579cb014c16a720260509e0b9161f99c32b598af2a330`.
The final 85864 quality checks and HF publication are pending; the local
publication watcher is running, with credentials retained locally and verified
temporary weights removed after upload. No weights or corpus bins are stored
in the development repository.

Run `runs/fresh4000-81864-20261007T1550` uses frozen source
`source-fresh80m-20261007T1550`, code commit
`3428f4ea502a965c7f9632ec106043a518c7799d`. The earlier argument-parser failure
made zero updates and its receipts are retained separately. See the
[fresh-window guide](../operators/rocm/FRESH_CORPUS_20261007.md) and
[startup evidence](../operators/rocm/results/rx7900xtx-20261007-fresh4000/README.md).

## Previous real-text releases

[ROCm step **77864**](../checkpoints/b0-rocm-realtext/step-77864/README.md),
[step **53307**](../checkpoints/b0-rocm-realtext/step-53307/README.md) and
[step **52616**](../checkpoints/b0-rocm-realtext/step-52616/README.md)
remain downloadable with their original hashes and explicit step aliases.
Step 52616 used the preceding packed-attention/native CPU Adam backend;
these historical provenance records are not rewritten by the new release.

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
