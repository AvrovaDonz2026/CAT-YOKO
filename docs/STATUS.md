# Status (2026-10-06)

GitHub training-progress pin. Spec knobs stay in [`FROZEN_SPEC.md`](FROZEN_SPEC.md).
Weights live on Hugging Face.

## Current published B0 snapshot

The recommended snapshot is **ROCm real-text B0 step 53307**, saved on
2026-10-04T05:37:22.748161Z. The 8B-token B0 envelope is still unfinished, and B1 has
not started. This immutable published save precedes the completed first corpus
pass. Both corpus passes have now completed; a 4000-update window in the
third pass is running as described below.

| Item | Value |
| --- | --- |
| Complete trainable-state overlay | [`checkpoints/b0-rocm-realtext/step-53307/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/trainable.pt) |
| Lightweight weights | [`weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/weights-only.pt), identical 132 BF16 tensors, no optimizer |
| Global step | **53307** |
| Total tokens in phase | **238,041,088** (includes earlier DummyStream history; 2.98% of 8B) |
| Actual packed real-text tokens | **75,796,480**; next unread row **18505 / 19531** |
| Native CPU Adam | **132** states, **264** finite FP32 moments, counter **18505** |
| Full snapshot SHA256 | `3dd9a62f7acfb4ae025ff44b0589b017104f07abc8b177e0098f5c4a910bd2b2` |
| Machine / runtime | RX 7900 XTX / gfx1100, PyTorch 2.9.1+ROCm 6.4, BF16 |
| Operators used for this published snapshot | Document-split packed FP32 attention, shared frozen expert storage, cached BF16 CPU-shadow Adam with native CPU FP32 moments |
| Fixed held-out NLL | **7.885217772929435** at step **53250**, 32 batches / 130877 valid loss tokens; 57 updates before the snapshot |

The full file contains trainable weights, Adam, RNG and the packed cursor; it
still requires MiniCPM5-2B-Base upcycling to reconstruct the frozen graph.
The light file cannot preserve the Adam trajectory. See the
[snapshot card](../checkpoints/b0-rocm-realtext/step-53307/README.md) and
[release manifest](../checkpoints/b0-rocm-realtext/step-53307/release.json) for
file sizes, base/tokenizer/data hashes and deployed-source hashes.

## Current ROCm continuation

The pilot has **79,998,976 unique training tokens** and **999,424 validation
tokens**, with 60% English web / 30% Chinese web / 10% L2 math and no code
slice. Corpus/tokenizer versions are in the
[manifest](../artifacts/rocm-rx7900xtx/realtext-20261002/manifest.json).
Phase counters also include earlier DummyStream history; repeated real-input
tokens are not new unique data.

The second corpus pass completed normally at **73864** on
**2026-10-06 02:00:44 Asia/Shanghai**, after all **19531** requested updates.
Absolute packed cursor and native Adam counter are **39062**, phase tokens
**322,242,560**, and cumulative real input is **159,997,952 tokens**, including
repetition. Final fixed held-out NLL is **7.486454654140886**, versus
**7.872862032828506** at the second-pass source; both use 32 batches / 130877
valid loss tokens. Best periodic NLL was **7.4756175927441015** at **73250**.
The final complete checkpoint contains all 132 BF16 trainable tensors and
264 retained CPU FP32 moments, with SHA256
`6dadd0450a25f598276fed4c5192b8564a89db7b3685c37eba86e6bf72214e1c`.
See the [second-pass report](../operators/rocm/CORPUS_PASS2_20261004.md) and
[new completion receipts](../operators/rocm/results/rx7900xtx-20261006-window4000/previous-completion/status.json).

The user approved another observation window. A detached supervisor has
launched **`runs/window4000-73864-20261006T0630`**, with frozen source
**`source-window4000-20261006T0630`**, from the complete **73864** state. It
requests exactly **4000** updates through **77864**, then stops for evaluation.
The packed stream repeats the same corpus in the same order, preserving Adam,
RNG, learning-rate/gate clocks and the original 8B B0 budget.

| Counter | Source → exact window target |
| --- | --- |
| Global step | **73864 → 77864** |
| Absolute packed cursor / Adam counter | **39062 → 43062** |
| Phase tokens, including earlier DummyStream history | **322,242,560 → 338,626,560** |
| Real input tokens consumed, including repetition | **159,997,952 → 176,381,952** |
| Unique training data | **79,998,976 tokens**, unchanged |

Document-split packed FP32 attention, shared frozen expert storage and cached
BF16 CPU-shadow Adam remain enabled, with native retained CPU FP32 moments
and ordinary `deterministic_algorithms=False` updates. The former
[operator adoption](../operators/rocm/OPERATOR_SWITCH_20261004.md) measured
2.55–3.03% whole-update gain and was explicitly user-requested.
Complete checkpoints save every **300 seconds**, retaining the latest
**three** rolling saves. The fixed source and independently verified recovery
point remain outside that rolling set. Fixed 32-batch evaluation runs every
**250** updates.

The new run binds the preceding completed production run, its full-model
checks and exact final state, and requires unchanged training mathematics.
It again checks all 132 gradients against the preceding packed production
attention before any updates. The earlier dense-native failure on two norm
gradients at source 54333 remains recorded; it is not relabelled as a pass.
Extra initial/final evaluation uses disjoint rows **32–63** of the same
validation file, separate from the repeatedly monitored rows **0–31**. This
is an additional pilot observation, not an external benchmark.
See the [4000-update scope and recovery policy](../operators/rocm/B0_WINDOW4000_20261006.md).
At **2026-10-06T07:11:48.088316+00:00**, the worker has completed **301** real
updates through **74165**. Fresh full-model comparison passes all 132
gradients and selected outputs with zero error against the preceding packed
production reference. Initial primary NLL is **7.4860189283560965**; the
additional disjoint slice has initial NLL **7.599186639848141**. Those different
slice levels must be compared with their own final scores, rather than each
other. Extra evaluation restores RNG/model/stream settings and advances only
its fresh validation cursor **32 → 64**. The independently inspected periodic
checkpoint at **74075** / cursor and Adam **39273** passes the complete CPU
state audit and native Adam load/export with all 264 moments byte-for-byte.
Python and CPU Torch RNG restoration pass; the saved GPU RNG schema is
checked without GPU allocation. The run has since verified a newer save at
**74132**. See the [runtime receipts](../operators/rocm/results/rx7900xtx-20261006-window4000/runtime/runtime_observed.json).

This remains **B0**; B1 has not started. Hugging Face's published **53307**
snapshot remains immutable, with its original release hashes and provenance.
The newer **73864** recovery state is remote and is not yet a new Hub release.
Use the [real-text recovery guide](../operators/rocm/REAL_TRAINING.md) with
explicit original `train.bin` / `eval.bin`; older default DummyStream
launchers do not restore this packed-stream trajectory.

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
