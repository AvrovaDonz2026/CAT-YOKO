# Ampere BF16 operator theoretical MFU (RTX 3090)

The 3090 **dense BF16 tensor** peak is **71.16 TFLOPS** (no 2:4 sparsity; large GEMMs measure about 72.3). HBM is 936 GB/s, ridge ≈ **76 FLOP/byte**. Theoretical MFU for an operator is the roofline:

\[
\mathrm{MFU}_{\mathrm{theo}} = \min\bigl(1,\; I / r\bigr),\quad I=\mathrm{FLOPs}/\mathrm{bytes},\quad r=\mathrm{peak}/\mathrm{BW}.
\]

Approaching theoretical MFU means approaching this roofline. It does not mean forcing 40% Kaplan or 100% peak on a seq=128 probe graph. Flash at seq=4096, hd=128 has IO-aware intensity \(I \approx S/4 = 1024\), already compute-bound, theoretical ~100%. At seq=128, \(I=32\), theoretical about 42%, then kernel launch knocks it into the 1% range.

`torch._grouped_mm` **only runs on SM90+**. On Ampere the API exists, the first call raises `RuntimeError`, and a per-layer try/except used to cost 2–3× more than going straight to padded bmm. `grouped_mm_available()` now gates on compute capability, so the 3090 uses bmm.

FlexAttention sliding window is about 1.7 ms on this torch 2.8 / sm_86 stack versus SDPA 0.02 ms: a speed trap. Do not swap in CSA/HCA. This is not a CSA CUDA kernel.

## Layout (activations, not overlay weights)

12B `qk_norm=True`. The old path fused QKV → `view+transpose` to `[B,H,S,D]` then RMSNorm / RoPE. RMSNorm and `torch.cat`-style `rotate_half` smash the **BSHD memory order** Flash wants (stride `(S·H·D, D, H·D, 1)`) into packed BHSD, copying Q/K every layer; `o_proj` then `contiguous().view` copies again.

Now:

- qk_norm + RoPE stay in contiguous `[B,S,H,D]` (last dim is `head_dim`)
- SDPA takes a `transpose` view only, no `.contiguous()`
- `o_proj` uses `reshape`; if Flash wrote the same BSHD memory, this step is a view
- fat-tile sliding window still needs packed BHSD (`unfold` along S) and packs only on the banded path; B0 covering windows do not take this path
- frozen MoE `bmm(x, W.transpose(1,2))` stays **TN** (K stride-1). Do not make `W.T` contiguous into NN
- SwiGLU uses `_silu_mul`: `SiLU(g)` writes into one buffer then `mul_(u)`; do not `silu inplace` on a `gu.chunk` view
- CUDA RMSNorm uses fused `F.rms_norm` (fp16/bf16 input, rstd still fp32) instead of `x.float()` on the whole activation; CPU still uses explicit fp32
- Indexer does one fp32 GEMM `cat(Wq,Wk)` and reads `x` once; both Linear modules remain (overlay / C-index parameter names are unchanged)
- CUDA `linear_cross_entropy` no longer copies chunk×V logits to fp32 (CE softmax accumulates fp32 in-kernel)
- Do **not** fold the three QKV Linears into one fused Linear module (the Hub overlay is still 132 tensors)

Do not re-benchmark or SIGKILL a 3090 B0 process that is already running. The last Flash GQA seq=4096 check at ~85% MFU remains the reference. This is activation layout / operators; it does not change the step **26940** weights. New `_silu_mul` / CUDA RMSNorm / fused indexer / CUDA CE land in git; picking up this round of operator work needs the next sibling resume (do not steal the training GPU just for operators).

## Blocks (theoretical)

Roofline from CPU `python3 -m cat_yoko.ampere_mfu` (versus 71.16 TFLOPS; indexer versus 35.58 TFLOPS FP32/TF32):

| Operator | Phase | bf16-probe theoretical MFU | 12B-shape theoretical MFU | Tuning (approach the roofline) |
| --- | --- | ---: | ---: | --- |
| fused QKV | A–E | 84.2% | 100% compute-bound | concat frozen weights once; no per-step `torch.cat` |
| dense Flash GQA | YOCO cross; window covers seq | 56.1% (launch then knocks it to ~1%) | 100%; measured up to ~87% of peak | Flash + `enable_gqa`, do not repeat KV; qk_norm/RoPE in BSHD |
| masked window | encoder `n_win<seq` | 37.4% (old S×S rows) | B0 `n_win=8192≥seq` uses Flash | **fat tiles 256** (only `seq≥2048`); 256-wide at seq=512 is still 0.14× S×S; CSA union remains S×S |
| CSA union | C-topk | 74.8% | 100% intensity, but the fused mask kernel cannot reach Flash | same; **do not** `enable_gqa+attn_mask` (Ampere silently falls back to the math kernel); repeat KV then Efficient/cuDNN |
| HCA concat | C-hca+ | 78.7% | same as left, longer k | static slot bias cache; do not swap CSA/HCA onto FlexAttention |
| MoE bmm | B2+ | 13.0% (12 tokens per expert, bandwidth-bound) | 100% intensity; uniform experts measure ~84% of peak | Ampere forbids grouped_mm; freeze experts and cache `gate‖up` |
| indexer fp32 | C-index | 38.9% (FP32 peak) | 100% compute-bound | scores stay fp32 (KEEP_HIGH_PREC); one `cat(Wq,Wk)` GEMM, `x` read once |

12B B0 training at `seq=4096`, `n_win=8192`: the encoder window covers the sequence, so the path is dense Flash, not masked rows. The probe deliberately uses `n_win<seq` so Theorem B's compression hole shows. Sliding window uses **256-wide fat tiles** when `n_win<seq` and `seq≥2048`. Cutting 32×64 tiles by `n_win` is a launch trap; 256-wide at seq=512 is still 0.14× S×S, and only seq=4096 / n_win=32 reaches 2.93×. This is not FlexAttention and not a CSA kernel. Short sequences still use one S×S mask. CSA/HCA union / concat still pay the S×S mask.

Continuing B0 on a 3090: **resume the Hub overlay into a separate directory**. Do not overwrite `checkpoints/b0-full` / Hub step **26940** (sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`). Ampere has no FP4 tensor cores, so continue with `--no-nvfp4` (the published C1+NVFP4 wall-clock conclusion is unchanged; this card just runs BF16). The 3090 run is a BF16 sibling (step **33800**, sha256 `2dc31406…`); it is not the published pin. ZeRO-3 only solves fitting in memory.

## Keep the GPU busy (3090 ZeRO-3)

The operators themselves (Flash GQA / fused QKV / MoE TN bmm) are already on the compute wall. Occupancy dropping to **0%** comes from idle gaps between steps, not from slow kernels:

1. **ZeRO prefetch is clipped by the live cap** (default `stage3_max_live_parameters=1e9`) — one frozen MoE layer is ~0.75e9, plus a 0.5e9 prefetch bucket overflows, prefetch silently does not run, and the GPU idles ~1s per step. Now `max_live/reuse=2e9`, prefetch bucket **5e8**. Keep `persistence` at **1e6**: 5e6 pins every 2048×2048 Q/O on-device, and the 3090 OOMs on the first-step `persistent all_gather` at 47.34/47.41 GiB. Experts at 12.6e6 still offload.
2. **CPU Adam is too slow** — ZeRO-3 shards the student into fragments, torch AdamW costs ~1s per step, and nvidia-smi reads 0%. Use **DeepSpeedCPUAdam** (still two decay / no-decay groups; do not `zero_force` them flat). The op needs JIT: the 3090 script puts miniconda `ninja` on PATH and preloads the system `libstdc++` (GLIBCXX_3.4.30). If that install fails, fall back to torch AdamW; the banner still says `adam=ds-cpu`.
3. **Per-step D2H** (nll / DS grad-norm / moe_utilization / `cuda.get_rng_state_all`). `LOG_EVERY=40`; CUDA RNG is snapshotted only on save.
4. **torch inductor 32 workers** steal CPU. `TORCH_COMPILE_DISABLE=1`. `CUDA_DEVICE_MAX_CONNECTIONS=32`. MoE top-k order changes every step, so ZeRO prefetch that follows the last trace misses and SMs idle ~1s. `leaf_module` treats `MoE` / `EncoderBlock` / `DecoderBlock` as whole-block prefetch units. Raising the DeepSpeed inflight H2D cap to 8 made idle flashes denser (1.38/min); keep the default of 2. `LOG_EVERY=40`. Do not persist experts.
5. DummyStream `doc_ids` stay on host; the next-batch H2D overlaps backward. MoE `counts.max()` is hidden on a side stream into the shared expert.
6. **ZeRO warmup**: after wrap and before the timed loop, run one synthetic-batch fwd+bwd (**do not** `engine.step()`; restore RNG). That moves allgather-trace recording and kernel JIT off the first step's 404 tok/s. The long middle stretch at ~647 tok/s is not a cold start; it is prefetch clipped by the live cap plus torch CPU Adam. Warmup cannot recover that stretch; occupancy v2 already pulled it back to ~719.

Do not turn off param offload to chase occupancy (a 49GiB card cannot hold 12B+activations). Do not use FlexAttention / a CSA kernel. `SAVE_EVERY=200` gathers only the 132 trainable tensors; do not use DeepSpeed's full-graph `_zero3_consolidated_16bit_state_dict` (`exclude_frozen` still allgathers the frozen 12B layer by layer, 6s of PCIe idle, SM 0%). The step loop still flashes **about 1s** of SM 0% when frozen-expert H2D does not overlap GEMM (power stays 260–300W). Whole-layer leaf + `LOG_EVERY=40` exist to cut how **often** that flash happens, not to turn PCIe off. Do not turn off param offload. Measured: before leaf ~**1.08/min**; MoE+Block leaf cap=2 about **0.67/min** / 754 tok/s; leaf-only MoE+cap=8 about **0.69/min** / 722 tok/s; Block leaf+cap=8 about **1.38/min** / 763 tok/s (faster but denser; cap=8 dropped). Current recipe (whole-layer leaf, inflight=2, `LOG_EVERY=40`, `gc.freeze`) over 6.5 min: **0.46/min** / ~750 tok/s.

## Sliding-window crossover (3090, 2026-09-20T01:06Z, H=16 hd=128 equal-head)

| seq | n_win | S×S | fat 256 | vs S×S |
| --- | ---: | ---: | ---: | ---: |
| 512 | 32 | 0.07 ms | 0.53 ms | **0.14×** (disabled) |
| 1024 | 32 | 0.19 ms | 0.55 ms | **0.35×** (disabled) |
| 2048 | 32 | 0.63 ms | 0.54 ms | 1.17× (threshold) |
| 4096 | 32 | 2.27 ms | 0.78 ms | **2.93×** |
| 4096 | 256 | 2.27 ms | 1.07 ms | 2.13× |
| 4096 | 8192 covering | 2.08 ms mask | 1.18 ms Flash | covering window uses Flash |

`BANDED_SEQ_MIN=2048`. B0 training windows cover the sequence and do not take this path.

## Measured (3090, 2026-09-19T17:28Z; sliding window remeasured 2026-09-20)

Calibration GEMM **72.12 TFLOPS**. `grouped_mm=False`.

| Operator | Measured | vs roofline |
| --- | --- | --- |
| fused QKV 12B shape | 74.66 T | ~105% (boost; frozen concat-once) |
| Flash GQA seq=4096 | 60.45 T | **85%** |
| MoE uniform bmm E=20 n=409 | 57.66 T | **81%** |
| CSA union seq=512 | 36.73 T | 52% (mask kernel, not Flash) |
| HCA concat seq=512 | 40.34 T | 57% |
| masked window seq=512 | 17.99 T | 25% |
| banded fat-tile seq=512 n_win=32 (disabled) | 2.02 T | 2.8% (0.14× S×S, hence `BANDED_SEQ_MIN=2048`) |
| masked window seq=2048 | 31.78 T | 45% |
| banded fat-tile seq=2048 n_win=32 | 32.50 T | 46% (1.17× S×S) |
| indexer fp32 12B shape | 14.28 T | 50% of FP32 roofline (d_idx=64 skinny K; theoretical table above was recomputed for fused QK, this row is still the old two-GEMM measurement) |
| all bf16-probe seq=128 | &lt;1 T | launch-bound; theoretical 30–84% is unreachable |

Ledger: [`artifacts/autodl-rtx3090/bf16-verify/mfu/`](../artifacts/autodl-rtx3090/bf16-verify/mfu/).

Earlier on the same card, before operator work: large GEMM 72.3T; Flash s=4096 62.9T; fused QKV slower with per-step cat; MoE 2–3× more expensive if it took the grouped_mm try/except path. FlexAttention sliding window ~1.7 ms vs SDPA 0.02 ms; do not use it.

## How to run

```bash
python3 -m cat_yoko.ampere_mfu
bash scripts/run_ampere_mfu.sh
# OUT=/root/autodl-tmp/bf16-verify/mfu/ledger.json

# 3090 same-phase continue (independent SAVE; read-only resume of Hub overlay; do not push-overwrite Hub)
# STEPS=8 bash scripts/run_b0_ampere_3090.sh
```

CPU only prints the theoretical table. Do not pull Ultra-FineWeb. Do not write a full-graph checkpoint. Do not `--save-full`. Do not overwrite Hub `b0-full`.
