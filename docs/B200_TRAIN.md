# Starting B200 training

Target GPU **NVIDIA B200 / SM 10.0** (same family B100/B300 is SM 10.3). Transformer Engine: NVFP4 **training** kernels exist only on SM 10.0 / 10.3. Default `NVFP4BlockScaling()` = 2D weight scaling + RHT on WGRAD + stochastic rounding on gradients. Do not apply 6000D `disable_rht` / `disable_2d` on this GPU.

sm_120 (RTX PRO 6000D) uses `Nvfp4Linear` E2M1/16 emulation; there is no SM100 TMEM/UMMA training kernel.

**The last Vast B200 has been recycled** (2026-09-19, SSH refused). Progress: [`STATUS.md`](STATUS.md). The next machine picks up from the Hub overlay; do not assume the old SSH Host still exists.

## Unknown next GPU

When the SKU is not yet chosen, adapt to **SM / VRAM / TE FPROP**, not Megatron EP/TP. `run_b0_full_b200.sh` exits 4 on anything other than 10.0/10.3. Next-machine entry:

```bash
python3 -m cat_yoko.hw_recipe --json
bash scripts/run_b0_next.sh
```

`cat_yoko.hw_recipe` picks a profile from family+GiB (`sm100_b200` / `sm120_6000d` / `hopper_h100` / `ada_tight` / `cpu`) and emits `cat_yoko.b0` seq / micro-batch / offload / grad-ckpt / `--try`. CPU prints JSON only and does not build 12B. `<40GiB` refuses the 8e9 envelope. Ampere/Ada published envelopes that offload the encoder attach a `deepspeed_zero` hint in JSON (default argv stays torch). The Megatron adapter (`ParallelPlan`, `--dump-megatron`) stays until there is a real multi-GPU node.

The rest of this page is **known B200 / SM100** operators and shortcuts.

## Resume B0 from the Hub overlay

6000D stopped at step **16020**. Nail before B200 recycle:

| Item | Value |
| --- | --- |
| Hub | https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| micro-batch | 2 (`MICRO_BATCH=1` fallback) |
| Logs | [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |

The folder note stays in sync with GitHub [`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md).

```bash
bash scripts/run_b0_next.sh
# known B200:
bash scripts/run_b200.sh
# or step by step:
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

Launch flags: **seq=4096**, `--micro-batch 2`, `--no-offload-encoder`, `--no-grad-ckpt`, `--no-save-full`, DummyStream, do not pull 50B Ultra-FineWeb.

Trainer: MiniCPM5 upcycle → overlay `trainable.pt` → restore `step` / `tokens_in_phase` / stream / RNG. Same phase must not resume the 32-step `--try` under `checkpoints/b0/`.

## Last measured run (B200, recycled)

nvcc 12.9 cubin, **micro-batch=2**: median about **15.7k tok/s** (~520ms/step, 8192 tok/step), Trainer **~138GiB**, nvidia-smi ~141/183GiB. Versus micro-batch=1 at 12.4k tok/s / 96GiB, throughput **+27%** (step time about 1.57×, not 2×). At step **22100**, `tokens_in_phase=90,392,576`, then +8192 per step.

Decoder stays bf16 under C1; encoder is NVFP4 FPROP. Versus B200 BF16 2.25 PFLOPS, the Kaplan B0 ledger is about **14.5% MFU**: encoder forward is only **17%** of B0 FLOPs. Do not also wrap the frozen decoder in NVFP4 — activations would quantize into the bf16 student and move Theorem A.

Hot path: `collapse_doc_ids`, vectorized `pad_packed_counts`, `repeat_interleave(..., output_size=)`, CE chunk cache, one D2H per step for nll/aux, CUDA SDPA prefers Flash/cuDNN (`sdpa_kernel` takes a **list**), frozen MoE skips z-loss.

`CAT_YOKO_TE_NVFP4=0` forces emulation. `FORCE_SM120=1` is required before a B200 launch script may run on sm_120.

## Operators

| Slot | B200 (SM100) | 6000D (sm_120) |
| --- | --- | --- |
| attn QKV/O, cross Q/O, cache KV, lm_head, shared SwiGLU | `TeNvfp4Linear`: one `copy_` into `te.Linear` at wrap, `te.autocast(recipe=NVFP4BlockScaling())` | `Nvfp4Linear` emulation |
| fused QKV / gate+up / cache KV | frozen: one fused `te.Linear`. Trainable: sequential native TE; `cache_k`/`cache_v` one cat GEMM | emulated fused cat |
| MoE routed experts | `te.GroupedLinear`; SM100 default RHT pads expert token count to **64**. Frozen gate+up become one GroupedLinear, and `weight{i}.requires_grad=False`. Do not stack TE master into bf16 `grouped_mm` | permute + `grouped_mm` emulation |
| router / embed / RMSNorm / qk_norm / SDPA | high precision | same |
| Attention topology | causal YOCO window + GQA 16/2/128; CSA not implemented | same |

When the leading dim is not a multiple of 16, **pad zero rows then slice back**; still uses the SM100 kernel. A single TE exception records that launch's shape only and does not disable process-wide NVFP4. On B0/B1 `detach_cache`, the encoder loop is `torch.no_grad()`, so frozen `TeNvfp4Linear` does not build a WGRAD graph.

Master weights remain bf16 Parameters. `state_dict` keys remain `q_proj.weight`.

Measured: torch 2.11.0+cu128, source-built TE 2.19 `@stable` SM100. **FPROP** works. **WGRAD** is `CUBLAS_STATUS_NOT_SUPPORTED` on cuBLAS 12.8.4.1. B0 frozen encoder only needs FPROP.

## TE install

PyPI `transformer-engine-torch` prebuilt `.so` hits `undefined symbol: CUDAErrorLogCapture` on torch 2.11. Must use `scripts/build_te_from_source.sh` (`--no-build-isolation --no-deps`, `NVTE_CUDA_ARCHS=100`).

CUTLASS SM100A `stg.256` needs **nvcc ≥ 12.9**. A CUDA 12.8 core hangs the GPU when quantizing B0 sizes (4096×2048). Install `cuda-nvcc-12-9 libcublas-dev-12-9 cuda-nvtx-12-9`. `MAX_JOBS` defaults to `nproc`. `te_linear_ready` uses **4096×2048**. CUDA 12.9 `lib64` goes on `LD_LIBRARY_PATH`. After the source install, rewrite dist-info, then import from `/tmp`.

## B1 / B2

Wait until the B0 envelope finishes, or until a free GPU can `--try` (do not steal a GPU that B0 still owns):

```bash
bash scripts/run_b1_try_b200.sh
bash scripts/run_b1_full_b200.sh
bash scripts/run_b2_try_b200.sh
bash scripts/run_b2_full_b200.sh
```

On B200, default is **no** CPU offload of encoder / per-layer offload / CPU Adam.

## Disk

Vast base-image container disks are often only **32GiB overlay**. `/workspace` is not necessarily a persistent volume. stop/start keeps the container disk; recycle/destroy wipes it. Do not write a 23GiB `latest.pt`. MiniCPM5 ≈5GiB + torch/TE + overlay must fit in that 32GiB.

Do not commit SSH passwords or deploy keys.
