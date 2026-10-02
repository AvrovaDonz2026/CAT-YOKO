# B200 training

Target GPU: **NVIDIA B200 / SM 10.0** (same-family B100/B300 is SM 10.3). Transformer Engine: NVFP4 **training** kernels exist only on SM 10.0 / 10.3. Default `NVFP4BlockScaling()` is 2D weight scaling + RHT on WGRAD + stochastic rounding on gradients. Do not apply the 6000D `disable_rht` / `disable_2d` knobs on this card.

sm_120 (RTX PRO 6000D) uses `Nvfp4Linear` E2M1/16 emulation. There is no SM100 TMEM/UMMA training kernel.

**The last Vast B200 instance was recycled** (2026-09-19, SSH refused). Progress is in [`STATUS.md`](STATUS.md). The next machine should pick up from the Hub overlay. Do not assume the old SSH Host still exists.

## Next GPU is unknown

Until the card is chosen, the adaptation surface is **SM / VRAM / TE FPROP**, not Megatron EP/TP. `run_b0_full_b200.sh` exits 4 on anything other than 10.0/10.3. Next-machine entrypoint:

```bash
python3 -m cat_yoko.hw_recipe --json
bash scripts/run_b0_next.sh
```

`cat_yoko.hw_recipe` picks a profile from family+GiB (`sm100_b200` / `sm120_6000d` / `hopper_h100` / `ada_tight` / `cpu`) and then emits `cat_yoko.b0` seq / micro-batch / offload / grad-ckpt / `--try`. CPU only prints JSON; it does not build 12B. `<40GiB` refuses the 8e9 envelope. Ampere/Ada published envelopes that offload the encoder attach a `deepspeed_zero` hint in the JSON (default argv is still torch). The Megatron adaptation surface (`ParallelPlan`, `--dump-megatron`) stays in place until a real multi-GPU node exists.

The rest of this file is **known B200 / SM100** operators and shortcut scripts.

## Resume B0 from the Hub overlay

6000D stopped at step **16020**. Pin at B200 release:

| Item | Value |
| --- | --- |
| Hub | https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| micro-batch | 2 (`MICRO_BATCH=1` fallback) |
| logs | [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md) |

Folder notes stay in sync with GitHub [`checkpoints/b0-full/README.md`](../checkpoints/b0-full/README.md). This Hub `b0-full` step **26940** overlay is the published pin. The RTX 3090 BF16 run is a sibling, not a replacement.

```bash
bash scripts/run_b0_next.sh
# Known B200:
bash scripts/run_b200.sh
# Or step by step:
bash scripts/upgrade_torch_te_b200.sh
python scripts/download_minicpm5.py --local-dir /workspace/hf/MiniCPM5-2B-Base
python scripts/download_hub_overlay.py --name b0-full --out-dir /workspace/runs/b0-full
bash scripts/run_b0_full_b200.sh
```

Launch args: **seq=4096**, `--micro-batch 2`, `--no-offload-encoder`, `--no-grad-ckpt`, `--no-save-full`, DummyStream. Do not pull 50B Ultra-FineWeb.

Trainer: MiniCPM5 upcycle → overlay `trainable.pt` → restore `step` / `tokens_in_phase` / stream / RNG. Do not resume the 32-step `--try` under `checkpoints/b0/` in the same phase.

## Last measured run (B200, recycled)

nvcc 12.9 cubin, **micro-batch=2**: median about **15.7k tok/s** (~520ms/step, 8192 tok/step), Trainer **~138GiB**, nvidia-smi ~141/183GiB. Versus micro-batch=1 at 12.4k tok/s / 96GiB, throughput is **+27%** (step time about 1.57×, not 2×). At step **22100**, `tokens_in_phase=90,392,576`; every later step adds +8192.

Decoder stays bf16 per C1; encoder uses NVFP4 FPROP. Relative to B200 BF16 2.25 PFLOPS, the Kaplan B0 ledger is about **14.5% MFU**: encoder forward is only **17%** of B0 FLOPs. Do not wrap the frozen decoder in NVFP4 either — activations would quantize into the bf16 student and move Theorem A.

Hot path: `collapse_doc_ids`, vectorized `pad_packed_counts`, `repeat_interleave(..., output_size=)`, CE chunk cache, one D2H per step for nll/aux, CUDA SDPA preferring Flash/cuDNN (`sdpa_kernel` takes a **list**), frozen MoE skips z-loss.

`CAT_YOKO_TE_NVFP4=0` forces emulation. `FORCE_SM120=1` is required before the B200 launch scripts are allowed to run on sm_120.

## Operators

| Slot | B200 (SM100) | 6000D (sm_120) |
| --- | --- | --- |
| attn QKV/O, cross Q/O, cache KV, lm_head, shared SwiGLU | `TeNvfp4Linear`: one `copy_` into `te.Linear` at wrap time, then `te.autocast(recipe=NVFP4BlockScaling())` | `Nvfp4Linear` emulation |
| fused QKV / gate+up / cache KV | Frozen: one fused `te.Linear`. Trainable: sequential native TE; `cache_k`/`cache_v` one cat GEMM | emulated fused cat |
| MoE routed experts | `te.GroupedLinear`; SM100 default RHT pads expert token counts to **64**. Frozen gate+up fuse into one GroupedLinear with `weight{i}.requires_grad=False`. Do not stack TE masters into bf16 `grouped_mm` | permute + `grouped_mm` emulation |
| router / embed / RMSNorm / qk_norm / SDPA | high precision | same as left |
| attention topology | causal YOCO window + GQA 16/2/128; CSA is not implemented | same as left |

When the leading dim is not a multiple of 16, **pad zero rows then slice back**, still using the SM100 kernel. A single TE exception records that one shape; it does not disable NVFP4 for the whole process. On B0/B1 `detach_cache`, the encoder runs entirely under `torch.no_grad()`, so frozen `TeNvfp4Linear` does not build a WGRAD graph.

Master weights remain bf16 Parameters. `state_dict` keys remain `q_proj.weight`.

Measured: torch 2.11.0+cu128, TE 2.19 `@stable` built from source for SM100. **FPROP** works. **WGRAD** on cuBLAS 12.8.4.1 returns `CUBLAS_STATUS_NOT_SUPPORTED`. B0 freezes the encoder and only needs FPROP.

## TE install

PyPI `transformer-engine-torch` prebuilt `.so` hits `undefined symbol: CUDAErrorLogCapture` on torch 2.11. Must use `scripts/build_te_from_source.sh` (`--no-build-isolation --no-deps`, `NVTE_CUDA_ARCHS=100`).

CUTLASS SM100A `stg.256` needs **nvcc ≥ 12.9**. A CUDA 12.8-built core hangs the GPU when quantizing at B0 size (4096×2048). Install `cuda-nvcc-12-9 libcublas-dev-12-9 cuda-nvtx-12-9`. `MAX_JOBS` defaults to `nproc`. `te_linear_ready` uses **4096×2048**. Put CUDA 12.9 `lib64` on `LD_LIBRARY_PATH`. After the source install, rewrite dist-info first, then import from `/tmp`.

## B1 / B2

Wait until the B0 envelope finishes, or use a free GPU for `--try` (do not contend with a B0 job that already owns the card):

```bash
bash scripts/run_b1_try_b200.sh
bash scripts/run_b1_full_b200.sh
bash scripts/run_b2_try_b200.sh
bash scripts/run_b2_full_b200.sh
```

On B200, the default is **no** CPU offload of the Encoder / per-layer offload / CPU Adam.

## Disk

Vast base-image container disks often have only a **32GiB overlay**. `/workspace` is not necessarily a persistent volume. stop/start keeps the container disk; recycle/destroy wipes it. Do not write a 23GiB `latest.pt`. MiniCPM5 ≈5GiB + torch/TE + overlay must fit in those 32GiB.

Do not write SSH passwords or deploy keys into the repo.
