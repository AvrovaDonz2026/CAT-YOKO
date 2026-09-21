# DeepSpeed ZeRO

Optional training backend. CI does not require DeepSpeed to be installed.

This is not Megatron EP/TP, not a CSA kernel, does not pull 50B, and does not write `--save-full`.
The C1 freeze boundary is unchanged. Hub overlays remain `{kind, trainable, extra, n_tensors, nbytes}`.
Before a ZeRO-3 save, **all ranks must gather** the 16-bit trainable weights; otherwise local shards will not match the 132 tensors.

## Why ZeRO first

A single 3090 48GiB card cannot hold a full 12B bf16 weight set plus Adam. The in-repo native
`--offload-encoder` / `--offload-blocks` / `--optim-cpu` path is **torch single-GPU**:
the frozen encoder is copied to CPU as a block, and B2 also shuttles layer by layer.

DeepSpeed ZeRO splits the work as follows:

| Stage | What it shards | Useful for single-GPU 12B? |
| --- | --- | --- |
| ZeRO-1 | Adam state | Helps trainable params only. B0 has ~219M trainable; the frozen 12B is **not** sharded |
| ZeRO-2 | + gradients | Same: frozen weights still sit in full on the GPU |
| ZeRO-3 + `offload_param` | the parameters themselves, to CPU | **This is the path that fits 24.5GiB of weights** |

So the 3090 recipe is **ZeRO-3 + optimizer CPU offload + param CPU offload**,
not ZeRO-1/2. ZeRO **fits the model in memory**; it does not make 8e9 B0 a reasonable wall-clock
on one 3090 (PCIe offload drives MFU into the single digits). **Prefetch has to actually run**: DeepSpeed defaults `stage3_max_live_parameters=1e9`. One frozen MoE layer is already ~0.75e9, so a 5e8 prefetch bucket is silently dropped by the live cap and the GPU sits idle ~1s per step. Current knobs are **max_live/reuse=2e9**, prefetch **5e8**, `persistence=1e6` (5e6 keeps every 2048×2048 tensor on-device and OOMs the 3090) plus **DeepSpeedCPUAdam** (still two decay groups; needs ninja to JIT). `CUDA_DEVICE_MAX_CONNECTIONS=32`. MoE expert order changes every step, so ZeRO prefetch that follows an old trace misses: `leaf_module` treats `MoE` / `EncoderBlock` / `DecoderBlock` as whole-block prefetch units. DeepSpeed inflight H2D defaults to 2; raising it to 8 made an 8 min sample **1.38/min** (denser idle flashes), so keep the default of 2. `LOG_EVERY=40`, then `gc.freeze()` after warmup. Current recipe over 6.5 min at 1 Hz: **0.46/min** / ~750 tok/s. After wrap, run one **ZeRO warmup** (synthetic batch fwd+bwd, no Adam). The middle stretch at ~647 tok/s is not a cold start. The Hub overlay is already 1.63% of B0; same-phase resume, do not restart the phase. Do not overwrite `b0-full`. The published pin remains Hub `b0-full` step **26940**; the 3090 BF16 run is a sibling.

## Install

```bash
pip install 'cat-yoko[deepspeed]'   # or pip install deepspeed
python3 -m cat_yoko.train --config 12b --dump-deepspeed --zero 3 --zero-offload-param
```

`--dump-deepspeed` **does not import DeepSpeed**, so CPU / CI can still print JSON.

## 3090 48GiB

```bash
python3 -m cat_yoko.b0 \
  --backend deepspeed --zero 3 --zero-offload --zero-offload-param \
  --device cuda --seq-len 4096 --micro-batch 1 \
  --resume /path/to/b0-full --upcycle-hf /path/to/MiniCPM5-2B-Base
```

When the phase CLI sees `--backend deepspeed`, it rewrites to
`--no-offload-encoder --no-offload-blocks --no-optim-cpu` so native
offload does not fight ZeRO for the same parameters.

Price-quote JSON for Ampere / Ada published envelopes that still offload the encoder
includes `recipe.deepspeed_zero.argv`; **default launch argv is still torch** and is not silently switched to ZeRO.

## Integration rules

1. **First** `apply_freeze` + NVFP4 wrap, **then** `deepspeed.initialize`.
2. Do not outsource DDP / FSDP. `--backend deepspeed --fsdp` is refused.
3. Turn off native `--offload-encoder` / `--offload-blocks` / `--optim-cpu`.
   Adam still uses the repo param groups (router does not decay). On CPU offload prefer
   `DeepSpeedCPUAdam` (keep both groups); `zero_force_ds_cpu_optimizer=false`
   so ZeRO does not collapse them into one group.
4. The micro-step loop stays in the trainer. At runtime DeepSpeed `gradient_accumulation_steps=1`;
   loss is still divided by `accum`, the same scaling as the torch path. GAS in the JSON dump is a mapping, not the runtime value.
5. Clipping is DeepSpeed `gradient_clipping` (default 1.0). Do not also call `clip_grad_norm_`.
6. Overlay: ZeRO-3 all-rank `GatheredParameters` on `requires_grad` (B0 132
   tensors). Do not use `_zero3_consolidated_16bit_state_dict` (that path still gathers the frozen
   12B layer by layer, and `SAVE_EVERY` idles ~6s on a 3090). Do not `--save-full`.
7. `--c1` / `--c1-smoke` cannot re-wrap in the same process as DeepSpeed (ZeRO-3 parameters are already partitioned).
   Run B0 → B1 → B2 in separate processes and `--resume` the overlay.
8. Keep the model's own `--grad-ckpt`. Do not enable DeepSpeed activation checkpointing.
9. This is not a Megatron EP/TP loop, not a CSA CUDA kernel, and it does not pull 50B Ultra-FineWeb.
10. Single-GPU `python -m cat_yoko.b0 --backend deepspeed` does not need the deepspeed launcher;
    `wrap_deepspeed` will `setdefault LOCAL_RANK=0`. Do not overwrite the Hub B0 overlay.

## Boundary versus torch / Megatron

| Backend | What it can do now |
| --- | --- |
| `--backend torch` | reference graph + DDP/FSDP + native CPU offload |
| `--backend deepspeed` | ZeRO-1/2/3 (including CPU offload). The training loop is this repo's trainer |
| `--backend megatron` | mapping / stub only; **no** training loop |

Single-GPU 32GB still uses `--try` seq=64; `hw_recipe` still refuses the 8e9 envelope for `<40GiB`.
48GiB is ≥ 40, so the recipe is marked published, but Ampere NVFP4 emulation is a speed trap; ZeRO only solves fitting in memory. See also `docs/DEEPSPEED_ZERO.md`.
