# DeepSpeed ZeRO: optional training backend

CI does not require DeepSpeed.

Not Megatron EP/TP, not a CSA kernel, does not pull 50B, does not `--save-full`.
C1 freeze boundaries are unchanged. The Hub overlay is still `{kind, trainable, extra, n_tensors, nbytes}`;
ZeRO-3 must **gather 16-bit trainable weights on all ranks** before save, or the local shard will not match the 132 tensors.

## Why ZeRO first

A single 3090 48GiB card cannot hold the full 12B bf16 weights + Adam. The native
`--offload-encoder` / `--offload-blocks` / `--optim-cpu` paths in this repo are **single-GPU torch**:
the frozen encoder is copied to CPU as a block, and B2 still shuttles layer by layer.

DeepSpeed ZeRO splits work as follows:

| Stage | What it shards | Useful for single-GPU 12B? |
| --- | --- | --- |
| ZeRO-1 | Adam state | only helps trainable params. B0 has ~219M; the frozen 12B **is not sharded** |
| ZeRO-2 | + gradients | same: frozen weights still sit whole on GPU |
| ZeRO-3 + `offload_param` | the parameters themselves, to CPU | **this is the path for the 24.5GiB weights** |

So the 3090 recipe is **ZeRO-3 + optimizer CPU offload + param CPU offload**,
not ZeRO-1/2. ZeRO **can fit in VRAM**; it cannot make 8e9 B0 a reasonable wall-clock
on one 3090 (PCIe offload knocks MFU into the single digits). The Hub overlay is already 1.63% of B0; same-phase resume, do not restart.

## Install

```bash
pip install 'cat-yoko[deepspeed]'   # or pip install deepspeed
python3 -m cat_yoko.train --config 12b --dump-deepspeed --zero 3 --zero-offload-param
```

`--dump-deepspeed` **does not import DeepSpeed**; CPU / CI can still emit JSON.

## 3090 48GiB

```bash
python3 -m cat_yoko.b0 \
  --backend deepspeed --zero 3 --zero-offload --zero-offload-param \
  --device cuda --seq-len 4096 --micro-batch 1 \
  --resume /path/to/b0-full --upcycle-hf /path/to/MiniCPM5-2B-Base
```

When the phase CLI sees `--backend deepspeed` it automatically switches to
`--no-offload-encoder --no-offload-blocks --no-optim-cpu`, so native offload
does not fight DeepSpeed for the same parameters.

Price-equivalent JSON for Ampere / Ada published envelopes that still offload the encoder
includes `recipe.deepspeed_zero.argv`; **the default launch argv stays torch** and is not silently switched to ZeRO.

## Integration contract

1. **`apply_freeze` + NVFP4 wrap first**, then `deepspeed.initialize`.
2. Do not outsource DDP / FSDP. `--backend deepspeed --fsdp` is refused.
3. Turn off native `--offload-encoder` / `--offload-blocks` / `--optim-cpu`.
   Adam still uses this repo's param groups (router no decay), `zero_force_ds_cpu_optimizer=false`.
4. The micro-step loop stays in the trainer. Runtime DeepSpeed `gradient_accumulation_steps=1`;
   loss is still divided by `accum`, same scaling as the torch path. GAS in the JSON dump is a mapping, not the runtime value.
5. Clipping is DeepSpeed `gradient_clipping` (default 1.0). Do not also `clip_grad_norm_`.
6. Overlay: ZeRO-3 all ranks go through `_zero3_consolidated_16bit_state_dict`; rank0 writes only
   `requires_grad` tensors. Do not `--save-full`.
7. `--c1` / `--c1-smoke` cannot re-wrap DeepSpeed in the same process (ZeRO-3 params are already partitioned).
   B0 → B1 → B2 are separate processes with `--resume` overlay.
8. Keep the model's own `--grad-ckpt`. Do not enable DeepSpeed activation checkpointing.
9. Not a Megatron EP/TP loop, not a CSA CUDA kernel, does not pull 50B Ultra-FineWeb.

## Boundary vs torch / Megatron

| Backend | What it can do now |
| --- | --- |
| `--backend torch` | reference graph + DDP/FSDP + native CPU offload |
| `--backend deepspeed` | ZeRO-1/2/3 (including CPU offload). The training loop is this repo's trainer |
| `--backend megatron` | mapping / stub only, **no** training loop |

Single-GPU 32GB still uses `--try` seq=64; `hw_recipe` still refuses the 8e9 envelope for `<40GiB`.
48GiB ≥ 40, so the recipe marks published, but NVFP4 emulation on Ampere is a speed trap; ZeRO only solves fitting.
