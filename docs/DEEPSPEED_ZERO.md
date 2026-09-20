"""DeepSpeed ZeRO：可选训练后端。CI 不强装 DeepSpeed。

不是 Megatron EP/TP，不是 CSA kernel，不拉 50B，不写 ``--save-full``。
C1 冻结边界不变。Hub overlay 仍是 ``{kind, trainable, extra, n_tensors, nbytes}``；
ZeRO-3 存盘前必须 **全 rank gather** 16-bit 可训练权重，否则本地 shard 对不上 132 张量。

## 为什么先做 ZeRO

单卡 3090 48GiB 塞不下完整 12B bf16 权重 + Adam。仓库里的 native
``--offload-encoder`` / ``--offload-blocks`` / ``--optim-cpu`` 是 **torch 单卡**
路径：冻住的 encoder 整块拷 CPU，B2 还逐层来回搬。

DeepSpeed ZeRO 的分工：

| 档 | 切什么 | 单卡 12B 有没有用 |
| --- | --- | --- |
| ZeRO-1 | Adam 状态 | 只帮可训练参数。B0 只有 ~219M，冻住的 12B **不切** |
| ZeRO-2 | + 梯度 | 同上，冻权重仍整份在 GPU |
| ZeRO-3 + ``offload_param`` | 参数本身切到 CPU | **这才是 24.5GiB 权重的路** |

所以 3090 的配方是 **ZeRO-3 + optimizer CPU offload + param CPU offload**，
不是 ZeRO-1/2。ZeRO **能塞进显存**；它不能让 8e9 B0 在一张 3090 上变成合理墙钟
（PCIe offload 会把 MFU 打到个位数）。**预取必须真的能跑**：DeepSpeed 默认 `stage3_max_live_parameters=1e9`，一层冻结 MoE 已经 ~0.75e9，5e8 预取桶会被 live cap 静默丢掉，GPU 每步空约 1s。现在 **max_live/reuse=2e9**、预取 **5e8**、`persistence=1e6`（5e6 把 2048×2048 全留卡，3090 会 OOM）+ **DeepSpeedCPUAdam**（仍两组 decay；需要 ninja 才能 JIT）。`CUDA_DEVICE_MAX_CONNECTIONS=32`。MoE 专家顺序每步都变，ZeRO 按旧 trace 预取会 miss：`leaf_module` 把 `MoE` / `EncoderBlock` / `DecoderBlock` 当整块 prefetch。wrap 后做一次 **ZeRO 预热**（合成 batch fwd+bwd，不 Adam），把 allgather 轨迹从第一步计时里拿掉。中间 ~647 tok/s 不是冷启动。Hub overlay 已经是 B0 的 1.63%，同阶段 resume，不要重开。不要覆盖 `b0-full`。

## 安装

```bash
pip install 'cat-yoko[deepspeed]'   # 或 pip install deepspeed
python3 -m cat_yoko.train --config 12b --dump-deepspeed --zero 3 --zero-offload-param
```

``--dump-deepspeed`` **不 import DeepSpeed**，CPU / CI 都能打 JSON。

## 3090 48GiB

```bash
python3 -m cat_yoko.b0 \
  --backend deepspeed --zero 3 --zero-offload --zero-offload-param \
  --device cuda --seq-len 4096 --micro-batch 1 \
  --resume /path/to/b0-full --upcycle-hf /path/to/MiniCPM5-2B-Base
```

phase CLI 看到 ``--backend deepspeed`` 会自动改成
``--no-offload-encoder --no-offload-blocks --no-optim-cpu``，避免和 native
offload 抢同一份参数。

等价格的 JSON 里，Ampere / Ada 且还在卸 encoder 的发布信封会带
``recipe.deepspeed_zero.argv``；**默认 launch argv 仍是 torch**，不会偷偷改成 ZeRO。

## 接入约定

1. **先** ``apply_freeze`` + NVFP4 wrap，**再** ``deepspeed.initialize``。
2. 不要外包 DDP / FSDP。``--backend deepspeed --fsdp`` 直接拒。
3. 关掉 native ``--offload-encoder`` / ``--offload-blocks`` / ``--optim-cpu``。
   Adam 仍走仓库的 param groups（router 不 decay）。CPU offload 时优先
   ``DeepSpeedCPUAdam``（两组都留着）；``zero_force_ds_cpu_optimizer=false``
   以免 ZeRO 压成一组。
4. 微步循环仍在 trainer 里。运行时 DeepSpeed ``gradient_accumulation_steps=1``，
   loss 仍除以 ``accum``，和 torch 路径同一套缩放。JSON dump 里的 GAS 是映射，不是运行时值。
5. clip 交给 DeepSpeed ``gradient_clipping``（默认 1.0）。不要再 ``clip_grad_norm_`` 一遍。
6. Overlay：ZeRO-3 全 rank 对 ``requires_grad`` 走 ``GatheredParameters``（B0 132
   张量），不要 ``_zero3_consolidated_16bit_state_dict``（那条仍会按层 gather 冻结
   12B，3090 上 ``SAVE_EVERY`` 会空 6s）。不要 ``--save-full``。
7. ``--c1`` / ``--c1-smoke`` 不能和 DeepSpeed 同进程重包（ZeRO-3 参数已经是 partitioned）。
   B0 → B1 → B2 分开进程，``--resume`` overlay。
8. 模型自己的 ``--grad-ckpt`` 留着。不要开 DeepSpeed activation checkpointing。
9. 不是 Megatron EP/TP loop，不是 CSA CUDA kernel，不拉 50B Ultra-FineWeb。
10. 单卡 ``python -m cat_yoko.b0 --backend deepspeed`` 不需要 deepspeed launcher；
    ``wrap_deepspeed`` 会 ``setdefault LOCAL_RANK=0``。不要覆盖 Hub B0 overlay。

## 和 torch / Megatron 的边界

| 后端 | 现在能做什么 |
| --- | --- |
| ``--backend torch`` | 参考图 + DDP/FSDP + native CPU offload |
| ``--backend deepspeed`` | ZeRO-1/2/3（含 CPU offload）。训练循环就是本仓库 trainer |
| ``--backend megatron`` | 只有 mapping / stub，**没有**训练循环 |

单卡 32GB 继续 ``--try`` seq=64；``hw_recipe`` 对 ``<40GiB`` 仍拒绝 8e9 信封。
48GiB ≥ 40，recipe 会标 published，但 Ampere 上 NVFP4 仿真是速度陷阱；ZeRO 只解决装得下。
