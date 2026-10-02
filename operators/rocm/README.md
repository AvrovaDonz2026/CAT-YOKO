# ROCm operator experiments

Standalone correctness and timing experiments for CAT-YOKO's 12B shapes,
measured on 2026-10-02 with a Radeon RX 7900 XTX, 24 GiB VRAM, and PyTorch
2.9.1+ROCm 6.4. Production training uses the verified baseline implementation.

The runner temporarily stopped this session's B0 process and resumed the same
process in `finally`; its allocations remained resident. This is a shared GPU,
and the experiment did not record every other GPU PID throughout the
measurements. These timings are observations, not guaranteed exclusive-GPU
measurements. Small differences, including the roughly 8% attention gain in one
layout, do not justify choosing a training implementation.

## Initial operator measurements

| Candidate | Baseline → candidate | Correctness and scope |
| --- | --- | --- |
| Merge identical frozen MoE experts; real MiniCPM5 layer 16, BF16, 4096 tokens, top-k 10 | Forward 23.29 → 4.17 ms, 5.58×; forward + input backward 46.89 → 7.31 ms, 6.42× | Output relative L2 0.248%; input-gradient relative L2 0.313%. Requires frozen, byte-identical experts. CPU offload transfers are excluded. |
| Query-chunked FP32 causal GQA; sequence 4096, chunk 512 | Forward approximately 17 → 12.3 ms; forward + backward approximately 35–36 → 37 ms; incremental peak 4208 → 668 MiB | Output/gradient relative L2 below 0.007%. Primarily a memory tradeoff. |
| Frozen QKV prepacking and pinned GPU buffer | Transfer + forward + input backward 4.31 → 4.13 ms, approximately 4% | Transferred weights are byte-identical; numerical results match the cached fused-projection baseline. |
| Shared gate/up prepacking and pinned GPU buffer | Transfer + forward + input backward 6.67 → 6.53 ms, approximately 2% | Same numerical checks as QKV. |
| Experimental AOTriton Flash BF16 causal GQA | Rejected for native backward | dQ relative L2 reaches approximately 55.8%. GQA, repeated KV, a contiguous-gradient hook, and different input layouts fail the gradient gate. |

The MoE result is for one GPU operator, not full-model throughput. Merging
identical experts changes BF16 reduction order and is numerically approximate,
not bitwise equivalent. Fixed relative-L2 limits are 1.5% for BF16 and `1e-4`
for FP32; ledgers also retain maximum absolute errors. The real weight test
uses a fixed random router because the B0 overlay does not contain frozen
router weights. Normalized gates, input gradients, and native MoE dispatch are
included. Diverged or trainable experts are unsupported.

Production already fuses QKV and gate/up. A comparison against separate GEMMs
does not measure the improvement available over current training. Observed
one-way transfers were approximately 3.3–3.5 GB/s; the device reported PCIe 4.0
x16. Transfer-inclusive prepacking differences were only 2–4%.

## Attention layout measurements

[`attention_layout.jsonl`](results/rx7900xtx-20261002/attention_layout.jsonl)
contains **72 rows: 45 passed, 18 numerical failures, 9 unsupported, and no
execution errors**. It covers BF16 sequences 128, 512, and 4096 in contiguous
BHSD, transposed BSHD, and mixed cross-cache layouts. In the cross-cache case,
Q/K have dense storage after normalization/RoPE and V retains the fused cache
projection's row stride. Candidates use identical input and upstream-gradient
values, with both contiguous and output-stride gradients.

The following are GPU-event medians for batch 1, 16 Q heads, 2 KV heads,
head dimension 128, sequence 4096, and the actual cross-cache layout. Copying,
KV repetition, and backward recomputation are included where applicable.
Incremental memory excludes allocations resident before the measurement.

| Candidate | Forward, ms | Forward + backward, ms | Incremental peak, MiB |
| --- | ---: | ---: | ---: |
| FP32 MATH baseline | 19.29 | 39.40 | 4208 |
| Grouped-query FP32 bmm | 18.26 | 56.88 | 4168 |
| Flash forward + FP32 MATH backward | 5.78 | 54.19 | 4224 |
| Same hybrid with contiguous input copies | 5.67 | 51.70 | 4224 |
| Flash forward + query-chunked backward | 5.33 | 71.60 | 684 |

Efficient attention does not support native GQA in this runtime. Repeating KV
allows it to run, but dQ relative L2 is **31%–56%** across the tested shapes.
Contiguous input copies and upstream gradients do not correct it. These
candidates are rejected.

Hybrid Flash forward passes with approximately 0.21% output relative-L2 error
at sequence 4096. Full FP32 recomputation gives reference-identical gradients;
chunked backward stays below 0.007% relative L2. Its backward is the FP32
reference derivative, rather than the exact derivative of the rounded Flash
result. It remains an experimental surrogate. The cross-cache training
operator is slower than the baseline; chunking reduces incremental peak memory
by approximately 84%. Faster forward-only measurements do not establish a
full-graph or continuation speedup.

## Full-model experiment status

The baseline B0 run has saved **step 33850**, or **158,345,216 tokens in phase**.
It is paused while another GPU job occupies approximately 22 GiB. Nine
compact-model tiny tests passed. Full 12B parity is waiting for GPU availability;
there is no completed full-graph parity result or compact continuation speedup
claim.

`compact_model.py` replaces only frozen, byte-identical routed B0 experts and
retains the native **132 trainable overlay keys** and parameter identities.
BF16 reductions remain approximate. It changes frozen full-state keys; rebuild
the native model before applying an overlay or entering B1/B2. Compact B1/B2 and
full compact-model checkpoint publication are unsupported.

`model_bench.py` compares the same source overlay and batches without optimizer
updates or clipping: loss, selected hidden/logit outputs, every trainable
gradient, and aggregate gradient. Default declared gates are loss absolute
error at most 0.02 and output/per-tensor/global gradient relative L2 at most 5%.
It starts bounded training only after all requested parity checks pass, unless
`--parity-only` is used. Resident training keeps CPU Adam and uses global
clipping; the baseline block-offload path used per-block clipping. Source
overlays contain no optimizer state, so Adam moments restart.

`run_compact_b0.py` is the separate bounded continuation launcher and assumes
full-model parity has already passed. `gpu_wait.py` reads other registered KFD
PIDs and requires two consecutive idle observations. Both launchers expose
`--wait-gpu-idle` and optional `--gpu-idle-max-wait` seconds. It does not signal
other processes or reserve the GPU after the check.

## Files and reproduction

- `frozen_moe.py`: byte-equality guards, normalized gates, native MoE output/input-gradient comparisons, and real base-weight slices.
- `attention_bench.py`: FP32 SDPA baseline plus an independent causal formula; Flash, repeated KV, gradient layouts, and checkpointed math chunks.
- `attention_layout.py`: Efficient attention, grouped-query FP32 bmm, copy-inclusive layouts, and Flash-forward/FP32-backward hybrids.
- `projection_bench.py`: real QKV and shared gate/up numerical checks plus pageable/pinned transfers.
- `run_benchmarks.py`: temporarily stops only an authorized `run_b0_rocm.py` process and resumes it after completion, timeout, or error.
- `compact_model.py`, `model_bench.py`, `run_compact_b0.py`, and `gpu_wait.py`: isolated compact B0 implementation, full-model parity, bounded continuation, and read-only availability waiting.
- [Result files](results/rx7900xtx-20261002/): JSON/JSONL ledgers include failed candidates and runner exit codes. The initial projection attempt exited 1; its retry exited 0.

```bash
python operators/rocm/frozen_moe.py \
  --device cuda --base /path/to/MiniCPM5-2B-Base --layer 16 \
  --modes bf16 --topk 10 --tokens 4096 --warmup 2 --iterations 5 \
  --json /path/to/operator-results/frozen_moe_actual.json

TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
python operators/rocm/attention_layout.py \
  --seq-lens 128 512 4096 --dtypes bf16 --layouts bhsd bshd cross_cache \
  --chunk-size 512 --warmup 2 --repeats 3 \
  --output /path/to/operator-results/attention_layout.jsonl

python operators/rocm/model_bench.py \
  --base /path/to/MiniCPM5-2B-Base --resume /path/to/b0/trainable.pt \
  --out /path/to/compact-parity --parity-seqs 64,256,4096 \
  --parity-only --wait-gpu-idle
```

After full-model parity passes, an independent bounded run can use:

```bash
python operators/rocm/run_compact_b0.py \
  --base /path/to/MiniCPM5-2B-Base --resume /path/to/b0/trainable.pt \
  --save-dir /path/to/compact-b0 --run-steps 10 --seq-len 4096 \
  --save-every 5 --wait-gpu-idle
```

All runs remain DummyStream experiments. Operator timing and tiny tests do not
establish language-quality improvement or full-model training throughput.
