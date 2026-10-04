# ROCm operator experiments

[The October 4 operator adoption](OPERATOR_SWITCH_20261004.md) switches the B0
continuation to document-split FP32 attention and cached BF16 CPU-shadow Adam,
after full-model gradient gates, byte-exact deterministic real updates and
native checkpoint recovery checks. The short whole-update comparison gains
2.55–3.03%; the user requested adoption below the 5% automatic selection gate.
[The preceding microbenchmarks](OPERATOR_ROUND3_20261004.md) retain their narrower
operator-only scope.

[Real-text B0 continuation](REAL_TRAINING.md) describes the bounded corpus,
held-out evaluation, optimizer-state recovery, and 24-hour training supervisor.

[Packed attention and clipping candidates](OPTIMIZATION_20261003.md) describes
the opt-in operators, full-step profiling, and the original deferred queue.
[The October 3 checkpoint-first evaluation](OPERATOR_SWITCH_20261003.md) records
their completed GPU checks and real-text training comparisons.

Standalone correctness and timing experiments for CAT-YOKO's 12B shapes,
measured on 2026-10-02 with a Radeon RX 7900 XTX, 24 GiB VRAM, and PyTorch
2.9.1+ROCm 6.4. The B0 continuation now uses the validated shared-storage layout
through the isolated runner; the native model remains the reference.

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
The baseline process was stopped after the completed save to release host
memory. Nine compact-model tiny tests passed, but full 12B sequence-4096
compact parity **failed**: forward+backward measured **34.986 → 4.189 seconds**
(8.35×), loss absolute error was **5.72e-5**, and aggregate gradient relative L2
was **2.95%**. The worst per-tensor gradient relative L2 was **15.69%**, and
**65 of 132 tensors exceeded the fixed 5% gate**. The candidate made **zero
optimizer updates** and was not adopted for continuation. The timed arithmetic
comparison is not a validated training speedup.

`compact_model.py` replaces only frozen, byte-identical routed B0 experts and
retains the native **132 trainable overlay keys** and parameter identities.
BF16 reductions remain approximate. It changes frozen full-state keys; rebuild
the native model before applying an overlay or entering B1/B2. Compact B1/B2 and
full compact-model checkpoint publication are unsupported.

`shared_storage_moe.py` shares byte-identical frozen expert weights while
retaining native MoE dispatch, batched GEMMs, gate multiplication, and reduction
order. Logical experts alias one master expert and packed weights broadcast
with zero expert stride; this preserves all native full-state and overlay keys.

The first ordinary-mode comparison failed the gradient gate: at sequence 4096,
34.63 → 6.39 seconds, aggregate gradient relative L2 3.42%, and 43 of 132 tensors
above 5%. It performed no updates. An isolated native BF16 `index_add_` probe
showed that repeated fixed inputs already vary under the reference's ordinary
atomic reduction policy. The standalone probe observed maximum repeat relative
L2 **0.284%** for identical expert outputs and **0.358%** for independent outputs;
deterministic mode repeated exactly in both cases. Its timings ran alongside
this session's continuation and are not operator speed comparisons.
[Original comparison](results/rx7900xtx-20261002/shared_storage_parity.json),
[standalone probe](results/rx7900xtx-20261002/index_add_probe.json), and
[earlier ad hoc probe](results/rx7900xtx-20261002/index_add_adhoc.json) are retained.

Strict deterministic full-model checks then **passed at sequences 64, 256, and
4096**: loss, sampled hidden states, selected logits, and all **132 trainable
gradients had exactly zero error**. The native 4096-token reference repeated
exactly too. No thresholds were relaxed. At 4096, native block-offload
forward+backward took **31.310 → 6.721 seconds**, a **4.66×** speedup for the shared,
resident layout. Parameter storage dropped from **24,500,762,624 to
4,418,435,072 bytes** (22.82 → 4.11 GiB), allowing the model to stay on the GPU.
This measures the layout plus removal of block offload, rather than an isolated
kernel improvement. [Full parity report](results/rx7900xtx-20261002/shared_storage_deterministic.json).

After validation, the runner restored the original deterministic-policy setting
(**false**) and continued from step **33850**, targeting **34802**, with resident
BF16 compute and CPU FP32 Adam. A recorded 20-step window reached a median
**584.0 tokens/s**, versus the native baseline's **131.3 tokens/s**, or **4.45×**
actual training throughput. Peak GPU allocation was **12,006.71 MiB**. The saved
step **33950** confirms optimizer updates and overlay saves; this is an ongoing
bounded run. [Metrics and status](../../artifacts/rocm-rx7900xtx/README.md).
Compute collapse remains unvalidated; shared-storage success does not promote
that separate implementation.

`model_bench.py` compares the same source overlay and batches without optimizer
updates or clipping: loss, selected hidden/logit outputs, every trainable
gradient, and aggregate gradient. Default declared gates are loss absolute
error at most 0.02 and output/per-tensor/global gradient relative L2 at most 5%.
It starts bounded training only after all requested parity checks pass, unless
`--parity-only` is used. Resident training keeps CPU Adam and uses global
clipping; the baseline block-offload path used per-block clipping. Source
overlays contain no optimizer state, so Adam moments restart.

`--moe-layout compact` is the default; `--moe-layout shared-storage` selects the
storage-preserving installer. Reports record the layout, available Git commit,
and benchmark/installer/native-MoE SHA256 hashes. Both layouts must retain the
same 132 trainable gradients and pass every requested sequence, including each
per-tensor 5% check, before bounded training (default 10 updates) can begin.

`--deterministic-parity` temporarily requires strict deterministic kernels,
restoring both enabled and warn-only settings in `finally` before training.
`--reference-repeat` also checks a repeated native 4096-token batch. Together,
these flags avoid mistaking reference atomic variation for a layout error.
Capture cleanup clears pending MoE statistics so parity batches cannot enter
subsequent utilization logs; frozen B0 router bias is never updated.

`run_compact_b0.py` installs compute collapse and assumes prior validation; use
`model_bench.py --moe-layout shared-storage` for the validated shared layout.
`gpu_wait.py` reads other registered KFD PIDs and requires two consecutive idle
observations. Both launchers expose `--wait-gpu-idle` and optional
`--gpu-idle-max-wait` seconds. It does not signal other processes or reserve the
GPU after the check.

## Files and reproduction

- [Round-three checkpoint-first adoption](OPERATOR_SWITCH_20261004.md): real-update arithmetic, synchronized old/new/old training comparison, and the active split-attention/cached-Adam continuation. `round3_candidate_bench.py` supplies the isolated entry; `run_round3_switch.py` supervises its acceptance and native recovery checks.
- [Checkpoint-first real-text operator evaluation](OPERATOR_SWITCH_20261003.md): packed attention, batched gradient clipping, full-model parity, and measured training comparisons.
- `run_operator_switch.py`: compare each candidate from one verified checkpoint, bracket trials with native baselines, and optionally continue with a validated winner. `--evaluate-only` writes the decision without starting continuation; `--preflight-dir` and `--baseline-run` can reuse validated results. The script never stops an existing GPU process.
- `frozen_moe.py`: byte-equality guards, normalized gates, native MoE output/input-gradient comparisons, and real base-weight slices.
- `attention_bench.py`: FP32 SDPA baseline plus an independent causal formula; Flash, repeated KV, gradient layouts, and checkpointed math chunks.
- `attention_layout.py`: Efficient attention, grouped-query FP32 bmm, copy-inclusive layouts, and Flash-forward/FP32-backward hybrids.
- `probe_index_add.py`: fixed native BF16 reductions under ordinary/deterministic policies, with identical and independent expert outputs.
- `projection_bench.py`: real QKV and shared gate/up numerical checks plus pageable/pinned transfers.
- `run_benchmarks.py`: temporarily stops only an authorized `run_b0_rocm.py` process and resumes it after completion, timeout, or error.
- `compact_model.py`, `shared_storage_moe.py`, `model_bench.py`, `run_compact_b0.py`, and `gpu_wait.py`: isolated B0 layouts, full-model parity, bounded continuation, and read-only availability waiting.
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
  --out /path/to/shared-storage-parity --moe-layout shared-storage \
  --parity-seqs 64,256,4096 --deterministic-parity --reference-repeat \
  --parity-only --wait-gpu-idle
```

To validate and continue the shared layout from the saved step 33850:

```bash
python operators/rocm/model_bench.py \
  --base /path/to/MiniCPM5-2B-Base \
  --resume /path/to/b0-layout-source/trainable_step_33850.pt \
  --out /path/to/shared-storage-continuation --moe-layout shared-storage \
  --parity-seqs 64,256,4096 --deterministic-parity --reference-repeat \
  --seq-len 4096 --run-steps 952 --save-every 10 --wait-gpu-idle

python operators/rocm/probe_index_add.py --device cuda --repeats 12 \
  --json /path/to/operator-results/index_add_probe.json
```

The October 2 runs above use DummyStream. Actual continuation throughput is
measured above; language-quality improvement has not been evaluated. CPU Adam
moments restart because the source overlay has no optimizer state. Resident
training uses global clipping and the baseline uses per-block clipping; recorded
gradient norms remain below the clipping threshold of 1.
