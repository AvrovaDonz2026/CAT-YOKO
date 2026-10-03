# Packed attention and gradient clipping candidates

These operators are isolated, opt-in experiments. The ongoing 24-hour B0 run
continues with its existing source and kernels. No GPU performance result has
yet been recorded for these candidates.

## Actual checkpoint configuration

The step-41300 checkpoint has sequence length 4096, **window 8192**, 16 query
heads, 2 KV heads, and head dimension 128. A smaller window would change the
model. Short-window tiling alone cannot accelerate this source configuration.

`packed_attention.py` instead factors packed attention into contiguous document
fragments, using FP32 causal MATH attention inside each fragment. It preserves
the original projections, q/k normalization, already-applied RoPE positions,
GQA, output cast, and output projection. Both encoder self-attention and decoder
cross-attention are covered. A single long document still requires quadratic
attention, and many short fragments introduce more launches.

Window widths smaller than the sequence also have a document-aware tiled path.
Extra attention bias, requested attention probabilities, unsupported shapes,
and noncontiguous reappearance of one document ID use the native fallback.
The document plan cache uses weak tensor identity and the mutation version;
inference tensors without version counters are not cached. Importing the module
does not install either patch. Context exit restores both patched methods.

`grad_norm.py` batches scalar norm readback per device. It retains each native
gradient norm's dtype/rounding, Python double-precision summation in parameter
order, and the original clipping coefficient and gradient multiplication.
It does not fuse norm kernels, alter Adam, or introduce FP32 master weights.

## Correctness and measurement

CPU checks cover FP32 and BF16 outputs and all Q/K/V gradients, packed layouts,
batch sizes, tile/document boundaries, dense document fragments, CrossAttention
input and projection gradients, no cross-document/future leakage, mutation
cache invalidation, and exception/nested context restoration. Gradient clipping
checks compare return values and gradient bytes against native, including
clipped/unclipped, mixed dtypes, empty and nonfinite cases.

`candidate_bench.py` captures the native reference **before** installing any
candidate. Packed parity searches at most 32 real 4096-token rows in its own
stream for a multi-document batch. It does not advance the later training
stream. The full-model comparison keeps the existing loss/output and all 132
per-tensor/global gradient gates, deterministic kernels, and native repeat.
No experimental updates are allowed if packed attention was not exercised.
`operators.json` records source hashes and actual optimized/fallback calls.

Passing CPU or operator checks does not prove full-model numerical equivalence.
Changes to matrix shapes can change BF16 results. Only the GPU full-model gate
allows independent bounded updates; it does not automatically replace the
production run or establish long-term convergence.

`profile_training.py` records four to eight complete updates after two real
warmups, restoring the source CPU FP32 Adam. It labels forward, vocabulary head
and CE, backward, clipping, Adam, data, checkpoint saving, and evaluation.
Steady fractions exclude saving/evaluation; CPU self time and summed linked GPU
work are separate quantities. Copy statistics overlap phase totals and include
dtype conversions. Profiler wall time includes instrumentation overhead and
is not the primary throughput comparison.

## Deferred GPU queue

`run_operator_queue.py` waits for the existing training supervisor to report
successful completion and observes KFD idleness. It never signals other GPU
jobs. It copies the final checkpoint into its own directory and verifies CPU
FP32 Adam and at least 20 remaining real rows before measurements. It refuses
to reset or silently wrap the source cursor.

The queue runs, sequentially:

1. Packed attention at windows 8192 and 32 in packed-QKV and cross-cache layouts.
2. Native versus batched clipping using all 132 actual checkpoint shapes.
3. A native shared-storage 20-update throughput run and a six-update profile.
4. A separate 20-update candidate run, with full parity before its first update.
5. A candidate profile only if that bounded run succeeds.

Only operators passing their GPU checks are included in the candidate run.
Both throughput runs resume the same immutable checkpoint, Adam and data cursor;
the first five updates are discarded. Reports retain rejected candidates.
Idle observations do not reserve the GPU, so PID evidence and traces still
need review before attributing small timing differences.

```bash
python operators/rocm/run_operator_queue.py \
  --training-run /path/to/b0-realtext-24h-20261002 \
  --source-dir /path/to/isolated-source \
  --base /path/to/MiniCPM5-2B-Base \
  --data /path/to/train.bin --eval-data /path/to/eval.bin \
  --out /path/to/new-operator-queue --python /path/to/rocm-python
```

Status is written to `status.json` while waiting and during each child stage.
Logs, operator ledgers, parity reports, profile traces, independent checkpoints,
and the final throughput comparison all stay beneath the queue output directory.
