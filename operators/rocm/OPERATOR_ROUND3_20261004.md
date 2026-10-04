# ROCm operator round 3 — document backward and CPU Adam transfers

This report describes the independent, opt-in candidates before adoption, when
the production source was `source-operators-round2-20261003T0903`. The later
[checkpoint-first adoption](OPERATOR_SWITCH_20261004.md) records the additional
full-model, real-update and recovery checks and the switch to both operators.
The numerical microchecks and timings below keep their operator-only scope.

Hardware: RX 7900 XTX, gfx1100, PyTorch 2.9.1+ROCm 6.4, BF16 parameters. The
isolated experiment source is `source-split-20261004T0310`. Raw evidence is in
[`results/rx7900xtx-20261004-round3`](results/rx7900xtx-20261004-round3).

## Attention: replace independent slices with one split

[`split_attention.py`](split_attention.py) partitions each FP32 Q/K/V tensor with
one `split(lengths, dim=-2)` instead of separately slicing each document. The
split backward concatenates disjoint gradients once. FP32 casts, GQA repetition,
MATH causal SDPA, document order, RoPE positions and output concatenations retain
the packed implementation's arithmetic. Only batch one with at least two valid,
contiguous document runs uses the candidate; other cases retain the existing
path. Importing the module installs nothing.

CPU validation passed 22 tests: nine new split tests and thirteen packed
regressions. They include an independent explicit dense-mask oracle, all input
gradients, FP32/BF16, document isolation, repeated-ID fallback, mutated plans,
CrossAttention parameter gradients and context restoration.

The GPU microcheck passed **14/14** comparisons with finite, **byte-exact output,
dQ, dK and dV**. Shapes are B=1, S=4096, Hq=16, Hkv=2, D=128, window=8192. QKV
values and upstream gradients are synthetic and identical between methods;
document boundaries come from five real corpus rows and two synthetic rows.
Q/K use dense BSHD storage after normalization/RoPE; V retains the fused QKV or
KV-cache row stride. Timing includes casts, GQA, fragment assembly, merge_heads
and complete input backward, with two warmups and seven repetitions per segment.

Each comparison brackets the candidate with two packed baselines. The following
numbers use the faster baseline median; positive values mean lower wall latency.
Cold measurements clear the document-plan cache before every invocation.

| Real row | Documents | Warm reduction, self / cross | Cold reduction, self / cross |
| --- | ---: | ---: | ---: |
| 8701 | 20 | -0.75% / -7.51% | 13.97% / 13.68% |
| 10222 | 8 | 9.38% / 9.24% | 11.25% / 6.74% |
| 14000 | 9 | 7.68% / 7.77% | 7.95% / 8.48% |
| 16000 | 4 | 3.15% / 2.72% | 2.37% / 2.16% |
| 19530 | 8 | 6.54% / 6.75% | 7.62% / 7.37% |

The sample median latency reduction is 6.64% warm and 7.78% cold. These are
unweighted operator observations, not corpus-weighted or full-training results.
Row 8701 has substantial warm-baseline drift: self 16.12→23.53 ms and cross
18.81→23.55 ms. Its warm measurements do not establish a stable regression or
gain. Uniform 64-token documents reduce latency by about 13–15%; uniform
1024-token documents by about 3%.

Autograd confirms the intended structural change: the baseline has three
`SliceBackward0` nodes per document, and the candidate has three
`SplitWithSizesBackward0` nodes in total. For 64 documents, this is 192 slices
versus three splits. This confirms the mechanism, not a whole-model speedup.

The first preflight required 6 GiB free and exited before sending any signal.
The live process retained GPU allocator storage, leaving about 3.17 GiB free.
The successful attempt conservatively filtered fixtures to 80% of available
memory using four FP32 score buffers plus 256 MiB for QKV/gradient assembly.
Real rows 12000, 17478 and 18000, and the synthetic single 4096-token document,
were skipped. There were no OOMs. The skipped single-document mode is covered by
CPU fallback tests, not by this GPU ledger.

Before measurement, step **52446** was hardlinked to a separate experiment path
and checked for all 132 native weights, 264 finite CPU FP32 moments, source Adam
counter **17644**, and packed cursor **17644**. SHA256:
`1d3044867075c0359555e381d12b218d59bb30c460178c42cc80cf6d7bf38ade`.
The GPU probe took 37.06 seconds. Its identity-checked controller resumed the
same worker, and a later observation recorded completed step **52498**, state R,
with only that worker owning the GPU. Controller code and receipts are retained.

## Adam: BF16 CPU shadow and direct writeback

[`cpu_adam_cached.py`](cpu_adam_cached.py) caches only a BF16 CPU parameter copy.
Every step starts from the latest BF16 value, converts freshly to FP32, executes
the original CPU Adam primitives in the same order, rounds back to BF16, and
copies directly to the GPU Parameter. It removes steady-state parameter D2H
and an extra GPU writeback temporary; it does not move Adam arithmetic to GPU
or keep a persistent FP32 master. The 219,212,288-parameter B0 cache is about
418 MiB of additional host memory.

Parameter version/storage changes and optimizer loads invalidate the cache.
Unversioned writes through `.data` or independent storage aliases require
explicit `installation.invalidate(parameter)`. The native per-block
`step_params` path is retained and invalidates affected cache entries. Parameter
identity, state IDs, decay groups and checkpoint format remain native.

Eight CPU checks passed, including seven arithmetic/lifecycle tests and source
syntax validation. A separate CPU smoke check passed the benchmark entry's
five-step parity, reset/bracketing and export checks.

The GPU experiment resumed the real **step 52493** checkpoint, SHA256
`7cff34b1ec782d41144d282088d825ce3eb5e77bbd74140fa10cb20381263349`.
All **132** parameters passed five consecutive updates with **byte-exact BF16
weights and both FP32 moments**, plus matching source counter increments.
The two timing terminal comparisons also passed, Parameter identities were
preserved, and all 264 exported moments remained finite CPU FP32. Source and
implementation hashes were checked again before publishing `complete`.
Gradients are fixed seeded BF16 fixtures, not gradients captured from training.

The synchronized optimizer-only B/C/B comparison resets every segment to the
same source weights and Adam state and applies the same five gradients:

| Segment | First update, ms | Updates 2–5 median, ms |
| --- | ---: | ---: |
| Native before | 1140.59 | 1134.48 |
| Cached CPU Adam | 1155.96 (cache build) | 1009.41 |
| Native after | 1157.04 | 1156.05 |

Against the faster native steady baseline, update latency decreases **11.02%**,
or about **125 ms per optimizer update** (1.124× optimizer-only throughput).
Timing includes D2H, native CPU math, direct H2D and completion synchronization;
fixture generation, comparisons, state resets and model forward/backward are
excluded. The first cache build is not counted as steady-state acceleration.
The private cache contains 132 BF16 CPU tensors / 438,424,576 bytes. Across
parity and timing, eight full cached updates avoided about 3.27 GiB of parameter
readback. Peak allocated GPU memory during timed updates was 1672.51 MiB;
gradient/template allocations already present at timer start are included in
that peak, not in the reported incremental delta.

This is an optimizer microbenchmark, **not an 11% training speedup**. The
controller took 170.88 seconds and resumed the same worker. A later observation
verified its full UID/start-time/command identity and completed step **52543**,
with only that worker owning the GPU. A new periodic checkpoint at **52506**
was already present, 203 seconds old at observation. The production command
continues to save every 300 seconds and retain the latest three checkpoints.

The archived controllers record the executed protocol; they are not general
launchers. A microprocess command snapshot can be empty during exec (as in the
Adam receipt), so reusing its watchdog cleanup logic requires waiting for a
stable nonempty command before recording that child's identity. Training-worker
identity was complete, and its automatic resume protection was independent of
the microprocess snapshot. Both actual probes finished normally and have
separate evidence of subsequent training updates.

## Acceptance required after these microbenchmarks

The next acceptance step was to capture native deterministic
reference repeats from the same fixed checkpoint, validate all 132 model
gradients without relaxing existing thresholds, and bracket synchronized full
training updates on the same data window, plus checkpoint export and resume
checks and the long-document shapes skipped here. The existing conservative
automatic selection gate was 5% whole-update improvement. The later
[adoption report](OPERATOR_SWITCH_20261004.md) records passed mathematical and
recovery checks and the user's explicit switch with a measured 2.55–3.03% gain,
below that automatic gate. B1 changes the frozen expert and memory assumptions
and requires separate validation.
