# Second operator evaluation on real-text B0 — October 3, 2026

**Retain the existing packed-attention and shared-storage MoE configuration.**
The new bucketed attention passed the numerical gates but reduced median
complete-update throughput by **1.37%** against the stronger bracketing baseline.
GPU FP32 Adam failed BF16 byte equality and was rejected before any trial
training. Short-document batching still has a useful isolated crossover to
investigate, but this round establishes no additional whole-training speedup.
The controller finished all three comparisons and a corrected GPU profile,
then resumed the fixed step-43491 source with the existing CPU FP32 optimizer.

## Fixed recovery source

Every comparison and the eventual continuation must restore the same durable
**step 43491** checkpoint, with **Adam step 8689** and next unread packed-data
cursor **8689**. Its SHA256 is:

```text
de5353a85b9478e033dbdb08b9675d0e1813d5dba7e3382ce7e6e50f89b156f7
```

The previous long run's last live metric was **step 43524**. Its **33 updates
after the durable checkpoint were not persisted** and must be recomputed when
continuation restores step 43491. Comparison updates do not advance the actual
continuation source, and no comparison endpoint substitutes for that source.

The environment remains **AMD Radeon RX 7900 XTX**, **PyTorch 2.9.1**, and
**ROCm 6.4**. Production attention semantics remain sequence length **4096**,
window **8192**, batch size 1, **16 query heads**, **2 KV heads**, and head
dimension **128**. The source checkpoint, base weights, train and held-out data,
and isolated source code are protected by the controller's identity and hash
checks. Checkpoint reclamation after verified continuation is recorded below.

## Candidates and completed checks

| Candidate or check | Observed result | Consequence |
| --- | --- | --- |
| New operator CPU suite | 23 passed; 1 GPU-specific test skipped | CPU behavior checked; GPU evidence is separate |
| Entry and profiler CPU suite | 10 passed | Entry guards and attribution regression checks passed |
| Controller CPU suite | 8 passed | Candidate admission, timing validation, fixed-source continuation, and lifecycle checks passed |
| GPU FP32 Adam numerical microbenchmark | First update produced 12 BF16 weight tensors that were not bitwise equal to the CPU reference | Rejected; no full-model optimizer trial updates |
| Packed versus bucketed attention microbenchmark | 20 result rows passed; 0 numerical failures; 0 errors | Eligible for full-model checks |
| Actual bucket execution in the microbenchmark | Bucketed path exercised in 6 cases | Batching was tested, rather than only its fallback |
| Real packed-data row 8689 | 4 document fragments; `no_batchable_fragments` fallback | This row does not demonstrate a bucketed-path speedup |

These three original CPU suites total **41 passed, with one GPU-specific skip**.
The later ordered candidate and stronger entry guards were tested separately in
an independent source directory: **47 passed, one GPU-specific skip** across
48 tests. The [later CPU ledger](results/rx7900xtx-20261003-round2/ordered/all_cpu_tests.log)
includes five ordered-update and checkpoint tests, plus the extra source/variant
binding regression. This later source does not replace the frozen training source.

The GPU Adam microbenchmark uses isolated copies of the fixed checkpoint's
parameters and moments with identical seeded BF16 gradients. Its gate requires
**all 132 BF16 weight tensors to be byte-identical after every update**, finite
FP32 moments within `atol=1e-8, rtol=1e-6`, matching Adam counters, and a valid CPU
FP32 state export. It does not maintain a persistent FP32 master-weight copy.
The first-update weight mismatch is sufficient to reject this candidate; the
gate has not been relaxed. Neither GPU Adam nor a GPU-Adam/bucketed combination
will proceed to full-model timing from these failed micro results.

The [measured optimizer ledger](results/rx7900xtx-20261003-round2/micro/gpu_adam.json)
shows that **all 132 first-moment tensors were byte-identical**. All 132
second-moment tensors differed in bits but passed the declared FP32 tolerance.
The 12 mismatched BF16 weight tensors had maximum relative-L2 error
**7.4534e-7** and maximum absolute error **3.0518e-5**. These small differences
still fail the byte-equality requirement. The standalone ordered-FP32 candidate expands the second-moment product and
parameter division in the native CPU expression order, uses device-tensor bias
correction division, and keeps the same first-moment update. It passed its CPU
multistep and lifecycle tests. GPU numerical evidence is recorded separately
below; this candidate has no production installation or measured whole-training
gain. Changing expression order does not prove cross-device byte equality.

The attention microbenchmark includes `packed_qkv` and `cross_cache` layouts,
one long document, six medium documents, many short documents, uneven document
lengths, and the actual row at the restored cursor. Its BF16 gate checks finite
output and **all dQ, dK, and dV gradients** against an independent dense FP32
document-mask reference. Every tensor must satisfy both
`abs(error) <= 0.02 + 0.02 * abs(reference)` and **relative-L2 error <= 1%**.
Fallbacks are explicit results and do not count as exercised bucketed calls.

The [attention ledger](results/rx7900xtx-20261003-round2/micro/bucketed.jsonl)
records the following **forward-plus-backward median wall times**, with two
warmup iterations and five measured repetitions. Arrows compare existing
packed attention to the bucketed candidate. Cold timings clear both plan
caches on every invocation. Peak allocated GPU memory has the isolated
attention scope, including live inputs, rather than the full model.

| Documents | Layout | Warm time, packed → bucketed (ms) | Warm operator speedup | Cold time, packed → bucketed (ms) | Peak allocated, packed → bucketed (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| One document of 4096 tokens | `packed_qkv` | 36.598 → 36.694 | 0.997× | 36.518 → 36.552 | 4308.0 → 4308.0 |
| One document of 4096 tokens | `cross_cache` | 36.659 → 36.633 | 1.001× | 36.549 → 36.738 | 4310.0 → 4310.0 |
| Six medium documents | `packed_qkv` | 9.375 → 11.799 | 0.795× | 10.017 → 12.414 | 526.6 → 943.2 |
| Six medium documents | `cross_cache` | 9.358 → 11.712 | 0.799× | 10.004 → 12.372 | 528.8 → 945.2 |
| 64 documents of 64 tokens | `packed_qkv` | 40.882 → 17.216 | 2.375× | 41.501 → 17.365 | 364.5 → 350.0 |
| 64 documents of 64 tokens | `cross_cache` | 49.094 → 17.213 | 2.852× | 41.297 → 17.855 | 366.5 → 352.0 |
| Eight uneven documents | `packed_qkv` | 20.590 → 20.635 | 0.998× | 21.303 → 21.097 | 2158.0 → 2203.5 |
| Eight uneven documents | `cross_cache` | 20.577 → 20.674 | 0.995× | 21.221 → 21.164 | 2160.0 → 2205.5 |
| Real row 8689, four fragments | `packed_qkv` | 16.115 → 16.130 | 0.999× | 16.700 → 16.644 | 1018.3 → 1018.3 |
| Real row 8689, four fragments | `cross_cache` | 16.044 → 16.155 | 0.993× | 16.732 → 16.632 | 1020.3 → 1020.3 |

The six medium lengths are `660,668,675,690,698,705`; the uneven lengths are
`2816,512,256,128,128,128,64,64`. Batching reduced the 64-short-document case
from 64 SDPA calls to four calls without padding, and provided the measured
2.37–2.85× operator gain. In contrast, combining the six medium documents into
one call increased padded score work by **6.59%**, doubled incremental peak
memory approximately, and made the `packed_qkv` operation **25.86% slower**.
Thus a small padding ratio alone does not establish a useful dispatch rule.
The eight uneven documents show no material benefit. One long document and
the real recovery row use the existing packed fallback.

The cross-cache short-document warm baseline differed from its cold baseline
despite the cold measurement doing more planning work. These are independent
timing samples: subtracting their medians would not isolate plan-copy cost.
Both warm and cold measurements show the short-document gain and medium-length
regression. GPU process snapshots were empty in the ledger, but the idle check
does not reserve the GPU or guarantee exclusive use throughout measurement.
These operator results do not establish whole-training gains.

## What the bucketed candidate changes

[Bucketed attention](bucketed_attention.py) groups compatible document lengths
into batches of **FP32 PyTorch math SDPA** calls. It is an opt-in layout and
batching candidate, rather than a new FlashAttention kernel. Right padding is
causal: a real query cannot see future padding keys. The implementation restores
the original document, batch, and token order before a single cast back to BF16.
Q/K/V are promoted to FP32 once, document boundaries remain isolated, and the
already-applied global RoPE positions are preserved.

The default plan limits padded score work to **1.25×** unpadded work per bucket,
at most **16 documents** per bucket, and **4,194,304 score elements per head**
for a multi-document bucket. A document covering at least **75%** of the sequence,
one long document, repeated noncontiguous document IDs, and plans without any
mergeable multi-document bucket retain the existing packed implementation.
These choices bound extra padding and avoid claiming gains on a path that
delegates every call.

The standalone benchmark records warm and cold-plan timings. Cold measurements
include document-plan transfer and construction; timed attention includes FP32
conversion, differentiable padding, layout assembly, GQA handling, and the
default `merge_heads` conversion. Projection layers and synthetic fixture
construction are outside that operator scope. These measurements cannot replace
complete-update training measurements.

## Full-model comparison and decision

[The controller](run_next_operator_round.py) compares three independently
restored runs in this order:

| Run | Active attention and optimizer | Planned updates | Complete-update median tokens/s | Complete-update aggregate tokens/s | Loop median tokens/s | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline before | Existing packed attention; CPU FP32 Adam | 20 | 801.577 | 788.815 | 801.125 | Validated |
| Bucketed candidate | Packed attention with opt-in document buckets; CPU FP32 Adam | 20 | 792.376 | 783.831 | 791.926 | Validated; rejected for performance |
| Baseline after | Existing packed attention; CPU FP32 Adam | 20 | 803.394 | 786.746 | 802.959 | Validated |

Against the faster corresponding baseline, the candidate changed median
complete-update throughput by **-1.371%**, aggregate throughput by **-0.632%**,
and median loop throughput by **-1.374%**. None meets the required +5%.
[The final decision](results/rx7900xtx-20261003-round2/comparison/decision.json)
selects the validated baseline. All three runs restored identical weights,
CPU FP32 Adam moments, counters, seed and the same 20 training rows.

The bucketed full-model parity loss absolute error was **0.0003672**, logits
relative-L2 error **1.223%**, final-hidden error **1.039%**, and global-gradient
error **1.508%**. All 132 individual gradients passed the unchanged 5% gate.
The native self-repeat was exact. Each trial's finite two-batch held-out
check and strict terminal CPU checkpoint inspection passed. These correctness
results permit comparison; they do not establish a language-quality improvement.

The first five updates of each run are discarded, leaving **15 measured
updates**. Each run must pass native self-repeat and the unchanged full-model
loss, logits, hidden-state, and **all-132-gradient** gates before optimizer
updates. Loss absolute error must be **<= 0.02**; selected logits, final hidden
state, each gradient, and the global gradient vector must have relative-L2 error
**<= 5%** and be finite. A saved trial checkpoint must validate its 132 BF16
trainable tensors, 264 finite CPU FP32 moment tensors, Adam counters, source
configuration, token accounting, and cursor advancement.

The source row is not bucketable. `BucketParityStream` therefore scans at most
**32 independent real parity rows** to select an eligible document layout. This
bounded search uses a separate parity stream and **does not advance the
restored Trainer training cursor**. A failed search refuses optimizer updates.
The extra `bucketed_calls > 0` gate must pass during parity. The end-of-run
count is cumulative across parity, evaluation and training, so it alone cannot
prove a training-scoped forward used the bucketed path. The 20-row planner
coverage below is static evidence, and the full-step timing is the measured
performance result. The scan
count and bucket layout are recorded for review.

[Complete-update timing](update_timing.py) synchronizes the GPU before the first
microbatch and after the optimizer update. It measures forward, backward,
next-batch prefetch, gradient handling, and the completed optimizer update.
Setup, parity, initial and final held-out evaluation, checkpoint saves,
post-update logs, LR/zero-grad preparation, and the already-prefetched current
batch are excluded. The separate loop metric is retained because explicit
synchronization can change pipeline behavior.

Adoption requires **all three throughput measures** to improve by **at least
5%** over the faster corresponding measure from the two valid baselines:
median complete-update tokens/s, aggregate complete-update tokens/s, and median
loop tokens/s. Aggregate throughput is recomputed from raw steady tokens and
elapsed time; incomplete or inconsistent samples are rejected. A microbenchmark
gain or a profiled timing cannot satisfy this decision gate. If no candidate
passes, the validated packed-attention baseline remains selected.

## Real-row coverage and the short-only candidate

The [actual comparison row plans](results/rx7900xtx-20261003-round2/bucketed_training_rows.json)
cover rows **8689–8708**: 157 fragments in 20 rows, of which 61 fragments have
length <= 256. The frozen policy marks **16 rows eligible**. Rows 8689 and 8706
fall back because no fragments can be combined, row 8702 is a single document,
and row 8707 crosses the long-document threshold.

The 16 eligible rows contain **40 multi-document buckets**: 24 pairs, 13 groups
of three, two groups of four, and one group of seven. **26 of these 40 buckets
have a maximum length above 256**. This is plan coverage, rather than a measured
breakdown of GPU time, but shows that most batches differ substantially from
the profitable synthetic case of 64 equally short documents.

The comparison and active long-run source stay frozen. A new standalone
candidate batches **only short fragments**, restricting
each group to a maximum padded length of **256**, at least **four members**,
and padded score work **<= 1.10×** its unpadded work, while keeping the existing
16-member and memory limits. Medium and long fragments continue through individual document SDPA calls.
The initial thresholds came from the first microbenchmark; the later
64/128/256-token crossover measurements below test this separate implementation.

[A CPU plan check](results/rx7900xtx-20261003-round2/short_training_rows.json)
applies the actual new implementation to the same 20 rows. Only **row 8701** retains
a qualifying group: lengths `133,138,140,143,146,146`, **846 tokens**, and a
padded score ratio of **1.071×**. The other 19 rows retain individual
document calls. This static **1-of-20 row coverage** is not performance
evidence. The crossover below measures lengths 64, 128 and 256 with groups of four,
eight and sixteen, including cold planning and the eligible real row.
Restricting batching may remove the
medium-length regression while leaving too little useful work for a material
whole-step gain.

The implementation promotes complete Q/K/V tensors to FP32 once, restores
original token order, and casts the assembled output back to BF16 once. Medium
and long fragments use individual document SDPA. A mixed row retains a qualifying
short group even when one long fragment dominates. The separate executor does
not replace the wide planner globally. Nine CPU tests check outputs and all
input/projection gradients, real right padding, causal/document isolation,
versioned weak caches, actual bucket execution and context restoration. Initial
fixture assumptions were corrected before GPU admission: the tight greedy
planner can split an 8-token fragment from 9-token fragments before a later
larger group forms. The final CPU suite passed all nine tests.

The [later GPU crossover ledger](results/rx7900xtx-20261003-round2/isolated_candidates/short_bucketed.jsonl)
contains **40 passing output/dQ/dK/dV result rows**, zero failures/errors, and
**20 exercised short-bucket cases**. It compares two QKV layouts and both
implementations across nine uniform length/count cases and real row 8701.
The same BF16 elementwise and 1% relative-L2 gates remain in force. The dense
reference ran on CPU with copies of the exact GPU-generated input and upstream
gradient; reference computation/transfers stayed outside candidate timing.
Candidate forward/backward, planning, FP32 conversion, GQA and merge-head copies
ran on GPU. Two warmups and five repetitions were measured per timing mode.

| Document length × count | `packed_qkv` warm F+B, packed → short (ms) | `cross_cache` warm F+B, packed → short (ms) |
| --- | ---: | ---: |
| 64 × 4 | 3.748 → 2.110 | 4.066 → 2.115 |
| 64 × 8 | 7.608 → 2.893 | 7.677 → 2.923 |
| 64 × 16 | 14.699 → 4.422 | 14.774 → 4.433 |
| 128 × 4 | 4.080 → 2.342 | 4.080 → 2.126 |
| 128 × 8 | 7.631 → 2.954 | 7.632 → 2.888 |
| 128 × 16 | 14.719 → 4.415 | 14.744 → 4.463 |
| 256 × 4 | 3.516 → 2.395 | 4.063 → 2.371 |
| 256 × 8 | 7.663 → 3.921 | 7.674 → 3.908 |
| 256 × 16 | 14.834 → 8.748 | 10.096 → 8.773 |
| Real row 8701 | 17.826 → 15.461 | 17.932 → 15.605 |

Uniform short cases show **1.15–3.33× operator speedup**. Real row 8701 reduces
warm forward/backward wall time by **13.26%** in packed QKV and **12.98%** in
cross cache (speed ratios 1.153× and 1.149×). Cold real-row times change
**18.139 → 16.032 ms** and **18.218 → 16.095 ms**. Incremental GPU allocation
increases by roughly 9–11 MiB on this real row. Some layout medians differ
substantially with only five samples; this is crossover evidence rather than
precise attribution of layout cost. The stopped training worker kept its GPU
allocator but issued no training work during the recorded isolated interval.

This candidate has **not** passed a full-model 132-gradient/training comparison
and has **not** been installed in the long run. Only 1/20 of the fixed comparison
rows has a qualifying group; the real-row operator gain cannot establish a 5%
whole-training gain. Any future installation still requires unchanged full-model
loss/logit/hidden/gradient gates and a new bracketing completed-update comparison.

## Profiler correction

The initial profiler attributed some parentless autograd-worker backward events
to `other`. Its old phase fractions are invalid and are not used in this report.
[The corrected profiler](profile_training.py) first follows labelled parent
scopes, then uses enclosing CPU event time ranges when a worker has no labelled
ancestor. Save and evaluation scopes take priority; otherwise the narrowest
phase scope wins. It records attribution counts and separate steady-step
operator tables, counting CPU-event self device time once.

The [corrected GPU profile](results/rx7900xtx-20261003-round2/comparison/profile_current/profile/summary.json)
completed six updates, discarding two warmup updates and recording steps
43494–43497. Its independent all-132-gradient gate, finite evaluation, and
six-update CPU Adam/cursor checkpoint inspection passed. It attributes 297,930
CPU events by labelled ancestry and 455,990 by enclosing CPU time ranges;
252 events remain unclassified.

| Steady phase | Share of linked device work |
| --- | ---: |
| Forward, excluding CE | 37.866% |
| Backward | 53.647% |
| CPU-offload optimizer's linked device work | 7.655% |
| Vocabulary head and CE | 0.775% |
| Gradient clipping | 0.052% |

The four steps contain **19.499 seconds of summed linked device work**.
`aten::bmm` contributes **9.862 seconds (50.58%)**, `aten::copy_` **1.839 seconds
(9.43%)**, and `aten::mm` **1.735 seconds (8.90%)**. These operator totals can
cover multiple model components; they do not isolate attention or physical
PCIe transfer. The transfer overlay includes dtype conversions and overlaps
phase totals. The profile also records **16.489 seconds** of self CPU time in
`hipMemcpyWithStream`, which can include host waiting and overlaps worker scopes.
These are useful optimization targets, **not fractions of GPU or CPU wall time**.
Profiled steps have instrumentation overhead and do not participate in selection.

## Reproduction entry points

Use an isolated source checkout and an idle GPU. Replace the example paths and
deadline with the actual frozen source, source checkpoint, base weights, corpus,
and completed microbenchmark directory. Outputs must be separate from every
protected input directory.

Attention microbenchmark, including the real recovery row and both layouts:

```bash
python operators/rocm/bucketed_attention_bench.py \
  --seq-len 4096 --window 8192 --dtypes bf16 \
  --layouts packed_qkv cross_cache \
  --cases one_long six_medium many_short uneven real \
  --data /path/to/corpus/train.bin --data-row 8689 --eos-id 1 \
  --wait-gpu-idle --output /path/to/microbench/bucketed.jsonl
```

Independent strict optimizer check; a numerical failure exits unsuccessfully
and must remain rejected:

```bash
python operators/rocm/gpu_adam_bench.py \
  --resume /path/to/checkpoints/trainable_step_43491.pt \
  --device cuda --steps 5 --moment-atol 1e-8 --moment-rtol 1e-6 \
  --wait-gpu-idle --json /path/to/microbench/gpu_adam.json
```

Reproduce the later low-memory numerical and short-only operator probes with
this repository revision (a failed numerical check must retain its nonzero exit):

```bash
python operators/rocm/gpu_adam_streamed_bench.py \
  --resume /path/to/checkpoints/trainable_step_43491.pt \
  --device cuda --json /path/to/separate-output/ordered-streamed.json

OMP_NUM_THREADS=1 python operators/rocm/short_bucket_attention_bench.py \
  --oracle-device cpu --data /path/to/corpus/train.bin --data-row 8701 \
  --wait-gpu-idle --output /path/to/separate-output/short-bucketed.jsonl
```

These standalone probes do not modify or install anything in another process.
Use an idle GPU for reproduction; the recorded short probe instead used the
explicitly authorized, automatically restored pause of its own training worker.

The controller consumes the completed microbenchmark bundle, including its
`status.json`, source identity, file hashes, and child exit receipts. Its
evaluation-only entry performs the gated comparisons without continuation:

```bash
python operators/rocm/run_next_operator_round.py \
  --source-dir /path/to/isolated-source \
  --python /path/to/rocm-venv/bin/python \
  --resume /path/to/checkpoints/trainable_step_43491.pt \
  --base /path/to/base-weights \
  --data /path/to/corpus/train.bin --eval-data /path/to/corpus/eval.bin \
  --micro-dir /path/to/completed-microbench \
  --out /path/to/round2-evaluation \
  --deadline-utc YYYY-MM-DDTHH:MM:SSZ --evaluate-only
```

For automatic continuation after a valid decision, use the same entry without
`--evaluate-only`, a future UTC deadline, and a new output directory. Continuation
restores the original source's Adam state, seed, and train cursor, not a trial's
terminal checkpoint. Synchronized comparison timing is **disabled** for the
continuation. Timed saving is requested **every 300 seconds**, at complete-update boundaries
and counted from the last successful publication, with **three periodic
checkpoints retained**, optimizer state included, and atomic `latest.pt`
publication. The deadline limits the training loop timer; model loading, parity, and
initial/final evaluation are outside that timer. The supervisor controls only children it
starts and preserves failed outputs for manual recovery.

Expected review files include `micro_validation.json`, both baseline validation
reports, candidate validation, each trial's `parity.json`, `operators.json`,
`next_operators.json`, `update_timing.json`, training metrics, CPU checkpoint
inspection logs, `decision.json`, child command/exit receipts, and supervisor
status. The [completed microbenchmark receipt](results/rx7900xtx-20261003-round2/micro/status.json)
records the fixed source identity, child exits, and measured operator source
hashes. Full-model artifacts are retained beside the final decision and corrected profile.
The measured-source manifest records the frozen remote files; subsequent gate
hardening and new standalone candidates are identified separately and do not
change the already measured or currently running source.

## Continuation and recovery receipts

The controller completed the corrected profile and started its continuation
child at **12:10 UTC**. Loading, independent reference checks, and a finite
32-batch held-out evaluation ran before optimizer updates. The initial NLL is
**8.123319** over **130,877 valid tokens**. This larger evaluation prefix cannot
be directly compared with the trials' two-batch NLL. The active configuration
uses packed attention, shared-storage MoE and CPU FP32 Adam, with synchronized
comparison instrumentation disabled. The source is step 43491, not a trial's
43511 checkpoint. The 33 unsaved updates from the prior run are recomputed.

The comparison supervisor alone was temporarily stopped at 09:59 UTC while
its already-running final-baseline child completed. An approval-service outage
prevented the requested resume until **12:02 UTC**. The restored session checked
the supervisor PID, UID, command and process start time before sending SIGCONT.
The [pause/resume receipt](results/rx7900xtx-20261003-round2/ordered_supervisor_pause.json)
preserves this interval. No unrelated GPU process was signalled. This interval
is controller waiting time, not training time or measured update elapsed time.

Periodic saves occur at complete-update boundaries after 300 seconds from the
last successful publication, with three numbered checkpoints, optimizer state
included and atomic latest publication. The first newly verified periodic
checkpoint is **step 43536**, with **Adam
step/cursor 8734**: exactly 45 updates and 184,320 tokens beyond the fixed source.
[Its strict CPU inspection](results/rx7900xtx-20261003-round2/ordered/continuation_checkpoint_check.log)
validated all 132 BF16 weights and 264 CPU FP32 moments, group mappings, seed,
configuration, token accounting and cursor. The next periodic save reached
43593. Checkpoint cleanup after later verification is recorded below.
Finite short-run checks do not establish long-run language quality.

The simultaneous all-parameter ordered Adam microbenchmark was admitted only
if at least 5 GiB remained free. Training allocator use left less room, so the
[first attempt](results/rx7900xtx-20261003-round2/ordered/probe_receipt.json)
aborted **before any pause or GPU optimizer update**. A later per-parameter
check keeps GPU copies small and preserves the strict numerical thresholds.
Its micro-only scope does not measure optimizer or whole-training speed.

Future optimizer admission now binds `native-order` to the exact three
implementation/reference source hashes and applies the full 132-parameter,
byte-equality, moment-tolerance gate to the terminal check as well as each
intermediate check. An `ordered-fp32` report cannot admit the original native-order
installer. These changes only strengthen future experiment admission; the
measured comparison and active long run retain their frozen source.
[The measured-source manifest](results/rx7900xtx-20261003-round2/measured_source.json)
keeps that distinction reviewable.


## Ordered optimizer outcome and final recovery snapshot

The [streamed GPU ledger](results/rx7900xtx-20261003-round2/isolated_candidates/gpu_adam_streamed.json)
checks one parameter at a time against the native CPU reference using each
source group's original hyperparameters, Adam counter and FP32 moments.
This bounds temporary GPU memory and measures no timing. It reuses strict BF16
byte equality and FP32 moment tolerances of `atol=1e-8, rtol=1e-6`.
The two cache projection parameters completed five updates. On the third
parameter, **decoder.0.cross_attn.q_proj.weight**, update **4** failed weight
byte equality: relative-L2 **7.5071e-7**, maximum absolute error **3.0518e-5**.
Its first moment was exact and its second moment remained within tolerance.
The run stopped immediately and exited 1. This **incomplete 3-parameter check**
is not a passed 132-parameter gate. Ordered GPU Adam remains rejected.

The combined isolated probe verified durable **step 43706** (Adam/cursor **8904**)
and SHA256 before its short pause. An independent 420-second watchdog backed
up the controller's `finally` resume; total probe work was bounded to 360 seconds.
The actual pause was **12:36:22–12:38:34 UTC**, about **132 seconds**. Both probes
finished and the same worker resumed successfully. The [probe receipt](results/rx7900xtx-20261003-round2/isolated_candidates/probe_receipt.json)
records the exact PID/UID/start time, commands, child exits and saved checkpoint
proof.
No unrelated GPU process was signalled. The fresh 13-test streamed/short suite
passed: together with the earlier independent suite, **60 tests passed and one
GPU-specific test was skipped**. GPU evidence is documented separately.

The [live snapshot](results/rx7900xtx-20261003-round2/live_snapshot.json) records
training at **step 43764**, 198,952,960 cumulative tokens, with packed attention,
CPU FP32 Adam, and three periodic recovery points. It is a timestamped progress
snapshot, not a claim that long training has finished. Periodic saves and final
held-out evaluation continue under the existing supervisor.

After verified continuation, [authorized reclamation](results/rx7900xtx-20261003-round2/cleanup_receipt.json)
removed four completed/validated comparison or corrected-profile checkpoint
files plus old periodic steps 43378 and 43435: **13,153,525,858 logical bytes
(12.25 GiB)**. Each removed file's SHA256 is recorded. The fixed step-43491 source
and current live recovery points remain intact, and the fixed source hash was
rechecked after cleanup. Trial validation reports remain available; reclaimed
weight files are no longer present for a second direct inspection.

The [artifact manifest](results/rx7900xtx-20261003-round2/MANIFEST.json) hashes
all retained reports. Source hashes distinguish the original measured/current
training source from the later isolated candidate source. No model weights,
corpus binaries, secrets or large profiler traces are included in this commit.

## Deadline completion and the next corpus continuation

The earlier continuation subsequently stopped normally at **step 45024** when
its **2.4993134067-hour training-loop limit** expired. Its final checkpoint was
saved at **14:45:30 UTC**, and the [controller status](results/rx7900xtx-20261003-round2/comparison/status.json)
recorded verified completion at **2026-10-03 14:47:28 UTC**. Strict CPU inspection
passed for all **132 BF16 weights and 264 CPU FP32 Adam moments**, with Adam
step and next unread packed-data cursor both **10222**. The fixed step-43491
source remains preserved.

The [completed run's held-out results](results/rx7900xtx-20261003-round2/comparison/continuation/parity.json)
use the same 32-batch prefix and **130,877 valid tokens**: initial NLL
**8.123319335125808** and final NLL **8.07905429250309**, a decrease of
**0.0442650426**. This records the outcome of that bounded continuation;
it does not establish final language quality or completion of the 8-billion-token
B0 recipe.

A new [continuation](results/rx7900xtx-20261003-round2/next_continuation/status.json)
restores the verified step-45024 checkpoint, SHA256
`655da42cf523ef7d512234247fcb229f70c8ab1c72a495fbb41cd82e778a4285`,
in `runs/longtrain-45024-20261003T1520`. It requests the remaining **9309** packed
rows, bounded to **step 54333**, Adam step **19531** and cursor **19531**. No
`--max-hours` limit is supplied. The terminal update skips the next-batch
prefetch, so this bound ends at the corpus boundary without reading a wrapped
row. The frozen packed-attention/shared-storage MoE and CPU FP32 Adam remain in
use, with saving every **300 seconds**, **three** recovery checkpoints and
32-batch held-out evaluation every 250 steps.

Startup passed the unchanged 4096-token, all-132-gradient parity gates. The new
initial held-out NLL is **8.08030031177656**; its small difference from the prior
final value is recorded explicitly and is not a claim of exact numerical replay.
The [progress snapshot](results/rx7900xtx-20261003-round2/next_continuation/live_snapshot.json)
at **2026-10-03 15:30:47 UTC** records **step 45091**, GPU utilization **97%**
and reported power **322 W**. It also lists the first periodic checkpoint,
**step 45079**, saved at **15:29:30 UTC** after 55 updates, with
**2,192,257,315 bytes**. These are progress and save observations, not an
additional operator speedup or a completed-run result.

The [first periodic checkpoint inspection](results/rx7900xtx-20261003-round2/next_continuation/periodic_45079.validation.json)
also passed the strict CPU gate for all 132 weights and 264 Adam moments.
Its Adam step/cursor are **10277**, and its token counter is **204,339,200**,
exactly **55 updates / 225,280 tokens** beyond the fixed step-45024 source.

The new supervisor's argument object omits `eval_data`, which affects its
post-training validation call. The training worker received the correct held-out
path and continues independently. An [independent final auditor](results/rx7900xtx-20261003-round2/next_continuation/final_audit_status.json)
has completed its source CPU preflight and is waiting for training completion;
it sends no signals and starts no training worker. It supplies the missing
validation argument and requires successful child exit, unchanged source/code
and corpus identity, all original parity gates, exactly 9309 consecutive updates,
and strict final weight/moment/token/cursor checks. It may publish authoritative
completion only after all checks pass, preserving the original supervisor status
and any validation error. Other failures remain failures. **This continuation
and its final audit are still pending at the recorded snapshot.**
