# Operator evaluation during a real-text B0 pause — October 3, 2026

**Packed attention alone was selected for real-text B0 continuation.** Its
unprofiled 20-update trial measured **820.219 tokens/s**, versus **579.804
tokens/s** for the faster of two native baselines, after discarding each run's
first five updates: **1.414649× throughput, a 41.4649% increase**. All five
full-model trials and their saved checkpoints validated. Batched gradient norm
and the combined candidate provided no additional whole-step benefit. Training
is **running with packed attention**, its first timed checkpoint at **step
42155** has passed strict CPU verification, and checkpoint cleanup has completed.
This phase's long-training target remains unfinished.

## Pause and recovery source

The user authorized pausing the CAT-YOKO long run to evaluate the new operators
and adopting a candidate if it improves training speed. The previous training
supervisor and operator queue were superseded under that authorization.

The fixed recovery source is the durable **step 42100** B0 checkpoint with the
next unread packed-data cursor **7298**. Its SHA256 is:

```text
14e845a3642ba802173797e2341b99a5084b03d267c991925e95da9821116238
```

Strict CPU inspection verified **132 BF16 trainable weight tensors**, **132 Adam
states**, and **264 finite CPU FP32 moment tensors**. The source preserves Adam
step counters and the packed-data cursor; experiments must restore this state
rather than restart moments or advance from another experiment's endpoint.

The last recorded metric before the pause was **step 42157**. The **57 updates
after step 42100 were not persisted** in the recovery source. Continuation
recomputed those updates and has now advanced beyond the old step 42157.
Checkpoints in the original directory were retained throughout evaluation;
older steps 41900 and 42000 were reclaimed only after the new continuation
checkpoint passed verification. The step-42100 recovery source remains intact.

Remote experiment output directory:

```text
/home/donz/cat-yoko-rocm-20261002/runs/operators-switch-20261003
```

## Measured operator results

Environment: **AMD Radeon RX 7900 XTX**, **PyTorch 2.9.1**, **ROCm 6.4**. The
attention test uses real packed-data row **7298**, with **six contiguous document
fragments**, batch size 1, sequence length **4096**, window **8192**, **16 query
heads**, **2 KV heads**, and head dimension **128**. The covering window retains
the checkpoint's attention semantics. Q/K/V values in this isolated benchmark
are seeded synthetic tensors; document boundaries come from the real corpus.

The candidate computes causal FP32 attention separately within each document,
preserving the original token order and already-applied RoPE positions. It
casts Q/K/V to FP32 once and accumulates their gradients before casting back to
BF16. The two layouts cover encoder packed QKV and decoder cross-cache inputs.

| Attention layout | Native forward + backward (ms) | Packed forward + backward (ms) | Operator speedup | Native peak allocated (MiB) | Packed peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `packed_qkv` | 36.866 | 12.831 | 2.873× | 4308.031 | 811.621 |
| `cross_cache` | 36.933 | 12.866 | 2.871× | 4310.031 | 814.505 |

Both layouts passed comparison against the independent dense FP32 masked
reference. BF16 output and **all dQ, dK, and dV gradients** were finite and met
both the elementwise bound `abs(error) <= 0.02 + 0.02 * abs(reference)` and the
relative-L2 bound **<= 1%**. The maximum Q/K/V gradient relative-L2 error was
**< 4.263e-5**. Peak memory above describes this isolated attention operation,
not the complete 12B model's training memory.

The gradient-norm candidate retains native per-tensor norms, Python summation
order, and gradient scaling, while batching scalar readback by device.

| Gradient-norm case | Native time (ms) | Batched time (ms) | Operator speedup |
| --- | ---: | ---: | ---: |
| Clipped | 7.88185 | 3.31315 | 2.379× |
| Unclipped | 6.68671 | 2.25070 | 2.971× |

Both cases returned the **same floating-point norm** and **byte-identical
gradients for all 132 trainable tensors**. These tests use seeded gradients
with the real checkpoint's tensor shapes. They exclude model forward/backward,
Adam, data loading, and checkpoint I/O.

## Full-model comparison and decision

Every comparison restored the same immutable step-42100 source, including Adam,
RNG, and cursor 7298, with the same real train and held-out data. Each run passed
native self-repeat and the full-model loss, output, and all-132-gradient checks
before applying optimizer updates. Packed-attention parity used a multi-document
4096-token row and confirmed that the optimized path was exercised.

Two native baselines bracketed the independently measured candidates:

| Run | Opt-in operators | Training updates | Median loop tokens/s, last 15 updates | Current status |
| --- | --- | ---: | ---: | --- |
| Baseline before | Native attention and gradient norm | 20 | 579.8038481235612 | Complete |
| Attention only | Packed attention | 20 | 820.2189673670778 | Complete |
| Norm only | Batched gradient norm | 20 | 579.7913052105006 | Complete |
| Combined | Packed attention and batched gradient norm | 20 | 819.5415239288179 | Complete |
| Baseline after | Native attention and gradient norm | 20 | 575.61727044889 | Complete |

The first baseline uses the previously validated shared-storage MoE layout,
with native attention at the actual **sequence length 4096 and window 8192**.
Native self-repeat and shared-storage layout parity both reported **zero loss
difference and zero global-gradient relative-L2 error**.

The attention-only full-model parity passed all **132 trainable-gradient
checks**. Its maximum per-tensor gradient relative-L2 error was
**0.0162485419730** (1.624854%), global-gradient relative-L2 error was
**0.006403872980606** (0.640387%), and absolute loss difference was
**0.000260353088379**. These are full-model parameter-gradient measurements;
they have a different scope from the isolated Q/K/V checks above.

The attention trial's peak GPU allocation was **11448.794921875 MiB** for the
complete training model. It should not be compared directly with an isolated
attention operation's allocation in the microbenchmark table. On the same fixed
held-out prefix of **two batches**, attention-only NLL was
**8.556895333008127 initially** and **8.478367215061164 after 20 updates**.
This finite short-run evaluation does not establish a language-quality gain.

Strict CPU inspection passed for the fixed source and **all five experiments'
final checkpoints**. Each final overlay has **132 BF16 weight tensors**, **132
Adam states with 264 CPU FP32 moments**, **step 42120**, and **stream cursor
7318**. Configuration consistency and the expected 20-update increase in token
counters also passed. This verifies saved state and continuation accounting;
it does not assert equality of terminal weights or moments between trials.

The comparison discarded the first **five updates** of each run and measured
the remaining **15 updates** using median loop tokens per second. The adoption
gate required passing the numerical checks and delivering **at least 5% higher
median throughput than the faster of the two baselines**. Separate attention,
norm, and combined measurements identify the source of the gain. Microbenchmark
speedups alone did not satisfy this gate. GPU process observations and raw timing
variability remain relevant when interpreting the comparison.

Attention alone delivered **1.4146490576452366×** the faster baseline's median
throughput, a **41.4649057645% increase**, and passed the adoption gate. Norm-only
throughput was effectively unchanged; combined throughput was about **0.083%
below attention alone** and supplied no additional demonstrated benefit. The
selected flags are therefore **`--packed-attention` only**, alongside the
existing shared-storage MoE layout. These throughput measurements exclude
initial/final evaluation and checkpoint I/O.

Optional profiling uses two warmup updates and four active updates. Profiling
adds tracing and shape-recording overhead; its throughput is not an adoption
measurement. Use profiles to identify where time is spent, and use the separate
unprofiled 20-update runs to select a candidate.

Strict deterministic algorithms were enabled temporarily for parity and
restored afterward. Actual training resumes the original **nondeterministic**
policy. In particular, byte-exact clipping on identical input gradients in the
norm microbenchmark does not imply byte-exact terminal training trajectories:
native BF16 atomic operations can produce different gradients across independent
training runs. The 20-update checks validate saved state and finite short-run
loss; they do not guarantee language quality or terminal weight/moment equality.

The new entry points passed **13 CPU regression tests**: nine operator-switch
tests, including actual serialization and corruption failures, and four
continuation lifecycle tests. Native source hashes and the raw artifact hashes
were verified. These checks support checkpoint integrity and lifecycle
behavior; GPU numerical and throughput evidence comes from the measured runs.

## Reproduce the evaluation and continuation

Use a frozen source directory, the fixed source checkpoint, and the exact
verified train/evaluation bins. Keep all outputs outside source, base, corpus,
and source-checkpoint directories. For an evaluation-only comparison:

```sh
/path/to/rocm-venv/bin/python /path/to/frozen-source/operators/rocm/run_operator_switch.py \
  --source-dir /path/to/frozen-source \
  --resume /path/to/fixed-checkpoint/trainable.pt \
  --base /path/to/MiniCPM5-2B-Base \
  --data /path/to/corpus/train.bin \
  --eval-data /path/to/corpus/eval.bin \
  --out /path/to/results/operator-comparison \
  --python /path/to/rocm-venv/bin/python \
  --deadline-utc 2026-10-03T14:40:00Z \
  --benchmark-steps 20 --evaluate-only
```

After a successful evaluation-only result, revalidate its decision and start a
separate continuation without rerunning benchmark updates:

```sh
/path/to/rocm-venv/bin/python /path/to/frozen-source/operators/rocm/continue_operator_switch.py \
  --comparison /path/to/results/operator-comparison \
  --source-dir /path/to/frozen-source \
  --base /path/to/MiniCPM5-2B-Base \
  --data /path/to/corpus/train.bin \
  --eval-data /path/to/corpus/eval.bin \
  --out /path/to/results/operator-continuation \
  --python /path/to/rocm-venv/bin/python \
  --deadline-utc 2026-10-03T14:40:00Z
```

The shown deadline identifies this run; choose an appropriate future deadline
for a later reproduction. The continuation entry checks comparison receipts,
the source SHA256, saved Adam/cursor state, and frozen code/data before launching.
Both entries may signal **only children they created**, and do not control
existing GPU jobs. If a continuation fails after recording updates or saving
state, its checkpoints and receipts remain available for manual recovery.

The five independent trial checkpoint files have now been reclaimed. Their
reports, metrics, operator-installation records, and verification receipts remain
available, but those recorded trials cannot be reused directly by
`continue_operator_switch.py`, which revalidates the checkpoint files themselves.
A fresh reproduction should rerun evaluation into a new output directory.

## Continuation and retention

The new continuation supervisor, **PID 1468849**, runs training child **PID
1468948** from the fixed **step-42100 recovery source and cursor 7298**, with
**packed attention only**. It restored the same CPU FP32 Adam state, seed, and
real data, and uses frozen source code and corpus files. It resumed the recovery
source rather than an independent 20-update experiment's endpoint.

The installation report confirms `packed_attention=true`,
`batched_grad_norm=false`, and `patches_installed_after_native_reference=true`.
Startup full-model parity again passed all **132 gradient checks**, with loss
difference **0.000260353088379** and global-gradient relative-L2 error
**0.006403872980606**. Initial evaluation on **32 fixed held-out batches**
reported NLL **8.16189351868786** over **130877 valid loss tokens**. This larger
evaluation prefix differs from the two-batch operator trials.

At launch, the corpus had **12233 remaining 4096-token rows**. The continuation is capped
by those rows and a remaining-time budget derived from the nominal **October 3,
2026, 14:40 UTC** deadline. The timer covers **`Trainer.run`**, with setup,
parity, and initial/final held-out evaluation outside its core deadline timer.
Those stages and completion of an update/save can extend the actual exit time;
14:40 UTC is not a guaranteed process-exit timestamp.

The continuation uses **300-second checkpoint intervals** and **keeps the latest
three numbered checkpoints**. Its first timed save, **step 42155**, passed strict
CPU verification: **132 finite BF16 trainable tensors**, **132 Adam states with
264 finite CPU FP32 moments**, **Adam step 7353**, **stream cursor 7353**, and
the expected **55-update increase in token counters**, with source configuration
and model name preserved. The latest pointer shares the numbered checkpoint's
inode, and the fixed source's SHA256 is unchanged.

At **2026-10-03 06:54:44 UTC**, training had completed **106 updates**, reaching
**step 42206** and passing the old pre-pause step 42157. The verified durable
checkpoint covers **55 updates**; subsequent in-memory updates await the next
timed save. Throughput changes with packed-document lengths: the observed
median after warmup was **779.62455 tokens/s**. Its rows differ from the fixed
20-update comparison, so it is not used to calculate a new baseline speedup.
The fixed-source, identical-row 20-update comparison remains **1.414649×**, but
it does not promise a 41% gain across the entire corpus.

After verifying step 42155, cleanup reclaimed **seven files**: five independent
trial `trainable.pt` files and original numbered checkpoints for steps **41900**
and **42000**. SHA256 values were recorded before deletion. The logical reclaimed
file size was **14.29187009576708 GiB**; this is not a measured filesystem-free-space
delta. The new step-42155 checkpoint and latest pointer, live run, and original
step-42100 recovery source were retained. All trial reports and metrics remain
available for review.

At this report's cutoff, packed attention is in use, continuation is **running**,
the first new timed checkpoint has passed verification, and authorized cleanup
has completed. This phase's long-training target has not been reached or
reported as complete.

## Raw evidence

The [artifact manifest](results/rx7900xtx-20261003/manifest.json) records file
hashes and the fixed source identity. Retained evidence includes the
[selection decision](results/rx7900xtx-20261003/decision.json),
[attention microbenchmark](results/rx7900xtx-20261003/packed_attention.jsonl),
[gradient-norm microbenchmark](results/rx7900xtx-20261003/gradient_norm.json),
[pause receipt](results/rx7900xtx-20261003/pause_receipt.json),
[first verified continuation checkpoint](results/rx7900xtx-20261003/first_checkpoint_verified.json),
and [checkpoint cleanup receipt](results/rx7900xtx-20261003/checkpoint_cleanup.json).

Each trial retains its full numerical report, per-update metrics, and
operator-installation record:

| Trial | Parity | Metrics | Operator installation |
| --- | --- | --- | --- |
| Baseline before | [Report](results/rx7900xtx-20261003/baseline_before_parity.json) | [Metrics](results/rx7900xtx-20261003/baseline_before_metrics.jsonl) | [Record](results/rx7900xtx-20261003/baseline_before_operators.json) |
| Attention | [Report](results/rx7900xtx-20261003/attention_parity.json) | [Metrics](results/rx7900xtx-20261003/attention_metrics.jsonl) | [Record](results/rx7900xtx-20261003/attention_operators.json) |
| Norm | [Report](results/rx7900xtx-20261003/norm_parity.json) | [Metrics](results/rx7900xtx-20261003/norm_metrics.jsonl) | [Record](results/rx7900xtx-20261003/norm_operators.json) |
| Combined | [Report](results/rx7900xtx-20261003/combined_parity.json) | [Metrics](results/rx7900xtx-20261003/combined_metrics.jsonl) | [Record](results/rx7900xtx-20261003/combined_operators.json) |
| Baseline after | [Report](results/rx7900xtx-20261003/baseline_after_parity.json) | [Metrics](results/rx7900xtx-20261003/baseline_after_metrics.jsonl) | [Record](results/rx7900xtx-20261003/baseline_after_operators.json) |

Relevant entries: [candidate_bench.py](candidate_bench.py),
[run_operator_switch.py](run_operator_switch.py),
[continue_operator_switch.py](continue_operator_switch.py),
[packed_attention_bench.py](packed_attention_bench.py),
[grad_norm_bench.py](grad_norm_bench.py),
[profile_training.py](profile_training.py), and
[the real-text continuation guide](REAL_TRAINING.md).
