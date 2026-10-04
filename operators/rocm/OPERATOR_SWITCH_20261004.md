# ROCm round-three operator adoption (2026-10-04)

The B0 real-text continuation now selects document-split FP32 attention and
cached BF16 CPU-shadow Adam, alongside the previously validated shared-storage
frozen MoE layout. The user explicitly requested this switch. The measured
whole-update gain is **2.55–3.03%** in a short bracketed window; it does **not**
meet the existing **5% automatic selection** gate. Numerical thresholds and
CPU FP32 Adam arithmetic were retained.

Hardware: Radeon RX 7900 XTX / gfx1100, PyTorch 2.9.1+ROCm 6.4. The initial
[operator microbenchmarks](OPERATOR_ROUND3_20261004.md) and their optimizer-only
11.02% result remain separate from the training measurements below.

## Checkpoint-first handoff

The original `longtrain-45024-20261003T1520` worker was stopped at its atomic
**step 53143** publication. The last completed update in its log was also
53143: **zero completed logged updates were rolled back**. A hardlink preserved
the full recovery source before CPU validation and termination of that owned
worker. The watchdog protected the temporary pause; unrelated jobs were not
signaled. The old supervisor/auditor ended after the intentional handoff, so
their historical `failed` states are not a completed-corpus verdict.

| Recovery item | Verified value |
| --- | --- |
| Global step | 53143 |
| Next unread packed row / Adam counter | 18341 / 18341 |
| Native overlay | 132 finite CPU BF16 tensors |
| Optimizer | 132 states, 264 finite CPU FP32 moments, native IDs and decay groups |
| Real-text tokens consumed | 75,124,736 |
| Phase token counter | 237,369,344, including earlier DummyStream history |
| Full source SHA256 | `40c8bbb93eb3c6e845232bdd3573dd3ac656d8e8f00d337342041da0f3652ee0` |

The saved RNG, packed cursor, architecture and token counters remain part of
the native overlay. Every experiment restarted from this same fixed source;
benchmark updates do not advance the production stream.

## Numerical and recovery evidence

The previously skipped long-document and single-document GPU fixtures now
pass **14/14**, with no errors or memory skips and byte-exact outputs, dQ, dK
and dV. This includes corpus rows 12000, 17478, 18000 and the current row
18341, plus uniform 64-, 1024- and 4096-token fixtures, in both actual QKV and
cross-cache layouts. Eight comparisons exercise split; six exercise the
single-document native fallback.

Full-model checks capture the unmodified native reference and repeat before
installing the candidates. All **132 trainable gradients**, aggregate gradient,
loss, sampled hidden states and logits pass the existing gates. Native
deterministic repeats have zero error. Relative to the unoptimized native
reference, the packed/shared candidate has aggregate gradient relative L2
**1.4949%** and loss absolute error **0.000927925**; it is not claimed to be
byte-exact against that reference. These values are identical across the
old/new/old timing trials. The original limits remain loss absolute error
0.02 and output/per-tensor/global gradient relative L2 5%.

Under the original ordinary training policy, the two old-operator 20-step
trials themselves do not reproduce byte-exact weights or moments. Their
maximum moment difference is approximately 0.00127714, and the original
terminal comparison fails. That background variation is recorded rather than
attributed to the new operators.

An additional controlled experiment holds strict deterministic algorithms
through **two real training updates**, with identical source, data and policy
for the old and new operators. Both end at step 53145 / cursor 18343. All
**132 BF16 weights and 264 FP32 moments are byte-exact**, with maximum moment
difference **0**. All 132 parameters were actually updated; the cached optimizer
loaded the source state once and recorded 264 GPU parameter updates, 132 cache
builds, 132 reuses and zero fallback updates. This is a two-step mathematical
check, not proof of byte-identical ordinary-policy 20-step trajectories.

Every terminal checkpoint also passes the CPU recovery audit: finite tensors,
CPU FP32 moments, native parameter mapping/options, counter increments, token
increments, configuration and packed cursor. The independent
[acceptance audit](results/rx7900xtx-20261004-round3-switch/acceptance_audit.json)
checks the source SHA binding and all consecutive policy receipts.

## Completed-update throughput

The comparison is old → new → old, **20 updates each**, from step 53143 and
the same real-data window. Discard the first five updates and measure the next
15. Both methods synchronize before forward and after the complete optimizer
update. Initialization, parity, evaluation, checkpoint writes and post-update
logs are excluded. LR/zero-grad and the already-prefetched current batch are
outside this timer; next-batch prefetch is included. This synchronized protocol
can change pipeline throughput and is not a long-run wall-clock estimate.

| Segment | Synchronized update median, tokens/s | Aggregate, tokens/s | Trainer-loop median, tokens/s |
| --- | ---: | ---: | ---: |
| Old before | 741.140 | 708.084 | 740.750 |
| Split attention + cached Adam | 763.612 | 726.336 | 763.217 |
| Old after | 740.042 | 708.301 | 739.661 |

Using the faster old baseline independently for each statistic gives **3.032%**
median synchronized gain, **2.546%** aggregate gain and **3.033%** loop-median
gain. The 5% automatic gate is false. Adoption is the user-requested choice
after numerical/recovery validation, not an automatic performance promotion.
The 20-step window does not establish corpus-wide or sustained acceleration.

## Applied continuation

The active frozen source is
`/home/donz/cat-yoko-rocm-20261002/source-round3-confirm-20261004T0455`.
The supervisor is PID **1716930** and its production child is PID **1719057**
at the recorded launch. Runtime state is
`runs/round3-confirm-20261004T0455/status.json`, and the training output is
`continuation/train/` inside that run.

Production resumes **53143**, discarding both benchmark trajectories, for
**1190** remaining updates to **54333 / packed cursor 19531**. It retains
the original ordinary training policy (`deterministic_algorithms=False`);
strict determinism is used for parity and the separate two-step control.
The command requests both `--split-attention` and `--cached-cpu-adam`.
Attention stays FP32 MATH, parameters stay BF16, and moments stay CPU FP32.
The private BF16 host cache uses about 418 MiB, is rebuilt after optimizer load,
and is absent from the checkpoint. There is no persistent FP32 weight master.

Complete checkpoints are scheduled every **300 seconds**, retaining the latest
**three** numbered publications. Fixed held-out evaluation uses **32 batches**
every **250** updates. The supervisor separately audits newly published
checkpoints and retains a hardlink to the latest verified recovery copy. It
also checks the final exact corpus/cursor/Adam boundary. There is no training
time cap and no B1 transition.

The first new-backend periodic checkpoint is **step 53201 / cursor and Adam
counter 18399**, verified at **2026-10-04T05:26:55Z**. Its 132 BF16 tensors and
264 finite CPU FP32 moments pass the complete source/delta audit. An independent
CPU process also loads it through native `CPUOffloadAdamW.load_state_dict` and
re-exports all 264 moments byte-exact with preserved groups. Python and CPU
Torch RNG states restore successfully; the one serialized HIP/CUDA RNG state
is validated as present, without GPU restoration in that CPU-only audit.
The full file is 2,192,257,315 bytes, SHA256
`9502e67bc5fb5fb22a816e42bbbeb3023bb14cfaa06b39fd7e132063dddf470b`.

At **2026-10-04T05:31:24Z**, production has completed **106** consecutive
updates to **53249** with both operator flags true and requested/actual
determinism false on every update. Only its worker is registered with KFD at
that observation. The
[production audit](results/rx7900xtx-20261004-round3-switch/production-snapshot/production_audit.json)
records actual command/process identity, full-model parity, 32-batch initial
evaluation and the first verified save. The initial fixed NLL is **7.916505**
at source step 53143, before these new updates; it is not a quality comparison
between kernels. The long run and its final boundary audit remain active.

The published Hugging Face step-52616 weights remain the earlier immutable
snapshot; this operator switch does not change that release's provenance.

## Code and archived execution

[`round3_candidate_bench.py`](round3_candidate_bench.py) installs the two
process-local contexts after the native/reference-repeat captures and before
Trainer optimizer construction/load. It preserves the full-model gates and
explicitly forwards the 300-second/three-checkpoint defaults to the inherited
runner. Requested flags, actual update policy, operator calls and source hashes
are recorded; importing the modules installs no patches.

[`run_round3_switch.py`](run_round3_switch.py) owns only the children it launches,
validates source/code integrity, and starts production from the fixed recovery
source. On failure or interruption it reaps its child, while preserving verified
checkpoints. Its `--confirm-from` mode revalidates a prior default-policy
comparison and adds the controlled real-update arithmetic check. The local
controller adds strict source-SHA and consecutive-policy receipt gates after
the deployed version was frozen. The independent acceptance audit checks those
same facts for this executed run; the frozen live source was not rewritten.

[Raw evidence](results/rx7900xtx-20261004-round3-switch) includes the boundary
capture, handoff, default-policy timing/parity/metrics, failed native repeat,
two-step control, decision, deployment hashes and exact executed controller/entry
copies. Weight files are kept on the remote machine, not in Git.

The entry and controller pass **27** standard-library integration/lifecycle
tests. GPU execution supplies the full-model and native checkpoint checks above;
the integration tests do not substitute for those checks.
