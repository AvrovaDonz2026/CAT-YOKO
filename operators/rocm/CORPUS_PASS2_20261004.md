# B0 continuation over a second pilot-corpus pass (2026-10-04)

The first packed real-text pass completed successfully at global step **54333**.
The user requested continued training. A separate supervisor continues from that
complete checkpoint to **73864**, without resetting the optimizer, RNG, packed
cursor or phase clocks. This is a second traversal of the same pilot, in the
same order; it does not add new unique training data or begin B1.

## Fixed recovery state

The preceding run is `runs/round3-confirm-20261004T0455`. Its supervisor exited
normally, recorded `status=complete`, and verified all 132 BF16 trainable tensors
and all 264 retained CPU FP32 Adam moments at the exact first-pass boundary.
The final fixed held-out NLL is **7.873593728100402**, measured at **54333** over
32 batches / 130877 valid loss tokens. This is the pilot validation set, not a
general language-quality benchmark.

The new run hardlinks the completed numbered checkpoint into its own immutable
`source_checkpoint/trainable.pt`. Its SHA256 is
`4966dcddbeb322b3f4e1b75e86cce66d3c5c95402346b853d4dee449a8fd796f`.
The new frozen source copies the preceding model and training implementation.
The supervisor and its audit helpers support an explicit absolute cursor
limit. A separate continuation entry selects the previously deployed packed
attention as the reference backend. Training mathematics are unchanged; the
preceding frozen sources and the failed startup are preserved.

| Counter | Source | Planned second-pass endpoint |
| --- | ---: | ---: |
| Global step | 54333 | 73864 |
| Packed cursor and native Adam counter | 19531 | 39062 |
| Tokens in phase / tokens seen | 242243584 | 322242560 |
| Real-text input tokens consumed, including repetition | 79998976 | 159997952 |
| Unique prepared real-text training tokens | 79998976 | 79998976 |
| Completed corpus passes | 1 | 2 |

The native packed stream already reads `i % nseq` and saves monotonically
increasing `i`. There is no cursor reset or data-loader change. Its 19531 rows
have width 4096 and stride 1. Train and evaluation hashes were recomputed and
match the previous release:

- `train.bin`: `0eb30dddd807f8b6aae74ebda0be655b66dae1bf8ceff2b21c5e40ff1095fca7`.
- `eval.bin`: `22b6312c4e426e49638fe6bbd57a1687ac2b6ee285515259de445c9473307fa6`.

## Running policy and recovery checks

The supervisor is `runs/pass2-54333-20261004T1045/status.json`; its frozen source
is `source-pass2-20261004T1045`. It waits for two idle KFD observations before
creating its own worker. A process lock prevents duplicate launches using the
same lock. It does not signal other GPU processes. The detached supervisor
continues after SSH disconnects and cleans up its own worker on failure or
interruption.

The worker retains document-split packed FP32 attention, shared frozen MoE
storage, and cached BF16 CPU-shadow Adam with native retained CPU FP32 moments.
Production keeps `deterministic_algorithms=False`; deterministic reference
checks are separate from the production updates. The existing 8B B0 token
budget, gate schedule and learning-rate clock continue from the saved state.
This run has no wall-clock cap and requests exactly **19531** additional updates.

Complete checkpoints save every **300 seconds**, retaining the latest **three**
periodic files. The fixed source and one independently verified recovery copy
remain outside that rolling set. Evaluation uses the original held-out data
every **250** updates, with **32** batches.

The new audit accepts a repeated pass only through explicit
`cpu_check(..., max_cursor=39062)`. Its default behavior still forbids first-pass
wrap. The explicit limit must be a positive integer whole-corpus multiple.
Source, requested window and each saved cursor must remain inside that limit.
All existing weight, moment, finite-value, nonnegative squared-moment,
optimizer-group, mapping, model-config, token and update-delta checks remain.

The supervisor requires consecutive update metrics starting at **54334**, both
new operator flags, CPU Adam, and the original production determinism policy.
Each periodic checkpoint is hardlinked before full CPU inspection; the verified
recovery copy is replaced only after inspection succeeds. Completion requires
the worker to exit successfully, full model parity and operator provenance,
all **19531** consecutive updates, and exact final step/cursor/Adam/token
counters. An incomplete or failed run is never labelled complete.

The published Hugging Face release remains immutable **step 53307**; the newer
54333 recovery state is currently remote. See [STATUS](../../docs/STATUS.md) for
the observed live progress and [the operator switch](OPERATOR_SWITCH_20261004.md)
for the original acceptance evidence and measured 2.55–3.03% whole-update gain.

## Validation

The first startup attempt, `runs/pass2-54333-20261004T1030`, failed the unchanged
dense-native parity gate before any training updates. Only
`decoder.0.cross_attn.q_norm.weight` and `k_norm.weight` exceeded the 0.05
per-tensor threshold, at **0.07754124371811304** and **0.07619966055779656**.
Global gradient relative L2 was **0.012760013312614109**, loss difference
**0.0010967254638671875**, and selected-output relative L2 **0.011346673668858167**.
The native repeat passed all 132 gradients exactly. This is a real failed gate,
retained under `failed-native-attempt/` in the evidence; it is not relabelled.

The recovery entry compares against the **previous packed production backend**.
During the independent offloaded reference captures, packed attention is
temporarily installed, without split attention or cached Adam. It is then
removed before the existing candidate installer applies shared MoE storage,
packed/split attention and cached CPU Adam. Thus the comparison still checks
the layout and split optimization against an independent predecessor, rather
than comparing the new implementation with itself. The reference backend,
actual call counts and source hashes are recorded explicitly. The failed
dense-native receipt is bound to the identical fixed source checkpoint.
The original 132 per-tensor/global gradient, output and loss thresholds remain.
Passing the production-reference check does not claim that the dense-native
comparison passed on this row or that packed attention is byte-exact to dense
attention for every model input.

Local supervision, audit and repository tests pass: **48** tests, with six
Torch-dependent fixtures skipped on the local machine. Remote CPU testing
passes all **32** audit and round-three tests, including those real Torch
checkpoint fixtures; all **11** continuation-entry and pass-supervisor tests
also pass remotely. A regression
test injects failure at the first running-state write and verifies that the
already-created worker is cleaned up. Source Python and CPU Torch RNG states
restore successfully, and the saved GPU RNG tensors pass schema checks; the
native trainer restores them on the GPU.

The production-reference GPU comparison passes all **132** gradients with
**zero** per-tensor and global relative L2 error. Loss difference, selected
logit error and final-hidden error are also zero. Both independent packed
reference captures are exercised, and their contexts restore before candidate
installation. Initial fixed held-out NLL is **7.872862032828506** at source
step **54333**, over the same 32 batches / 130877 valid loss tokens.

At **2026-10-04T11:11:22.228406+00:00**, the worker has completed **67** real
updates through **54400**, recording both new operator flags, CPU Adam and
the ordinary production policy on every update. Its initial gate and LR
continue from the checkpoint clocks. Supervisor **21834** and worker **21873**
are alive; only the worker is registered on KFD at this observation.

The first periodic checkpoint is **54388**, after **55** new updates, with
absolute cursor and Adam counter **19586**. Its complete file is **2192257315**
bytes, SHA256
`89765a12dbe7b4177216e3961efb84526f9e1cca0ff0823cee480235fd57c3b5`.
The supervisor's full CPU state audit passes. A separate fixed hardlink also
passes native CPU Adam load/export with all **264** FP32 moments byte-exact,
unchanged optimizer groups, successful Python and CPU Torch RNG restoration,
and valid saved GPU RNG tensors. This audit allocates no GPU memory and does
not claim to exercise GPU RNG restoration itself. The rolling save policy
continues; this first inspected recovery point is additionally retained.

All launch, completion, failed-native and runtime receipts are archived in
[the evidence directory](results/rx7900xtx-20261004-corpus-pass2/README.md).
The archived live files are observations of a continuing run, not a final
second-pass completion claim.
