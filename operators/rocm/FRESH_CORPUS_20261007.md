# Fresh B0 corpus window — 2026-10-07

Fresh data preparation and its audits completed, and the new B0 window started.
The startup snapshot reached step **81945**, with **81** consecutive updates, at
**2026-10-07 16:04:25 UTC**, continuing from the complete native checkpoint at
step **81864**. The authorized endpoint is **85864**, using fresh training rows
**0–3999** once each: **16,384,000** training tokens at sequence length 4096.
The first timed recovery checkpoint passed its native CPU audit. Final quality
and the step-85864 publication remain pending.

The live remote run is `runs/fresh4000-81864-20261007T1550`, with frozen source
`source-fresh80m-20261007T1550`, under `/home/donz/cat-yoko-rocm-20261002`.
Implementation commit: `3428f4ea502a965c7f9632ec106043a518c7799d`.

## Data and provenance

The completed corpus contains **19,531 training rows / 79,998,976 tokens /
92,264 documents** and **244 validation rows / 999,424 tokens / 1,171 documents**,
from preparation targets of 80M training and 1M validation tokens. The
English/Chinese/math mixture retains the previous **60:30:10** weights and the
same tokenizer files, vocabulary and EOS ID. The output manifest, rather than
the rounded token target, records the completed row counts and SHA256 digests.

New named Parquet shards use the previous dataset revisions. Their official
Hugging Face file sizes and whole-file SHA256 identifiers are pinned in a source
lock. The default reader requests bounded HTTP ranges and checks HTTP 206,
exact `Content-Range` including the locked total size, identity encoding and
exact response length. It refuses a whole-file GET fallback. Reading selected
ranges **does not compute or prove the complete downloaded file's SHA256**;
`input_files_read[].full_file_verified` records that distinction. Complete local
shards, when supplied instead, must pass the official size and full SHA256 check.
The finished packed binaries and document hash ledgers receive full local
SHA256 checks.

Exact full-document SHA256 exclusion covers both old training and old validation
documents. All four new/old split intersections must be zero, and the fresh
training and fresh validation ledgers must also be disjoint. This establishes
exact-text exclusion; it does not detect near duplicates or semantic overlap.
The old corpus and its evidence remain immutable. Preparation publishes its
complete manifest only after checking the output files and rechecking the old
exclusion evidence and source lock.

## Explicit checkpoint handoff

The source checkpoint remains immutable. It carries all **132 trainable BF16
tensors**, native Adam state with **264 FP32 moment tensors**, parameter groups,
RNG state and global/phase clocks. B0, shared frozen MoE storage and the accepted
split-attention/cached-CPU-Adam operators continue. Learning rate and gate use
the existing clocks and schedule; neither is restarted for the new corpus.

The previous stream has absolute cursor **47062**, stride 1 and 19,531 old rows.
The new adapter explicitly maps that cursor to fresh row 0:

```text
fresh_row = absolute_cursor - 47062
initial: absolute_cursor 47062 -> fresh_row 0
final:   absolute_cursor 51062 -> next_fresh_row 4000
```

There is no modulo and no wrapping. The serialized stream binds the new corpus
SHA256, row count, cursor origin, logical row and handoff SHA256. Wrong corpus,
offset, cursor or identity fails with `RuntimeError`, including during trainer
restore. The first legacy source state is accepted only through this explicit
handoff. The original Adam counter **47062** remains continuous to **51062**.

| Counter | Source | Exact target |
| --- | ---: | ---: |
| Global update | 81864 | 85864 |
| Native Adam / absolute stream cursor | 47062 | 51062 |
| Phase tokens / total tokens seen | 355,010,560 | 371,394,560 |
| Real text tokens | 192,765,952 | 209,149,952 |
| New corpus next row | 0 | 4000 |

`data_handoff.json` records the immutable source checkpoint, old/new corpus and
fresh validation digests, origins and exact endpoint. Supervisor status uses
`data_handoff`, `data_handoff_path` and `data_handoff_sha256`. CPU native audits
check the complete checkpoint without resetting or rewriting its tensor state.
The worker additionally records which physical fresh rows training consumed.

This adapter and supervisor implement a **bounded 4000-update window**. A future
continuation or recovery needs a controlled resume that validates the saved
handoff and its corpus-local position. Feeding this checkpoint directly to the
legacy packed-stream path would lose the explicit mapping and is unsupported.

## Evaluation and recovery

Initial and final paired evaluations use three independent fixed slices:

1. Old validation rows 0–31: primary continuity comparison (`parity.json`).
2. Old validation rows 32–63: additional continuity comparison (`quality.json`).
3. Fresh validation rows 0–31: new-data comparison (`fresh_quality.json`).

All three initial evaluations completed at step **81864**, using 32 batches
each. These are this run's actual paired baselines:

| Initial validation slice | NLL | Valid tokens |
| --- | ---: | ---: |
| Old primary, rows 0–31 | 7.291744287131125 | 130,877 |
| Old additional, rows 32–63 | 7.388477264365619 | 130,878 |
| Fresh, rows 0–31 | 7.371492111944588 | 130,918 |

The initial old-data values differ slightly from the previous run's final
evaluations; comparisons for this window must use these new initial receipts.
The GPU numerical acceptance checks passed for all **132 trainable tensors**;
the reported output and gradient differences were zero. No final evaluation
receipt was available at the startup snapshot above.

Fresh evaluation restores training RNG, model mode and evaluation settings, and
does not advance the training stream. Completion requires all three receipts,
the existing all-132-tensor numerical operator checks, and the exact native
checkpoint endpoint. New quality numbers can only be reported after these
evaluations run; no result is implied by data preparation or this plan.

The worker saves complete native checkpoints every **300 seconds**, keeping the
last **three** numbered saves. The supervisor protects the immutable source and
a separately verified recovery hardlink. Its preflight requires free disk space
for five source-sized checkpoints plus 1 GiB for atomic writes and recovery.
Training waits for an idle GPU under the existing run lock and does not control
other training processes.

The first timed complete save is **step 81921**, audited at
**2026-10-07T16:02:21.836238Z**. Native Adam and absolute stream cursor are
**47119**, mapping to next fresh row **57**. Its 132 weights, 264 FP32 moments,
clocks and RNG schema passed CPU checks. The full file contains **2,192,257,635
bytes**, SHA256 `4f5052e4bc3f633888e579cb014c16a720260509e0b9161f99c32b598af2a330`.
This is a recorded rolling recovery point; later saves advance the latest link.
The [startup receipts](results/rx7900xtx-20261007-fresh4000/README.md) preserve
the checkpoint identity and the actual initial evaluations.

Weights stay on the remote host and Hugging Face; the local development machine
does not retain checkpoint weights. The final native/export publication enters
the HF publication queue after run completion and validation. The local watcher
is already waiting with the exact source, corpus-handoff SHA and endpoint;
successful Hub verification clears temporary local weight files. Remote cleanup
must retain recovery points until the relevant uploaded artifacts are verified.

## Launch interface

Select the actual frozen source and new run directories through `SOURCE` and
`RUN`; their timestamped identifiers may change during preparation. The run must
contain its immutable step-81864 checkpoint at `source_checkpoint/trainable.pt`
(a remote hardlink avoids an extra copy). Launch only after the fresh manifest is
complete and the CPU checks and disk preflight pass.

An earlier launch, `runs/fresh4000-81864-20261007T1510` with frozen source ending
`T1535`, stopped before model/GPU execution and made zero optimizer updates. Its
failure receipts remain intact. The wrapper now uses `allow_abbrev=False` so
native `--data` cannot be consumed as an abbreviation of `--data-handoff`;
supervisor-command round-trip and actual production-parser CPU tests cover this
failure. The live `T1550` run uses the corrected wrapper.

```sh
ROOT=/home/donz/cat-yoko-rocm-20261002
PY=/home/donz/revelation-rocm-venv/bin/python
: "${SOURCE:?Set the frozen source directory}"
: "${RUN:?Set the new fresh-window run directory}"

"$PY" -u "$SOURCE/operators/rocm/run_fresh_corpus_window.py" \
  --source-dir "$SOURCE" \
  --resume "$RUN/source_checkpoint/trainable.pt" \
  --base "$ROOT/hf/MiniCPM5-2B-Base" \
  --old-data "$ROOT/data/phase-b-real-20261002/train.bin" \
  --data "$ROOT/data/phase-b-fresh80m-20261007/train.bin" \
  --eval-data "$ROOT/data/phase-b-real-20261002/eval.bin" \
  --fresh-eval-data "$ROOT/data/phase-b-fresh80m-20261007/eval.bin" \
  --corpus-manifest "$ROOT/data/phase-b-fresh80m-20261007/manifest.json" \
  --out "$RUN" \
  --lock-file "$ROOT/run-locks/cat-yoko-b0.lock" \
  --accepted-run "$ROOT/runs/window4000-77864-20261006T1410" \
  --run-updates 4000 --python "$PY"
```
