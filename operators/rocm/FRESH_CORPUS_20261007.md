# Fresh B0 corpus window — 2026-10-07

At the time this note was written, fresh data preparation was in progress. The
4000-update training window had not started, and no fresh-corpus quality result
was available. The authorized experiment continues B0 from the complete native
checkpoint at step **81864** to step **85864**, using fresh training rows
**0–3999** once each: **16,384,000** training tokens at sequence length 4096.

## Data and provenance

The preparation target is 80M training tokens and 1M validation tokens. Complete
4096-token rows yield **19,531 training rows / 79,998,976 tokens** and
**244 validation rows / 999,424 tokens** when those targets are reached. The
English/Chinese/math mixture retains the previous **60:30:10** weights and the
same tokenizer files, vocabulary and EOS ID. The output manifest, rather than
this target, records the actual completed row counts and SHA256 digests.

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

Weights stay on the remote host and Hugging Face; the local development machine
does not retain checkpoint weights. The final native/export publication enters
the HF publication queue after run completion and validation. Remote cleanup
must retain recovery points until the relevant uploaded artifacts are verified.

## Launch interface

Select the actual frozen source and new run directories through `SOURCE` and
`RUN`; their timestamped identifiers may change during preparation. The run must
contain its immutable step-81864 checkpoint at `source_checkpoint/trainable.pt`
(a remote hardlink avoids an extra copy). Launch only after the fresh manifest is
complete and the CPU checks and disk preflight pass.

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
