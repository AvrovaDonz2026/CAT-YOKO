# Real-text B0 continuation on ROCm

Continue the native B0 overlay with `shared-storage`, preserving routing and
the original frozen expert arithmetic. Real packed text participates in the
deterministic loss, output, and all-trainable-gradient comparison before any
optimizer update. This is separate from the earlier DummyStream performance
experiment and the rejected compute-collapse candidate.

The October 2 continuation starts from the completed **step 34802** overlay.
Keep the original overlay and MiniCPM5 base files available for recovery.

## Data

Prepare a bounded raw-text corpus with the local MiniCPM5 tokenizer:

```sh
python -m operators.rocm.prepare_real_data \
  --tokenizer-dir "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --out-dir "$YOKO_WORK/data/phase-b-real-20261002" \
  --endpoint https://hf-mirror.com
```

The default endpoint is Hugging Face. The optional mirror changes only the
download host; the manifest retains the official source URLs and revisions.
HTTP range reading verifies the official file size and records that a full
file SHA256 was not verified. Locally supplied Parquet shards must match both
the pinned size and the official LFS SHA256.

This run uses **60% English Ultra-FineWeb, 30% Chinese Ultra-FineWeb, and 10%
UltraData-Math L2**, allocated by tokens in complete 4096-token rows. This is
a web/math pilot rather than the published recipe's 5% StarCoder mixture.
Only two explicitly named shards per source are eligible; the reader stops
when the quotas are met. The actual totals are **79,998,976 training tokens**
and **999,424 validation tokens**. Sources and file identities are defined in
`prepare_real_data.py` and copied into each generated manifest.

Normalize complete documents with `str.strip()` and split using
`int(SHA256(text_utf8), 16) % 1000 < 10` for validation. Global exact-text
deduplication uses the same hash without a source salt. The two selected hash
sets must have no intersection. This checks exact duplicates; near duplicates
remain possible. Each split is packed separately, with EOS=1 and no chat
template or automatic BOS. Added special tokens are included in the full
130560-token vocabulary check. Row locations are shuffled using a fixed seed.

The data directory contains the source manifest, tokenizer-file hashes,
token-bin hashes, document-hash evidence, and preparation status. Keep generated
corpora and checkpoints outside the source repository.

## Start and resume

`run_real_longtrain.py` first validates the completed corpus, runs five real-text
updates, checks the saved optimizer state, then resumes into the longer run.

```sh
python -m operators.rocm.run_real_longtrain \
  --work-dir "$YOKO_WORK" \
  --python /home/donz/revelation-rocm-venv/bin/python \
  --base "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --data-dir "$YOKO_WORK/data/phase-b-real-20261002" \
  --resume "$YOKO_WORK/runs/b0-shared-storage-deterministic/train/trainable.pt" \
  --out "$YOKO_WORK/runs/b0-realtext-24h-20261002" \
  --max-hours 24
```

The wall-clock cap applies to the long training loop, including its periodic
validation and checkpoint overhead, after preparation, parity, and the smoke
run. It stops at the next completed optimizer update and saves the correct
next unread data cursor. A single update or validation can overrun the cap.
The additional update limit also prevents repeating this corpus. B0's 8B-token
gate and learning-rate schedule stay intact; this run does not enter B1/B2.

The source step-34802 overlay contains no optimizer, so Adam starts fresh for
the five-update smoke run. New overlays include CPU FP32 moments and their
step counters. The long run restores those moments and the packed-data cursor.
An incompatible supplied optimizer is an error when optimizer saving is
requested. Frozen base weights are reconstructed rather than serialized into
each overlay. Use the explicit final `trainable.pt` path for future resumes.

The run launched on October 2 saves every 100 updates and retains three numbered checkpoints,
evaluates 32 fixed validation batches every 250 updates, and records initial
and final held-out NLL. Periodic evaluation samples **131072 input tokens**
from the independent validation bin; valid loss tokens exclude boundaries.
Logs report valid token counts, actual throughput, and peak memory. Packed
multi-document masks can change both speed and memory compared with DummyStream;
the earlier 4.4x result is not a promised real-text speedup.

## Five-minute checkpoint saves

New launches of the supervisor default to `--save-every-seconds 300 --keep-last 3`.
The existing running process cannot change its interval through a source edit;
the new setting takes effect on the next launch or continuation.

For a continuation from a real-text checkpoint with Adam, use `model_bench.py`
with the normal base/data/evaluation arguments and add:

```sh
--save-every 0 --save-every-seconds 300 --keep-last 3 --save-optim
```

Timed saves happen after completed optimizer updates. The interval starts after
the previous checkpoint successfully finishes, so update and I/O time can make
the wall-clock spacing longer than five minutes. The prefetch cursor and RNG are
captured at the same boundary as step-based saves. CPU FP32 Adam moments and
step counters remain in each overlay.

Numbered files are completed before a temporary hardlink or copy atomically
replaces `trainable.pt`/`latest.pt`. Only then does retention remove older
numbered checkpoints from the same save directory. A failed save or publication
keeps the previous latest checkpoint and does not advance the save timer.
Timed final saves also use numbered files, so they enter the same retention
policy; the latest hardlink does not consume another checkpoint's storage.
Source checkpoints in other directories and the base model are not reclaimed.
Timed saving currently supports one training process.

The supervisor records process IDs and stage status under its output directory,
with separate smoke and long-run logs. Failed preparation or parity prevents
training; it never substitutes random data. Other GPU processes are observed
by the existing idle wait and are not suspended or stopped.
