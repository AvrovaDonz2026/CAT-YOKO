# Real-text B0 continuation, October 2, 2026

These files record completed data preparation and the five-update real-text
check before a bounded 24-hour continuation. They do not record completion of
the long run. No corpus text, model weights, or optimizer tensors are checked in.

| Record | Evidence |
| --- | --- |
| `manifest.json` | Fixed source revisions and eligible shards; tokenizer-file hashes; 79,998,976 training and 999,424 held-out tokens; selected document hashes have zero intersection |
| `data_verification.json` | Independent supervisor verification of bin SHA256, sidecars, document-hash files, and train/validation separation |
| `parity.json` | Full deterministic real-data comparisons at 64 and 4096; native 4096 reference repeat; all 132 trainable gradients and sampled outputs match exactly; five completed updates |
| `metrics.jsonl` | Steps 34803–34807; median loop throughput 582.49 tokens/s; final training NLL 10.969116 |
| `smoke_checkpoint.json` | Saved step 34807; all 132 trainable tensors finite; 132 distinct Adam states, 264 finite CPU FP32 moments; packed cursor `i=5` |
| `long-start.json` | Live snapshot at 2026-10-02T14:41:11Z: long run had reached 34828; restored source contains Adam; 4096 real-data parity passes exactly; initial 32-batch validation NLL 10.743914 over 130877 valid tokens |

The source step-34802 overlay contains no optimizer, so Adam starts fresh for
these five updates. Subsequent continuation restores this new optimizer state.
The same four held-out batches contain **16,362 valid loss tokens**: NLL
**11.531286 → 10.758360**. Five updates and a small fixed held-out sample do not
establish broad language capability or a completed training recipe.

The 4096-token forward/backward comparison took **31.4732 → 6.8177 seconds**,
including the change from block offload to resident shared storage. It is not
an isolated kernel speed measurement. Deterministic mode applies only to the
parity comparison and is restored before training. CPU Adam and global gradient
clipping remain enabled. The five-update loop took 80.82 seconds including
saving a roughly 2 GiB optimizer-bearing overlay every update; those smoke-save
overheads are excluded from the per-step `tok_s` values. The long run saves
every 100 updates instead.

CPU validation included the optimizer, checkpoint, additional-update,
deadline/prefetch, trainer, and real-data entry regressions. The initial suite
passed 106 tests and 28 subtests; after restore-reporting and corpus additions,
the affected suite passed 90 tests and 16 subtests. These suites overlap and
must not be added together. The finalized corpus and supervisor tests also
passed 18 standard-library tests locally.

[Continuation guide](../../../operators/rocm/REAL_TRAINING.md) explains the
data mixture, exact-text split, partial HTTP verification, wall-clock scope,
optimizer recovery, and remote logs.
