# Fresh B0 startup evidence

At **2026-10-07T16:04:25.594763Z** (October 8, 00:04:25 Asia/Shanghai), the
independent **81864 → 85864** window had completed **81** consecutive updates
through **81945**. These are live startup receipts, not completion or final
quality results. No weights or corpus binaries are included.

[startup/snapshot.json](startup/snapshot.json) binds the captured file sizes and
SHA256 identities. Metrics are the complete JSONL prefix at capture; training
continues beyond that prefix. The archived source deployment pins implementation
commit `3428f4ea502a965c7f9632ec106043a518c7799d`. The
[new corpus audit](../../../../artifacts/rocm-rx7900xtx/freshtext-20261007/README.md)
rechecks complete packed bins and all exact-document intersections on the host.

| Initial evaluation at 81864 | NLL | Valid tokens |
| --- | ---: | ---: |
| Old rows 0–31 | 7.291744287131125 | 130,877 |
| Old rows 32–63 | 7.388477264365619 | 130,878 |
| Fresh rows 0–31 | 7.371492111944588 | 130,918 |

All three use 32 batches. The fresh and additional evaluations restore RNG,
model mode and evaluation settings without consuming training rows. Numerical
preflight passed all 132 gradient tensors with zero reported output/gradient
error against the preceding packed-production reference. No new operator
speedup is claimed.

[startup/first_checkpoint_receipt.json](startup/first_checkpoint_receipt.json)
records the first complete save, **81921**, verified at
**2026-10-07T16:02:21.836238Z**: native Adam/cursor **47119**, next fresh row
**57**, 132 BF16 weights, 264 CPU FP32 moments and RNG schema. The audited file
was **2,192,257,635 bytes**, SHA256
`4f5052e4bc3f633888e579cb014c16a720260509e0b9161f99c32b598af2a330`.
Its latest hardlink advances with subsequent five-minute saves; keep3 applies.

[startup/hf-queue](startup/hf-queue) records the local publication watcher
waiting for validated completion at 85864. Credentials remain local. Publication
requires the final native audit, all three paired evaluations and the exercised
fresh-row receipt; verified temporary local weights are then cleared.

[zero-update-failure](zero-update-failure) preserves the previous launch stopped
by an argument-prefix parsing error before model execution or optimizer updates.
The corrected wrapper disables argument abbreviation, with actual production
parser and 17 remote CPU checks passed. Neither that failure nor historical
dense-native failures were relabelled successful.
