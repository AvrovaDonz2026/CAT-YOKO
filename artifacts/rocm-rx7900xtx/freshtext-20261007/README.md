# Fresh B0 text prepared on 2026-10-07

The remote preparation and independent CPU audit completed. The original
tokenizer, 4096-token int32 packing and English/Chinese/math weights 60:30:10
are retained. No text binaries or checkpoint weights are stored here.

| Output | Tokens | Rows | Documents | SHA256 |
| --- | ---: | ---: | ---: | --- |
| Training | 79,998,976 | 19,531 | 92,264 | `1391928b2d01abd3b2c35f1b708619e3d352fa6bdbf8e222296c591c2f02bb63` |
| Validation | 999,424 | 244 | 1,171 | `504c29980b65ee62ee6c16d379451cb5e09b095995397f1cab35669282b6f597` |

[manifest.json](manifest.json) records the source revisions, packing recipe,
tokenizer hashes and finished output identities. [source-lock.json](source-lock.json)
pins the official size and SHA256 of eligible new shards 3 and 4; preparation
reached its quotas using shard 3 of each source. Public mirror HTTP ranges were
read with the bounded retrying `requests` reader. Complete source Parquet SHA256
was not recomputed; the packed output binaries were hashed in full.

[audit.json](audit.json) independently rechecks both output digests, legal token
IDs and all five document-hash intersections: new train/new eval, and each new
split against both old splits. All are zero. Exclusion removed 357 documents
matching old training and 2 matching old validation before tokenization. These
checks cover exact full text; near duplicates and semantic overlap are not tested.

The [fresh-window guide](../../../operators/rocm/FRESH_CORPUS_20261007.md) describes
the 81864-to-85864 continuation, explicit row mapping and three paired evaluations.
