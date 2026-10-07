# Completed B0 step 81864

Global step **81864**, Adam counter / absolute packed cursor **47062**. Next corpus row **8000**; 79,998,976 unique training tokens, 192,765,952 packed inputs including repeated passes. B0 continues; this is not B1.

`trainable.pt` is the native trainable overlay plus optimizer, RNG and cursor: 2,192,257,315 bytes, SHA256 `20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc`. `weights-only.pt` has identical 132 BF16 weights and no Adam history: 438,478,163 bytes, SHA256 `74d8a60b9e43613e917a5debd52d27776604ea69e1564e4a891b4f99dd906085`.

Reconstruct the frozen 12B graph by MiniCPM5-2B-Base upcycling. These files are overlays. The complete recovery snapshot contains 264 CPU FP32 Adam moments; native load/export and light-weight byte comparisons pass. GPU RNG is serialized and schema-checked on CPU; this publisher does not restore GPU state.

Final primary held-out NLL: **7.29112438273621** on rows 0–31; additional NLL: **7.387959246989646** on rows 32–63. Both are slices of the same pilot validation corpus, not external benchmarks.

See `release.json` for verified base/data/source identities and the immutable `source/` archive. Restore the packed absolute cursor without resetting Adam or RNG; streams read `i % nseq`. The deployed `source/operators/rocm/production_continuation.py` retains the accepted packed reference, split attention, shared frozen storage and cached CPU Adam. Its `--accepted-run` receipt references the archived complete-run evidence. Complete original run directories and corpora remain necessary for that supervisor provenance check; the standard checkpoint loader can restore the overlay from the recorded base, configuration and native Adam.
