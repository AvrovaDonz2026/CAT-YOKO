# B0 real-text ROCm snapshots

The [model card on Hugging Face](https://huggingface.co/AvrovaDonz/CAT-YOKO)
identifies the latest completed and published snapshot. Each step directory
keeps its own weights, recovery metadata and SHA256 identities.
GitHub stores code and compact release receipts; weight files remain on Hub.

The newly verified release is [step 81864](step-81864/README.md), saved on
**2026-10-06T20:58:48.075579Z** after a completed 4000-update window.

| Artifact | Hub path | Use |
| --- | --- | --- |
| Complete recovery | [step-81864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/trainable.pt) | 2,192,257,315 bytes; 132 BF16 weights, 264 native CPU FP32 Adam moments, RNG and absolute cursor |
| Weights only | [step-81864/weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/weights-only.pt) | 438,478,163 bytes; the same 132 weights byte-for-byte, without optimizer history |
| Release identities | [step-81864/release.json](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/release.json) | Artifact, base, tokenizer, data, code archive and complete-run checks |

Full recovery SHA256: `20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc`.
Weights-only SHA256: `74d8a60b9e43613e917a5debd52d27776604ea69e1564e4a891b4f99dd906085`.

These are native B0 trainable overlays. Reconstruct the frozen graph by
upcycling MiniCPM5-2B-Base before loading either file. The complete 12B graph
is not uploaded; the weights-only file cannot restore the saved Adam trajectory.

The absolute packed cursor and native Adam counter are **47062**, next
physical row **8000 / 19531**. Actual real-input consumption is
**192,765,952 tokens**, including repetitions of the **79,998,976-token**
pilot. Phase clocks total **355,010,560** and include earlier DummyStream
history. The 8B-token B0 envelope remains unfinished; B1 has not started.

Final fixed NLL is **7.29112438273621** on validation rows 0–31;
additional rows 32–63 score **7.387959246989646**. Both use 32 batches from
the same validation file and have been observed in prior windows;
neither is an external capability benchmark.

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext-81864 --out-dir /workspace/hub/step-81864
python scripts/download_hub_overlay.py --name b0-rocm-realtext-81864-weights --out-dir /workspace/hub/step-81864-weights
```

The default `b0-rocm-realtext` and `b0-rocm-realtext-weights` aliases also
select 81864. The downloader writes either selected file as local
`trainable.pt`; use separate directories for the two variants. Explicit step
aliases remain stable. Complete original corpus/base/tokenizer identities
and native CPU Adam are required to preserve the meaning of the packed cursor.
See the [release card](step-81864/README.md) and
[real-text recovery guide](../../operators/rocm/REAL_TRAINING.md).

The operators remain document-split packed FP32 attention, shared frozen MoE
storage and cached BF16 CPU-shadow Adam with retained native FP32 moments.
Complete saves remained every 300 seconds / keep3 throughout the completed
4000-update window. The [operator switch report](../../operators/rocm/OPERATOR_SWITCH_20261004.md)
records the original numerical/recovery acceptance and user-requested
adoption after a 2.55–3.03% short-window gain. No new speed claim is made.

Historical [step 77864](step-77864/README.md),
[step 53307](step-53307/README.md),
[step 52616](step-52616/README.md),
[B200 step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full)
and [RTX 3090 step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16)
retain their original artifacts, hashes and explicit downloader aliases.
