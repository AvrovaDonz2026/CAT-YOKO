# B0 real-text ROCm snapshots

The recommended release is [step 53307](step-53307/README.md), saved on **2026-10-04T05:37:22.748161Z**. Weights are hosted at [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-rocm-realtext); GitHub stores these cards and release metadata.

| Artifact | Hub path | Use |
| --- | --- | --- |
| Recovery snapshot | [`step-53307/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/trainable.pt) | 132 BF16 overlay weights plus CPU FP32 Adam, counters, RNG and packed cursor; 2,192,257,315 bytes |
| Smaller overlay | [`step-53307/weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/weights-only.pt) | The same 132 weights, verified byte-for-byte, without Adam; 438,477,679 bytes |
| Manifest | [`step-53307/release.json`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/release.json) | Artifact, source, base, tokenizer, data and recovery identities |

These are B0 trainable overlays. Loading either requires the repository's CAT-YOKO graph upcycled from **MiniCPM5-2B-Base**; the frozen 12B graph is not serialized into the files. The recovery file preserves native CPU Adam state. The smaller overlay is useful when only weights are needed and does not provide complete optimizer recovery.

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir /workspace/hub/b0-rocm-realtext-step-53307
# Weights-only alternative:
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir /workspace/hub/b0-rocm-realtext-step-53307-weights
```

The downloader writes the selected artifact as local `trainable.pt`; select a separate output directory for each variant. Follow the [step card](step-53307/README.md) for hashes and the explicit base/data/evaluation recovery command.

At this snapshot, global step is **53307**, native Adam counter and next unread packed cursor are **18505**, and actual packed training consumption is **75,796,480 tokens**. Cumulative phase and total clocks are **238,041,088 tokens**, including earlier DummyStream history. The bounded pilot contains 79,998,976 train / 999,424 validation tokens with a 60% English web / 30% Chinese web / 10% L2 math mixture and no code slice. B0's planned 8B-token budget is unfinished; B1 has not started in this release.

The runtime uses RX 7900 XTX `gfx1100`, PyTorch 2.9.1 + ROCm 6.4, BF16, document-split packed FP32 MATH attention, shared-storage frozen MoE and cached BF16 CPU-shadow Adam. Native CPU FP32 moments and checkpoint format remain unchanged; production keeps `deterministic_algorithms=False`. The [switch report](../../operators/rocm/OPERATOR_SWITCH_20261004.md) records all-132-gradient gates, two controlled byte-exact real updates and the user-requested adoption below the 5% automatic speed gate. The measured short-window whole-update gain was 2.55–3.03%, not a corpus-wide result.

The latest fixed evaluation was NLL **7.885217772929435** at **53250**, 57 updates before this snapshot, over 32 batches / 130877 valid loss tokens. From the saved cursor, **1026** updates complete the first 19531-row pass at global step 54333 without wrapping; the step card supplies the new operator recovery command with 300-second saves / keep3 and evaluation every 250 updates over 32 batches.

Historical [ROCm step52616](step-52616/README.md) retains both original artifacts, hashes and its preceding packed-attention/native CPU Adam provenance. The explicit download aliases `b0-rocm-realtext-52616` and `b0-rocm-realtext-52616-weights` remain available.

Historical [`b0-full` step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full) and [`b0-3090-bf16` step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) remain available. Their DummyStream launcher instructions describe those historical releases; use the real-text recovery recipe for this one.
