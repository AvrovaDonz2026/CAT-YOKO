# B0 real-text ROCm snapshots

The recommended release is [step 52616](step-52616/README.md), saved on **2026-10-04T03:44:25.977204Z**. Weights are hosted at [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-rocm-realtext); GitHub stores these cards and release metadata.

| Artifact | Hub path | Use |
| --- | --- | --- |
| Recovery snapshot | [`step-52616/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/trainable.pt) | 132 BF16 overlay weights plus CPU FP32 Adam, counters, RNG and packed cursor; 2,192,257,315 bytes |
| Smaller overlay | [`step-52616/weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/weights-only.pt) | The same 132 weights, verified byte-for-byte, without Adam; 438,477,743 bytes |
| Manifest | [`step-52616/release.json`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/release.json) | Artifact, source, base, tokenizer, data and recovery identities |

These are B0 trainable overlays. Loading either requires the repository's CAT-YOKO graph upcycled from **MiniCPM5-2B-Base**; the frozen 12B graph is not serialized into the files. The recovery file preserves native CPU Adam state. The smaller overlay is useful when only weights are needed and does not provide complete optimizer recovery.

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir /workspace/hub/b0-rocm-realtext-step-52616
# Weights-only alternative:
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir /workspace/hub/b0-rocm-realtext-step-52616-weights
```

The downloader writes the selected artifact as local `trainable.pt`; select a separate output directory for each variant. Follow the [step card](step-52616/README.md) for hashes and the explicit base/data/evaluation recovery command.

At this snapshot, global step is **52616**, native Adam counter and next unread packed cursor are **17814**, and actual packed training consumption is **72,966,144 tokens**. Cumulative phase and total clocks are **235,210,752 tokens**, including earlier DummyStream history. The bounded pilot contains 79,998,976 train / 999,424 validation tokens with a 60% English web / 30% Chinese web / 10% L2 math mixture and no code slice. B0's planned 8B-token budget is unfinished; B1 has not started in this release.

The runtime uses RX 7900 XTX `gfx1100`, PyTorch 2.9.1 + ROCm 6.4, BF16, packed FP32 MATH attention, shared-storage frozen MoE and original CPU FP32 Adam. Round-three operator candidates are not installed in the published snapshot.

Historical [`b0-full` step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full) and [`b0-3090-bf16` step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) remain available. Their DummyStream launcher instructions describe those historical releases; use the real-text recovery recipe for this one.
