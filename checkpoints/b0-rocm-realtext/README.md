# B0 real-text ROCm snapshots

The [model card on Hugging Face](https://huggingface.co/AvrovaDonz/CAT-YOKO)
identifies the latest completed and published snapshot. Each step directory
keeps its own weights, recovery metadata and SHA256 identities.
GitHub stores code and compact release receipts; weight files remain on Hub.

The newly verified release is [step 77864](step-77864/README.md), saved on
**2026-10-06T13:02:17.286196Z** after a completed 4000-update window.

| Artifact | Hub path | Use |
| --- | --- | --- |
| Complete recovery | [step-77864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/trainable.pt) | 2,192,257,315 bytes; 132 BF16 weights, 264 native CPU FP32 Adam moments, RNG and absolute cursor |
| Weights only | [step-77864/weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/weights-only.pt) | 438,478,163 bytes; the same 132 weights byte-for-byte, without optimizer history |
| Release identities | [step-77864/release.json](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/release.json) | Artifact, base, tokenizer, data, code archive and complete-run checks |

These are native B0 trainable overlays. Reconstruct the frozen graph by
upcycling MiniCPM5-2B-Base before loading either file. The complete 12B graph
is not uploaded; the weights-only file cannot restore the saved Adam trajectory.

The absolute packed cursor and native Adam counter are **43062**, next
physical row **4000 / 19531**. Actual real-input consumption is
**176,381,952 tokens**, including repetitions of the **79,998,976-token**
pilot. Phase clocks total **338,626,560** and include earlier DummyStream
history. The 8B-token B0 envelope remains unfinished; B1 has not started.

Final fixed NLL is **7.3636053664583905** on validation rows 0–31;
additional rows 32–63 score **7.503450781393409**. Both use 32 batches from
the same validation file, and neither is an external capability benchmark.

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext-77864 --out-dir /workspace/hub/step-77864
python scripts/download_hub_overlay.py --name b0-rocm-realtext-77864-weights --out-dir /workspace/hub/step-77864-weights
```

The downloader writes either selected file as local `trainable.pt`; use
separate directories for the two variants. Explicit step aliases remain
stable. Consult the Hub model card for releases newer than this checkout's
pinned default. Complete original corpus/base/tokenizer identities and native
CPU Adam are required to preserve the meaning of the packed cursor.
See the [release card](step-77864/README.md) and
[real-text recovery guide](../../operators/rocm/REAL_TRAINING.md).

The operators remain document-split packed FP32 attention, shared frozen MoE
storage and cached BF16 CPU-shadow Adam with retained native FP32 moments.
The [operator switch report](../../operators/rocm/OPERATOR_SWITCH_20261004.md)
records the original numerical/recovery acceptance and user-requested
adoption after a 2.55–3.03% short-window gain. No new speed claim is made.

Historical [step 53307](step-53307/README.md),
[step 52616](step-52616/README.md),
[B200 step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full)
and [RTX 3090 step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16)
retain their original artifacts and hashes.
