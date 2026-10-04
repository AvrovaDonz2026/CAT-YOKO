# CAT-YOKO B0 recovery snapshot: step 52616

Saved on **2026-10-04T03:44:25.977204Z** during real-text training on an AMD RX 7900 XTX. This immutable step directory contains a complete B0 recovery overlay and a smaller weights-only variant. Both require frozen weights reconstructed from MiniCPM5-2B-Base; neither contains the full 12B graph.

| | |
| --- | --- |
| Phase / global step | B0 / 52616 |
| Native Adam counter / next unread packed cursor | 17814 / 17814 |
| Cumulative phase / total token clocks | 235,210,752; includes earlier DummyStream history |
| Actual real-text consumption | 72,966,144 packed input tokens |
| Trainable weights | 132 BF16 tensors; identical in both artifacts byte-for-byte |
| Optimizer in recovery file | Original CPU FP32 Adam; 132 states / 264 finite FP32 moment tensors, counters and groups |
| Additional recovery state | Python, torch and device RNG; saved model configuration, seed, gate and packed cursor |
| Prepared corpus | 79,998,976 train tokens / 999,424 validation tokens; sequence length 4096; EOS=1 |
| Actual pilot mixture | 60% English Ultra-FineWeb / 30% Chinese Ultra-FineWeb / 10% UltraData-Math L2; no code slice |
| Device and runtime | RX 7900 XTX `gfx1100`; PyTorch 2.9.1 + ROCm 6.4; BF16 |
| Training operators | Packed FP32 MATH attention; shared-storage frozen MoE; original CPU FP32 Adam |
| Code pin | [`341e0ed`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/341e0ed); deployed source hashes are also recorded in [release.json](release.json) |

The 8B-token B0 budget is unfinished, and this release does not enter B1. Round-three operator candidates were not installed for the updates represented by this snapshot.

## Files and verification

| File | Bytes | SHA256 |
| --- | ---: | --- |
| [`trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/trainable.pt) | 2,192,257,315 | `d4c4898be1cd248b2742bd9705a11de8af37003a0d209a91347498d47ec181df` |
| [`weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/weights-only.pt) | 438,477,743 | `a627dfa07a402fe6c3ed9cd5524c7b8cda518dffdfcd03ff3f32759d8a9a7570` |

`trainable.pt` contains native trainable weights, optimizer moments and recovery metadata. `weights-only.pt` contains the same weights and checkpoint metadata with no Adam state; loading it restarts the optimizer. CPU inspection verified all 132 weights and 264 moment tensors in the recovery file, including state mappings, finite values, counters and cursor. The weights-only export was compared against all 132 recovery weights byte-for-byte.

The required base is [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base). Its `model.safetensors` SHA256 is `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`. Base configuration, tokenizer, packed corpus and deployed source identities are recorded in [release.json](release.json). Use the repository's upcycling code and checkpoint configuration rather than `AutoModelForCausalLM.from_pretrained` on this model repository.

## Download and recover B0

Use the current repository checkout for the updated download aliases; the release manifest records the training code pin and exact deployed source hashes. The following paths are examples on a new machine:

```sh
export YOKO_WORK=/workspace/cat-yoko
python scripts/download_minicpm5.py --check-hash --local-dir "$YOKO_WORK/hf/MiniCPM5-2B-Base"
python scripts/download_hub_overlay.py --name b0-rocm-realtext-52616 --out-dir "$YOKO_WORK/hub/b0-rocm-realtext-step-52616"
```

Prepare the original pilot with the [real-text guide](https://github.com/AvrovaDonz2026/CAT-YOKO/blob/main/operators/rocm/REAL_TRAINING.md), then verify the training/evaluation bins and tokenizer files against the release manifest. This snapshot restores a cursor into a **19531-row** training bin with stride 1. The stream loader does not authenticate the corpus contents, so a different `.bin` does not provide the same data continuation merely because it has the same shape. The historical DummyStream and DeepSpeed launchers are not the complete recovery path for this native CPU Adam snapshot.

After verifying the original base and corpus, resume with the validated packed/shared-storage path:

```sh
python -m operators.rocm.candidate_bench \
  --base "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --resume "$YOKO_WORK/hub/b0-rocm-realtext-step-52616/trainable.pt" \
  --out "$YOKO_WORK/runs/b0-resume-step-52616" \
  --data "$YOKO_WORK/data/phase-b-real-20261002/train.bin" \
  --eval-data "$YOKO_WORK/data/phase-b-real-20261002/eval.bin" \
  --eos-id 1 --seq-len 4096 --parity-seqs 4096 \
  --moe-layout shared-storage --packed-attention \
  --deterministic-parity --reference-repeat \
  --run-steps 1717 \
  --save-optim --save-every 0 --save-every-seconds 300 --keep-last 3 \
  --eval-every 250 --eval-batches 32 --wait-gpu-idle
```

This runner reconstructs the MiniCPM5-upcycled graph, loads the overlay, validates deterministic native-reference outputs and all trainable gradients, and restores the original CPU FP32 Adam for B0. It does not enable the round-three candidates. Use a new output directory and the explicit source file. From this exact snapshot, **1717 additional updates** reach global step 54333 and the end of the first packed-corpus pass, without intentionally starting another pass. Keep the training data and checkpoint available during recovery. Timed saves occur after completed updates, include Adam/RNG/cursor and retain three numbered recovery points; update and write time can make spacing exceed 300 seconds.

For a smaller weights-only download:

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext-52616-weights --out-dir "$YOKO_WORK/hub/b0-rocm-realtext-step-52616-weights"
```

The downloader writes either selected artifact as local `trainable.pt`. Use separate output directories for the two variants. The smaller variant is suitable for loading weights and inspection; its missing moments prevent exact continuation of the saved optimizer trajectory.

## Held-out observation

The latest fixed evaluation before the saved snapshot ran at **step 52500**, giving NLL **7.900367058954696** over **32 batches / 130,877 valid next-token loss tokens**. Validation uses an independent bin; this is a finite sample of that bin, not a general generation/reasoning benchmark or a full validation-corpus score. The release weights were saved 116 updates after that evaluation. No later evaluation or B1 result is attributed to this snapshot.

Historical [B200 step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full) and [RTX 3090 step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) remain under their original paths. Code and derived weights use Apache-2.0.
