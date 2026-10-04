# CAT-YOKO B0 recovery snapshot: step 53307

Saved on **2026-10-04T05:37:22.748161Z** during real-text training on an AMD RX 7900 XTX. This immutable step directory contains a complete B0 recovery overlay and a smaller weights-only variant. Both require frozen weights reconstructed from MiniCPM5-2B-Base; neither contains the full 12B graph.

| | |
| --- | --- |
| Phase / global step | B0 / 53307 |
| Native Adam counter / next unread packed cursor | 18505 / 18505 |
| Cumulative phase / total token clocks | 238,041,088; includes earlier DummyStream history |
| Actual real-text consumption | 75,796,480 packed input tokens |
| Trainable weights | 132 BF16 tensors; identical in both artifacts byte-for-byte |
| Optimizer in recovery file | Native CPU FP32 Adam; 132 states / 264 finite FP32 moment tensors, counters and groups |
| Additional recovery state | Python, torch and device RNG; saved model configuration, seed, gate and packed cursor |
| Prepared corpus | 79,998,976 train tokens / 999,424 validation tokens; sequence length 4096; EOS=1 |
| Actual pilot mixture | 60% English Ultra-FineWeb / 30% Chinese Ultra-FineWeb / 10% UltraData-Math L2; no code slice |
| Device and runtime | RX 7900 XTX `gfx1100`; PyTorch 2.9.1 + ROCm 6.4; BF16 |
| Training operators | Document-split packed FP32 MATH attention; shared-storage frozen MoE; cached BF16 CPU-shadow Adam with native CPU FP32 moments |
| Code pin | [`b27e889`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/b27e889291476f9c8c0b296d97f6250982a582e1); deployed source hashes are also recorded in [release.json](release.json) |

The 8B-token B0 budget is unfinished, and this release does not enter B1. Both new operator flags were verified for all **164** saved updates after the **53143** handoff. Production retains its original `deterministic_algorithms=False` policy. The cached BF16 CPU shadow is rebuilt after optimizer load; no persistent FP32 weight master is introduced, and the native optimizer checkpoint format is unchanged.

## Files and verification

| File | Bytes | SHA256 |
| --- | ---: | --- |
| [`trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/trainable.pt) | 2,192,257,315 | `3dd9a62f7acfb4ae025ff44b0589b017104f07abc8b177e0098f5c4a910bd2b2` |
| [`weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/weights-only.pt) | 438,477,679 | `050a114b43234a441520703ebd9a6b8abfebceaa9ac191fb8d934b3204608ab7` |

`trainable.pt` contains native trainable weights, optimizer moments and recovery metadata. `weights-only.pt` contains the same weights and checkpoint metadata with no Adam state; loading it restarts the optimizer. CPU inspection verified all 132 weights and 264 moment tensors in the recovery file, including state mappings, finite values, counters and cursor. The weights-only export was compared against all 132 recovery weights byte-for-byte.

The required base is [`openbmb/MiniCPM5-2B-Base`](https://huggingface.co/openbmb/MiniCPM5-2B-Base). Its `model.safetensors` SHA256 is `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`. Base configuration, tokenizer, packed corpus and deployed source identities are recorded in [release.json](release.json). Use the repository's upcycling code and checkpoint configuration rather than `AutoModelForCausalLM.from_pretrained` on this model repository.

## Download and recover B0

Use the current repository checkout for the updated download aliases; the release manifest records the training code pin and exact deployed source hashes. The following paths are examples on a new machine:

```sh
export YOKO_WORK=/workspace/cat-yoko
python scripts/download_minicpm5.py --check-hash --local-dir "$YOKO_WORK/hf/MiniCPM5-2B-Base"
python scripts/download_hub_overlay.py --name b0-rocm-realtext-53307 --out-dir "$YOKO_WORK/hub/b0-rocm-realtext-step-53307"
```

Prepare the original pilot with the [real-text guide](https://github.com/AvrovaDonz2026/CAT-YOKO/blob/main/operators/rocm/REAL_TRAINING.md), then verify the training/evaluation bins and tokenizer files against the release manifest. This snapshot restores a cursor into a **19531-row** training bin with stride 1. The stream loader does not authenticate the corpus contents, so a different `.bin` does not provide the same data continuation merely because it has the same shape. The historical DummyStream and DeepSpeed launchers are not the complete recovery path for this native CPU Adam snapshot.

After verifying the original base and corpus, resume with the validated packed/shared-storage path:

```sh
python -m operators.rocm.round3_candidate_bench \
  --base "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --resume "$YOKO_WORK/hub/b0-rocm-realtext-step-53307/trainable.pt" \
  --out "$YOKO_WORK/runs/b0-resume-step-53307" \
  --data "$YOKO_WORK/data/phase-b-real-20261002/train.bin" \
  --eval-data "$YOKO_WORK/data/phase-b-real-20261002/eval.bin" \
  --eos-id 1 --seq-len 4096 --parity-seqs 4096 \
  --moe-layout shared-storage --packed-attention \
  --split-attention --cached-cpu-adam \
  --deterministic-parity --reference-repeat \
  --run-steps 1026 \
  --save-optim --save-every 0 --save-every-seconds 300 --keep-last 3 \
  --eval-every 250 --eval-batches 32 --wait-gpu-idle
```

This runner reconstructs the MiniCPM5-upcycled graph, loads the overlay, validates deterministic native-reference outputs and all trainable gradients, installs the document-split attention and cached CPU-shadow Adam contexts, and restores native CPU FP32 Adam for B0. The parity contexts restore the ordinary training policy before updates; the example does not enable deterministic training or synchronized timing instrumentation. Use a new output directory and the explicit source file. From this exact snapshot, **1026 additional updates** reach global step **54333**, Adam counter and packed cursor **19531**, ending the first corpus pass without wrapping. No time cap is applied. Keep the training data and checkpoint available during recovery. Timed saves occur after completed updates, include Adam/RNG/cursor and retain three numbered recovery points; update and write time can make spacing exceed 300 seconds.

For a smaller weights-only download:

```sh
python scripts/download_hub_overlay.py --name b0-rocm-realtext-53307-weights --out-dir "$YOKO_WORK/hub/b0-rocm-realtext-step-53307-weights"
```

The downloader writes either selected artifact as local `trainable.pt`. Use separate output directories for the two variants. The smaller variant is suitable for loading weights and inspection; its missing moments prevent exact continuation of the saved optimizer trajectory.

## Operator acceptance scope

The [switch report](https://github.com/AvrovaDonz2026/CAT-YOKO/blob/b27e889291476f9c8c0b296d97f6250982a582e1/operators/rocm/OPERATOR_SWITCH_20261004.md) records the numerical and recovery gates. The full-model comparison covered all 132 trainable gradients; global relative L2 error was **1.4949%** and loss absolute error **0.000927925**, within the existing 5% gradient/output and 0.02 loss limits. Long-document and single-document GPU fixtures passed 14/14 without skips: eight exercised split and six checked native fallback.

Under the same controlled deterministic policy, **two real updates** preserved all 132 BF16 weights and 264 CPU FP32 moments byte-for-byte. Ordinary 20-step native repeats were not byte-exact, so that controlled result is limited to two updates and does not establish a byte-exact ordinary training trajectory. Short old/new/old whole-update windows measured **2.55–3.03%** gain, below the existing 5% automatic selection gate. Adoption followed the explicit user request after numerical and recovery acceptance; no sustained full-corpus speed or language-quality gain is claimed.

Operator implementations and evidence are also available at [AvrovaDonz/CAT-YOKO-KERNEL](https://huggingface.co/AvrovaDonz/CAT-YOKO-KERNEL). These PyTorch operators use the in-repository recovery entry above. The reference code pin covers core and operator code; [release.json](release.json) records exact deployed source hashes, including the frozen supervisor that predates additional source/policy evidence checks.

## Held-out observation

The latest fixed evaluation before the saved snapshot ran at **step 53250**, giving NLL **7.885217772929435** over **32 batches / 130,877 valid next-token loss tokens**. Validation uses an independent bin; this is a finite sample of that bin, not a general generation/reasoning benchmark or a full validation-corpus score. The release weights were saved **57 updates** after that evaluation. No later evaluation or B1 result is attributed to this snapshot.

Historical [ROCm step52616](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-rocm-realtext/step-52616) retains its original complete and weights-only artifacts, SHA pins and preceding packed-attention/native CPU Adam provenance. Fixed `b0-rocm-realtext-52616` download aliases remain available.

Historical [B200 step 26940](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full) and [RTX 3090 step 33800](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) remain under their original paths. Code and derived weights use Apache-2.0.
