# Hugging Face Hub (weights)

Model repository: [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO).
Code, documentation and compact evidence stay on GitHub. Weight files are on
Hub; GitHub does not use LFS. Code, derived weights and MiniCPM5-2B-Base are
Apache-2.0.

## Recommended B0 snapshot

The latest recommended release is **ROCm real-text step 53307**, captured on
2026-10-04. It is an immutable snapshot while training continues.

| File | Role |
| --- | --- |
| [`step-53307/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/trainable.pt) | 2,192,257,315 bytes; 132 BF16 trainable tensors + 264 native CPU FP32 Adam moments, RNG, step and packed cursor |
| [`step-53307/weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-53307/weights-only.pt) | 438,477,679 bytes; the same 132 tensors, no Adam history |
| [`release.json`](../checkpoints/b0-rocm-realtext/step-53307/release.json) | Both SHA256 values, model/base/tokenizer/data/source hashes and verified counters |

```bash
python scripts/download_hub_overlay.py --out-dir runs/b0-rocm-realtext
# optional lightweight file, for weight loading rather than complete Adam recovery:
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir runs/b0-rocm-realtext-weights
```

Both files are native trainable overlays. Reconstruct frozen weights from
MiniCPM5-2B-Base, then load the overlay. They are not a standalone 12B full graph
or an AutoModelForCausalLM architecture. Complete packed-stream recovery also
requires the original train/eval/tokenizer hashes and native CPU Adam; old
DummyStream launchers do not preserve the meaning of this packed cursor.
See the [snapshot card](../checkpoints/b0-rocm-realtext/step-53307/README.md)
and [real-text guide](../operators/rocm/REAL_TRAINING.md).

The global step is **53307**; Adam and packed cursor are **18505**. Cumulative
**238,041,088** phase tokens include DummyStream history; actual packed real-text
training totals **75,796,480** tokens. The latest fixed pilot evaluation is
NLL **7.885217772929435** at **53250**, 57 updates before the snapshot, over
32 batches / 130877 valid loss tokens; it is not a downstream benchmark.

This snapshot uses document-split FP32 attention and cached BF16 CPU-shadow
Adam with shared frozen expert storage. Native CPU FP32 moments and checkpoint
format remain intact. Two controlled deterministic real updates matched all
132 weights and 264 moments byte-for-byte; production retains its ordinary
`deterministic_algorithms=False` policy. The **2.55–3.03%** short-window
whole-update gain did not reach the 5% automatic gate; adoption was
user-requested after numerical and recovery acceptance. See the
[switch evidence](../operators/rocm/OPERATOR_SWITCH_20261004.md) and
[operator repository](https://huggingface.co/AvrovaDonz/CAT-YOKO-KERNEL).
The step card uses `round3_candidate_bench --split-attention --cached-cpu-adam`
for the remaining **1026** updates to **54333**, preserving the original packed
corpus, CPU FP32 Adam, 300-second saves / keep3 and 250-update / 32-batch eval.

## Historical snapshots

The historical weight files and SHA pins remain available:

- [ROCm step52616](../checkpoints/b0-rocm-realtext/step-52616/README.md): full recovery SHA256 `d4c4898be1cd248b2742bd9705a11de8af37003a0d209a91347498d47ec181df`; preceding packed-attention/native CPU Adam backend. Explicit `--name b0-rocm-realtext-52616` and `--name b0-rocm-realtext-52616-weights` preserve access to both original files.
- `checkpoints/b0-full/`: Vast B200 step26940, SHA256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`; weights only, DummyStream, recycled machine.
- `checkpoints/b0-3090-bf16/`: RTX3090 step33800, SHA256 `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690`; historical BF16 sibling.
- `checkpoints/b0/`: 6000D32-step try; `checkpoints/b0-nvfp4-try/`: two-step try.
- B1/B2 training weights have not been published.

Explicit `download_hub_overlay.py --name b0-full` and `--name b0-3090-bf16`
continue to fetch those historical files. The default now selects the new
ROCm complete state. Machine history is in [STATUS.md](STATUS.md).

## Publishing

Publish a curated payload containing only model cards, release metadata and
explicit weight files, not the whole GitHub checkout. `hf upload` can commit
that payload once to `AvrovaDonz/CAT-YOKO`. Use an immutable step directory and
update the root model card and download pin together.

The legacy `scripts/push_to_hf.sh` remains available for explicitly chosen
historical paths; it also ships the root card, b0-full card and license. It
uses the existing HF SSH key, which remains outside the checkout. Existing
HF CLI login credentials likewise stay outside source control.

Uploads use `huggingface.co`. Do not reconnect the retired AutoDL machines.
