# Hugging Face Hub (weights)

Model repository: [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO).
Code, documentation and compact evidence stay on GitHub. Weight files are on
Hub; GitHub does not use LFS. Code, derived weights and MiniCPM5-2B-Base are
Apache-2.0.

## Recommended B0 snapshot

The latest recommended release is **ROCm real-text step 52616**, captured on
2026-10-04. It is an immutable snapshot while training continues.

| File | Role |
| --- | --- |
| [`step-52616/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/trainable.pt) | 2,192,257,315 bytes; 132 BF16 trainable tensors + 264 native CPU FP32 Adam moments, RNG, step and packed cursor |
| [`step-52616/weights-only.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-52616/weights-only.pt) | 438,477,743 bytes; the same 132 tensors, no Adam history |
| [`release.json`](../checkpoints/b0-rocm-realtext/step-52616/release.json) | Both SHA256 values, model/base/tokenizer/data/source hashes and verified counters |

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
See the [snapshot card](../checkpoints/b0-rocm-realtext/step-52616/README.md)
and [real-text guide](../operators/rocm/REAL_TRAINING.md).

The global step is 52616; Adam and packed cursor are 17814. Cumulative
235,210,752 phase tokens include DummyStream history; actual packed real-text
training totals 72,966,144 tokens. The latest fixed pilot evaluation before the
snapshot is NLL7.900367 at step52500, not a downstream benchmark.

## Historical snapshots

The historical weight files and SHA pins remain available:

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
