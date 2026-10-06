# Hugging Face Hub (weights)

Model repository: [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO).
Code, documentation and compact evidence stay on GitHub. Weight files are on
Hub; GitHub does not use LFS. Code, derived weights and MiniCPM5-2B-Base are
Apache-2.0.

## Recommended B0 snapshot

The current pinned release is **ROCm real-text step 77864**, saved on
**2026-10-06T13:02:17.286196Z** at the end of a completed 4000-update window.
The [Hub model card](https://huggingface.co/AvrovaDonz/CAT-YOKO) identifies
newer completed releases if training has advanced beyond this checkout.

| File | Role |
| --- | --- |
| [step-77864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/trainable.pt) | 2,192,257,315 bytes; 132 BF16 trainable tensors + 264 native CPU FP32 Adam moments, RNG and absolute packed cursor |
| [step-77864/weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-77864/weights-only.pt) | 438,478,163 bytes; the same 132 tensors byte-for-byte, no Adam history |
| [release.json](../checkpoints/b0-rocm-realtext/step-77864/release.json) | Artifact SHA256, base/tokenizer/data identities, deployed-source archive and complete-run validation |

```bash
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir runs/b0-rocm-realtext
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir runs/b0-rocm-realtext-weights
```

Both files are native trainable overlays. Reconstruct frozen weights from
MiniCPM5-2B-Base, then load the overlay. Complete packed-stream recovery also
requires original train/eval/tokenizer identities and native CPU Adam. The
light file cannot preserve the optimizer trajectory, and neither contains
the complete 12B graph. See the [snapshot card](../checkpoints/b0-rocm-realtext/step-77864/README.md)
and [real-text guide](../operators/rocm/REAL_TRAINING.md).

At **77864**, Adam and absolute packed cursor are **43062**, with next
physical row **4000 / 19531**. Phase tokens are **338,626,560**, including
prior DummyStream history; real packed input is **176,381,952**, including
repetitions of the **79,998,976 unique training tokens**.
Final primary NLL is **7.3636053664583905**, and additional validation rows
32–63 score **7.503450781393409**. Both are finite 32-batch observations from
the same pilot validation corpus, not external capability benchmarks.

The backend remains split packed FP32 attention, shared frozen expert
storage and cached BF16 CPU-shadow Adam with retained native FP32 moments.
The completed 4000 updates preserve the ordinary False deterministic
policy. See the [completion report](../operators/rocm/B0_WINDOW4000_20261006.md)
and original [operator acceptance](../operators/rocm/OPERATOR_SWITCH_20261004.md).

A further **77864 → 81864** window is running. Its detached local uploader
waits for exact 4000-update completion, complete CPU state and paired final
quality checks, then exports remotely and uploads with local HF login.
The local machine must remain running and connected. Check
[the new continuation report](../operators/rocm/B0_WINDOW4000_77864_20261006.md)
for the launcher and monitor identities.

## Historical snapshots

The historical weight files and SHA pins remain available:

- [ROCm step53307](../checkpoints/b0-rocm-realtext/step-53307/README.md): both immutable files and explicit `b0-rocm-realtext-53307` / `b0-rocm-realtext-53307-weights` download aliases remain unchanged.
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

The completed-run publisher is `operators/rocm/publish_completed_run.py`.
Remote `--prepare-only` performs complete CPU/native optimizer checks and
builds an allowlisted immutable payload. Local `--publish-prepared` uses
existing local HF login, a parent-commit lock, immutable-file checks and
commit-pinned SHA/card verification. `scripts/watch_rocm_hf_publish.py`
connects those steps after the approved run completes; HF credentials never
leave the local machine. Source archives and compact receipts are included
under each release directory, while GitHub tracks no weight binaries.
