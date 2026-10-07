# Hugging Face Hub (weights)

Model repository: [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO).
Code, documentation and compact evidence stay on GitHub. Weight files are on
Hub; GitHub does not use LFS. Code, derived weights and MiniCPM5-2B-Base are
Apache-2.0.

## Recommended B0 snapshot

The current pinned release is **ROCm real-text step 81864**, saved on
**2026-10-06T20:58:48.075579Z** after a completed 4000-update window. The
[Hub model card](https://huggingface.co/AvrovaDonz/CAT-YOKO) and immutable
release manifest identify the verified recovery artifact.

Publication was verified at immutable Hub revision
[`da001379d7bcd27c9348bf2ecf99036e6825dde9`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/da001379d7bcd27c9348bf2ecf99036e6825dde9/checkpoints/b0-rocm-realtext/step-81864).
Both checkpoint LFS sizes/SHA256 and the commit-pinned cards and manifest
match the curated payload.

| File | Role |
| --- | --- |
| [step-81864/trainable.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/trainable.pt) | 2,192,257,315 bytes; 132 BF16 trainable tensors + 264 native CPU FP32 Adam moments, RNG and absolute packed cursor |
| [step-81864/weights-only.pt](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-rocm-realtext/step-81864/weights-only.pt) | 438,478,163 bytes; the same 132 tensors byte-for-byte, no Adam history |
| [release.json](../checkpoints/b0-rocm-realtext/step-81864/release.json) | Artifact SHA256, base/tokenizer/data identities, deployed-source archive and complete-run validation |

Full recovery SHA256: `20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc`.
Weights-only SHA256: `74d8a60b9e43613e917a5debd52d27776604ea69e1564e4a891b4f99dd906085`.

For this training workflow, run these downloads on the training machine.
Keep checkpoint copies on that machine and Hugging Face; the local publishing
machine does not keep a permanent weight copy.

```bash
python scripts/download_hub_overlay.py --name b0-rocm-realtext --out-dir runs/b0-rocm-realtext
python scripts/download_hub_overlay.py --name b0-rocm-realtext-weights --out-dir runs/b0-rocm-realtext-weights
# Explicit immutable aliases:
python scripts/download_hub_overlay.py --name b0-rocm-realtext-81864 --out-dir runs/step-81864
python scripts/download_hub_overlay.py --name b0-rocm-realtext-81864-weights --out-dir runs/step-81864-weights
```

Both files are native trainable overlays. Reconstruct frozen weights from
MiniCPM5-2B-Base, then load the overlay. Complete packed-stream recovery also
requires original train/eval/tokenizer identities and native CPU Adam. The
light file cannot preserve the optimizer trajectory, and neither contains
the complete 12B graph. See the [snapshot card](../checkpoints/b0-rocm-realtext/step-81864/README.md)
and [real-text guide](../operators/rocm/REAL_TRAINING.md).

At **81864**, Adam and absolute packed cursor are **47062**, with next
physical row **8000 / 19531**. Phase tokens are **355,010,560**, including
prior DummyStream history; real packed input is **192,765,952**, including
repetitions of the **79,998,976 unique training tokens**. This is still B0.

The **77864 → 81864** window completed all **4000** updates. Primary
rows 0–31 improved **7.363943263994131 → 7.29112438273621**; additional
rows 32–63 improved **7.5028788111342815 → 7.387959246989646**. Each slice
has 32 batches, respectively 130877 and 130878 valid loss tokens. Both slices
come from the same pilot validation corpus and were observed in prior
windows; they are not external capability benchmarks.

The backend remains split packed FP32 attention, shared frozen expert
storage and cached BF16 CPU-shadow Adam with retained native FP32 moments.
All 4000 updates preserve the ordinary False deterministic policy and
complete saves every **300 seconds / keep3**. Completed-state CPU audit,
native Adam load/export and all-132 light-weight byte comparisons pass;
paired-quality checks preserve cursor and RNG. Publication uses local HF
login and commit-pinned artifact checks. See the
[completion report](../operators/rocm/B0_WINDOW4000_77864_20261006.md) and
original [operator acceptance](../operators/rocm/OPERATOR_SWITCH_20261004.md).

## Historical snapshots

The historical weight files and SHA pins remain available:

- [ROCm step77864](../checkpoints/b0-rocm-realtext/step-77864/README.md): complete recovery SHA256 `adf13e2e44a1fcbabbfc1f60cfab2cc61459d0949c95ec9e6e332287691a8676`; weights-only SHA256 `0ad37d86006fe68d1cf861d4519484ed5ef5bf882af19f803cd3a056db53e84e`. Explicit `b0-rocm-realtext-77864` / `b0-rocm-realtext-77864-weights` aliases retain both original files.
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

Local weight files are temporary upload staging. After commit-pinned Hub
verification passes, remove the task's local full and weights-only binaries
and transport archives, keeping logs, manifests, hashes and publication
receipts. Complete checkpoints and recovery copies remain on the training
machine and Hugging Face. This is the storage policy and cleanup procedure;
the completion of a particular cleanup is recorded separately.
