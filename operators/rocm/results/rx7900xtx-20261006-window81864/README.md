# B0 step 77864 to 81864

This user-authorized window completed exactly 4000 updates, preserving full
Adam/RNG/cursor and the preceding production operators. The archived startup
observation is 77999 / 135 new updates at 2026-10-06T14:54:50Z; complete
checkpoint 77980 passed the independent supervisor CPU audit.

`launch/` contains the frozen source inventory, detached commands, source and
periodic complete-state checks, full-model gates and initial held-out baselines.
`local-upload-watch/` records the detached local uploader that waits for exact
81864 completion and paired quality/full checkpoint acceptance. It uses local
HF login; no HF credentials are sent to the remote trainer. These retain the
historical launch state; final publication receipts are in `publication/`.

`completion/` records the final supervisor state at
**2026-10-06T21:01:53.537453Z** (**2026-10-07 05:01:53 CST**), both paired
evaluations, complete metrics, operator reports and the final CPU audit.
Its transfer archive SHA256 is
`78546c820528d709c5b7919043e2aeb143e79c30f1a4ccebf4b697ced6cc734d`.
The launch timestamps and early observations above retain their historical
meaning; they are not the final state.

| Completed state | Value |
| --- | ---: |
| Global step | 77864 → 81864 |
| Consecutive completed updates | 4000, steps 77865–81864 |
| Absolute packed cursor / native Adam counter | 43062 → 47062 |
| Phase tokens including earlier dummy history | 338626560 → 355010560 |
| Real input tokens including repetitions | 176381952 → 192765952 |
| Unique prepared training tokens | 79998976, unchanged |
| Final next physical packed row | 8000 |
| Final CPU-verified native state | 132 trainable weights / 132 Adam states / 264 CPU FP32 moments |

Primary fixed validation NLL improved
**7.363943263994131 → 7.29112438273621** (32 batches, **130877** valid tokens).
The disjoint rows 32–63 improved
**7.5028788111342815 → 7.387959246989646** (32 batches, **130878** valid tokens).
Both initial/final pairs passed; the extra evaluation restored RNG and
evaluation state and left the training stream untouched. These are two slices
of the same pilot validation corpus, not an external benchmark or proof of
general quality or B0 convergence.

Full checkpoints used the **300-second** cadence and **three** rolling saves,
with the fixed and verified recovery states retained. Final status is
`complete`, child PID is cleared, and the stop reason is `steps`.
The actual endpoint **47062** remains below the whole-corpus audit ceiling
**58593**. Split attention/cached CPU Adam and the original **False** training
policy appear in every update receipt; the optimizer's **528000** parameter
updates match **132 × 4000**, with no fallback and retained CPU FP32 moments.

The **81864** Hub release is published and verified at commit
[`da001379d7bcd27c9348bf2ecf99036e6825dde9`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/da001379d7bcd27c9348bf2ecf99036e6825dde9/checkpoints/b0-rocm-realtext/step-81864).
The local watcher recorded `published_verified` at **2026-10-07T14:19:34Z**.
`publication/` archives only the final watcher status, the staging binding,
the explicit prepared file inventory and the verified Hub receipt. It does
not duplicate checkpoint binaries or the source tree.

| Artifact | Bytes | Commit-pinned Hub LFS SHA256 |
| --- | ---: | --- |
| Complete `trainable.pt` | 2192257315 | `20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc` |
| Weights-only `weights-only.pt` | 438478163 | `74d8a60b9e43613e917a5debd52d27776604ea69e1564e4a891b4f99dd906085` |

Both LFS sizes and SHA256 match the CPU-verified prepared files. The root
model card, immutable snapshot card and release manifest were downloaded at
this commit and compared byte-for-byte. The lightweight overlay retains the
same 132 BF16 tensors and omits optimizer history. See the
[publication receipt](publication/publish.json) and
[window report](../../B0_WINDOW4000_77864_20261006.md) for provenance and scope.
The family index was separately verified at
[`fcc6b963a078316562889da77b0e1f54f24167c7`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/fcc6b963a078316562889da77b0e1f54f24167c7/checkpoints/b0-rocm-realtext/README.md),
preserving the sizes and SHA256 of all **12** historical and new weight files.

No checkpoint binaries or authentication credentials are tracked here.
