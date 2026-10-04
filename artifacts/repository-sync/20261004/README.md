# CAT-YOKO repository synchronization (2026-10-04)

The GitHub code/docs, Hugging Face model overlay and Hugging Face operator
archive are synchronized around the validated round-three B0 continuation.
The implementation reference is GitHub commit
[`b27e889`](https://github.com/AvrovaDonz2026/CAT-YOKO/tree/b27e889291476f9c8c0b296d97f6250982a582e1).
This follow-up GitHub commit updates release cards, fixed download aliases and
these receipts; it does not change the active frozen training implementation.

| Repository | Published artifact | Verification |
| --- | --- | --- |
| [GitHub CAT-YOKO](https://github.com/AvrovaDonz2026/CAT-YOKO) | Release cards, default step-53307 download pins, fixed step-52616/53307 aliases and synchronization records | Card/manifest/counter/CLI checks and existing publishing regressions |
| [HF CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/5f2fd476c4d669617f758c6f5cffd5e59b007bfb) | Step 53307 full native recovery overlay and byte-identical lightweight weights; updated cards | Two complete local SHA256 values match Hub LFS SHA/size; six commit-pinned documents match; all six prior weight files retain their hashes |
| [HF CAT-YOKO-KERNEL](https://huggingface.co/AvrovaDonz/CAT-YOKO-KERNEL/tree/e060c89b22b846c3eec5530438469ca8dba46ba7) | Split attention, cached CPU Adam, matching core, isolated runner/tests and numerical/timing evidence | All 196 published files downloaded at the exact revision and SHA-verified; 64 prior files, including all 12 HF builder files, preserved; zero deletions |

[Model receipt](model-publish.json),
[kernel receipt](kernel-publish.json),
[kernel file plan](kernel-plan.json),
[local kernel checks](kernel-local-checks.json), and
[published kernel checks](kernel-download-checks.json) contain the precise
identities and validation scopes. The
[step-53307 release manifest](../../../checkpoints/b0-rocm-realtext/step-53307/release.json)
records base, tokenizer, original packed corpus, source hashes, RNG/Adam/cursor
and the immutable kernel revision.

Step 53307 was fixed from a completed numbered checkpoint while training
continued. Its Adam counter / next unread packed row is 18505, real-text tokens
75,796,480 and cumulative phase clock 238,041,088. The latest fixed held-out
evaluation preceding it is NLL 7.885217772929435 at step 53250, 57 updates
earlier, across 32 batches / 130877 valid tokens. The new operators' short-window
2.55–3.03% gain and two-step deterministic arithmetic check retain the scope in
the [adoption report](../../../operators/rocm/OPERATOR_SWITCH_20261004.md).

The kernel archive contains source, not a newly built `get_kernel()` package.
The model artifacts are trainable overlays requiring MiniCPM5-2B-Base, not a
standalone full 12B graph. No checkpoint files or credentials are stored in
GitHub. Prior weights and their model/operator provenance remain available.

[Observed continuation](training-observed.json) confirms that publication did
not pause training and the split/cached backend, native CPU FP32 moments,
300-second saves and retention of three periodic checkpoints remain active.
The active supervisor still targets step 54333 / packed row 19531 and audits
the final boundary; this synchronization is not a completed-corpus or B1 claim.
