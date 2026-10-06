# B0 4000-update observation window

The user approved continued training from complete **step 73864**. The new
run requests exactly **4000** updates through **77864**, retaining Adam, RNG,
the absolute cursor, the existing B0 schedules, split packed FP32 attention,
shared frozen MoE storage and cached CPU Adam with native FP32 moments.

| Directory | Receipt scope |
| --- | --- |
| `previous-completion/` | The preceding second pass completed all 19531 updates, with final state 73864 / cursor and Adam 39062 and fixed held-out NLL 7.486454654140886 |
| `source/` | Fixed complete 73864 checkpoint SHA, native CPU Adam load/export and RNG recovery audit, original-data hashes and the exact 4000-update plan |
| `launch/` | Detached supervisor/worker commands, frozen source inventory, CPU source/window audits and 39 passing remote supervision/accepted-run/quality tests |
| `runtime/` | Running status, full-model zero-error gates, primary/additional initial NLL, consecutive metrics and independent native recovery inspection of periodic save 74075 |

The fixed source SHA256 is
`6dadd0450a25f598276fed4c5192b8564a89db7b3685c37eba86e6bf72214e1c`.
Its full file is **2192257315 bytes**, with 132 BF16 trainable tensors and 264
CPU FP32 Adam moments. Native Adam load/export preserves every moment
byte-for-byte. Python and CPU Torch RNG restoration pass; saved GPU RNG
tensors are checked without GPU allocation in the CPU audit.

The new supervisor binds the previous completed run and unchanged training
mathematics. All 132 full-model gradient/output gates retain their original
thresholds. The additional initial/final NLL uses rows **32–63** of the same
validation corpus, separate from the primary prefix rows **0–31**; it is not
an external benchmark. The window stops at cursor **43062**; its whole-corpus
CPU audit ceiling **58593** does not increase the requested update budget.

The active run is `runs/window4000-73864-20261006T0630`, using independent
`source-window4000-20261006T0630`. It schedules full checkpoints every 300
seconds, retaining three rolling saves plus fixed/verified recovery points.
No checkpoint binary or authentication credential is stored in this directory.
See [the window report](../../B0_WINDOW4000_20261006.md) for current observations.

The archived running status observes **74165** / **301** new updates at
**2026-10-06T07:11:48.088316Z**, with latest verified checkpoint **74132**.
`runtime/runtime_observed.json` records an earlier **74108** / **244** update
inspection at **07:06:36 UTC**. It binds the separately pinned **74075** save,
whose full-state and native Adam load/export checks pass. All 264 native CPU
FP32 moments remain byte-identical after recovery; Python/CPU Torch RNG
restoration and saved GPU RNG schema checks pass without GPU allocation.
The runtime files were copied sequentially while training continued; their
last steps can differ. Final evaluation/completion receipts do not yet exist.
`file-manifest.json` hashes each archived text artifact.
