# B0 4000-update observation window

The user approved continued training from complete **step 73864**. The run
completed exactly **4000** updates through **77864** at
**2026-10-06T13:05:21.161521Z** (**21:05:21 Asia/Shanghai**), retaining Adam, RNG,
the absolute cursor, the existing B0 schedules, split packed FP32 attention,
shared frozen MoE storage and cached CPU Adam with native FP32 moments.

| Directory | Receipt scope |
| --- | --- |
| `previous-completion/` | The preceding second pass completed all 19531 updates, with final state 73864 / cursor and Adam 39062 and fixed held-out NLL 7.486454654140886 |
| `source/` | Fixed complete 73864 checkpoint SHA, native CPU Adam load/export and RNG recovery audit, original-data hashes and the exact 4000-update plan |
| `launch/` | Detached supervisor/worker commands, frozen source inventory, CPU source/window audits and 39 passing remote supervision/accepted-run/quality tests |
| `runtime/` | Historical startup status, full-model zero-error gates, primary/additional initial NLL, consecutive metrics and independent native recovery inspection of periodic save 74075 |
| `completion/` | Verified complete status, all 4000 metrics, final native 132-weight/264-moment audit, operator policy, and completed paired primary/additional evaluation |

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

The completed run is `runs/window4000-73864-20261006T0630`, using independent
`source-window4000-20261006T0630`. It scheduled full checkpoints every 300
seconds, retaining three rolling saves plus fixed/verified recovery points.
No checkpoint binary or authentication credential is stored in this directory.
See [the window report](../../B0_WINDOW4000_20261006.md) for the complete scope.

The final state is **77864** / absolute cursor and Adam **43062** / phase
tokens **338626560**. Cumulative real-input consumption is **176381952**,
including repeats of the same **79998976** unique prepared training tokens.
The full checkpoint is **2192257315 bytes**, SHA256
`adf13e2e44a1fcbabbfc1f60cfab2cc61459d0949c95ec9e6e332287691a8676`.
All 132 BF16 weights, 264 native CPU FP32 moments and exact final counters pass
the final CPU audit. Both additional evaluations pass with quality status
**completed**; the supervisor records `final_quality_done=true`.

Primary held-out NLL decreased **7.4860189283560965 → 7.3636053664583905**
(32 batches / 130877 valid loss tokens). Additional disjoint rows **32–63**
decreased **7.599186639848141 → 7.503450781393409**
(32 batches / 130878 valid loss tokens). Each slice compares its own initial
and final score; relative perplexity reductions are **11.52176%** and
**9.12960%**, respectively. Primary periodic NLL medians were
**7.475844210160067 → 7.412587716432117** across the first/final eight scores.
This remains a same-validation-corpus pilot observation, not a downstream
benchmark or a completed overall B0 phase. The separate
[publication receipt](../../../../checkpoints/b0-rocm-realtext/step-77864/publish.json)
records the subsequently verified **77864** Hub release; **53307** retains
its original immutable artifacts.

The archived historical startup status observes **74165** / **301** new updates at
**2026-10-06T07:11:48.088316Z**, with latest verified checkpoint **74132**.
`runtime/runtime_observed.json` records an earlier **74108** / **244** update
inspection at **07:06:36 UTC**. It binds the separately pinned **74075** save,
whose full-state and native Adam load/export checks pass. All 264 native CPU
FP32 moments remain byte-identical after recovery; Python/CPU Torch RNG
restoration and saved GPU RNG schema checks pass without GPU allocation.
The runtime files were copied sequentially while training continued; their
last steps can differ. The separate `completion/` receipts supersede those
historical running observations for the completed window.
`file-manifest.json` hashes each archived text artifact.
