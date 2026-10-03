# RX 7900 XTX real-text operator evaluation, October 3

These are measured reports from the checkpoint-first B0 evaluation, not model
weights. See [the experiment report](../../OPERATOR_SWITCH_20261003.md) for the
method, numerical gates, scope and adopted operator.

Each of the five trials restored the same step-42100 checkpoint and CPU FP32
Adam state, then trained on the same 20 consecutive real packed rows. Metrics
retain all updates; the decision discards the first five. `decision.json`
selects packed attention alone against the faster of two native baselines.

`first_checkpoint_verified.json` records the actual continuation's first timed
save at step 42155, including strict CPU weight, Adam and cursor checks.
`checkpoint_cleanup.json` records reclamation after that verification. The
trial checkpoint files were subsequently removed; their measured reports and
metrics remain, and the original step-42100 recovery source was retained.

The terminal-variation reports describe separate runs under the restored
nondeterministic training policy. Byte-exact clipping on identical gradients
does not establish byte-exact whole-training trajectories.

`manifest.json` records report SHA256 hashes and the 13 new CPU test results.
The long continuation remains running; these artifacts do not report completion
of the long-training stage or a whole-corpus speedup.
