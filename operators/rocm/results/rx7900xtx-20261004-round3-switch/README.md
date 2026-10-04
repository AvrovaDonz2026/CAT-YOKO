# Round-three switch evidence

See [the adoption report](../../OPERATOR_SWITCH_20261004.md) for interpretation.

- `timing-default-policy/`: old/new/old 20-step runs, all full-model parity and
  timing ledgers, checkpoint-first handoff, and the failed old/old terminal
  repeat under the ordinary training policy.
- `deterministic-confirmation/`: old/new two-step real training controls,
  byte-exact terminal comparison, full-model parity, complete CPU recovery
  audits and the explicit user-requested adoption decision.
- `deployed/`: exact frozen controller and entry files used for each phase.
  The current local controller later adds stricter policy/source evidence gates;
  it is not retroactively claimed to be the deployed version.
- `source_hash_audit.json`: independent live checkpoint and deployed code hashes.
- `acceptance_audit.json`: independently recomputed timing, source binding and
  consecutive real-update policy checks, with a link to completed production
  snapshot acceptance.
- `production-snapshot/`: 106 consecutive ordinary-policy updates to 53249,
  the verified first periodic checkpoint at 53201, native Adam load/export
  and CPU RNG checks, actual process identity and the updated handoff pointer.
  This is a running snapshot, not the final 54333 boundary audit.

Checkpoint `.pt` files and credentials are excluded. Run paths/PIDs are snapshots,
not permanent process identities. Historical failed/superseded supervisors do
not describe the status of the new production continuation.
