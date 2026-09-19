#!/usr/bin/env bash
# Pull the newest complete B0 overlay from SSH Host `vast-b200`.
#
# Does not kill the live trainer. Does not copy HuggingFace keys to Vast.
# Host alias only — no machine IPs in this file.
#
# keep-last=2 rotates numbered files about every 10s. SIGINT does not save.
# Wait until trainable_step_*.pt mtime age >= 5s, copy that file (not the
# hardlink pointer), then scp. Staging on the remote is /tmp so rotation
# cannot delete the bytes mid-transfer.
#
# Usage:
#   ./scripts/pull_vast_b0_overlay.sh
#   OUT=checkpoints/b0-full/trainable.pt ./scripts/pull_vast_b0_overlay.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOST="${VAST_SSH_HOST:-vast-b200}"
REMOTE_DIR="${VAST_B0_DIR:-/workspace/runs/b0-full}"
REMOTE_TMP="${VAST_B0_TMP:-/tmp/cat-yoko-hub-overlay.pt}"
OUT="${OUT:-$ROOT/checkpoints/b0-full/trainable.pt}"
MIN_AGE="${MIN_AGE_SECS:-5}"

die() {
  echo "error: $*" >&2
  exit 1
}

command -v ssh >/dev/null 2>&1 || die "ssh not found"
command -v scp >/dev/null 2>&1 || die "scp not found"
[[ "$OUT" == *.pt ]] || die "OUT must end in .pt (got $OUT)"
mkdir -p "$(dirname "$OUT")"

ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" "bash -s" <<REMOTE
set -euo pipefail
DIR=$(printf '%q' "$REMOTE_DIR")
TMP=$(printf '%q' "$REMOTE_TMP")
MIN_AGE=$(printf '%q' "$MIN_AGE")
READY=""
for i in \$(seq 1 50); do
  newest=\$(ls -1 "\$DIR"/trainable_step_*.pt 2>/dev/null | sed 's/.*step_//' | sed 's/\\.pt//' | sort -n | tail -1)
  [ -n "\$newest" ] || { echo "wait i=\$i no overlay yet"; sleep 0.4; continue; }
  f="\$DIR/trainable_step_\${newest}.pt"
  age=\$(( \$(date +%s) - \$(stat -c %Y "\$f") ))
  size=\$(stat -c %s "\$f")
  if [ "\$age" -ge "\$MIN_AGE" ] && [ "\$size" -gt 400000000 ]; then
    echo "READY step=\$newest age=\${age}s size=\$size"
    cp -a "\$f" "\$TMP"
    READY=1
    break
  fi
  echo "wait i=\$i step=\$newest age=\${age}s size=\$size"
  sleep 0.4
done
[ -n "\$READY" ] || { echo "error: no stable overlay in \$DIR" >&2; ls -l "\$DIR"/trainable*.pt >&2 || true; exit 1; }
if [ -x /venv/main/bin/python3 ]; then
  PY=/venv/main/bin/python3
else
  PY=python3
fi
"\$PY" - "\$TMP" <<'PY'
import json, sys
path = sys.argv[1]
try:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(json.dumps({"error": str(exc), "path": path}))
    raise SystemExit(0)
extra = obj.get("extra") or {}
print(json.dumps({
    "kind": obj.get("kind"),
    "n_tensors": obj.get("n_tensors"),
    "nbytes": obj.get("nbytes"),
    "step": extra.get("step"),
    "tokens_in_phase": extra.get("tokens_in_phase"),
    "tokens_seen": extra.get("tokens_seen"),
}, default=str))
PY
REMOTE

scp -o BatchMode=yes "$HOST:$REMOTE_TMP" "$OUT"
ssh -o BatchMode=yes "$HOST" "rm -f $(printf '%q' "$REMOTE_TMP")"
sha256sum "$OUT"
ls -l "$OUT"
echo "wrote $OUT"
