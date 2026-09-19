#!/bin/bash
# Dedicated-dir mini-train proof of attention / YOCO / PDSA-in-graph / C1 chain.
# DummyStream only. Does not download Ultra-FineWeb. Does not persist a full-graph checkpoint.
# Does not write passwords or host keys.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
if [[ -z "${PY:-}" ]]; then
  if [[ -x /root/miniconda3/bin/python3 ]]; then
    PY=/root/miniconda3/bin/python3
  else
    PY=python3
  fi
fi
OUT="${OUT:-/root/autodl-tmp/plan-verify/runs}"
LOG="${LOG:-/root/autodl-tmp/plan-verify/plan_verify.log}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-2}"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== plan-verify $(date -u +%Y-%m-%dT%H:%M:%SZ) device=${DEVICE} steps=${STEPS} ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
fi
if [[ -d /root/autodl-tmp ]]; then
  df -h /root/autodl-tmp | tail -1 || true
fi
cd "$ROOT"
"$PY" -m cat_yoko.plan_verify --out "$OUT" --device "$DEVICE" --steps "$STEPS"
ec=$?
echo "=== exit ${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [[ -f "$OUT/ledger.json" ]]; then
  "$PY" - "$OUT/ledger.json" <<'PY'
import json, sys
blob = json.load(open(sys.argv[1]))
print(f"ledger ok={blob['ok']} claims={blob['n_claims']} fail={blob['n_fail']} deferred={blob['n_deferred']}")
for row in blob.get("failed") or []:
    print("FAIL", row["name"], row["observed"])
PY
fi
ls -ld "$OUT" || true
exit "$ec"
