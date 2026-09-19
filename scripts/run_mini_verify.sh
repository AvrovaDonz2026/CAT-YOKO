#!/usr/bin/env bash
# Isolated BF16 + INT8 mini full-pipeline verify. No hostnames, no secrets.
# Does not download Ultra-FineWeb. DummyStream. Do not pass --save-full.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-/root/autodl-tmp/mini-verify/runs}"
PY="${PY:-python3}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$OUT"
cd "$ROOT"
echo "=== mini-verify $(date -u +%Y-%m-%dT%H:%M:%SZ) root=$ROOT out=$OUT ==="
extra=()
if [[ "${TWELVE:-0}" == "1" ]]; then
  extra+=(--12b)
fi
"$PY" -m cat_yoko.mini_verify --out "$OUT" --device cuda --recipe both --steps "${STEPS:-2}" "${extra[@]}" "$@"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "$ec"
