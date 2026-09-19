#!/bin/bash
# Ampere BF16 operator roofline. DummyStream not involved. Does not download Ultra-FineWeb.
# Not a CSA CUDA kernel. Does not write a full-graph checkpoint.
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
OUT="${OUT:-/root/autodl-tmp/bf16-verify/mfu/ledger.json}"
LOG="${LOG:-/root/autodl-tmp/bf16-verify/mfu/ampere_mfu.log}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== ampere-mfu $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
fi
cd "$ROOT"
"$PY" -m cat_yoko.ampere_mfu --out "$OUT"
ec=$?
echo "=== exit ${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "$ec"
