#!/bin/bash
# AutoDL C1 B0 --try on RTX 6000D: real MiniCPM5 upcycle if weights land.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
# shellcheck disable=SC1091
source "$ROOT/scripts/autodl_env.sh"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
PY="${PY:-/root/miniconda3/bin/python3}"
SAVE="${SAVE:-/root/autodl-tmp/runs/b0}"
LOG="${LOG:-/root/autodl-tmp/runs/b0_try.log}"
LOCAL="${LOCAL:-/root/autodl-tmp/hf/MiniCPM5-2B-Base}"
mkdir -p "$SAVE"
exec > >(tee -a "$LOG") 2>&1
echo "=== B0 try $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
df -h /root/autodl-tmp | tail -1
UPCYCLE_ARGS=(--dummy-upcycle)
if "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL"; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL"
else
  echo "MiniCPM5 download failed; dummy-upcycle"
fi
echo "argv extra: ${UPCYCLE_ARGS[*]}"
cd "$ROOT"
"$PY" -m cat_yoko.b0 --try --save-dir "$SAVE" "${UPCYCLE_ARGS[@]}"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
