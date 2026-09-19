#!/bin/bash
# B200 C1 B2 published envelope (15e9 DummyStream). Resume B1 overlay + MiniCPM5.
# 192GiB: no per-block offload, GPU Adam, no grad-ckpt.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${WORK:-/workspace}"
[ -d "$WORK" ] || WORK="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$WORK/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [ -x /venv/main/bin/python ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
PY="${PY:-python3}"
SAVE="${SAVE:-$WORK/runs/b2}"
RESUME="${RESUME:-$WORK/runs/b1}"
LOG="${LOG:-$WORK/runs/b2_full.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
mkdir -p "$SAVE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B2 B200 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
if [ ! -f "$RESUME/trainable.pt" ] && [ ! -f "$RESUME/latest.pt" ] && ! ls "$RESUME"/trainable_step_*.pt >/dev/null 2>&1; then
  echo "missing B1 overlay in $RESUME" >&2
  exit 1
fi
if [ ! -d "$LOCAL" ]; then
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || exit 2
fi
cd "$ROOT"
"$PY" -m cat_yoko.b2 \
  --resume "$RESUME" \
  --save-dir "$SAVE" \
  --upcycle-hf "$LOCAL" \
  --no-offload-blocks \
  --no-optim-cpu \
  --no-grad-ckpt \
  --log-every 1
ec=$?
echo "=== B2 B200 exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
