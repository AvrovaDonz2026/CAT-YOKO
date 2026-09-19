#!/bin/bash
# B200 B2 GPU smoke: 32 steps, seq=64. Resume B1 overlay.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${WORK:-/workspace}"
[ -d "$WORK" ] || WORK="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$WORK/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [ -x /venv/main/bin/python ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
PY="${PY:-python3}"
SAVE="${SAVE:-$WORK/runs/b2-try}"
RESUME="${RESUME:-$WORK/runs/b1}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
LOG="${LOG:-$WORK/runs/b2_try.log}"
mkdir -p "$SAVE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B2 try B200 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [ ! -f "$RESUME/trainable.pt" ] && ! ls "$RESUME"/trainable_step_*.pt >/dev/null 2>&1 && [ ! -d "$WORK/runs/b1-try" ]; then
  echo "missing B1 overlay $RESUME" >&2
  exit 1
fi
if [ ! -f "$RESUME/trainable.pt" ] && [ -d "$WORK/runs/b1-try" ]; then
  RESUME="$WORK/runs/b1-try"
fi
UPCYCLE=(--dummy-upcycle)
[ -d "$LOCAL" ] && UPCYCLE=(--upcycle-hf "$LOCAL")
cd "$ROOT"
"$PY" -m cat_yoko.b2 --try --resume "$RESUME" --save-dir "$SAVE" \
  --no-offload-blocks --no-optim-cpu --no-grad-ckpt "${UPCYCLE[@]}"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "$ec"
