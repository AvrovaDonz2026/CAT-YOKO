#!/bin/bash
# B200 C1 B1 published envelope (27e9 DummyStream). Resume B0 overlay + MiniCPM5.
# 192GiB: encoder on GPU, no block offload, GPU Adam, no grad-ckpt.
# Keep --try via run_b1_try_b200.sh. Does not download 50B tokens.
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
SAVE="${SAVE:-$WORK/runs/b1}"
RESUME="${RESUME:-$WORK/runs/b0-full}"
LOG="${LOG:-$WORK/runs/b1_full.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
mkdir -p "$SAVE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B1 B200 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
if [ ! -f "$RESUME/trainable.pt" ] && [ ! -f "$RESUME/latest.pt" ] && ! ls "$RESUME"/trainable_step_*.pt >/dev/null 2>&1; then
  echo "missing B0 overlay in $RESUME" >&2
  exit 1
fi
if [ ! -d "$LOCAL" ]; then
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || exit 2
fi
cd "$ROOT"
"$PY" -m cat_yoko.b1 \
  --resume "$RESUME" \
  --save-dir "$SAVE" \
  --upcycle-hf "$LOCAL" \
  --no-offload-encoder \
  --no-optim-cpu \
  --no-grad-ckpt \
  --log-every 1
ec=$?
echo "=== B1 B200 exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
