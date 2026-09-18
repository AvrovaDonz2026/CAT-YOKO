#!/bin/bash
# AutoDL RTX 6000D: published C1 B0 (8e9 DummyStream tokens, seq=4096).
#
# Not --try. Overlay-only saves. MiniCPM5 upcycle. Encoder GEMMs NVFP4.
# Does not download Ultra-FineWeb. Does not write 23GiB latest.pt.
# Does not overwrite /root/autodl-tmp/runs/b0 (32-step --try).
# GitHub fetch hangs on this box — overlay the tree via tar/scp first.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
# shellcheck disable=SC1091
source "$ROOT/scripts/autodl_env.sh"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-/root/miniconda3/bin/python3}"
SAVE="${SAVE:-/root/autodl-tmp/runs/b0-full}"
PREV="${PREV:-/root/autodl-tmp/runs/b0}"
LOG="${LOG:-/root/autodl-tmp/runs/b0_full.log}"
LOCAL="${LOCAL:-/root/autodl-tmp/hf/MiniCPM5-2B-Base}"
SEQ="${SEQ:-4096}"
SAVE_EVERY="${SAVE_EVERY:-20}"
KEEP_LAST="${KEEP_LAST:-2}"

mkdir -p "$SAVE"
exec > >(tee -a "$LOG") 2>&1
echo "=== B0 published $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv,noheader
df -h /root/autodl-tmp | tail -1
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

if [ ! -d "$LOCAL" ]; then
  echo "missing MiniCPM5 at $LOCAL" >&2
  exit 2
fi

RESUME_ARGS=()
if [ -f "$SAVE/trainable.pt" ]; then
  RESUME_ARGS=(--resume "$SAVE")
  echo "resume published overlay $SAVE/trainable.pt"
elif [ -f "$PREV/trainable.pt" ]; then
  RESUME_ARGS=(--resume "$PREV")
  echo "resume 32-step overlay $PREV/trainable.pt (same-phase B0 + MiniCPM5 upcycle)"
else
  echo "fresh MiniCPM5 upcycle, no overlay"
fi

run_b0() {
  local seq="$1"
  echo "B0 argv seq=$seq tokens=8e9 save-every=$SAVE_EVERY no-offload-encoder"
  cd "$ROOT"
  "$PY" -m cat_yoko.b0 \
    --save-dir "$SAVE" \
    --seq-len "$seq" \
    --save-every "$SAVE_EVERY" \
    --keep-last "$KEEP_LAST" \
    --no-offload-encoder \
    --upcycle-hf "$LOCAL" \
    --log-every 1 \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
}

ec=1
# Fall back only if this start wrote no overlay (OOM before step 20).
# A mid-run kill must not restart at a smaller seq and drop tokens_in_phase.
for seq in "$SEQ" 2048 1024 512; do
  run_b0 "$seq"
  ec=$?
  echo "=== seq=$seq exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ls -lh "$SAVE" || true
  if [ "$ec" -eq 0 ]; then
    exit 0
  fi
  if [ -f "$SAVE/trainable.pt" ]; then
    echo "overlay exists; not falling back seq. Re-run this script to resume."
    exit "$ec"
  fi
  echo "seq=$seq wrote no overlay; trying shorter seq"
done
exit "$ec"
