#!/bin/bash
# AutoDL C1 B1 --try on RTX 6000D: resume B0 overlay + MiniCPM5 upcycle.
# Keep --try (32 steps, seq=64) for GPU smoke. Full 27B tokens is H100-scale.
# CLI 注入 --no-save-full --save-trainable，不写 23GiB latest.pt。
# DummyStream only. Overlay to HuggingFace checkpoints/b1/.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
# shellcheck disable=SC1091
source "$ROOT/scripts/autodl_env.sh"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
PY="${PY:-/root/miniconda3/bin/python3}"
SAVE="${SAVE:-/root/autodl-tmp/runs/b1}"
LOG="${LOG:-/root/autodl-tmp/runs/b1_try.log}"
LOCAL="${LOCAL:-/root/autodl-tmp/hf/MiniCPM5-2B-Base}"
B0_FULL="${B0_FULL:-/root/autodl-tmp/runs/b0-full}"
B0="${B0:-/root/autodl-tmp/runs/b0}"
mkdir -p "$SAVE"
exec > >(tee -a "$LOG") 2>&1
echo "=== B1 try $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "HF_ENDPOINT=${HF_ENDPOINT}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
df -h /root/autodl-tmp | tail -1
has_overlay() {
  local d="$1"
  [ -d "$d" ] || return 1
  [ -f "$d/trainable.pt" ] && return 0
  [ -f "$d/latest.pt" ] && return 0
  ls "$d"/trainable_step_*.pt >/dev/null 2>&1
}

if [ -z "${RESUME:-}" ]; then
  if has_overlay "$B0_FULL"; then
    RESUME="$B0_FULL"
  else
    RESUME="$B0"
  fi
fi
echo "resume $RESUME"
if ! has_overlay "$RESUME"; then
  echo "missing B0 overlay in $RESUME (need trainable.pt / latest.pt / trainable_step_*.pt)"
  exit 1
fi
UPCYCLE_ARGS=(--dummy-upcycle)
if [ -d "$LOCAL" ]; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL"
elif "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL"; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL"
else
  echo "MiniCPM5 download failed; dummy-upcycle"
fi
echo "argv extra: ${UPCYCLE_ARGS[*]}"
cd "$ROOT"
"$PY" -m cat_yoko.b1 --try --resume "$RESUME" --save-dir "$SAVE" "${UPCYCLE_ARGS[@]}"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
