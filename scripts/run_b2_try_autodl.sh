#!/bin/bash
# AutoDL C1 B2 --try on RTX 6000D: resume B1 overlay, unfreeze all, NVFP4, block offload.
# Published: 15B tokens, detach=False, gate=1.0, student nvfp4, offload_blocks, CPU Adam, accum=1.
# Attention stays causal YOCO WindowAttention + CrossAttention + fp32 SDPA.
# MiniCPM5 upcycle fills encoder+embed under the B1 overlay.
# DummyStream overlay-only. Does not pull origin. Does not write a 23GiB full graph.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
# shellcheck disable=SC1091
source "$ROOT/scripts/autodl_env.sh"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
PY="${PY:-/root/miniconda3/bin/python3}"
RESUME="${RESUME:-/root/autodl-tmp/runs/b1}"
SAVE="${SAVE:-/root/autodl-tmp/runs/b2}"
LOG="${LOG:-/root/autodl-tmp/runs/b2_try.log}"
LOCAL="${LOCAL:-/root/autodl-tmp/hf/MiniCPM5-2B-Base}"
mkdir -p "$SAVE"
exec > >(tee -a "$LOG") 2>&1
echo "=== B2 try $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "HF_ENDPOINT=${HF_ENDPOINT}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
df -h /root/autodl-tmp | tail -1
if [ ! -d "$RESUME" ]; then
  echo "missing B1 overlay dir $RESUME (sibling B1 / parent B0 must finish first)"
  exit 2
fi
if [ ! -f "$RESUME/trainable.pt" ] && [ ! -f "$RESUME/latest.pt" ]; then
  echo "no trainable.pt / latest.pt in $RESUME"
  ls -lh "$RESUME" || true
  exit 2
fi
# B1 overlay is decoder+lm_head+norm; encoder+embed come from MiniCPM5 (same as B1 handoff).
UPCYCLE_ARGS=(--dummy-upcycle)
if [ -d "$LOCAL" ]; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL for encoder+embed under B1 overlay"
elif "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL"; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL for encoder+embed under B1 overlay"
else
  echo "MiniCPM5 missing at $LOCAL; dummy-upcycle + B1 overlay"
fi
echo "resume $RESUME"
echo "argv extra: ${UPCYCLE_ARGS[*]}"
cd "$ROOT"
"$PY" -m cat_yoko.b2 --try --resume "$RESUME" --save-dir "$SAVE" "${UPCYCLE_ARGS[@]}"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
