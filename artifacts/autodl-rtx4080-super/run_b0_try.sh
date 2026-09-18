#!/bin/bash
# AutoDL C1 B0 --try: MiniCPM5 upcycle if hf-mirror works, else dummy.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_DISABLE_XET=1
export PYTHONPATH=/root/autodl-tmp/CAT-YOKO
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python3
ROOT=/root/autodl-tmp/CAT-YOKO
SAVE=/root/autodl-tmp/runs/b0
LOG=/root/autodl-tmp/runs/b0_try.log
mkdir -p "$SAVE" "$HF_HOME" /root/autodl-tmp/hf/MiniCPM5-2B-Base
exec > >(tee -a "$LOG") 2>&1
echo "=== B0 try $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
df -h /root/autodl-tmp | tail -1
UPCYCLE_ARGS=(--dummy-upcycle)
if "$PY" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "openbmb/MiniCPM5-2B-Base",
    local_dir="/root/autodl-tmp/hf/MiniCPM5-2B-Base",
)
print("downloaded", p)
PY
then
  UPCYCLE_ARGS=(--upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base)
  echo "using MiniCPM5-2B-Base from hf-mirror"
else
  echo "hf-mirror download failed; dummy-upcycle"
fi
echo "argv extra: ${UPCYCLE_ARGS[*]}"
cd "$ROOT"
"$PY" -m cat_yoko.b0 --try --save-dir "$SAVE" "${UPCYCLE_ARGS[@]}"
ec=$?
echo "=== exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
