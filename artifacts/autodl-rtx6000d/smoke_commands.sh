#!/bin/bash
# WESTE RTX 6000D — run AFTER origin has cursor/nvfp4-train-6000d-02c6.
# python is /root/miniconda3/bin/python
# Do NOT download Ultra-FineWeb. Do NOT write 23GiB full graphs.
# Existing overlay: /root/autodl-tmp/runs/b0 (32-step MiniCPM5 --try). Keep it.
set -euo pipefail

source /root/autodl-tmp/cat-yoko-env.sh
export PATH="/root/miniconda3/bin:$PATH"
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/CAT-YOKO
HF=/root/autodl-tmp/hf/MiniCPM5-2B-Base

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
test "$HF_ENDPOINT" = "https://hf-mirror.com"
test "$HF_HUB_DISABLE_XET" = "1"

cd "$ROOT"
# Remote was on cursor/nvfp4-c1-theory-02c6 with a dirty NVFP4 working tree.
git stash push -u -m "weste local nvfp4 smoke" || true
git fetch origin cursor/nvfp4-train-6000d-02c6
git checkout cursor/nvfp4-train-6000d-02c6
git pull --ff-only origin cursor/nvfp4-train-6000d-02c6
git log -1 --oneline

"$PY" -m pip install -e .

mkdir -p /root/autodl-tmp/runs/gpu_smoke /root/autodl-tmp/runs/b0-nvfp4
"$PY" -m cat_yoko.gpu_smoke --json | tee /root/autodl-tmp/runs/gpu_smoke/tiny_nvfp4.json

# 12B recipe already has use_nvfp4=True. Pass the flag only if this checkout grew one.
EXTRA=()
if "$PY" -m cat_yoko.b0 --help 2>&1 | grep -q -- '--use-nvfp4'; then
  EXTRA+=(--use-nvfp4)
fi

# Short B0 --try. Default --try is 32 steps / seq=64 / trainable.pt overlay (~0.42GiB).
# For a 2-step disk-safe probe, add --steps 2 (already ran once on this box).
"$PY" -m cat_yoko.b0 --try \
  --upcycle-hf "$HF" \
  --save-dir /root/autodl-tmp/runs/b0-nvfp4 \
  "${EXTRA[@]}"
