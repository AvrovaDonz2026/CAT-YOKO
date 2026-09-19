#!/bin/bash
# One-shot B200 bring-up: torch+TE, MiniCPM5, Hub B0 overlay, published B0.
# Does not download 50B Ultra-FineWeb. Overlay-only checkpoints.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${WORK:-/workspace}"
[ -d "$WORK" ] || WORK="$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$WORK/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$WORK/runs" "$WORK/hf" "$HF_HOME"
echo "=== B200 bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
bash "$ROOT/scripts/upgrade_torch_te_b200.sh"
PY="${PY:-python3}"
if [ -x /venv/main/bin/python ]; then
  PY=/venv/main/bin/python
fi
"$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$WORK/hf/MiniCPM5-2B-Base"
"$PY" "$ROOT/scripts/download_hub_overlay.py" --name b0-full --out-dir "$WORK/runs/b0-full"
exec bash "$ROOT/scripts/run_b0_full_b200.sh"
