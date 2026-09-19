#!/bin/bash
# NVIDIA B200 (SM 10.0 / 10.3): published C1 B0, 8e9 DummyStream tokens, seq=4096.
#
# Hardware NVFP4 via te.Linear + default NVFP4BlockScaling (RHT+2D+SR).
# Resume Hub overlay checkpoints/b0-full (step 16020) after MiniCPM5 upcycle.
# Does not download Ultra-FineWeb. Does not write 23GiB latest.pt.
# 192GiB HBM: encoder on GPU, no activation checkpoint.
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
SAVE="${SAVE:-$WORK/runs/b0-full}"
LOG="${LOG:-$WORK/runs/b0_full.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
SEQ="${SEQ:-4096}"
SAVE_EVERY="${SAVE_EVERY:-20}"
KEEP_LAST="${KEEP_LAST:-2}"

has_overlay() {
  local d="$1"
  [ -d "$d" ] || return 1
  [ -f "$d/trainable.pt" ] && return 0
  [ -f "$d/latest.pt" ] && return 0
  ls "$d"/trainable_step_*.pt >/dev/null 2>&1
}

newest_trainable() {
  local d="$1"
  ls -1 "$d"/trainable_step_*.pt 2>/dev/null | sort -V | tail -1
}

publish_trainable_pointer() {
  local d="$1"
  local newest
  newest="$(newest_trainable "$d")"
  if [ -n "$newest" ]; then
    ln -f "$newest" "$d/trainable.pt"
    echo "publish $newest -> $d/trainable.pt"
  fi
}

mkdir -p "$SAVE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B0 B200 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv,noheader
df -h / "$WORK" | tail -5
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
CAP="$("$PY" -c "import torch; c=torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo missing)"
case "$CAP" in
  10.0|10.3) echo "SM $CAP NVFP4 training recipe (default RHT+2D+SR)" ;;
  12.0)
    echo "SM 12.0 is 6000D; this launcher is for B200. Set FORCE_SM120=1 to continue with emulation." >&2
    if [ "${FORCE_SM120:-0}" != "1" ]; then
      exit 4
    fi
    ;;
  *)
    echo "unexpected compute cap $CAP (want 10.0/10.3)" >&2
    if [ "${FORCE_SM120:-0}" != "1" ]; then
      exit 4
    fi
    ;;
esac

if [ ! -d "$LOCAL" ] || [ ! -f "$LOCAL/model.safetensors" ]; then
  echo "MiniCPM5 missing at $LOCAL; downloading"
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || {
    echo "MiniCPM5 download failed" >&2
    exit 2
  }
fi

if ! has_overlay "$SAVE"; then
  echo "no local overlay in $SAVE; pulling Hub checkpoints/b0-full"
  "$PY" "$ROOT/scripts/download_hub_overlay.py" --name b0-full --out-dir "$SAVE" || {
    echo "Hub overlay download failed (fresh MiniCPM5 upcycle, no resume)" >&2
  }
fi

publish_trainable_pointer "$SAVE"

RESUME_ARGS=()
if has_overlay "$SAVE"; then
  RESUME_ARGS=(--resume "$SAVE")
  echo "resume overlay $SAVE"
else
  echo "fresh MiniCPM5 upcycle, no overlay"
fi

echo "B0 argv seq=$SEQ tokens=8e9 no-offload-encoder no-grad-ckpt"
cd "$ROOT"
"$PY" -m cat_yoko.b0 \
  --save-dir "$SAVE" \
  --seq-len "$SEQ" \
  --save-every "$SAVE_EVERY" \
  --keep-last "$KEEP_LAST" \
  --no-offload-encoder \
  --no-grad-ckpt \
  --upcycle-hf "$LOCAL" \
  --log-every 1 \
  "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
ec=$?
echo "=== B0 B200 exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
