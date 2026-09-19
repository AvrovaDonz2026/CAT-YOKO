#!/bin/bash
# Unknown next GPU: probe SM/VRAM, dispatch published C1 B0 (or --try if <40GiB).
#
# Not Megatron. Not a CSA kernel. Does not download Ultra-FineWeb.
# Does not write 23GiB latest.pt. Resume Hub checkpoints/b0-full only —
# never Hub checkpoints/b0 (32-step --try).
#
# Card-named scripts (run_b0_full_b200.sh / run_b0_full_autodl.sh) stay as
# shortcuts once you know the SKU. This is the entry when you do not.
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
for _d in \
  /usr/local/cuda-12.9/lib64 \
  /usr/local/cuda-12.8/lib64 \
  /venv/main/lib/python3.12/site-packages/nvidia/cublas/lib \
  /venv/main/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
  /venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib; do
  if [ -d "$_d" ]; then
    export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
unset _d
if [ -x /venv/main/bin/python ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
PY="${PY:-python3}"
SAVE="${SAVE:-$WORK/runs/b0-full}"
LOG="${LOG:-$WORK/runs/b0_full.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
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
echo "=== B0 next-card $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv,noheader || true
df -h / "$WORK" | tail -5
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

HW_FLAGS=()
if [ "${TRY:-0}" = "1" ]; then
  HW_FLAGS+=(--try)
fi
if [ "${PROBE_TE:-0}" = "1" ]; then
  HW_FLAGS+=(--probe-te)
fi
# shellcheck disable=SC1090
eval "$("$PY" -m cat_yoko.hw_recipe --shell "${HW_FLAGS[@]+"${HW_FLAGS[@]}"}")"
echo "hw profile=$CAT_YOKO_HW_PROFILE family=$CAT_YOKO_HW_FAMILY launch=$CAT_YOKO_HW_LAUNCH try=$CAT_YOKO_HW_TRY nvfp4=$CAT_YOKO_HW_NVFP4 argv=$CAT_YOKO_HW_ARGV"
echo "$CAT_YOKO_HW_JSON"

if [ "${CAT_YOKO_HW_LAUNCH:-0}" != "1" ]; then
  echo "no CUDA recipe; not building 12B. JSON dumped above." >&2
  exit 3
fi

if [ -n "${CAT_YOKO_HW_TE_ENV:-}" ]; then
  export CAT_YOKO_TE_NVFP4="$CAT_YOKO_HW_TE_ENV"
  echo "CAT_YOKO_TE_NVFP4=$CAT_YOKO_TE_NVFP4"
fi

if [ ! -d "$LOCAL" ] || [ ! -f "$LOCAL/model.safetensors" ]; then
  echo "MiniCPM5 missing at $LOCAL; downloading"
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || {
    echo "MiniCPM5 download failed" >&2
    exit 2
  }
fi

if ! has_overlay "$SAVE"; then
  echo "no local overlay in $SAVE; pulling Hub checkpoints/${CAT_YOKO_HW_RESUME_HUB:-b0-full}"
  "$PY" "$ROOT/scripts/download_hub_overlay.py" --name "${CAT_YOKO_HW_RESUME_HUB:-b0-full}" --out-dir "$SAVE" || {
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

# Env overrides after the probe (MICRO_BATCH / SEQ) without rewriting the recipe.
# Do not pass --save-full. Do not resume checkpoints/b0.
EXTRA=()
if [ -n "${SEQ:-}" ]; then
  EXTRA+=(--seq-len "$SEQ")
fi
if [ -n "${MICRO_BATCH:-}" ]; then
  EXTRA+=(--micro-batch "$MICRO_BATCH")
fi

# CAT_YOKO_HW_ARGV is a shell-quoted flag string from hw_recipe --shell.
# shellcheck disable=SC2086
set -- $CAT_YOKO_HW_ARGV
echo "B0 argv $* extra=${EXTRA[*]-} no-save-full Hub=${CAT_YOKO_HW_RESUME_HUB:-b0-full}"
cd "$ROOT"
"$PY" -m cat_yoko.b0 \
  --save-dir "$SAVE" \
  --save-every "$SAVE_EVERY" \
  --keep-last "$KEEP_LAST" \
  --upcycle-hf "$LOCAL" \
  --log-every 1 \
  "$@" \
  "${EXTRA[@]+"${EXTRA[@]}"}" \
  "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
ec=$?
echo "=== B0 next-card exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
exit "$ec"
