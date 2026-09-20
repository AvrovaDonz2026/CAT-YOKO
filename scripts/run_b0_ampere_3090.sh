#!/bin/bash
# RTX 3090 Ampere BF16: resume published B0 overlay, write a NEW overlay dir.
#
# Does not overwrite Hub checkpoints/b0-full (step 26940 / sha256 7eebc9a4…).
# Resume dir is read-only after download. Save dir is a sibling, not the Hub pointer.
# DummyStream only. Does not download Ultra-FineWeb. Does not persist a full-graph checkpoint.
# Ampere has no FP4 tensor core: --no-nvfp4 (published C1+NVFP4 wall-clock is unchanged).
# ZeRO-3 + CPU offload to fit 24.5GiB weights. Not Megatron. Not a CSA kernel.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${WORK:-/root/autodl-tmp}"
[ -d "$WORK" ] || WORK="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# MiniCPM5: leave HF_ENDPOINT unset so download_minicpm5.py can pick
# hf-mirror on AutoDL. Hub overlay is a user repo on huggingface.co.
export HF_HOME="${HF_HOME:-$WORK/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HUB_SHA="${HUB_SHA:-7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955}"
if [[ -z "${PY:-}" ]]; then
  if [[ -x /root/miniconda3/bin/python3 ]]; then
    PY=/root/miniconda3/bin/python3
  else
    PY=python3
  fi
fi

RESUME="${RESUME:-$WORK/hub-b0-full}"
SAVE="${SAVE:-$WORK/b0-3090-bf16}"
LOG="${LOG:-$WORK/b0-3090-bf16/b0_3090.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
SEQ="${SEQ:-4096}"
SAVE_EVERY="${SAVE_EVERY:-2}"
KEEP_LAST="${KEEP_LAST:-2}"
STEPS="${STEPS:-8}"

hub_overlay_forbidden() {
  local p="$1"
  case "$p" in
    *checkpoints/b0-full*|*checkpoints/b0/*)
      return 0
      ;;
  esac
  return 1
}

if [[ "$(readlink -f "$SAVE" 2>/dev/null || echo "$SAVE")" == "$(readlink -f "$RESUME" 2>/dev/null || echo "$RESUME")" ]]; then
  echo "SAVE must not be RESUME; refusing to overwrite the Hub overlay copy" >&2
  exit 2
fi
if hub_overlay_forbidden "$SAVE"; then
  echo "SAVE=$SAVE would clobber the published B0 overlay; pick a sibling dir" >&2
  exit 2
fi

mkdir -p "$SAVE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B0 Ampere 3090 BF16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv,noheader || true
fi
df -h / "$WORK" | tail -5 || true
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

if ! "$PY" -c "import deepspeed" >/dev/null 2>&1; then
  echo "DeepSpeed missing; pip install 'deepspeed>=0.14'"
  "$PY" -m pip install 'deepspeed>=0.14' || {
    echo "DeepSpeed install failed; cannot shard 12B on 48GiB without ZeRO-3" >&2
    exit 3
  }
fi

if [[ ! -d "$LOCAL" ]] || [[ ! -f "$LOCAL/model.safetensors" ]]; then
  echo "MiniCPM5 missing at $LOCAL; downloading"
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || {
    echo "MiniCPM5 download failed" >&2
    exit 2
  }
fi

if [[ ! -f "$RESUME/trainable.pt" ]]; then
  echo "pull Hub b0-full overlay into read copy $RESUME"
  HF_ENDPOINT="${HUB_ENDPOINT:-https://huggingface.co}" \
    "$PY" "$ROOT/scripts/download_hub_overlay.py" --name b0-full --out-dir "$RESUME" || {
    echo "Hub overlay download failed; refusing a fresh upcycle that would drop step 26940" >&2
    exit 2
  }
fi
got=$("$PY" -c "import hashlib,sys; h=hashlib.sha256();
f=open(sys.argv[1],'rb');
[h.update(c) for c in iter(lambda:f.read(1<<20), b'')];
print(h.hexdigest())" "$RESUME/trainable.pt")
if [[ "$got" != "$HUB_SHA" ]]; then
  echo "Hub copy sha256 $got != $HUB_SHA; refusing to train on a mutated overlay" >&2
  exit 2
fi
chmod a-w "$RESUME/trainable.pt" 2>/dev/null || true
echo "Hub overlay copy ok sha256=$got (read-only; will not write this file)"

echo "resume (read-only Hub copy) $RESUME"
echo "save (new overlay, not Hub) $SAVE"
echo "B0 Ampere argv seq=$SEQ steps=$STEPS no-nvfp4 zero-3"

cd "$ROOT"
"$PY" -m cat_yoko.b0 \
  --backend deepspeed --zero 3 --zero-offload --zero-offload-param \
  --no-nvfp4 \
  --device cuda --dtype bf16 \
  --seq-len "$SEQ" --micro-batch 1 --grad-ckpt \
  --steps "$STEPS" \
  --save-dir "$SAVE" \
  --save-every "$SAVE_EVERY" \
  --keep-last "$KEEP_LAST" \
  --resume "$RESUME" \
  --upcycle-hf "$LOCAL" \
  --log "$SAVE/metrics.jsonl" \
  --log-every 1
ec=$?
echo "=== B0 Ampere 3090 exit ${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
ls -lh "$RESUME/trainable.pt" || true
exit "$ec"
