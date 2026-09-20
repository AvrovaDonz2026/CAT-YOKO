#!/bin/bash
# RTX 3090 Ampere BF16: resume published B0 overlay, write a NEW overlay dir.
#
# Does not overwrite Hub checkpoints/b0-full (step 26940 / sha256 7eebc9a4…).
# Hub copy is always verified read-only. SAVE is a sibling. Same-dir resume is
# allowed on that sibling (STEPS=0 keeps going toward 8e9).
# DummyStream only (thinking mix: 5% hashed code snippets). Does not download Ultra-FineWeb.
# Does not persist a full-graph checkpoint.
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
# cpu_adam JIT looks up `ninja` on PATH. AutoDL SSH PATH has no miniconda.
PY_BIN="$(dirname "$PY")"
if [[ -x "$PY_BIN/ninja" || -x "$PY_BIN/python3" ]]; then
  export PATH="$PY_BIN:$PATH"
fi
# Miniconda libstdc++ stops at GLIBCXX_3.4.26; cpu_adam.so needs 3.4.30.
SYS_STDCPP="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if [[ -f "$SYS_STDCPP" ]]; then
  export LD_PRELOAD="${SYS_STDCPP}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

HUB_COPY="${HUB_COPY:-$WORK/hub-b0-full}"
RESUME="${RESUME:-$HUB_COPY}"
SAVE="${SAVE:-$WORK/b0-3090-bf16}"
LOG="${LOG:-$WORK/b0-3090-bf16/b0_3090.log}"
LOCAL="${LOCAL:-$WORK/hf/MiniCPM5-2B-Base}"
SEQ="${SEQ:-4096}"
# 50 步约 4–5 分钟一次 ZeRO-3 gather，GPU 占用会掉到 0 约 6s。
SAVE_EVERY="${SAVE_EVERY:-200}"
KEEP_LAST="${KEEP_LAST:-2}"
# 每步 D2H / DS grad-norm 会把 GPU 打到 0%。20 步打一行仍能看 tok/s。
LOG_EVERY="${LOG_EVERY:-20}"
# STEPS is extra optimizer steps after resume. 0 = run until the 8e9 envelope
# (or the instance dies). --steps is an absolute cap; Hub is already 26940.
STEPS="${STEPS:-8}"
# Inductor 32 workers steal CPU from ZeRO pin_memory copies.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
# Multiple copy engines so ZeRO H2D overlaps GEMM. NCCL often sets this to 1.
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
# DeepSpeedCPUAdam / pin_memory copies. Too many OpenMP threads steal from PCIe.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

hub_overlay_forbidden() {
  local p="$1"
  case "$p" in
    *checkpoints/b0-full*|*checkpoints/b0/*|*hub-b0-full*)
      return 0
      ;;
  esac
  return 1
}

save_abs="$(readlink -f "$SAVE" 2>/dev/null || echo "$SAVE")"
resume_abs="$(readlink -f "$RESUME" 2>/dev/null || echo "$RESUME")"
if hub_overlay_forbidden "$SAVE" || hub_overlay_forbidden "$save_abs"; then
  echo "SAVE=$SAVE would clobber the published B0 overlay; pick a sibling dir" >&2
  exit 2
fi
if [[ "$save_abs" == "$resume_abs" ]] && hub_overlay_forbidden "$RESUME"; then
  echo "SAVE must not be the Hub overlay copy; refusing to overwrite B0 26940" >&2
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
# DeepSpeedCPUAdam JIT-builds cpu_adam. Without ninja it silently falls back
# to torch AdamW on ZeRO CPU shards (~1s/step, GPU util 0%).
if ! "$PY" -c "from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability()" >/dev/null 2>&1; then
  echo "ninja missing; installing so DeepSpeedCPUAdam can JIT"
  "$PY" -m pip install ninja || echo "ninja install failed; Adam stays torch"
fi
if "$PY" -c "import torch; from deepspeed.ops.adam import DeepSpeedCPUAdam; DeepSpeedCPUAdam([torch.nn.Parameter(torch.zeros(8))], lr=1e-3)" >/dev/null 2>&1; then
  echo "DeepSpeedCPUAdam op ready"
else
  echo "DeepSpeedCPUAdam unavailable; trainer will use torch AdamW"
fi

if [[ ! -d "$LOCAL" ]] || [[ ! -f "$LOCAL/model.safetensors" ]]; then
  echo "MiniCPM5 missing at $LOCAL; downloading"
  "$PY" "$ROOT/scripts/download_minicpm5.py" --local-dir "$LOCAL" || {
    echo "MiniCPM5 download failed" >&2
    exit 2
  }
fi

sha256_file() {
  "$PY" -c "import hashlib,sys; h=hashlib.sha256();
f=open(sys.argv[1],'rb');
[h.update(c) for c in iter(lambda:f.read(1<<20), b'')];
print(h.hexdigest())" "$1"
}

if [[ ! -f "$HUB_COPY/trainable.pt" ]]; then
  echo "pull Hub b0-full overlay into read copy $HUB_COPY"
  HF_ENDPOINT="${HUB_ENDPOINT:-https://huggingface.co}" \
    "$PY" "$ROOT/scripts/download_hub_overlay.py" --name b0-full --out-dir "$HUB_COPY" || {
    echo "Hub overlay download failed; refusing a fresh upcycle that would drop step 26940" >&2
    exit 2
  }
fi
hub_got="$(sha256_file "$HUB_COPY/trainable.pt")"
if [[ "$hub_got" != "$HUB_SHA" ]]; then
  echo "Hub copy sha256 $hub_got != $HUB_SHA; refusing to train on a mutated overlay" >&2
  exit 2
fi
chmod a-w "$HUB_COPY/trainable.pt" 2>/dev/null || true
echo "Hub overlay copy ok sha256=$hub_got (read-only; will not write this file)"

if [[ ! -f "$RESUME/trainable.pt" ]]; then
  echo "resume overlay missing at $RESUME" >&2
  exit 2
fi
if hub_overlay_forbidden "$RESUME"; then
  echo "resume (read-only Hub copy) $RESUME"
else
  echo "resume (sibling overlay, not Hub) $RESUME"
fi
echo "save (new overlay, not Hub) $SAVE"

MORE_ARGS=()
if [[ "$STEPS" != "0" ]]; then
  MORE_ARGS+=(--more-steps "$STEPS")
  echo "B0 Ampere argv seq=$SEQ more-steps=$STEPS save-every=$SAVE_EVERY no-nvfp4 zero-3"
else
  echo "B0 Ampere argv seq=$SEQ more-steps=unlimited (8e9 envelope) save-every=$SAVE_EVERY no-nvfp4 zero-3"
fi

cd "$ROOT"
"$PY" -m cat_yoko.b0 \
  --backend deepspeed --zero 3 --zero-offload --zero-offload-param \
  --no-nvfp4 \
  --device cuda --dtype bf16 \
  --seq-len "$SEQ" --micro-batch 1 --grad-ckpt \
  "${MORE_ARGS[@]}" \
  --save-dir "$SAVE" \
  --save-every "$SAVE_EVERY" \
  --keep-last "$KEEP_LAST" \
  --log-every "$LOG_EVERY" \
  --resume "$RESUME" \
  --upcycle-hf "$LOCAL" \
  --log "$SAVE/metrics.jsonl"
ec=$?
echo "=== B0 Ampere 3090 exit ${ec} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" || true
ls -lh "$RESUME/trainable.pt" || true
exit "$ec"
