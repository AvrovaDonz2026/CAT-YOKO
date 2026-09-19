#!/bin/bash
# B200 / SM100: torch cu128 + Transformer Engine pytorch extension.
# TE 2.19 pytorch is an sdist: MUST use --no-build-isolation so the
# extension links against the venv torch (uv isolation pulls a newer
# libtorch and then import hits CUDAErrorLogCapture undefined).
# Does not download 50B tokens.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
if [ -x /venv/main/bin/python ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
  PY="$(command -v python)"
fi
LOG="${LOG:-$ROOT/runs/b200_upgrade.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== B200 upgrade $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true
"$PY" -c "import sys; print('python', sys.executable, sys.version.split()[0])"
if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install)
else
  PIP=("$PY" -m pip install)
fi
export TMPDIR="${TMPDIR:-/dev/shm}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
CUDNN_PATH="$("$PY" -c "import nvidia.cudnn,sys; p=list(getattr(nvidia.cudnn,'__path__',[]) or []); print(p[0] if p else '')" 2>/dev/null || true)"
if [ -n "$CUDNN_PATH" ]; then
  export CUDNN_PATH CUDNN_HOME="$CUDNN_PATH"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
  echo "CUDNN_PATH=$CUDNN_PATH"
fi
echo "install torch cu128"
"${PIP[@]}" torch --index-url https://download.pytorch.org/whl/cu128
echo "install TE build deps"
"${PIP[@]}" ninja packaging pybind11 einops nvidia-cudnn-frontend nvdlfw-inspect
echo "install transformer_engine matching torch CUDA"
CUDA_MM="$("$PY" -c "import torch; print((torch.version.cuda or '12.8').split('.')[0])" 2>/dev/null || echo 12)"
if [ "$CUDA_MM" = "13" ]; then
  TE_EXTRA="transformer_engine[pytorch,core-cu13]"
else
  TE_EXTRA="transformer_engine[pytorch,core-cu12]"
fi
echo "TE extra $TE_EXTRA (torch CUDA major $CUDA_MM) no-build-isolation NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
uv pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch 2>/dev/null || true
if ! "${PIP[@]}" --no-build-isolation "$TE_EXTRA"; then
  echo "TE extra failed; core + --no-deps torch extension"
  "${PIP[@]}" "transformer-engine-cu${CUDA_MM}" transformer-engine
  "${PIP[@]}" --no-build-isolation --no-deps transformer-engine-torch || true
fi
# [pytorch] extra on TE 2.19 also pulls the newest CUDA core; drop the mismatch.
if [ "$CUDA_MM" = "12" ]; then
  uv pip uninstall -y transformer-engine-cu13 2>/dev/null || true
else
  uv pip uninstall -y transformer-engine-cu12 2>/dev/null || true
fi
echo "install transformers / hub"
"${PIP[@]}" "transformers>=4.51" safetensors accelerate huggingface_hub
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
"$PY" -c "import transformer_engine as te, transformer_engine.pytorch; print('te', getattr(te, '__version__', 'ok'))" || echo "TE pytorch import failed"
PROBE_JSON="${PROBE_JSON:-$ROOT/runs/NVFP4_PROBE.json}"
mkdir -p "$(dirname "$PROBE_JSON")"
if [ -f "$ROOT/scripts/probe_nvfp4_hw.py" ]; then
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$ROOT/scripts/probe_nvfp4_hw.py" | tee "$PROBE_JSON"
fi
echo "=== B200 upgrade exit 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h / | tail -1
exit 0
