#!/bin/bash
# Build Transformer Engine pytorch extension against the *venv* torch.
# PyPI transformer-engine-torch 2.19 ships a prebuilt .so that imports as
# undefined CUDAErrorLogCapture on torch 2.11.0+cu128. Official docs:
#   pip install --no-build-isolation git+...@stable
# with NVTE_FRAMEWORK=pytorch. Compile only SM100 (B200). Uses /dev/shm.
# Does not download 50B tokens.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
if [ -x /venv/main/bin/python ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
  PY="$(command -v python)"
fi
LOG="${LOG:-$ROOT/runs/te_src_build.log}"
SRC="${TE_SRC:-/dev/shm/TransformerEngine}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== TE source build $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
export TMPDIR="${TMPDIR:-/dev/shm}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
CUDNN_PATH="$("$PY" -c "import nvidia.cudnn; p=list(getattr(nvidia.cudnn,'__path__',[]) or []); print(p[0] if p else '')" 2>/dev/null || true)"
if [ -n "$CUDNN_PATH" ]; then
  export CUDNN_PATH CUDNN_HOME="$CUDNN_PATH"
  export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
fi
if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install)
else
  PIP=("$PY" -m pip install)
fi
"${PIP[@]}" ninja packaging pybind11
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone --branch stable --recursive --depth 1 --shallow-submodules \
    https://github.com/NVIDIA/TransformerEngine.git "$SRC"
  git -C "$SRC" submodule update --init --recursive
fi
echo "TE src $(git -C "$SRC" rev-parse --short HEAD) NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS"
cd "$SRC"
"${PIP[@]}" --no-build-isolation --no-deps .
"$PY" -c "import transformer_engine as te, transformer_engine.pytorch as tep; print('te', getattr(te,'__version__','ok'), 'pytorch', tep)"
echo "=== TE source build exit 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit 0
