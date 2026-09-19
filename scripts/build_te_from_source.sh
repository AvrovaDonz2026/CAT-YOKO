#!/bin/bash
# Build Transformer Engine pytorch extension against the *venv* torch.
# PyPI transformer-engine-torch 2.19 ships a prebuilt .so that imports as
# undefined CUDAErrorLogCapture on torch 2.11.0+cu128. Official docs:
#   pip install --no-build-isolation git+...@stable
# with NVTE_FRAMEWORK=pytorch. Compile only SM100 (B200). Uses /dev/shm.
#
# CUTLASS SM100A stg.256 / ldg.256 is gated on nvcc **12.9+**. CUDA 12.8
# builds a core whose 16×128 FPROP works but B0-sized NVFP4 quantize
# prints CUTE_ARCH_STORE256_SM100A_ENABLED and then launch-fails.
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
# nvcc 12.9+ is required for CUTE_ARCH_STORE256_SM100A_ENABLED. The 12.9
# apt package is compiler-only (no cuBLAS). Keep CUDA_HOME on a full
# toolkit (12.8 on Vast) and put 12.9 nvcc first on PATH.
NVCC_BIN=""
for cand in /usr/local/cuda-12.9/bin/nvcc /usr/local/cuda-13.0/bin/nvcc \
            /usr/local/cuda-13.1/bin/nvcc /usr/local/cuda-13.2/bin/nvcc; do
  if [ -x "$cand" ]; then
    NVCC_BIN="$cand"
    break
  fi
done
if [ -z "${CUDA_HOME:-}" ]; then
  if [ -f /usr/local/cuda-12.8/include/cublas_v2.h ]; then
    CUDA_HOME=/usr/local/cuda-12.8
  elif [ -f /usr/local/cuda/include/cublas_v2.h ]; then
    CUDA_HOME=/usr/local/cuda
  elif [ -n "$NVCC_BIN" ]; then
    CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
  else
    CUDA_HOME=/usr/local/cuda
  fi
fi
export CUDA_HOME
if [ -z "$NVCC_BIN" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  NVCC_BIN="$CUDA_HOME/bin/nvcc"
fi
export PATH="$(dirname "${NVCC_BIN:-$CUDA_HOME/bin/nvcc}"):$CUDA_HOME/bin:${PATH:-}"
export CUDACXX="${NVCC_BIN:-$CUDA_HOME/bin/nvcc}"
export CMAKE_CUDA_COMPILER="$CUDACXX"
export TMPDIR="${TMPDIR:-/dev/shm}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
export NVTE_CUDA_INCLUDE_PATH="${NVTE_CUDA_INCLUDE_PATH:-$CUDA_HOME/include}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
echo "nvcc $($CUDACXX --version | tail -1) CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX"
NVCC_MAJ_MIN="$("$CUDACXX" --version | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1\2/p' | head -1)"
if [ -n "$NVCC_MAJ_MIN" ] && [ "$NVCC_MAJ_MIN" -lt 129 ]; then
  echo "ERROR: nvcc $NVCC_MAJ_MIN < 12.9; SM100A stg.256 will be compiled out. Install cuda-nvcc-12-9." >&2
  exit 2
fi
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
# Force a clean CUDA compile so a previous 12.8 cubin is not reused.
rm -rf "$SRC/build" "$SRC/transformer_engine/common/build"
find "$SRC" -name CMakeCache.txt -delete 2>/dev/null || true
cd "$SRC"
"${PIP[@]}" --no-build-isolation --no-deps .
"$PY" - <<'PY'
"""Source metapackage + leftover PyPI cu12 can trip TE's PyPI sanity check
(version suffix mismatch, or two libtransformer_engine.so cores). Keep the
SM100 source .so, drop a fat wheel_lib copy, pin Version to the cu12 core.
"""
from pathlib import Path
import transformer_engine as te

root = Path(te.__file__).resolve().parent
sos = list(root.rglob("libtransformer_engine.so"))
print("te cores", [(str(p.relative_to(root)), round(p.stat().st_size / 1024 / 1024, 1)) for p in sos])
if len(sos) > 1:
    keep = None
    for p in sos:
        if "wheel_lib" not in p.parts:
            keep = p
            break
    if keep is None:
        keep = min(sos, key=lambda p: p.stat().st_size)
    for p in sos:
        if p != keep:
            print("remove extra TE core", p, "MiB", round(p.stat().st_size / 1024 / 1024, 1))
            p.unlink()
site = root.parent
for meta in site.glob("transformer_engine-*.dist-info/METADATA"):
    if "cu12" in meta.parent.name or "cu13" in meta.parent.name or "torch" in meta.parent.name:
        continue
    text = meta.read_text()
    lines = []
    changed = False
    for line in text.splitlines(True):
        if line.startswith("Version:") and "+" in line:
            ver = line.split(":", 1)[1].strip().split("+", 1)[0]
            line = f"Version: {ver}\n"
            changed = True
        lines.append(line)
    if changed:
        meta.write_text("".join(lines))
        print("patched", meta, "Version to match cu core")
for wheel in site.glob("transformer_engine-*.dist-info/WHEEL"):
    if "cu12" in wheel.parent.name or "cu13" in wheel.parent.name or "torch" in wheel.parent.name:
        continue
    text = wheel.read_text()
    if "Root-Is-Purelib: false" in text:
        wheel.write_text(text.replace("Root-Is-Purelib: false", "Root-Is-Purelib: true"))
        print("patched", wheel, "Root-Is-Purelib")
print("te", getattr(te, "__version__", "ok"))
import transformer_engine.pytorch as tep
print("pytorch", tep)
PY
echo "=== TE source build exit 0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit 0
