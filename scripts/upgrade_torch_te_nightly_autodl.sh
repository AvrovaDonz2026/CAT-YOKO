#!/bin/bash
# AutoDL: install PyTorch nightly (cu130) + Transformer Engine into a *new*
# venv. Does not touch the running B0 process (that still uses miniconda 2.8).
#
# After probe, restart B0 with:
#   PY=/root/autodl-tmp/venv-nightly/bin/python bash scripts/run_b0_full_autodl.sh
#
# Does not download 50B Ultra-FineWeb. Does not write passwords.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
VENV="${VENV:-/root/autodl-tmp/venv-nightly}"
BASE_PY="${BASE_PY:-/root/miniconda3/bin/python3}"
LOG="${LOG:-/root/autodl-tmp/runs/nightly_upgrade.log}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/nightly/cu130}"
mkdir -p "$(dirname "$LOG")" "$VENV"
exec > >(tee -a "$LOG") 2>&1
echo "=== nightly upgrade $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h / /root/autodl-tmp | tail -2
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

if [ ! -x "$BASE_PY" ]; then
  echo "missing $BASE_PY" >&2
  exit 2
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv $VENV"
  "$BASE_PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel ninja packaging
echo "install torch nightly from $TORCH_INDEX"
python -m pip install --pre torch --index-url "$TORCH_INDEX"

# TE: CUDA 13 core + PyTorch extension rebuilt against this nightly.
# Previous 2.8+cu128 install had transformer_engine.pytorch .so missing /
# ncclCommWindowRegister. Uninstall any leftover cu12 meta packages first.
python -m pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch transformer-engine-jax || true
echo "install transformer_engine[pytorch,core-cu13]"
if ! python -m pip install --no-build-isolation "transformer_engine[pytorch,core-cu13]"; then
  echo "pypi TE extra failed; trying GitHub @main"
  export NVTE_FRAMEWORK=pytorch
  export MAX_JOBS="${MAX_JOBS:-2}"
  export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-1}"
  python -m pip install --no-build-isolation "git+https://github.com/NVIDIA/TransformerEngine.git@main"
fi

echo "=== probe ==="
PROBE_JSON="$VENV/NVFP4_PROBE.json"
if [ -f "$ROOT/scripts/probe_nvfp4_hw.py" ]; then
  python "$ROOT/scripts/probe_nvfp4_hw.py" | tee "$PROBE_JSON"
else
  python - <<'PY' | tee "$PROBE_JSON"
import json, sys, torch
out={"torch": torch.__version__, "cuda": torch.version.cuda,
     "cap": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
print(json.dumps(out, indent=2))
PY
fi

python - <<'PY'
import json, os, sys
from pathlib import Path
venv = Path(sys.prefix)
probe = venv / "NVFP4_PROBE.json"
ok = venv / "NVFP4_PROBE_OK"
ok.unlink(missing_ok=True)
data = {}
if probe.is_file():
    try:
        data = json.loads(probe.read_text())
    except Exception as e:
        print("probe json parse fail", e)
        sys.exit(3)
print("torch", data.get("torch"), "cuda", data.get("cuda"), "cap", data.get("cap"))
print("float4_cast", data.get("float4_cast"), "te_pytorch", data.get("te_pytorch"),
      "te_nvfp4_linear", data.get("te_nvfp4_linear"), "hw_nvfp4_gemm", data.get("hw_nvfp4_gemm"))
if data.get("torch") and data.get("cap"):
    ok.write_text("ok\n")
    print("wrote", ok)
    sys.exit(0)
sys.exit(4)
PY
ec=$?
echo "=== nightly upgrade exit $ec $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
df -h /root/autodl-tmp | tail -1
exit "$ec"
