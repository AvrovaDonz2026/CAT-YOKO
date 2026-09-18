#!/bin/bash
# AutoDL RTX 6000D: NVFP4 wrap smoke, then a short B0/B1 --try.
# Attention stays published causal YOCO; this only exercises Nvfp4Linear GEMMs.
set -uo pipefail
ROOT="${ROOT:-/root/autodl-tmp/CAT-YOKO}"
# shellcheck disable=SC1091
source "$ROOT/scripts/autodl_env.sh"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
PY="${PY:-/root/miniconda3/bin/python3}"
SAVE="${SAVE:-/root/autodl-tmp/runs/nvfp4}"
LOG="${LOG:-/root/autodl-tmp/runs/nvfp4_smoke.log}"
LOCAL="${LOCAL:-/root/autodl-tmp/hf/MiniCPM5-2B-Base}"
mkdir -p "$SAVE"
exec > >(tee -a "$LOG") 2>&1
echo "=== NVFP4 smoke $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv,noheader
df -h /root/autodl-tmp | tail -1
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
"$PY" -c "import transformer_engine, sys; print('te', getattr(transformer_engine, '__version__', 'ok'))" || echo "TE not importable; emulation path"
cd "$ROOT"
"$PY" -m unittest tests.test_nvfp4_linear tests.test_attention_plan -v
"$PY" -m cat_yoko.gpu_smoke --json | tee "$SAVE/gpu_smoke_tiny.json"
UPCYCLE_ARGS=(--dummy-upcycle)
if [ -d "$LOCAL" ]; then
  UPCYCLE_ARGS=(--upcycle-hf "$LOCAL")
  echo "using MiniCPM5-2B-Base from $LOCAL"
fi
"$PY" -m cat_yoko.b0 --try --save-dir "$SAVE/b0" "${UPCYCLE_ARGS[@]}" --steps 8
ec0=$?
"$PY" -m cat_yoko.b1 --try --save-dir "$SAVE/b1" --resume "$SAVE/b0" --steps 4 || true
echo "=== b0 exit $ec0 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$SAVE" "$SAVE/b0" 2>/dev/null || true
exit "$ec0"
