#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/CAT-YOKO
PY=/root/miniconda3/bin/python3
echo "=== START $(date -Is) ==="
echo "ALLOC_BEFORE_IMPORT=${PYTORCH_CUDA_ALLOC_CONF-}"
$PY - << 'PY'
from cat_yoko.trainer import enable_expandable_segments
import os, torch
print("ALLOC", enable_expandable_segments())
print("ENV", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("TOTAL_GIB", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
print("BF16", torch.cuda.is_bf16_supported())
PY
echo "=== CLI_NO_SEQLEN ==="
set +e
$PY -m cat_yoko.train --config 12b --device cuda --dtype bf16 --steps 1 --micro-batch 1
echo CLI_NO_SEQLEN_EXIT:$?
echo "=== CLI_TEACHER ==="
$PY -m cat_yoko.train --config 12b --device cuda --dtype bf16 --steps 1 --micro-batch 1 --seq-len 64 --teacher-hf
echo CLI_TEACHER_EXIT:$?
set -e
echo "=== TINY ==="
set +e
$PY -m cat_yoko.gpu_smoke --json
echo TINY_EXIT:$?
set -e
echo "=== IDLE_AFTER_TINY ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== B0 ==="
set +e
$PY -m cat_yoko.gpu_smoke --middle --phase B0 --seq-len 64 --json
echo B0_EXIT:$?
set -e
echo "=== IDLE_AFTER_B0 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== B1 ==="
set +e
$PY -m cat_yoko.gpu_smoke --middle --phase B1 --seq-len 64 --json
echo B1_EXIT:$?
set -e
echo "=== IDLE_AFTER_B1 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== DONE $(date -Is) ==="
