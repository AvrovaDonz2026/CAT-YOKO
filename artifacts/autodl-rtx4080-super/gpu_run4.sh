#!/bin/bash
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/CAT-YOKO
PY=/root/miniconda3/bin/python3
echo "=== START $(date -Is) ==="
$PY - << 'PY'
from cat_yoko.gpu_smoke import cuda_info
from cat_yoko.trainer import enable_expandable_segments
print("ALLOC", enable_expandable_segments())
print("INFO", cuda_info())
PY
echo "=== TINY ==="
set +e
$PY -m cat_yoko.gpu_smoke --json
echo TINY_EXIT:$?
set -e
echo "=== IDLE ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "=== DONE $(date -Is) ==="
