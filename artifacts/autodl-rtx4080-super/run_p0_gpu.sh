#!/bin/bash
set -u
export PYTHONPATH=/root/autodl-tmp/CAT-YOKO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/CAT-YOKO
PY=/root/miniconda3/bin/python3
OUT=/tmp/p0_gpu
mkdir -p "$OUT"
run() {
  local name=$1; shift
  echo "START $name $(date -u +%H:%M:%S)" | tee "$OUT/${name}.log"
  $PY -m cat_yoko.gpu_smoke "$@" --json > "$OUT/${name}.json" 2>> "$OUT/${name}.log"
  echo EXIT:$? | tee -a "$OUT/${name}.log"
}
run tiny
run b0 --middle --phase B0 --seq-len 64
run b1 --middle --phase B1 --seq-len 64
run b2 --middle --phase B2 --seq-len 64
run c1 --c1 --seq-len 64
echo DONE $(date -u +%H:%M:%S) | tee "$OUT/done"
