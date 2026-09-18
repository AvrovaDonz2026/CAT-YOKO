#!/bin/bash
set -u
export PYTHONPATH=/root/autodl-tmp/CAT-YOKO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/CAT-YOKO
PY=/root/miniconda3/bin/python3
OUT=/tmp/minicpm5_cuda
mkdir -p "$OUT"
run() {
  local name=$1; shift
  echo "START $name $(date -u +%H:%M:%S)" | tee "$OUT/${name}.log"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$OUT/${name}.log"
  $PY -m cat_yoko.gpu_smoke "$@" --json > "$OUT/${name}.json" 2>> "$OUT/${name}.log"
  echo EXIT:$? | tee -a "$OUT/${name}.log"
  $PY -c "import gc,torch; gc.collect(); torch.cuda.empty_cache(); print(\"allocGiB\", round(torch.cuda.memory_allocated()/1024**3,3))" | tee -a "$OUT/${name}.log"
}
run b0 --middle --phase B0 --seq-len 64
run b1 --middle --phase B1 --seq-len 64
run b2 --middle --phase B2 --seq-len 64
run c1 --c1 --seq-len 64
run b0s2 --middle --phase B0 --steps 2 --seq-len 64
echo DONE $(date -u +%H:%M:%S) | tee "$OUT/done"
