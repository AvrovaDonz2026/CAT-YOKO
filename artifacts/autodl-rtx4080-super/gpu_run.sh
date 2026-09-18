#!/bin/bash
# AutoDL GPU verify for CAT-YOKO-12B. Do not commit this host script.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/autodl-tmp/CAT-YOKO
PY=/root/miniconda3/bin/python3
LOG=/root/autodl-tmp/gpu_run.log
exec > >(tee -a "$LOG") 2>&1
cd /root/autodl-tmp/CAT-YOKO
echo "===== START $(date -Is) ====="
tar -xzf /root/autodl-tmp/cat-yoko-sync.tgz
$PY -c "import cat_yoko.checkpoint as c; print('checkpoint', c.publish_latest, c.resolve_resume_path)"

echo "===== MOVE OLD /tmp CKPT ====="
mkdir -p /root/autodl-tmp/old_b0ckpt
if [ -f /tmp/b0ckpt/step_1.pt ]; then
  mv /tmp/b0ckpt/step_1.pt /root/autodl-tmp/old_b0ckpt/step_1.pt
fi
rm -rf /tmp/b0ckpt
df -h /tmp /root/autodl-tmp /
ls -lh /root/autodl-tmp/old_b0ckpt || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

echo "===== 1. RESUME old CPU step_1.pt (no save) ====="
$PY -m cat_yoko.train --config 12b --phase B0 --device cuda --dtype bf16 \
  --steps 2 --seq-len 64 --micro-batch 1 --accum 1 \
  --resume /root/autodl-tmp/old_b0ckpt/step_1.pt
echo "RESUME_OLD_EXIT:$?"
$PY -c "import torch; torch.cuda.synchronize(); print('idle_mib', torch.cuda.memory_allocated()//1024**2)"

echo "===== 2. TINY CUDA ====="
$PY -m cat_yoko.gpu_smoke
echo "TINY_EXIT:$?"

echo "===== 3. 12B B0 save hardlink on autodl-tmp ====="
rm -rf /root/autodl-tmp/b0ckpt
$PY -m cat_yoko.gpu_smoke --middle --phase B0 --seq-len 64 --steps 1 \
  --save-dir /root/autodl-tmp/b0ckpt --json
echo "SAVE_EXIT:$?"
echo "--- ckpt dir ---"
ls -lh /root/autodl-tmp/b0ckpt || true
stat -c '%n ino=%i nlink=%h size=%s' /root/autodl-tmp/b0ckpt/* || true
df -h /root/autodl-tmp /tmp

echo "===== 4. 12B B0 resume from directory ====="
$PY -m cat_yoko.gpu_smoke --middle --phase B0 --seq-len 64 --steps 2 \
  --resume /root/autodl-tmp/b0ckpt --json
echo "RESUME_DIR_EXIT:$?"

echo "===== 5. 12B B1 isolated ====="
$PY -m cat_yoko.gpu_smoke --middle --phase B1 --seq-len 64 --json
echo "B1_EXIT:$?"

echo "===== 6. 12B B2 isolated ====="
$PY -m cat_yoko.gpu_smoke --middle --phase B2 --seq-len 64 --json
echo "B2_EXIT:$?"

echo "===== 7. 12B C1 chain ====="
$PY -m cat_yoko.gpu_smoke --c1 --seq-len 64 --json
echo "C1_EXIT:$?"

echo "===== END $(date -Is) ====="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
df -h /tmp /root/autodl-tmp /
ls -lh /root/autodl-tmp/b0ckpt || true
echo DONE
