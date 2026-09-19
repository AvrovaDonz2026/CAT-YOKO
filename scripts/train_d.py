#!/usr/bin/env python3
"""Phase D. python3 scripts/train_d.py --try --stage 8k --save-dir checkpoints/d-8k
python3 scripts/train_d.py --try --chain --save-dir checkpoints/d"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.phase_train import main_d

if __name__ == "__main__":
    raise SystemExit(main_d(sys.argv[1:]))
