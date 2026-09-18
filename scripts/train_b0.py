#!/usr/bin/env python3
"""Published C1 B0. python3 scripts/train_b0.py --try --save-dir checkpoints/b0"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.phase_train import main_b0

if __name__ == "__main__":
    raise SystemExit(main_b0(sys.argv[1:]))
