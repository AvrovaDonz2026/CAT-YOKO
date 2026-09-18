#!/usr/bin/env python3
"""Published C1 B1. python3 scripts/train_b1.py --resume checkpoints/b0"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.phase_train import main_b1

if __name__ == "__main__":
    raise SystemExit(main_b1(sys.argv[1:]))
