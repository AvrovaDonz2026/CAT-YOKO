#!/usr/bin/env python3
"""Phase E. python3 scripts/train_e.py --try --save-dir checkpoints/e"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.phase_train import main_e

if __name__ == "__main__":
    raise SystemExit(main_e(sys.argv[1:]))
