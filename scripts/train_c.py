#!/usr/bin/env python3
"""Phase C. python3 scripts/train_c.py --try --stage indexer --save-dir checkpoints/c-index
python3 scripts/train_c.py --try --chain --save-dir checkpoints/c"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.phase_train import main_c

if __name__ == "__main__":
    raise SystemExit(main_c(sys.argv[1:]))
