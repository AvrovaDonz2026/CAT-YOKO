#!/usr/bin/env python3
"""Entry: python3 scripts/gpu_smoke.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.gpu_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
