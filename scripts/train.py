#!/usr/bin/env python3
"""Entry point: python3 scripts/train.py --config tiny --phase B0 --steps 3"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.train import main

if __name__ == "__main__":
    raise SystemExit(main())
