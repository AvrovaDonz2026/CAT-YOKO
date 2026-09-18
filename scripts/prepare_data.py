#!/usr/bin/env python3
"""Entry: python3 scripts/prepare_data.py --mix local --local texts.jsonl --tokenizer dummy --out t.bin"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.prepare import main

if __name__ == "__main__":
    raise SystemExit(main())
