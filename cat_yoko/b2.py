"""C1 B2: 15B tokens, full model, block offload. ``python3 -m cat_yoko.b2 --resume checkpoints/b1``."""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_b2

if __name__ == "__main__":
    raise SystemExit(main_b2(sys.argv[1:]))
