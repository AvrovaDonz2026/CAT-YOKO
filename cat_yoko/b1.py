"""C1 B1: 27B tokens, decoder+lm_head+final RMSNorm, nvfp4.

Resume = MiniCPM5 upcycle + B0 ``trainable.pt`` overlay::

    python3 -m cat_yoko.b1 --try --resume checkpoints/b0 --upcycle-hf MiniCPM5
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_b1

if __name__ == "__main__":
    raise SystemExit(main_b1(sys.argv[1:]))
