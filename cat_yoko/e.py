"""Phase E: WSD decay to ~1/100 peak LR.

``python3 -m cat_yoko.e --try``
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_e

if __name__ == "__main__":
    raise SystemExit(main_e(sys.argv[1:]))
