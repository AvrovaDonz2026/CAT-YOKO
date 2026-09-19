"""Phase E: WSD decay to ~1/100 peak LR.

``python3 -m cat_yoko.e --try``
Prepare ``--mix phase-e`` (HQ web + math + code + UltraChat-as-text).
Does not download those datasets in CI. ``sparse=hca``; inherits ``use_kda``.
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_e

if __name__ == "__main__":
    raise SystemExit(main_e(sys.argv[1:]))
