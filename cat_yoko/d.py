"""Phase D long-context curriculum: 8K → 32K → 128K.

``python3 -m cat_yoko.d --try --stage 8k``
``--stage 32k|128k`` or ``--chain`` (8k→32k→128k).
MiniCPM5 ``rope_theta=5e6``; no extra NTK table.
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_d

if __name__ == "__main__":
    raise SystemExit(main_d(sys.argv[1:]))
