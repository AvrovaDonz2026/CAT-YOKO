"""C1 B0: 8B tokens, new-modules only, student bf16.

``python3 -m cat_yoko.b0`` is the published 8e9-token envelope (seq=4096).
``--try`` is the 32-step / seq=64 overlay path. B200: ``--no-offload-encoder
--no-grad-ckpt``. 6000D: ``--no-offload-encoder`` (emulation).
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_b0

if __name__ == "__main__":
    raise SystemExit(main_b0(sys.argv[1:]))
