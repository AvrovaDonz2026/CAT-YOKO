"""Phase C: indexer KL → top-k mask → HCA pool → 8K window.

``python3 -m cat_yoko.c --try --stage indexer``
``--stage kda|indexer|topk|hca|win`` or ``--chain`` (indexer→topk→hca→win).
Implement KDA on B (``python3 -m cat_yoko.b0 --use-kda``); C only lights.
``--chain`` after a ``use_kda`` overlay: C-kda first, then CSA/HCA.
No CSA CUDA kernel: theorem-B union mask + HCA concat only.
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_c

if __name__ == "__main__":
    raise SystemExit(main_c(sys.argv[1:]))
