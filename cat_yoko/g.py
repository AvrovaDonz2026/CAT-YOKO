"""Phase G RL: GRPO (default) or DPO.

``python3 -m cat_yoko.g --try``
``--algo dpo``. Dummy RLVR reward for --try; PDSA / PS-PPO stay off.
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_g

if __name__ == "__main__":
    raise SystemExit(main_g(sys.argv[1:]))
