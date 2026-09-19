"""Phase F SFT: loss on response tokens only.

``python3 -m cat_yoko.f --try``
DummyStream masks the prompt prefix (labels=-100). jsonl may ship ``labels``,
``prompt_ids``/``response_ids``, or tokenized ``messages``.
Prepare: ``python3 -m cat_yoko.prepare --mix phase-f --out sft.jsonl``
(UltraChat turns → packed jsonl; not downloaded here). ``sparse=hca``.
"""

from __future__ import annotations

import sys

from cat_yoko.phase_train import main_f

if __name__ == "__main__":
    raise SystemExit(main_f(sys.argv[1:]))
