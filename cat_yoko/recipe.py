"""Published Phase B data mix: OpenBMB Ultra-FineWeb + UltraData-Math.

Code (StarCoder) is an optional extra — not OpenBMB. Default mix is 100% OpenBMB.
UltraChat / 指令对话是 Phase F/G，不进这张表。
"""

from __future__ import annotations

from dataclasses import dataclass

# Apache-2.0 MiniCPM5. Base = upcycle/teacher weights; instruct = tokenizer.
MINICPM_HF = "openbmb/MiniCPM5-2B-Base"
MINICPM_TOKENIZER = "openbmb/MiniCPM5-2B"


@dataclass(frozen=True)
class Source:
    key: str
    repo: str
    weight: float
    config: str | None = None
    split: str = "train"
    text_fields: tuple[str, ...] = ("text", "content")
    notes: str = ""


# Ultra-FineWeb paper mix was 60% en / 30% zh / 10% code. We keep en/zh and
# give the last 10% to UltraData-Math so the default is all-OpenBMB.
PHASE_B = (
    Source("ultrafineweb-en", "openbmb/Ultra-FineWeb", 0.60, config="en"),
    Source("ultrafineweb-zh", "openbmb/Ultra-FineWeb", 0.30, config="zh"),
    Source(
        "ultradata-math",
        "openbmb/UltraData-Math",
        0.10,
        config="l2",
        text_fields=("content", "text"),
        notes="L2 quality-selected math; fall back to default split if config missing",
    ),
)

PHASE_B_WITH_CODE = (
    Source("ultrafineweb-en", "openbmb/Ultra-FineWeb", 0.50, config="en"),
    Source("ultrafineweb-zh", "openbmb/Ultra-FineWeb", 0.30, config="zh"),
    Source("ultradata-math", "openbmb/UltraData-Math", 0.10, config="l2", text_fields=("content", "text")),
    Source(
        "starcoder",
        "bigcode/starcoderdata",
        0.10,
        text_fields=("content", "text"),
        notes="not OpenBMB; MiniCPM / Ultra-FineWeb eval mix used 10% code",
    ),
)

MIXES = {"phase-b": PHASE_B, "phase-b-code": PHASE_B_WITH_CODE}


def mix_named(name: str) -> tuple[Source, ...]:
    if name not in MIXES:
        raise KeyError(f"unknown mix {name}; choose from {sorted(MIXES)}")
    return MIXES[name]
