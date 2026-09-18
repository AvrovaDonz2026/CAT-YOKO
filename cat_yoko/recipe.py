"""Published Phase B data mix: OpenBMB Ultra-FineWeb + UltraData-Math.

Code (StarCoder) is an optional extra — not OpenBMB. Default mix is 100% OpenBMB.
UltraChat / 指令对话是 Phase F/G，不进这张表。
"""

from __future__ import annotations

from dataclasses import dataclass

# Apache-2.0 MiniCPM5. Base = upcycle/teacher weights; instruct = tokenizer.
# MiniCPM-2B / MiniCPM3 / MiniCPM4 are rejected at load time (GML or wrong graph).
MINICPM5_HF = "openbmb/MiniCPM5-2B-Base"
MINICPM5_TOKENIZER = "openbmb/MiniCPM5-2B"
MINICPM_HF = MINICPM5_HF
MINICPM_TOKENIZER = MINICPM5_TOKENIZER
MINICPM5_HIDDEN = 2048
MINICPM5_VOCAB = 130_560
MINICPM5_LAYERS = 42
MINICPM5_HEADS = 16
MINICPM5_KV_HEADS = 2
MINICPM5_INTERMEDIATE = 6144


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
        notes="not OpenBMB; MiniCPM5 / Ultra-FineWeb eval mix used 10% code",
    ),
)

MIXES = {"phase-b": PHASE_B, "phase-b-code": PHASE_B_WITH_CODE}


def mix_named(name: str) -> tuple[Source, ...]:
    if name not in MIXES:
        raise KeyError(f"unknown mix {name}; choose from {sorted(MIXES)}")
    return MIXES[name]


_LEGACY_MINICPM = ("minicpm2b", "minicpm3", "minicpm4")


def _compact_model_id(source: str) -> str:
    return str(source).lower().replace("_", "").replace("-", "").replace("/", "")


def is_minicpm5_id(source: str) -> bool:
    return "minicpm5" in _compact_model_id(source)


def assert_minicpm5_id(source: str, *, kind: str = "checkpoint") -> str:
    """Reject MiniCPM-2B / 3 / 4 Hub ids and paths. Local anonymous dirs pass."""
    text = str(source)
    compact = _compact_model_id(text)
    if "minicpm5" in compact:
        return text
    for marker in _LEGACY_MINICPM:
        if marker in compact:
            raise ValueError(
                f"{text} is not MiniCPM5-2B ({kind}). "
                f"upcycle/teacher = {MINICPM5_HF}; tokenizer = {MINICPM5_TOKENIZER} "
                "(Apache-2.0 Llama GQA). MiniCPM-2B-sft-bf16 is GML and is rejected."
            )
    return text


def assert_minicpm5_hf_config(cfg: object) -> None:
    """Fail closed if a loaded HF config is MiniCPM-2B MHA (d=2304) or another graph."""
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    kv = int(getattr(cfg, "num_key_value_heads", 0) or 0)
    heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    mid = int(getattr(cfg, "intermediate_size", 0) or 0)
    bad: list[str] = []
    if hidden and hidden != MINICPM5_HIDDEN:
        bad.append(f"hidden_size={hidden} (want {MINICPM5_HIDDEN})")
    if vocab and vocab != MINICPM5_VOCAB:
        bad.append(f"vocab_size={vocab} (want {MINICPM5_VOCAB})")
    if kv and kv != MINICPM5_KV_HEADS:
        bad.append(f"num_key_value_heads={kv} (want {MINICPM5_KV_HEADS} GQA)")
    if heads and heads != MINICPM5_HEADS:
        bad.append(f"num_attention_heads={heads} (want {MINICPM5_HEADS})")
    if layers and layers != MINICPM5_LAYERS:
        bad.append(f"num_hidden_layers={layers} (want {MINICPM5_LAYERS})")
    if mid and mid != MINICPM5_INTERMEDIATE:
        bad.append(f"intermediate_size={mid} (want {MINICPM5_INTERMEDIATE})")
    if bad:
        raise ValueError(
            "checkpoint is not MiniCPM5-2B Llama GQA: " + "; ".join(bad)
        )
