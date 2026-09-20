"""Published Phase B thinking mix: OpenBMB web/math plus a modest code slice.

The model is meant to do coding work, so default Phase B/C is not 0% code.
StarCoder is not OpenBMB; do not download it in CI / this VM. Optional
``phase-b-code`` raises the slice to 10% (Ultra-FineWeb paper eval mix).
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


# Thinking (Phase B/C): keep math at 10%, shave 5% off English web for code.
# Not the Ultra-FineWeb paper's 10% code slot (that is optional phase-b-code).
# DummyStream uses the same fraction with builtin snippets — no Hub download.
THINK_CODE_FRAC = 0.05

PHASE_B = (
    Source("ultrafineweb-en", "openbmb/Ultra-FineWeb", 0.55, config="en"),
    Source("ultrafineweb-zh", "openbmb/Ultra-FineWeb", 0.30, config="zh"),
    Source(
        "ultradata-math",
        "openbmb/UltraData-Math",
        0.10,
        config="l2",
        text_fields=("content", "text"),
        notes="L2 quality-selected math; fall back to default split if config missing",
    ),
    Source(
        "starcoder",
        "bigcode/starcoderdata",
        THINK_CODE_FRAC,
        text_fields=("content", "text"),
        notes="modest thinking-mix code; not OpenBMB; not downloaded in CI / this VM",
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
        notes="optional extra 10% code (Ultra-FineWeb paper eval mix); not OpenBMB",
    ),
)

# Original short snippets for DummyStream (no StarCoder download). Tile to seq.
THINK_CODE_SNIPPETS = (
    "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n",
    "class Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n        return self.n\n",
    "def qsort(xs):\n    if len(xs) < 2:\n        return xs\n    p = xs[0]\n    return qsort([x for x in xs[1:] if x < p]) + [p] + qsort([x for x in xs[1:] if x >= p])\n",
    "function clamp(x, lo, hi) {\n  return Math.min(hi, Math.max(lo, x));\n}\n",
    "int gcd(int a, int b) {\n  while (b) { int t = a % b; a = b; b = t; }\n  return a;\n}\n",
    "set -euo pipefail\nsum=0\nfor n in 1 2 3 4; do\n  sum=$((sum + n))\ndone\necho \"$sum\"\n",
    "SELECT id, name FROM users WHERE active = 1 ORDER BY id LIMIT 20;\n",
    "fn fib(n: u32) -> u32 {\n    if n < 2 { return n; }\n    fib(n - 1) + fib(n - 2)\n}\n",
    "from typing import Iterable\n\ndef uniq(xs: Iterable[str]) -> list[str]:\n    seen: set[str] = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
    "const merge = (a, b) => {\n  const out = [];\n  let i = 0, j = 0;\n  while (i < a.length && j < b.length) {\n    out.push(a[i] <= b[j] ? a[i++] : b[j++]);\n  }\n  return out.concat(a.slice(i), b.slice(j));\n};\n",
)

PHASE_F = (
    Source(
        "ultrachat",
        "openbmb/UltraChat",
        1.0,
        text_fields=("data", "content", "text"),
        notes="Phase F SFT; not downloaded in CI / this VM",
    ),
)

# Long-context D: same OpenBMB web + extra code for repo-length concat.
# Not downloaded in CI. Sidecar 4K bins are re-windowed at train time.
PHASE_D = (
    Source("ultrafineweb-en", "openbmb/Ultra-FineWeb", 0.45, config="en"),
    Source("ultrafineweb-zh", "openbmb/Ultra-FineWeb", 0.20, config="zh"),
    Source(
        "ultradata-math",
        "openbmb/UltraData-Math",
        0.10,
        config="l2",
        text_fields=("content", "text"),
    ),
    Source(
        "starcoder",
        "bigcode/starcoderdata",
        0.25,
        text_fields=("content", "text"),
        notes="long-context code concat; not OpenBMB; not downloaded in CI / this VM",
    ),
)

PHASE_E = (
    Source("ultrafineweb-en", "openbmb/Ultra-FineWeb", 0.30, config="en"),
    Source("ultrafineweb-zh", "openbmb/Ultra-FineWeb", 0.15, config="zh"),
    Source(
        "ultradata-math",
        "openbmb/UltraData-Math",
        0.25,
        config="l2",
        text_fields=("content", "text"),
        notes="WSD decay: math; not downloaded in CI / this VM",
    ),
    Source(
        "starcoder",
        "bigcode/starcoderdata",
        0.15,
        text_fields=("content", "text"),
        notes="WSD decay code; not OpenBMB; not downloaded in CI / this VM",
    ),
    Source(
        "ultrachat",
        "openbmb/UltraChat",
        0.15,
        text_fields=("data", "content", "text"),
        notes="instruction precursor packed as text; Phase F tokenizes as SFT",
    ),
)

MIXES = {
    "phase-b": PHASE_B,
    "phase-b-code": PHASE_B_WITH_CODE,
    "phase-c": PHASE_B,
    "phase-d": PHASE_D,
    "phase-e": PHASE_E,
    "phase-f": PHASE_F,
    "phase-g": PHASE_F,
}


def mix_named(name: str) -> tuple[Source, ...]:
    if name not in MIXES:
        raise KeyError(f"unknown mix {name}; choose from {sorted(MIXES)}")
    return MIXES[name]


SFT_MIXES = frozenset({"phase-f", "phase-g"})


def default_mix(phase: str) -> str:
    """Prepare mix for a training phase. Does not download."""
    p = str(phase)
    if p.startswith("C"):
        return "phase-c"
    if p.startswith("D"):
        return "phase-d"
    if p == "E":
        return "phase-e"
    if p == "F":
        return "phase-f"
    if p.startswith("G"):
        return "phase-g"
    return "phase-b"


def is_sft_mix(name: str) -> bool:
    return str(name) in SFT_MIXES


def mix_code_weight(sources: tuple[Source, ...] | None = None) -> float:
    """StarCoder (or other non-OpenBMB code) weight in a mix. Default: thinking Phase B."""
    src = PHASE_B if sources is None else sources
    return float(sum(s.weight for s in src if s.key == "starcoder"))


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
