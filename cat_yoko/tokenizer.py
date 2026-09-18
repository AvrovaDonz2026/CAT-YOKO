"""Tokenizers for Phase B. Production: MiniCPM-2B; tests: byte-hash dummy."""

from __future__ import annotations

from typing import Protocol

from cat_yoko.config import CATYokoConfig

MINICPM_TOKENIZER = "openbmb/MiniCPM-2B-sft-bf16"


class Tokenizer(Protocol):
    vocab_size: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...


class HashTokenizer:
    """Deterministic stand-in so prepare/train tests never hit HuggingFace."""

    def __init__(self, vocab_size: int, eos_id: int = 2) -> None:
        if vocab_size < 8:
            raise ValueError("vocab_size too small")
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.name = "dummy"

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for b in text.encode("utf-8") or b"\x00":
            t = b % self.vocab_size
            if t == self.eos_id:
                t = (t + 1) % self.vocab_size
            ids.append(t)
        ids.append(self.eos_id)
        return ids


class MiniCPMTokenizer:
    def __init__(self, name: str = MINICPM_TOKENIZER) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MiniCPM tokenizer needs transformers: pip install 'cat-yoko[data]'"
            ) from exc
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        eos = self.tok.eos_token_id
        self.eos_id = int(eos) if eos is not None else 2
        self.vocab_size = int(getattr(self.tok, "vocab_size", None) or len(self.tok))
        self.name = name

    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if not ids:
            return [self.eos_id]
        if ids[-1] != self.eos_id:
            ids = list(ids) + [self.eos_id]
        return [int(t) for t in ids]


def load_tokenizer(name: str, cfg: CATYokoConfig | None = None) -> Tokenizer:
    if name in {"dummy", "hash"}:
        vocab = cfg.vocab_size if cfg is not None else 128
        return HashTokenizer(vocab)
    return MiniCPMTokenizer(name)
