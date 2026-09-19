"""Tokenize OpenBMB mixes into a packed int32 bin the C1 trainer can mmap."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path
from typing import Iterator

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import sidecar_path
from cat_yoko.recipe import MINICPM5_TOKENIZER, MIXES, Source, mix_named
from cat_yoko.tokenizer import HashTokenizer, Tokenizer, load_tokenizer


def row_text(row: dict, fields: tuple[str, ...]) -> str | None:
    for k in fields:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def iter_local_jsonl(path: Path, fields: tuple[str, ...] = ("text", "content")) -> Iterator[str]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "tokens" in obj or "input_ids" in obj:
                continue
            t = row_text(obj, fields)
            if t:
                yield t


def iter_hf_texts(src: Source) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install 'cat-yoko[data]' to stream HuggingFace datasets") from exc
    kwargs: dict = {"split": src.split, "streaming": True}
    if src.config:
        try:
            ds = load_dataset(src.repo, src.config, **kwargs)
        except Exception:
            ds = load_dataset(src.repo, **kwargs)
    else:
        ds = load_dataset(src.repo, **kwargs)
    for row in ds:
        t = row_text(dict(row), src.text_fields)
        if t:
            yield t


def _pick(keys: list[str], weights: list[float], rng: random.Random) -> str:
    total = sum(weights)
    x = rng.random() * total
    acc = 0.0
    for k, w in zip(keys, weights, strict=True):
        acc += w
        if x <= acc:
            return k
    return keys[-1]


def mix_documents(
    iters: dict[str, Iterator[str]],
    weights: dict[str, float],
    rng: random.Random,
) -> Iterator[tuple[str, str]]:
    keys = list(iters)
    w = [weights[k] for k in keys]
    while True:
        k = _pick(keys, w, rng)
        try:
            text = next(iters[k])
        except StopIteration:
            keys = [x for x in keys if x != k]
            if not keys:
                return
            w = [weights[x] for x in keys]
            continue
        yield k, text


def write_packed_bin(
    docs: Iterator[list[int]],
    out: Path,
    seq_len: int,
    max_tokens: int,
) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    written = 0
    nseq = 0
    with out.open("wb") as f:
        for doc in docs:
            for t in doc:
                buf.append(int(t))
                if len(buf) == seq_len:
                    f.write(struct.pack("<" + "i" * seq_len, *buf))
                    written += seq_len
                    nseq += 1
                    buf = []
                    if written >= max_tokens:
                        return written, nseq
    return written, nseq


def prepare(
    *,
    mix: str,
    out: Path,
    max_tokens: int,
    seq_len: int,
    tokenizer: Tokenizer,
    local_jsonl: Path | None = None,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    if mix == "local":
        if local_jsonl is None:
            raise ValueError("--local jsonl required for mix=local")

        def docs() -> Iterator[list[int]]:
            for text in iter_local_jsonl(local_jsonl):
                yield tokenizer.encode(text)

        sources = ["local"]
        weights_used = {"local": 1.0}
    else:
        recipe = mix_named(mix)
        iters = {s.key: iter_hf_texts(s) for s in recipe}
        weights_used = {s.key: s.weight for s in recipe}

        def docs() -> Iterator[list[int]]:
            for _key, text in mix_documents(iters, weights_used, rng):
                yield tokenizer.encode(text)

        sources = [s.repo for s in recipe]

    written, nseq = write_packed_bin(docs(), out, seq_len, int(max_tokens))
    meta = {
        "out": str(out),
        "mix": mix,
        "sources": sources,
        "weights": weights_used,
        "seq_len": seq_len,
        "tokens": written,
        "sequences": nseq,
        "eos_id": tokenizer.eos_id,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": getattr(tokenizer, "name", type(tokenizer).__name__),
    }
    meta_path = sidecar_path(out)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pack OpenBMB Ultra-FineWeb (+ Math) for CAT-YOKO")
    p.add_argument("--mix", choices=sorted(set(MIXES) | {"local"}), default="phase-b")
    p.add_argument("--local", type=Path, default=None, help="jsonl with {text|content} for mix=local")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=float, default=1e8, help="stop after this many packed tokens")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--config", choices=["12b", "tiny"], default="12b")
    p.add_argument(
        "--tokenizer",
        default=MINICPM5_TOKENIZER,
        help="HF id, or 'dummy' for tests",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    cfg = CATYokoConfig.tiny() if args.config == "tiny" else CATYokoConfig.middle_12b()
    seq_len = args.seq_len or cfg.seq_len
    tok = load_tokenizer(args.tokenizer, cfg)
    meta = prepare(
        mix=args.mix,
        out=args.out,
        max_tokens=int(args.max_tokens),
        seq_len=seq_len,
        tokenizer=tok,
        local_jsonl=args.local,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    if args.config == "12b" and not isinstance(tok, HashTokenizer) and tok.vocab_size not in {cfg.vocab_size, 130560}:
        print(
            f"warning: tokenizer vocab {tok.vocab_size} != MiniCPM5-2B {cfg.vocab_size}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
