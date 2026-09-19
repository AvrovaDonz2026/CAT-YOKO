"""Tokenize OpenBMB mixes into a packed int32 bin the C1 trainer can mmap."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from pathlib import Path
from typing import Iterator, TypeVar

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import sidecar_path
from cat_yoko.recipe import (
    MINICPM5_TOKENIZER,
    MIXES,
    Source,
    is_sft_mix,
    mix_named,
)
from cat_yoko.sft import encode_sft, iter_sft_pairs, pack_sft_pairs, sft_turns, turns_to_text
from cat_yoko.tokenizer import HashTokenizer, Tokenizer, load_tokenizer

T = TypeVar("T")


def row_text(row: dict, fields: tuple[str, ...]) -> str | None:
    for k in fields:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    turns = sft_turns(row)
    if turns:
        return turns_to_text(turns)
    return None


def iter_local_rows(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def iter_local_jsonl(path: Path, fields: tuple[str, ...] = ("text", "content")) -> Iterator[str]:
    for obj in iter_local_rows(path):
        if "tokens" in obj or "input_ids" in obj or "prompt_ids" in obj:
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
        obj = dict(row)
        t = row_text(obj, src.text_fields)
        if t:
            yield t


def iter_hf_rows(src: Source) -> Iterator[dict]:
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
        obj = dict(row)
        if isinstance(obj, dict):
            yield obj


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
    iters: dict[str, Iterator[T]],
    weights: dict[str, float],
    rng: random.Random,
) -> Iterator[tuple[str, T]]:
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


def write_sft_jsonl(
    pairs: Iterator[tuple[list[int], list[int]]],
    out: Path,
    max_tokens: int,
) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    nseq = 0
    with out.open("w") as f:
        for ids, labels in pairs:
            if not ids:
                continue
            f.write(json.dumps({"tokens": ids, "labels": labels}, separators=(",", ":")) + "\n")
            written += len(ids)
            nseq += 1
            if written >= max_tokens:
                return written, nseq
    return written, nseq


def _sft_pairs(
    *,
    mix: str,
    tokenizer: Tokenizer,
    seq_len: int,
    local_jsonl: Path | None,
    rng: random.Random,
) -> Iterator[tuple[list[int], list[int]]]:
    if mix == "local":
        if local_jsonl is None:
            raise ValueError("--local jsonl required for mix=local")
        yield from iter_sft_pairs(iter_local_rows(local_jsonl), tokenizer, seq_len)
        return
    recipe = mix_named(mix)
    iters = {s.key: iter_hf_rows(s) for s in recipe}
    weights = {s.key: s.weight for s in recipe}
    for _key, row in mix_documents(iters, weights, rng):
        if not isinstance(row, dict):
            continue
        turns = sft_turns(row)
        if not turns:
            continue
        packed = encode_sft(tokenizer, turns, seq_len)
        if packed is not None:
            yield packed


def _sft_out_path(out: Path) -> Path:
    if out.suffix.lower() in {".jsonl", ".json"}:
        return out
    return out.with_suffix(".jsonl")


def prepare(
    *,
    mix: str,
    out: Path,
    max_tokens: int,
    seq_len: int,
    tokenizer: Tokenizer,
    local_jsonl: Path | None = None,
    seed: int = 0,
    sft: bool = False,
) -> dict:
    rng = random.Random(seed)
    sft_mode = bool(sft) or is_sft_mix(mix)
    if mix == "local":
        if local_jsonl is None:
            raise ValueError("--local jsonl required for mix=local")
        sources = ["local"]
        weights_used = {"local": 1.0}
    else:
        recipe = mix_named(mix)
        sources = [s.repo for s in recipe]
        weights_used = {s.key: s.weight for s in recipe}

    out = Path(out)
    if sft_mode:
        dest = _sft_out_path(out)
        written, nseq = write_sft_jsonl(
            pack_sft_pairs(
                _sft_pairs(
                    mix=mix,
                    tokenizer=tokenizer,
                    seq_len=seq_len,
                    local_jsonl=local_jsonl,
                    rng=rng,
                ),
                seq_len,
            ),
            dest,
            int(max_tokens),
        )
        fmt = "sft_jsonl"
        out = dest
    else:
        if mix == "local":

            def docs() -> Iterator[list[int]]:
                for text in iter_local_jsonl(local_jsonl):
                    yield tokenizer.encode(text)

        else:
            recipe = mix_named(mix)
            iters = {s.key: iter_hf_texts(s) for s in recipe}

            def docs() -> Iterator[list[int]]:
                for _key, text in mix_documents(iters, weights_used, rng):
                    yield tokenizer.encode(text)

        written, nseq = write_packed_bin(docs(), out, seq_len, int(max_tokens))
        fmt = "packed_bin"
        dest = out

    meta = {
        "out": str(dest),
        "mix": mix,
        "format": fmt,
        "sources": sources,
        "weights": weights_used,
        "seq_len": seq_len,
        "tokens": written,
        "sequences": nseq,
        "eos_id": tokenizer.eos_id,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": getattr(tokenizer, "name", type(tokenizer).__name__),
    }
    meta_path = sidecar_path(dest)
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
    p.add_argument(
        "--sft",
        action="store_true",
        help="write response-only jsonl (default for --mix phase-f / phase-g)",
    )
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
        sft=bool(args.sft),
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
