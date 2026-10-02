"""Prepare a bounded, reproducible raw-text corpus; no torch dependency.

Run from this repository's root. Store generated bins and document-hash evidence
outside published code directories. --manifest-only writes a plan without loading
datasets/transformers or contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlsplit

VOCAB_SIZE = 130560
EOS_ID = 1
SPLIT_MODULUS = 1000
EVAL_BUCKETS = 10
FILE_SHA256 = {
    "data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet":
        "2fd70197e66dff7b568e46fc578aedfdb20b35245dd191c5f912e09c6184aac7",
    "data/ultrafineweb_en/ultrafineweb-en-part-0002-of-2048.parquet":
        "8ecfe2c281428fda74c1c3516897915542a1d3d50ae482756b86d793136a27aa",
    "data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet":
        "8e58378a56b7f0a7a64390b400d45601506630cecf23dc3e9bfa255fd8be9a57",
    "data/ultrafineweb_zh/ultrafineweb-zh-part-002-of-256.parquet":
        "7205a2ee83f2dafb24a89ecfd2ebaf70f1a0706ceb6092fc9411c71e118d4c41",
    "data/UltraData-Math-L2-preview/UltraData-Math-L2-part-00001-of-00138.parquet":
        "03128309b74e1d12bde64df52ff13796df63ebbd9c0abcb93ade19b034c107e9",
    "data/UltraData-Math-L2-preview/UltraData-Math-L2-part-00002-of-00138.parquet":
        "912a83ff96546b61d7d69ae7f4c33223aca9701707ed25f89b169baa60901527",
}


@dataclass(frozen=True)
class Source:
    key: str
    repo: str
    revision: str
    weight: int
    files: tuple[tuple[str, int], ...]

    @property
    def urls(self) -> list[str]:
        return [f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{name}"
                for name, _ in self.files]


SOURCES = (
    Source("en", "openbmb/Ultra-FineWeb", "02c85641e3d19a854be2e09139c25adaa9518063", 6, (
        ("data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet", 1300415046),
        ("data/ultrafineweb_en/ultrafineweb-en-part-0002-of-2048.parquet", 1297717678),
    )),
    Source("zh", "openbmb/Ultra-FineWeb", "02c85641e3d19a854be2e09139c25adaa9518063", 3, (
        ("data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet", 1269012674),
        ("data/ultrafineweb_zh/ultrafineweb-zh-part-002-of-256.parquet", 1268857869),
    )),
    Source("math", "openbmb/UltraData-Math", "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87", 1, (
        ("data/UltraData-Math-L2-preview/UltraData-Math-L2-part-00001-of-00138.parquet", 415426460),
        ("data/UltraData-Math-L2-preview/UltraData-Math-L2-part-00002-of-00138.parquet", 342973907),
    )),
)


def document_hash(text: str) -> str:
    """Normalize only with str.strip(); preserve internal whitespace and code."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def document_split(digest: str) -> str:
    return "eval" if int(digest, 16) % SPLIT_MODULUS < EVAL_BUCKETS else "train"


def row_quotas(tokens: int, seq_len: int, sources: tuple[Source, ...] = SOURCES) -> dict[str, int]:
    """Floor total tokens to full rows; allocate rows by largest remainder."""
    if tokens <= 0 or seq_len <= 0 or tokens < seq_len:
        raise ValueError("token budget must contain at least one positive-length row")
    total_rows = tokens // seq_len
    total_weight = sum(source.weight for source in sources)
    if not sources or total_weight <= 0 or any(source.weight <= 0 for source in sources):
        raise ValueError("source weights must be positive")
    if len({source.key for source in sources}) != len(sources):
        raise ValueError("source keys must be unique")
    rows = {source.key: total_rows * source.weight // total_weight for source in sources}
    ordered = sorted(range(len(sources)),
                     key=lambda i: (-(total_rows * sources[i].weight % total_weight), i))
    for i in ordered[:total_rows - sum(rows.values())]:
        rows[sources[i].key] += 1
    return rows


def make_plan(*, train_tokens: int = 80_000_000, eval_tokens: int = 1_000_000,
              seq_len: int = 4096, seed: int = 20261002,
              sources: tuple[Source, ...] = SOURCES) -> dict:
    train_rows, eval_rows = (row_quotas(n, seq_len, sources) for n in (train_tokens, eval_tokens))
    return {
        "schema_version": 1, "status": "planned", "seq_len": seq_len, "seed": seed,
        "requested_tokens": {"train": train_tokens, "eval": eval_tokens},
        "target_tokens": {"train": sum(train_rows.values()) * seq_len,
                          "eval": sum(eval_rows.values()) * seq_len},
        "normalization": "Python str.strip(), UTF-8; internal whitespace unchanged",
        "split_rule": "int(SHA256(normalized_text_utf8), 16) % 1000 < 10 => eval; else train",
        "split_seed": None, "dedup": "global exact normalized-text SHA256, without source salt",
        "mix_rule": "per-source packed rows, shuffled row locations using random.Random(seed)",
        "hash_evidence_scope": "full document hashes, including any quota-truncated final document",
        "eos_id": EOS_ID, "vocab_size": VOCAB_SIZE,
        "sources": [{"key": source.key, "repo": source.repo, "revision": source.revision,
                     "weight": source.weight / sum(s.weight for s in sources),
                     "text_field": "content", "license": "apache-2.0",
                     "license_note": "Ultra-FineWeb also requires checking underlying source licenses",
                     "files": [{"path": name, "bytes": size, "url": url,
                                "official_sha256": FILE_SHA256.get(name)}
                               for (name, size), url in zip(source.files, source.urls)],
                     "row_quotas": {"train": train_rows[source.key], "eval": eval_rows[source.key]},
                     "token_quotas": {"train": train_rows[source.key] * seq_len,
                                      "eval": eval_rows[source.key] * seq_len}}
                    for source in sources],
    }


def normalize_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query
            or parsed.fragment or parsed.username or parsed.password or parsed.path not in ("", "/")):
        raise ValueError("endpoint must be an HTTP(S) origin without a path or credentials")
    return endpoint.rstrip("/")


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


class RowWriter:
    def __init__(self, path: Path, seq_len: int, rows: int):
        self.path, self.seq_len, self.target_rows = path, seq_len, rows
        self.rows = 0
        self.buffer: list[int] = []
        self.handle = path.open("wb")
        self.format = "<" + "i" * seq_len

    @property
    def full(self) -> bool:
        return self.rows >= self.target_rows

    def add(self, tokens: list[int]) -> None:
        offset = 0
        while offset < len(tokens) and not self.full:
            take = min(self.seq_len - len(self.buffer), len(tokens) - offset)
            self.buffer.extend(tokens[offset:offset + take])
            offset += take
            if len(self.buffer) == self.seq_len:
                self.handle.write(struct.pack(self.format, *self.buffer))
                self.rows += 1
                self.buffer.clear()

    def close(self) -> None:
        self.handle.close()


def merge_rows(paths: dict[str, Path], quotas: dict[str, int], destination: Path,
               seq_len: int, seed: int) -> dict:
    order = [(key, row) for key, count in quotas.items() for row in range(count)]
    random.Random(seed).shuffle(order)
    handles = {key: path.open("rb") for key, path in paths.items()}
    temporary = destination.with_name(destination.name + ".partial")
    row_bytes = seq_len * 4
    try:
        with temporary.open("wb") as output:
            for key, row in order:
                handle = handles[key]
                handle.seek(row * row_bytes)
                data = handle.read(row_bytes)
                if len(data) != row_bytes:
                    raise RuntimeError(f"short packed row in {paths[key]} at {row}")
                output.write(data)
    finally:
        for handle in handles.values():
            handle.close()
    expected = len(order) * row_bytes
    if temporary.stat().st_size != expected:
        raise RuntimeError("merged corpus size does not match row quotas")
    os.replace(temporary, destination)
    return {"path": destination.name, "sequences": len(order), "tokens": len(order) * seq_len,
            "bytes": expected, "sha256": file_digest(destination)}


def iter_source_rows(source: Source, log: Callable[[dict], None] = emit, *,
                     endpoint: str = "https://huggingface.co", source_dir: Path | None = None,
                     reader: str = "http") -> Iterable[Mapping]:
    """HTTP byte ranges avoid Hub file-tree discovery for this huge dataset.

    Local shards must match the official LFS SHA256. The optional datasets reader
    preserves the same fixed URLs and content-only streaming, without fallback.
    """
    endpoint = normalize_endpoint(endpoint)
    for (name, expected_bytes), url in zip(source.files, source.urls):
        download_url = endpoint + url.removeprefix("https://huggingface.co")
        event = {"event": "open_file", "source": source.key, "url": url,
                 "download_url": download_url, "expected_bytes": expected_bytes,
                 "official_sha256": FILE_SHA256.get(name), "full_file_verified": False}
        if source_dir is not None:
            from pyarrow import parquet

            path = Path(source_dir) / name
            if not path.is_file():
                path = Path(source_dir) / Path(name).name
            if not path.is_file():
                raise FileNotFoundError(f"required local shard is missing: {name} under {source_dir}")
            actual_digest = file_digest(path)
            expected_digest = FILE_SHA256.get(name)
            if path.stat().st_size != expected_bytes or not expected_digest or actual_digest != expected_digest:
                raise ValueError(f"local shard does not match pinned official size/SHA256: {path}")
            log({**event, "local_path": str(path.resolve()), "actual_sha256": actual_digest,
                 "full_file_verified": True})
            with path.open("rb") as handle:
                for batch in parquet.ParquetFile(handle).iter_batches(batch_size=128, columns=["content"]):
                    yield from ({"content": text} for text in batch.column(0).to_pylist())
        elif reader == "http":
            import fsspec
            from pyarrow import parquet

            log(event)
            with fsspec.open(download_url, mode="rb", block_size=16 << 20,
                             cache_type="readahead", timeout=30) as handle:
                if handle.size != expected_bytes:
                    raise ValueError(f"HTTP shard size differs from pinned official size: {url}")
                for batch in parquet.ParquetFile(handle).iter_batches(batch_size=128, columns=["content"]):
                    yield from ({"content": text} for text in batch.column(0).to_pylist())
        elif reader == "datasets":
            from datasets import load_dataset

            log(event)
            dataset = load_dataset("parquet", data_files={"train": [download_url]}, split="train",
                                   streaming=True, columns=["content"])
            yield from dataset
        else:
            raise ValueError(f"unknown reader: {reader}")


def prepare_corpus(*, out_dir: Path, tokenizer, tokenizer_metadata: dict | None = None,
                   train_tokens: int = 80_000_000, eval_tokens: int = 1_000_000,
                   seq_len: int = 4096, seed: int = 20261002,
                   sources: tuple[Source, ...] = SOURCES,
                   endpoint: str = "https://huggingface.co", source_dir: Path | None = None,
                   reader: str = "http",
                   row_loader: Callable[[Source], Iterable[Mapping]] | None = None,
                   log: Callable[[dict], None] = emit) -> dict:
    plan = make_plan(train_tokens=train_tokens, eval_tokens=eval_tokens, seq_len=seq_len,
                     seed=seed, sources=sources)
    plan["download_endpoint"] = normalize_endpoint(endpoint)
    plan["source_dir"] = str(Path(source_dir).resolve()) if source_dir is not None else None
    plan["reader"] = "local_pyarrow" if source_dir is not None else reader
    plan["input_files_read"] = []
    if int(tokenizer.vocab_size) != VOCAB_SIZE or int(tokenizer.eos_id) != EOS_ID:
        raise ValueError("tokenizer must have vocab_size=130560 and eos_id=1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.bin", "eval.bin", "manifest.json", "train.docs.sha256", "eval.docs.sha256"):
        if (out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite existing corpus evidence: {out_dir / name}")
    plan["tokenizer"] = tokenizer_metadata or {"kind": type(tokenizer).__name__}
    plan["preparer_sha256"] = file_digest(Path(__file__))
    plan["software"] = {"python": __import__("sys").version.split()[0]}
    for package in ("datasets", "transformers"):
        try:
            plan["software"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(out_dir / "plan.json", plan)
    stats = {}
    seen: set[str] = set()
    selected = {split: set() for split in ("train", "eval")}
    status = {"status": "running", "source_stats": stats}
    write_json(out_dir / "status.json", status)
    def source_log(event):
        if event.get("event") == "open_file":
            plan["input_files_read"].append(event)
        log(event)

    row_loader = row_loader or (lambda source: iter_source_rows(
        source, source_log, endpoint=endpoint, source_dir=source_dir, reader=reader))
    last_log = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=out_dir) as temporary:
            tmp_dir = Path(temporary)
            paths = {split: {} for split in selected}
            hash_handles = {split: (out_dir / f"{split}.docs.sha256").open("w", encoding="ascii")
                            for split in selected}
            try:
                for source, info in zip(sources, plan["sources"]):
                    counters = {"scanned_documents": 0, "empty_documents": 0, "duplicate_documents": 0,
                                "quota_skipped_documents": 0, "train_documents": 0, "eval_documents": 0,
                                "internal_eos_tokens": 0, "train_rows": 0, "eval_rows": 0,
                                "max_document_chars": 0, "max_document_tokens": 0}
                    stats[source.key] = counters
                    writers = {}
                    for split in selected:
                        path = tmp_dir / f"{source.key}.{split}.bin"
                        paths[split][source.key] = path
                        writers[split] = RowWriter(path, seq_len, info["row_quotas"][split])
                    status["active_source"] = source.key
                    log({"event": "source_start", "source": source.key, "row_quotas": info["row_quotas"]})
                    write_json(out_dir / "status.json", status)
                    rows = None
                    try:
                        rows = iter(row_loader(source)) if not all(w.full for w in writers.values()) else iter(())
                        for row in rows:
                            counters["scanned_documents"] += 1
                            text = row.get("content")
                            if not isinstance(text, str) or not text.strip():
                                counters["empty_documents"] += 1
                            else:
                                text = text.strip()
                                digest = document_hash(text)
                                if digest in seen:
                                    counters["duplicate_documents"] += 1
                                else:
                                    seen.add(digest)
                                    split = document_split(digest)
                                    writer = writers[split]
                                    if writer.full:
                                        counters["quota_skipped_documents"] += 1
                                    else:
                                        ids = list(tokenizer.encode(text))
                                        if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in ids):
                                            raise ValueError(f"invalid tokenizer ID in {source.key}")
                                        if not ids or ids[-1] != EOS_ID:
                                            ids.append(EOS_ID)
                                        counters["max_document_chars"] = max(counters["max_document_chars"], len(text))
                                        counters["max_document_tokens"] = max(counters["max_document_tokens"], len(ids))
                                        counters["internal_eos_tokens"] += ids[:-1].count(EOS_ID)
                                        writer.add(ids)
                                        selected[split].add(digest)
                                        hash_handles[split].write(digest + "\n")
                                        counters[f"{split}_documents"] += 1
                            for split, writer in writers.items():
                                counters[f"{split}_rows"] = writer.rows
                            now = time.monotonic()
                            if now - last_log >= 10 or counters["scanned_documents"] % 50_000 == 0:
                                log({"event": "progress", "source": source.key, **counters})
                                write_json(out_dir / "status.json", status)
                                last_log = now
                            if all(writer.full for writer in writers.values()):
                                break
                        if not all(writer.full for writer in writers.values()):
                            actual = {split: writer.rows for split, writer in writers.items()}
                            raise RuntimeError(f"bounded source {source.key} exhausted: rows={actual}, "
                                               f"required={info['row_quotas']}; extend pinned file manifest")
                    finally:
                        for writer in writers.values():
                            writer.close()
                        close = getattr(rows, "close", None)
                        if close is not None:
                            close()
                    log({"event": "source_complete", "source": source.key, **counters})
                    write_json(out_dir / "status.json", status)
            finally:
                for handle in hash_handles.values():
                    handle.close()
            intersection = len(selected["train"] & selected["eval"])
            if intersection:
                raise RuntimeError(f"train/eval document hash intersection is {intersection}")
            plan["hashes"] = {split: {"path": f"{split}.docs.sha256", "documents": len(selected[split]),
                                     "sha256": file_digest(out_dir / f"{split}.docs.sha256")}
                              for split in selected}
            plan["hash_intersection"] = intersection
            plan["source_stats"] = stats
            plan["unique_scanned_documents"] = len(seen)
            plan["outputs"] = {}
            for split in selected:
                quotas = {info["key"]: info["row_quotas"][split] for info in plan["sources"]}
                output = merge_rows(paths[split], quotas, out_dir / f"{split}.bin", seq_len, seed)
                plan["outputs"][split] = output
                write_json(out_dir / f"{split}.bin.meta.json", {
                    **output, "format": "packed_bin", "seq_len": seq_len, "eos_id": EOS_ID,
                    "vocab_size": VOCAB_SIZE, "tokenizer": plan["tokenizer"].get("directory", "local"),
                    "split": split, "manifest": "manifest.json", "source_row_quotas": quotas,
                })
        plan["status"] = "complete"
        write_json(out_dir / "manifest.json", plan)
        write_json(out_dir / "status.json", {"status": "complete", "outputs": plan["outputs"],
                                            "hash_intersection": 0, "source_stats": stats})
        log({"event": "complete", "out_dir": str(out_dir), "outputs": plan["outputs"],
             "hash_intersection": 0})
        return plan
    except Exception as error:
        write_json(out_dir / "status.json", {**status, "status": "failed",
                                            "error_type": type(error).__name__, "error": str(error)})
        log({"event": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def validate_tokenizer(tokenizer) -> dict:
    # HF vocab_size excludes added special tokens; len includes the complete
    # embedding vocabulary used by MiniCPM5 (130072 base + 488 added tokens).
    vocabulary = tokenizer.get_vocab()
    ids = set(vocabulary.values())
    if len(tokenizer) != VOCAB_SIZE or ids != set(range(VOCAB_SIZE)) or tokenizer.eos_token_id != EOS_ID:
        raise ValueError("local MiniCPM5 tokenizer must have 130560 contiguous IDs and eos_token_id=1")
    return {"base_vocab_size": int(tokenizer.vocab_size), "total_vocab_size": len(tokenizer)}


def load_local_tokenizer(directory: Path):
    from transformers import AutoTokenizer

    path = Path(directory).resolve()
    if not path.is_dir():
        raise ValueError(f"tokenizer directory does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    vocabulary_metadata = validate_tokenizer(tokenizer)

    class RawTextTokenizer:
        vocab_size, eos_id = VOCAB_SIZE, EOS_ID

        def encode(self, text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    files = {name: file_digest(path / name) for name in
             ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model")
             if (path / name).is_file()}
    if not files:
        raise ValueError("local tokenizer has no files available for provenance hashes")
    return RawTextTokenizer(), {"directory": str(path), "files_sha256": files, **vocabulary_metadata,
                                "add_special_tokens": False, "append_eos_if_missing": EOS_ID}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--train-tokens", type=int, default=80_000_000)
    parser.add_argument("--eval-tokens", type=int, default=1_000_000)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--source-dir", type=Path,
                        help="local parquet shards under original subpaths or basenames; verify official SHA256")
    parser.add_argument("--reader", choices=("http", "datasets"), default="http")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest_only:
        plan = make_plan(train_tokens=args.train_tokens, eval_tokens=args.eval_tokens,
                         seq_len=args.seq_len, seed=args.seed)
        plan["download_endpoint"] = normalize_endpoint(args.endpoint)
        plan["source_dir"] = str(args.source_dir.resolve()) if args.source_dir else None
        plan["reader"] = "local_pyarrow" if args.source_dir else args.reader
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if any((args.out_dir / name).exists() for name in ("manifest.json", "train.bin", "eval.bin")):
            raise FileExistsError("refusing to replace a completed corpus plan")
        write_json(args.out_dir / "plan.json", plan)
        write_json(args.out_dir / "status.json", {"status": "planned", "downloaded": False})
        emit(plan)
        return 0
    if args.tokenizer_dir is None:
        parser.error("--tokenizer-dir is required unless --manifest-only is used")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "train.bin", "eval.bin", "train.docs.sha256", "eval.docs.sha256"):
        if (args.out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite existing corpus evidence: {args.out_dir / name}")
    try:
        tokenizer, metadata = load_local_tokenizer(args.tokenizer_dir)
        prepare_corpus(out_dir=args.out_dir, tokenizer=tokenizer, tokenizer_metadata=metadata,
                       train_tokens=args.train_tokens, eval_tokens=args.eval_tokens,
                       seq_len=args.seq_len, seed=args.seed, endpoint=args.endpoint,
                       source_dir=args.source_dir, reader=args.reader)
    except Exception as error:
        # Import/tokenizer failures happen before prepare_corpus starts.
        status_path = args.out_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        write_json(status_path, {**status, "status": "failed",
                   "error_type": type(error).__name__, "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
