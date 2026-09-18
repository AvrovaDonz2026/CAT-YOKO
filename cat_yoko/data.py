"""Pretrain data: dummy stream, jsonl / int32 memmap, packing with document ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch


def read_int32_bin(path: Path) -> torch.Tensor:
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} is not a multiple of 4 bytes (int32 tokens)")
    return torch.frombuffer(bytearray(data), dtype=torch.int32).clone()


def labels_with_doc_boundaries(ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """Ignore next-token loss across packed document boundaries."""
    labels = ids.clone()
    if ids.numel() <= 1:
        return labels
    cross = doc_ids[1:] != doc_ids[:-1]
    labels[1:][cross] = -100
    return labels


def pack_documents(
    documents: list[list[int] | torch.Tensor],
    seq_len: int,
    *,
    drop_last: bool = True,
) -> list[dict[str, torch.Tensor]]:
    ids_buf: list[int] = []
    doc_buf: list[int] = []
    out: list[dict[str, torch.Tensor]] = []
    for d, doc in enumerate(documents):
        toks = doc.tolist() if isinstance(doc, torch.Tensor) else list(doc)
        for t in toks:
            ids_buf.append(int(t))
            doc_buf.append(d)
            if len(ids_buf) == seq_len:
                ids = torch.tensor(ids_buf, dtype=torch.long)
                docs = torch.tensor(doc_buf, dtype=torch.long)
                out.append(
                    {
                        "input_ids": ids,
                        "doc_ids": docs,
                        "labels": labels_with_doc_boundaries(ids, docs),
                    }
                )
                ids_buf, doc_buf = [], []
    if ids_buf and not drop_last:
        pad = seq_len - len(ids_buf)
        ids_buf.extend([0] * pad)
        doc_buf.extend([doc_buf[-1]] * pad)
        ids = torch.tensor(ids_buf, dtype=torch.long)
        docs = torch.tensor(doc_buf, dtype=torch.long)
        labels = labels_with_doc_boundaries(ids, docs)
        labels[-pad:] = -100
        out.append({"input_ids": ids, "doc_ids": docs, "labels": labels})
    return out


class DummyStream:
    """Infinite random single-document sequences (smoke / L0)."""

    def __init__(self, vocab_size: int, seq_len: int, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.gen = torch.Generator().manual_seed(seed)

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        ids = torch.randint(
            0,
            self.vocab_size,
            (micro_batch, self.seq_len),
            generator=self.gen,
        )
        docs = torch.arange(micro_batch).unsqueeze(1).expand_as(ids)
        return {
            "input_ids": ids.to(device),
            "labels": ids.clone().to(device),
            "doc_ids": docs.to(device),
        }


def _jsonl_docs(path: Path) -> Iterator[list[int]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            toks = obj.get("tokens", obj.get("input_ids", obj.get("ids")))
            if toks is None:
                raise ValueError(f"jsonl row missing tokens/input_ids: {path}")
            yield [int(t) for t in toks]


def _bin_docs(path: Path, seq_len: int, eos_id: int | None) -> Iterator[list[int]]:
    mm = read_int32_bin(path)
    if eos_id is None:
        n = (mm.numel() // seq_len) * seq_len
        for i in range(0, n, seq_len):
            yield mm[i : i + seq_len].tolist()
        return
    doc: list[int] = []
    for t in mm.tolist():
        doc.append(int(t))
        if int(t) == eos_id:
            yield doc
            doc = []
    if doc:
        yield doc


class FileStream:
    """Cycles packed sequences from jsonl or int32 `.bin`."""

    def __init__(self, path: Path, seq_len: int, *, eos_id: int | None = None) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.eos_id = eos_id
        suffix = self.path.suffix.lower()
        docs = (
            list(_jsonl_docs(self.path))
            if suffix in {".jsonl", ".json"}
            else list(_bin_docs(self.path, seq_len, eos_id))
        )
        if suffix in {".jsonl", ".json"} or eos_id is not None:
            self._packed = pack_documents(docs, seq_len, drop_last=True)
        else:
            self._packed = []
            for i, row in enumerate(docs):
                ids = torch.tensor(row, dtype=torch.long)
                doc_ids = torch.full_like(ids, i)
                self._packed.append(
                    {"input_ids": ids, "doc_ids": doc_ids, "labels": ids.clone()}
                )
        if not self._packed:
            raise ValueError(f"no sequences packed from {self.path}")
        self._i = 0

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        rows = []
        for _ in range(micro_batch):
            rows.append(self._packed[self._i % len(self._packed)])
            self._i += 1
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]).to(device),
            "labels": torch.stack([r["labels"] for r in rows]).to(device),
            "doc_ids": torch.stack([r["doc_ids"] for r in rows]).to(device),
        }


def open_stream(
    data: Path | None,
    vocab_size: int,
    seq_len: int,
    *,
    seed: int = 0,
    eos_id: int | None = None,
) -> DummyStream | FileStream:
    if data is None:
        return DummyStream(vocab_size, seq_len, seed=seed)
    return FileStream(Path(data), seq_len, eos_id=eos_id)
