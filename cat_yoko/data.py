"""Pretrain data: dummy stream, jsonl / int32 memmap, packing with document ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch


def sidecar_path(path: Path) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".meta.json")


def sidecar_meta(path: Path | None) -> dict:
    if path is None:
        return {}
    meta_path = sidecar_path(Path(path))
    if not meta_path.is_file():
        return {}
    try:
        obj = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def resolve_eos(path: Path | None, eos_id: int | None) -> int | None:
    """Prefer an explicit --eos; else PackedBin sidecar ``*.bin.meta.json``."""
    if eos_id is not None or path is None:
        return eos_id
    meta = sidecar_meta(path)
    if meta.get("eos_id") is None:
        return None
    return int(meta["eos_id"])


def resolve_seq_len(path: Path | None, seq_len: int) -> int:
    """Packed bins must be viewed with the seq_len they were written with."""
    meta = sidecar_meta(path)
    if meta.get("seq_len") is None:
        return seq_len
    packed = int(meta["seq_len"])
    if packed <= 0:
        raise ValueError(f"sidecar seq_len must be positive, got {packed}")
    return packed


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

    def state_dict(self) -> dict:
        return {"kind": "dummy", "gen": self.gen.get_state()}

    def load_state_dict(self, st: dict) -> None:
        if st.get("kind") not in (None, "dummy"):
            return
        if st.get("gen") is not None:
            self.gen.set_state(st["gen"].cpu())

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

    def __init__(
        self,
        path: Path,
        seq_len: int,
        *,
        eos_id: int | None = None,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.stride = max(int(num_shards), 1)
        self._i = int(shard_id) % self.stride
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

    def state_dict(self) -> dict:
        return {"kind": "file", "i": self._i, "stride": self.stride}

    def load_state_dict(self, st: dict) -> None:
        if st.get("kind") not in (None, "file", "packed"):
            return
        if "i" in st:
            self._i = int(st["i"])

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        rows = []
        for _ in range(micro_batch):
            rows.append(self._packed[self._i % len(self._packed)])
            self._i += self.stride
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]).to(device),
            "labels": torch.stack([r["labels"] for r in rows]).to(device),
            "doc_ids": torch.stack([r["doc_ids"] for r in rows]).to(device),
        }


def doc_ids_from_eos(ids: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    """Document index increases after each EOS (packed pretrain rows)."""
    if eos_id is None:
        return torch.zeros_like(ids)
    hits = (ids == eos_id).to(torch.long)
    return hits.cumsum(dim=-1) - hits


class PackedBinStream:
    """Memory-map a seq_len-packed int32 bin. Does not load the whole corpus."""

    def __init__(
        self,
        path: Path,
        seq_len: int,
        *,
        eos_id: int | None = None,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.stride = max(int(num_shards), 1)
        self._i = int(shard_id) % self.stride
        nbytes = self.path.stat().st_size
        if nbytes % 4:
            raise ValueError(f"{self.path} is not int32-aligned")
        ntok = nbytes // 4
        self.nseq = ntok // seq_len
        if self.nseq <= 0:
            raise ValueError(f"{self.path} shorter than one sequence of {seq_len}")
        usable = self.nseq * seq_len
        try:
            flat = torch.from_file(str(self.path), shared=False, size=usable, dtype=torch.int32)
        except (RuntimeError, SystemError, TypeError):
            flat = read_int32_bin(self.path)[:usable]
        self.rows = flat.view(self.nseq, seq_len)

    def state_dict(self) -> dict:
        return {"kind": "packed", "i": self._i, "stride": self.stride, "nseq": self.nseq}

    def load_state_dict(self, st: dict) -> None:
        if st.get("kind") not in (None, "packed", "file"):
            return
        if "i" in st:
            self._i = int(st["i"])

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        idx = []
        i = self._i
        for _ in range(micro_batch):
            idx.append(i % self.nseq)
            i += self.stride
        self._i = i
        ids = torch.stack([self.rows[j].to(dtype=torch.long) for j in idx]).to(device)
        docs = doc_ids_from_eos(ids, self.eos_id)
        labels = labels_with_doc_boundaries(ids, docs) if self.eos_id is not None else ids.clone()
        return {"input_ids": ids, "labels": labels, "doc_ids": docs}


def open_stream(
    data: Path | None,
    vocab_size: int,
    seq_len: int,
    *,
    seed: int = 0,
    eos_id: int | None = None,
    rank: int = 0,
    world: int = 1,
) -> DummyStream | FileStream | PackedBinStream:
    if data is None:
        return DummyStream(vocab_size, seq_len, seed=seed)
    path = Path(data)
    eos = resolve_eos(path, eos_id)
    packed_seq = resolve_seq_len(path, seq_len)
    world = max(int(world), 1)
    rank = int(rank) % world
    if path.suffix.lower() in {".bin", ".tok"}:
        return PackedBinStream(
            path, packed_seq, eos_id=eos, shard_id=rank, num_shards=world
        )
    return FileStream(path, packed_seq, eos_id=eos, shard_id=rank, num_shards=world)
