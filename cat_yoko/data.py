"""Pretrain data: dummy stream, jsonl / int32 memmap, packing with document ids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from cat_yoko.sft import pack_sft_pairs


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
    """Sidecar pack width when present; else ``seq_len``.

    Phase D/C-win pass an explicit view length into ``open_stream`` and
    re-window the flat int32 stream (concat 4K rows to 8K/32K/128K).
    """
    meta = sidecar_meta(path)
    if meta.get("seq_len") is None:
        return int(seq_len)
    packed = int(meta["seq_len"])
    if packed <= 0:
        raise ValueError(f"sidecar seq_len must be positive, got {packed}")
    return packed


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Host→device copy. Pin + non_blocking when the target is CUDA."""
    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        tensor = value
        if want_cuda and tensor.device.type == "cpu":
            if not tensor.is_pinned():
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    pass
            out[key] = tensor.to(device, non_blocking=True)
        else:
            out[key] = tensor.to(device)
    return out


def read_int32_bin(path: Path) -> torch.Tensor:
    data = Path(path).read_bytes()
    if len(data) % 4:
        raise ValueError(f"{path} is not a multiple of 4 bytes (int32 tokens)")
    return torch.frombuffer(bytearray(data), dtype=torch.int32).clone()


def labels_with_doc_boundaries(ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """Ignore next-token loss across packed document boundaries.

    Operates on the last dim so ``PackedBinStream.batch`` can pass ``[B, S]``.
    ``doc_ids[1:]`` would be the batch axis and silently skip a micro_batch=1 row.
    """
    labels = ids.clone()
    if ids.shape[-1] <= 1:
        return labels
    cross = doc_ids[..., 1:] != doc_ids[..., :-1]
    labels[..., 1:] = labels[..., 1:].masked_fill(cross, -100)
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

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        seed: int = 0,
        *,
        shard_id: int = 0,
        num_shards: int = 1,
        response_only: bool = False,
        prompt_frac: float = 0.5,
        needle: bool = False,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.stride = max(int(num_shards), 1)
        self.shard_id = int(shard_id) % self.stride
        # Offset by rank so DDP ranks with the same seed do not emit identical batches.
        self.gen = torch.Generator().manual_seed(int(seed) + self.shard_id)
        self.response_only = bool(response_only)
        self.prompt_frac = float(prompt_frac)
        self.needle = bool(needle)

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
        if self.needle and self.seq_len >= 2:
            mid = self.seq_len // 2
            ids[:, mid] = self.vocab_size - 1
        docs = torch.arange(micro_batch).unsqueeze(1).expand_as(ids)
        labels = ids.clone()
        if self.response_only:
            cut = max(int(self.seq_len * self.prompt_frac), 1)
            labels[:, :cut] = -100
        return to_device(
            {
                "input_ids": ids,
                "labels": labels,
                "doc_ids": docs,
            },
            device,
        )


def _ids_and_sft_labels(messages: list) -> tuple[list[int], list[int]] | None:
    ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        piece = m.get("ids") or m.get("input_ids") or m.get("tokens")
        if piece is None:
            continue
        piece = [int(t) for t in piece]
        role = str(m.get("role") or m.get("from") or "").lower()
        ids.extend(piece)
        if role in {"assistant", "gpt", "bot", "model"}:
            labels.extend(piece)
        else:
            labels.extend([-100] * len(piece))
    if not ids or all(x == -100 for x in labels):
        return None
    return ids, labels


def _jsonl_docs(path: Path) -> Iterator[tuple[list[int], list[int] | None]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt_ids" in obj and "response_ids" in obj:
                prompt = [int(t) for t in obj["prompt_ids"]]
                resp = [int(t) for t in obj["response_ids"]]
                yield prompt + resp, ([-100] * len(prompt) + resp)
                continue
            msgs = obj.get("messages") or obj.get("conversations")
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
                packed = _ids_and_sft_labels(msgs)
                if packed is not None:
                    yield packed
                    continue
            toks = obj.get("tokens", obj.get("input_ids", obj.get("ids")))
            if toks is None:
                raise ValueError(f"jsonl row missing tokens/input_ids: {path}")
            labels = obj.get("labels")
            yield [int(t) for t in toks], ([int(t) for t in labels] if labels is not None else None)


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


def _pad_row(ids: list[int], labels: list[int] | None, seq_len: int, i: int) -> dict[str, torch.Tensor]:
    if len(ids) >= seq_len:
        ids = ids[:seq_len]
        lab = (labels[:seq_len] if labels is not None else ids)
    else:
        pad = seq_len - len(ids)
        ids = ids + [0] * pad
        if labels is None:
            lab = ids[:]
            lab[-pad:] = [-100] * pad
        else:
            lab = labels + [-100] * pad
            lab = lab[:seq_len]
    t_ids = torch.tensor(ids, dtype=torch.long)
    t_lab = torch.tensor(lab, dtype=torch.long)
    docs = torch.full_like(t_ids, i)
    return {"input_ids": t_ids, "labels": t_lab, "doc_ids": docs}


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
        response_only: bool = False,
        prompt_frac: float = 0.5,
    ) -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.stride = max(int(num_shards), 1)
        self._i = int(shard_id) % self.stride
        had_labels = False
        suffix = self.path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            rows = list(_jsonl_docs(self.path))
            labeled = [lab is not None for _, lab in rows]
            had_labels = any(labeled)
            if had_labels:
                packed_pairs = pack_sft_pairs(
                    ((toks, lab) for toks, lab in rows if lab is not None),
                    seq_len,
                )
                self._packed = [
                    _pad_row(toks, lab, seq_len, i)
                    for i, (toks, lab) in enumerate(packed_pairs)
                ]
            else:
                docs = [toks for toks, _ in rows]
                self._packed = pack_documents(docs, seq_len, drop_last=True)
        else:
            docs = list(_bin_docs(self.path, seq_len, eos_id))
            if eos_id is not None:
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
        if response_only and not had_labels:
            cut = max(int(seq_len * float(prompt_frac)), 1)
            for row in self._packed:
                lab = row["labels"].clone()
                lab[:cut] = -100
                row["labels"] = lab

    def state_dict(self) -> dict:
        return {"kind": "file", "i": self._i, "stride": self.stride}

    def load_state_dict(self, st: dict) -> None:
        kind = st.get("kind")
        if kind == "packed":
            raise ValueError(
                "FileStream cannot restore PackedBinStream cursor (kind='packed')"
            )
        if kind not in (None, "file"):
            return
        if "stride" in st and int(st["stride"]) != self.stride:
            raise ValueError(
                f"FileStream cannot restore cursor: checkpoint stride={st['stride']} "
                f"!= stream stride={self.stride}; would desync DDP shards"
            )
        if "i" in st:
            self._i = int(st["i"])

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        rows = []
        for _ in range(micro_batch):
            rows.append(self._packed[self._i % len(self._packed)])
            self._i += self.stride
        return to_device(
            {
                "input_ids": torch.stack([r["input_ids"] for r in rows]),
                "labels": torch.stack([r["labels"] for r in rows]),
                "doc_ids": torch.stack([r["doc_ids"] for r in rows]),
            },
            device,
        )


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
        kind = st.get("kind")
        if kind == "file":
            raise ValueError(
                "PackedBinStream cannot restore FileStream cursor (kind='file')"
            )
        if kind not in (None, "packed"):
            return
        if "stride" in st and int(st["stride"]) != self.stride:
            raise ValueError(
                f"PackedBinStream cannot restore cursor: checkpoint stride={st['stride']} "
                f"!= stream stride={self.stride}; would desync DDP shards"
            )
        if "i" in st:
            self._i = int(st["i"])

    def batch(self, micro_batch: int, device: str) -> dict[str, torch.Tensor]:
        idx = []
        i = self._i
        for _ in range(micro_batch):
            idx.append(i % self.nseq)
            i += self.stride
        self._i = i
        ids = torch.stack([self.rows[j].to(dtype=torch.long) for j in idx])
        docs = doc_ids_from_eos(ids, self.eos_id)
        labels = labels_with_doc_boundaries(ids, docs) if self.eos_id is not None else ids.clone()
        return to_device({"input_ids": ids, "labels": labels, "doc_ids": docs}, device)


def open_stream(
    data: Path | None,
    vocab_size: int,
    seq_len: int,
    *,
    seed: int = 0,
    eos_id: int | None = None,
    rank: int = 0,
    world: int = 1,
    response_only: bool = False,
    prompt_frac: float = 0.5,
    needle: bool = False,
) -> DummyStream | FileStream | PackedBinStream:
    world = max(int(world), 1)
    rank = int(rank) % world
    if data is None:
        return DummyStream(
            vocab_size,
            seq_len,
            seed=seed,
            shard_id=rank,
            num_shards=world,
            response_only=response_only,
            prompt_frac=prompt_frac,
            needle=needle,
        )
    path = Path(data)
    eos = resolve_eos(path, eos_id)
    view = int(seq_len)
    if path.suffix.lower() in {".bin", ".tok"}:
        return PackedBinStream(
            path, view, eos_id=eos, shard_id=rank, num_shards=world
        )
    return FileStream(
        path,
        view,
        eos_id=eos,
        shard_id=rank,
        num_shards=world,
        response_only=response_only,
        prompt_frac=prompt_frac,
    )
