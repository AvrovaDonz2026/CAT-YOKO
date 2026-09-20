#!/usr/bin/env python3
"""DDP sharding and resume for DummyStream / PackedBinStream / FileStream."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.data import DummyStream, FileStream, PackedBinStream, open_stream, utf8_tile_ids
from cat_yoko.recipe import THINK_CODE_FRAC, THINK_CODE_SNIPPETS


def _write_bin(path: Path, toks: list[int]) -> None:
    path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))


class DummyStreamTests(unittest.TestCase):
    def test_seed_constructor_still_works(self) -> None:
        stream = DummyStream(32, 8, seed=0)
        batch = stream.batch(2, "cpu")
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertTrue(torch.equal(batch["labels"], batch["input_ids"]))
        self.assertEqual(batch["doc_ids"][:, 0].tolist(), [0, 1])
        self.assertTrue((batch["doc_ids"][0] == 0).all())
        self.assertTrue((batch["doc_ids"][1] == 1).all())

    def test_open_stream_default_matches_constructor(self) -> None:
        a = DummyStream(32, 8, seed=5)
        b = open_stream(None, 32, 8, seed=5)
        self.assertTrue(
            torch.equal(a.batch(1, "cpu")["input_ids"], b.batch(1, "cpu")["input_ids"])
        )

    def test_ranks_same_seed_are_not_identical(self) -> None:
        a = open_stream(None, 64, 8, seed=7, rank=0, world=2)
        b = open_stream(None, 64, 8, seed=7, rank=1, world=2)
        self.assertFalse(
            torch.equal(a.batch(1, "cpu")["input_ids"], b.batch(1, "cpu")["input_ids"])
        )

    def test_same_rank_and_seed_is_deterministic(self) -> None:
        a = DummyStream(32, 8, seed=7, shard_id=1, num_shards=4)
        b = DummyStream(32, 8, seed=7, shard_id=1, num_shards=4)
        self.assertTrue(
            torch.equal(a.batch(2, "cpu")["input_ids"], b.batch(2, "cpu")["input_ids"])
        )

    def test_state_dict_restores_generator(self) -> None:
        a = DummyStream(32, 8, seed=3)
        a.batch(1, "cpu")
        st = a.state_dict()
        self.assertEqual(st["kind"], "dummy")
        b = DummyStream(32, 8, seed=99)
        b.load_state_dict(st)
        self.assertTrue(
            torch.equal(a.batch(1, "cpu")["input_ids"], b.batch(1, "cpu")["input_ids"])
        )

    def test_ignores_packed_kind(self) -> None:
        stream = DummyStream(32, 8, seed=0)
        before = stream.gen.get_state().clone()
        stream.load_state_dict({"kind": "packed", "i": 9})
        self.assertTrue(torch.equal(stream.gen.get_state(), before))

    def test_thinking_dummy_defaults_to_modest_code_frac(self) -> None:
        stream = DummyStream(32, 8, seed=0)
        self.assertAlmostEqual(stream.code_frac, THINK_CODE_FRAC)
        needle = DummyStream(32, 8, seed=0, needle=True)
        sft = DummyStream(32, 8, seed=0, response_only=True)
        self.assertEqual(needle.code_frac, 0.0)
        self.assertEqual(sft.code_frac, 0.0)
        via_open = open_stream(None, 32, 8, seed=0)
        self.assertAlmostEqual(via_open.code_frac, THINK_CODE_FRAC)

    def test_code_frac_one_uses_only_snippets(self) -> None:
        allowed = {tuple(utf8_tile_ids(s, 32, 16)) for s in THINK_CODE_SNIPPETS}
        stream = DummyStream(32, 16, seed=1, code_frac=1.0)
        batch = stream.batch(8, "cpu")
        for row in batch["input_ids"]:
            self.assertIn(tuple(row.tolist()), allowed)
        self.assertTrue((batch["doc_ids"][0] == 0).all())
        self.assertNotIn(2, batch["input_ids"][0].tolist())

    def test_modest_code_frac_is_not_all_or_nothing(self) -> None:
        allowed = {tuple(utf8_tile_ids(s, 32, 8)) for s in THINK_CODE_SNIPPETS}
        stream = DummyStream(32, 8, seed=0)
        n_code = 0
        n = 200
        for _ in range(n):
            row = stream.batch(1, "cpu")["input_ids"][0]
            if tuple(row.tolist()) in allowed:
                n_code += 1
        self.assertGreater(n_code, 0)
        self.assertLess(n_code, n // 2)


class PackedBinStreamResumeTests(unittest.TestCase):
    def test_stride_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            _write_bin(path, list(range(32)))
            stream = PackedBinStream(path, 8, shard_id=0, num_shards=2)
            with self.assertRaises(ValueError) as ctx:
                stream.load_state_dict({"kind": "packed", "i": 1, "stride": 4})
            self.assertIn("stride", str(ctx.exception).lower())
            self.assertEqual(stream._i, 0)

    def test_rejects_file_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            _write_bin(path, list(range(32)))
            stream = PackedBinStream(path, 8, shard_id=0, num_shards=2)
            with self.assertRaises(ValueError) as ctx:
                stream.load_state_dict({"kind": "file", "i": 99, "stride": 2})
            self.assertIn("file", str(ctx.exception).lower())
            self.assertEqual(stream._i, 0)

    def test_ignores_dummy_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            _write_bin(path, list(range(32)))
            stream = PackedBinStream(path, 8)
            stream.load_state_dict({"kind": "dummy", "i": 7})
            self.assertEqual(stream._i, 0)

    def test_world_gt_nseq_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            seq_len = 4
            _write_bin(path, list(range(8)))  # nseq = 2
            stream = PackedBinStream(path, seq_len, shard_id=0, num_shards=4)
            self.assertEqual(stream.nseq, 2)
            batch = stream.batch(3, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (3, seq_len))
            self.assertEqual(batch["input_ids"][0].tolist(), list(range(4)))
            self.assertEqual(batch["input_ids"][1].tolist(), list(range(4)))

    def test_same_stride_restores_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            _write_bin(path, list(range(64)))
            stream = PackedBinStream(path, 8, shard_id=1, num_shards=2)
            stream.batch(1, "cpu")
            st = stream.state_dict()
            restored = PackedBinStream(path, 8, shard_id=1, num_shards=2)
            restored.load_state_dict(st)
            self.assertEqual(restored._i, stream._i)


class FileStreamResumeTests(unittest.TestCase):
    def test_rejects_packed_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "docs.jsonl"
            path.write_text(
                json.dumps({"tokens": list(range(16))})
                + "\n"
                + json.dumps({"tokens": list(range(16, 32))})
                + "\n"
            )
            stream = FileStream(path, 16)
            with self.assertRaises(ValueError) as ctx:
                stream.load_state_dict({"kind": "packed", "i": 99, "stride": 1})
            self.assertIn("packed", str(ctx.exception).lower())
            self.assertEqual(stream._i, 0)

    def test_stride_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "docs.jsonl"
            path.write_text(json.dumps({"tokens": list(range(16))}) + "\n")
            stream = FileStream(path, 16, shard_id=0, num_shards=2)
            with self.assertRaises(ValueError) as ctx:
                stream.load_state_dict({"kind": "file", "i": 1, "stride": 4})
            self.assertIn("stride", str(ctx.exception).lower())
            self.assertEqual(stream._i, 0)


if __name__ == "__main__":
    unittest.main()
