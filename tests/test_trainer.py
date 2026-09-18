#!/usr/bin/env python3
"""Trainer loop: packing, checkpoint, accum, CLI safety."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import FileStream, PackedBinStream, pack_documents, sidecar_meta
from cat_yoko.loss import kd_kl, kd_weight
from cat_yoko.optim import adamw_param_groups, wsd_lr
from cat_yoko.train import main
from cat_yoko.trainer import Trainer, auto_accum, train_loop


class PackTests(unittest.TestCase):
    def test_boundary_labels_are_ignored(self) -> None:
        packed = pack_documents([[1, 2, 3], [4, 5, 6, 7]], seq_len=4)
        self.assertEqual(packed[0]["input_ids"].tolist(), [1, 2, 3, 4])
        self.assertEqual(packed[0]["doc_ids"].tolist(), [0, 0, 0, 1])
        self.assertEqual(int(packed[0]["labels"][3]), -100)
        self.assertEqual(int(packed[0]["labels"][1]), 2)

    def test_jsonl_roundtrip(self) -> None:
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "docs.jsonl"
            rows = [{"tokens": list(range(8, 24))}, {"tokens": list(range(24, 40))}]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            stream = FileStream(path, cfg.seq_len)
            batch = stream.batch(1, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 16))

    def test_int32_bin_chunks(self) -> None:
        cfg = CATYokoConfig.tiny()
        toks = list(range(32))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            stream = FileStream(path, cfg.seq_len)
            batch = stream.batch(2, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (2, 16))
            self.assertEqual(batch["input_ids"][0].tolist(), list(range(16)))


class StreamShardTests(unittest.TestCase):
    def test_packed_shards_are_disjoint(self) -> None:
        cfg = CATYokoConfig.tiny()
        toks = list(range(64))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            a = PackedBinStream(path, cfg.seq_len, shard_id=0, num_shards=2)
            b = PackedBinStream(path, cfg.seq_len, shard_id=1, num_shards=2)
            ba = a.batch(1, "cpu")["input_ids"][0].tolist()
            bb = b.batch(1, "cpu")["input_ids"][0].tolist()
            self.assertEqual(ba, list(range(16)))
            self.assertEqual(bb, list(range(16, 32)))

    def test_sidecar_seq_len_wins(self) -> None:
        toks = list(range(32))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            (Path(str(path) + ".meta.json")).write_text(json.dumps({"seq_len": 8, "eos_id": 2}))
            self.assertEqual(sidecar_meta(path)["seq_len"], 8)
            from cat_yoko.data import open_stream

            stream = open_stream(path, vocab_size=128, seq_len=16)
            batch = stream.batch(1, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 8))


class LoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)

    def test_accum_runs(self) -> None:
        nll = train_loop(self.cfg, "B0", steps=1, device="cpu", accum=2, micro_batch=1)
        self.assertTrue(nll > 0)

    def test_token_budget_stops(self) -> None:
        tr = Trainer(self.cfg, "B0", "cpu", tokens=64, micro_batch=2, accum=1)
        out = tr.run()
        self.assertGreaterEqual(out.tokens_seen, 64)
        self.assertGreaterEqual(out.step, 1)

    def test_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg, "B0", "cpu", steps=1, accum=1, save_dir=save, save_every=1, seed=1
            ).run()
            ckpt = save / "latest.pt"
            self.assertTrue(ckpt.is_file())
            out = Trainer(
                self.cfg, "B0", "cpu", steps=2, accum=1, resume=ckpt, seed=1
            ).run()
            self.assertEqual(out.step, 2)

    def test_resume_b0_into_b1_starts_new_phase(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg, "B0", "cpu", steps=1, accum=1, save_dir=save, save_every=1, seed=1
            ).run()
            out = Trainer(
                self.cfg, "B1", "cpu", steps=1, accum=1, resume=save / "latest.pt", seed=1
            ).run()
            self.assertEqual(out.step, 1)
            self.assertEqual(out.phase, "B1")

    def test_resume_advances_packed_cursor(self) -> None:
        cfg = CATYokoConfig.tiny()
        toks = list(range(64))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            save = td / "run"
            Trainer(
                cfg, "B0", "cpu", steps=1, accum=1, micro_batch=2, data=path, save_dir=save, save_every=1
            ).run()
            ckpt = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["extra"]["stream"]["i"], 2)
            out = Trainer(
                cfg, "B0", "cpu", steps=2, accum=1, micro_batch=2, data=path, resume=save / "latest.pt"
            ).run()
            self.assertEqual(out.step, 2)

    def test_grad_ckpt_b2(self) -> None:
        nll = train_loop(self.cfg, "B2", steps=1, device="cpu", accum=1, grad_ckpt=True)
        self.assertTrue(nll > 0)

    def test_seq_len_override_mismatch_raises(self) -> None:
        toks = list(range(32))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            (Path(str(path) + ".meta.json")).write_text(json.dumps({"seq_len": 16}))
            with self.assertRaises(ValueError):
                Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, data=path, seq_len=8)

    def test_log_includes_grad_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, log_path=log).run()
            row = json.loads(log.read_text().splitlines()[0])
            self.assertIn("grad_norm", row)
            self.assertIn("tok_s", row)
            self.assertIn("aux", row)
            self.assertEqual(row["adam"], "gpu")

    def test_wsd_b1_offset_skips_warmup(self) -> None:
        lr = wsd_lr(8e9, self.cfg, "B1")
        self.assertAlmostEqual(lr, self.cfg.lr)

    def test_kd_weight_needs_more_than_one_step(self) -> None:
        self.assertEqual(kd_weight(0, 1, 0.5), 0.0)
        self.assertGreater(kd_weight(0, 8, 0.5), 0.3)
        self.assertEqual(kd_kl(torch.zeros(2, 4), torch.zeros(2, 4), 2.0).shape, ())

    def test_auto_accum_tiny(self) -> None:
        self.assertEqual(auto_accum(self.cfg, micro_batch=2, world=1), 4)

    def test_adamw_skips_norm_decay(self) -> None:
        from cat_yoko.model import CATYokoForCausalLM
        from cat_yoko.freeze import apply_freeze

        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B0")
        groups = adamw_param_groups(model, 0.1)
        self.assertEqual(len(groups), 2)


class CliTests(unittest.TestCase):
    def test_tiny_save_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            code = main(
                [
                    "--config",
                    "tiny",
                    "--phase",
                    "B0",
                    "--steps",
                    "1",
                    "--accum",
                    "1",
                    "--save-dir",
                    str(td / "run"),
                    "--log",
                    str(td / "m.jsonl"),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((td / "run" / "latest.pt").is_file())
            self.assertTrue((td / "m.jsonl").is_file())

    def test_12b_refuses_full_envelope(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "12b", "--phase", "B0"])

    def test_12b_cpu_steps_need_cuda(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "12b", "--phase", "B0", "--steps", "1"])


if __name__ == "__main__":
    unittest.main()
