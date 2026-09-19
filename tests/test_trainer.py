#!/usr/bin/env python3
"""Trainer loop: packing, checkpoint, accum, CLI safety."""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import FileStream, PackedBinStream, pack_documents, sidecar_meta
from cat_yoko.loss import kd_kl, kd_weight
from cat_yoko.optim import adamw_param_groups, wsd_lr
from cat_yoko.teacher import DummyTeacher
from cat_yoko.train import main, resolve_12b_accum
from cat_yoko.trainer import Trainer, auto_accum, train_loop
from cat_yoko.upcycle import dummy_minicpm_state


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

    def test_sidecar_seq_len_wins_without_override(self) -> None:
        toks = list(range(32))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            (Path(str(path) + ".meta.json")).write_text(json.dumps({"seq_len": 8, "eos_id": 2}))
            self.assertEqual(sidecar_meta(path)["seq_len"], 8)
            from cat_yoko.data import open_stream, resolve_seq_len

            self.assertEqual(resolve_seq_len(path, 16), 8)
            stream = open_stream(path, vocab_size=128, seq_len=16)
            batch = stream.batch(1, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (1, 16))
            self.assertEqual(batch["input_ids"][0].tolist(), list(range(16)))


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

    def test_latest_hardlinks_matching_step_ckpt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg, "B0", "cpu", steps=1, accum=1, save_dir=save, save_every=1, seed=1
            ).run()
            step = save / "step_1.pt"
            latest = save / "latest.pt"
            self.assertTrue(step.is_file())
            self.assertTrue(latest.samefile(step))

    def test_latest_is_fresh_save_when_step_lags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg, "B0", "cpu", steps=3, accum=1, save_dir=save, save_every=2, seed=1
            ).run()
            self.assertTrue((save / "step_2.pt").is_file())
            self.assertFalse((save / "step_3.pt").is_file())
            latest = save / "latest.pt"
            self.assertTrue(latest.is_file())
            self.assertFalse(latest.samefile(save / "step_2.pt"))
            ckpt = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["extra"]["step"], 3)

    def test_resume_directory_without_latest_uses_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg, "B0", "cpu", steps=1, accum=1, save_dir=save, save_every=1, seed=1
            ).run()
            (save / "latest.pt").unlink()
            self.assertTrue((save / "step_1.pt").is_file())
            out = Trainer(
                self.cfg, "B0", "cpu", steps=2, accum=1, resume=save, seed=1
            ).run()
            self.assertEqual(out.step, 2)

    def test_keep_last_latest_shares_inode_with_newest_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg,
                "B0",
                "cpu",
                steps=3,
                accum=1,
                save_dir=save,
                save_every=1,
                save_keep=2,
            ).run()
            self.assertEqual(
                [p.name for p in sorted(save.glob("step_*.pt"))],
                ["step_2.pt", "step_3.pt"],
            )
            self.assertTrue((save / "latest.pt").samefile(save / "step_3.pt"))

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
            self.assertEqual(ckpt["extra"]["seed"], 0)
            self.assertEqual(ckpt["extra"]["cfg"]["name"], "tiny")
            self.assertIn("rng_py", ckpt["extra"])
            out = Trainer(
                cfg, "B0", "cpu", steps=2, accum=1, micro_batch=2, data=path, resume=save / "latest.pt"
            ).run()
            self.assertEqual(out.step, 2)

    def test_resume_stream_kind_mismatch_skips_cursor(self) -> None:
        cfg = CATYokoConfig.tiny()
        toks = list(range(64))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            save = td / "run"
            Trainer(
                cfg, "B0", "cpu", steps=1, accum=1, micro_batch=1, data=path, save_dir=save, save_every=1
            ).run()
            jsonl = td / "docs.jsonl"
            jsonl.write_text(json.dumps({"tokens": list(range(cfg.seq_len))}) + "\n")
            out = Trainer(
                cfg, "B0", "cpu", steps=1, accum=1, micro_batch=1, data=jsonl, resume=save / "latest.pt"
            ).run()
            self.assertEqual(out.step, 1)

    def test_grad_ckpt_b2(self) -> None:
        nll = train_loop(self.cfg, "B2", steps=1, device="cpu", accum=1, grad_ckpt=True)
        self.assertTrue(nll > 0)

    def test_seq_len_override_rewindows_packed_bin(self) -> None:
        toks = list(range(32))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            (Path(str(path) + ".meta.json")).write_text(json.dumps({"seq_len": 16}))
            tr = Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, data=path, seq_len=8)
            self.assertEqual(tr.seq_len, 8)
            nll = tr.run().nll
            self.assertTrue(math.isfinite(nll))

    def test_log_includes_grad_norm(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, log_path=log).run()
            row = json.loads(log.read_text().splitlines()[0])
            self.assertIn("grad_norm", row)
            self.assertIn("tok_s", row)
            self.assertIn("aux", row)
            self.assertEqual(row["adam"], "gpu")
            self.assertIn("ppl", row)
            self.assertIn("moe_cv", row)
            self.assertIn("n_valid", row)
            self.assertEqual(row["kd_w"], 0.0)
            self.assertEqual(row["world"], 1)
            self.assertGreater(row["n_valid"], 0)

    def test_eval_every_without_eval_data_skips_dummy_stream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, log_path=log, eval_every=1).run()
            row = json.loads(log.read_text().splitlines()[0])
            ev = row.get("eval_nll")
            self.assertTrue(
                ev is None or (isinstance(ev, float) and math.isnan(ev)),
                msg=row,
            )
            self.assertNotIn("eval_nll", row)

    def test_eval_nll_is_token_weighted_and_skips_empty(self) -> None:
        tr = Trainer(self.cfg, "B0", "cpu", steps=1, eval_data=Path("unused.bin"))

        class FakeStream:
            def batch(self, micro_batch, device):
                return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

        class FakeModel:
            def __init__(self, seq):
                self.training = True
                self._seq = list(seq)
                self.i = 0

            def eval(self):
                self.training = False

            def train(self, mode: bool = True):
                self.training = bool(mode)

            def __call__(self, **kwargs):
                item = self._seq[self.i]
                self.i += 1
                return item

        tr._open = lambda path, seed: FakeStream()
        seq = [
            {"nll": torch.tensor(2.0), "n_valid": torch.tensor(10)},
            {"nll": torch.tensor(4.0), "n_valid": torch.tensor(0)},
            {"nll": torch.tensor(1.0), "n_valid": torch.tensor(30)},
        ]
        got = tr._eval_nll(FakeModel(seq), batches=3)
        # (2.0*10 + 1.0*30) / 40 = 1.25; empty n_valid skipped. Not mean (2+4+1)/3.
        self.assertAlmostEqual(got, 1.25)
        none_tr = Trainer(self.cfg, "B0", "cpu", steps=1, eval_data=None)
        self.assertTrue(math.isnan(none_tr._eval_nll(FakeModel([]), batches=1)))
        empty = [
            {"nll": torch.tensor(3.0), "n_valid": torch.tensor(0)},
            {"nll": torch.tensor(9.0), "n_valid": torch.tensor(0)},
        ]
        self.assertTrue(math.isnan(tr._eval_nll(FakeModel(empty), batches=2)))

    def test_token_mean_nll_empty_rank_does_not_poison(self) -> None:
        from cat_yoko.trainer import token_mean_nll

        # rank0: nll=2 over 10 tokens; rank1: no valid tokens → (0, 0), not nan.
        self.assertAlmostEqual(token_mean_nll(2.0 * 10 + 0.0, 10.0 + 0.0), 2.0)
        self.assertTrue(math.isnan(token_mean_nll(0.0, 0.0)))
        self.assertTrue(math.isnan(token_mean_nll(float("nan"), 0.0)))
        tr = Trainer(self.cfg, "B0", "cpu", steps=1)
        self.assertAlmostEqual(tr._allreduce_token_nll(20.0, 10.0), 2.0)
        self.assertTrue(math.isnan(tr._allreduce_token_nll(0.0, 0.0)))

    def test_open_dummy_does_not_double_offset_rank_seed(self) -> None:
        from cat_yoko.data import DummyStream

        tr = Trainer(self.cfg, "B0", "cpu", steps=1)
        tr.rank = 1
        tr.world = 2
        got = tr._open(None, tr.seed).batch(1, "cpu")["input_ids"]
        exp = DummyStream(
            self.cfg.vocab_size, tr.seq_len, seed=tr.seed, shard_id=1, num_shards=2
        ).batch(1, "cpu")["input_ids"]
        self.assertTrue(torch.equal(got, exp))
        doubled = DummyStream(
            self.cfg.vocab_size, tr.seq_len, seed=tr.seed + 1, shard_id=1, num_shards=2
        ).batch(1, "cpu")["input_ids"]
        self.assertFalse(torch.equal(got, doubled))

    def test_log_jsonl_is_strict_json(self) -> None:
        from cat_yoko.trainer import _json_safe

        self.assertIsNone(_json_safe(float("nan")))
        self.assertIsNone(_json_safe(float("inf")))
        self.assertIsNone(_json_safe(float("-inf")))
        self.assertEqual(_json_safe(1.25), 1.25)
        self.assertIsNone(_json_safe({"ppl": float("nan")})["ppl"])
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            tr = Trainer(self.cfg, "B0", "cpu", steps=1, log_path=log)
            tr._log(
                {
                    "name": "tiny",
                    "phase": "B0",
                    "step": 1,
                    "steps_or_inf": 1,
                    "nll": 1.0,
                    "ppl": float("nan"),
                    "aux": float("inf"),
                    "gate": 0.0,
                    "grad_norm": 0.0,
                    "moe_cv": 0.0,
                    "trainable_m": 1.0,
                    "lr": 1e-4,
                    "fp8": False,
                    "nvfp4": False,
                    "tokens_seen": 1.0,
                    "tok_s": 1.0,
                    "mem_mib": 0.0,
                    "path": Path("/tmp/x"),
                }
            )
            raw = log.read_text().splitlines()[0]
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            row = json.loads(raw)
            self.assertIsNone(row["ppl"])
            self.assertIsNone(row["aux"])
            self.assertEqual(row["path"], "/tmp/x")

    def test_built_line_path_does_not_crash(self) -> None:
        tr = Trainer(self.cfg, "B0", "cpu", steps=1)

        class M:
            def param_count(self):
                return 123

        tr._print_built(M(), n_train=1, adam_state="gpu")

    def test_eval_nll_lands_in_jsonl(self) -> None:
        toks = list(range(64))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            log = td / "m.jsonl"
            data = td / "tok.bin"
            data.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            Trainer(
                self.cfg,
                "B0",
                "cpu",
                steps=1,
                accum=1,
                log_path=log,
                eval_every=1,
                eval_data=data,
            ).run()
            row = json.loads(log.read_text().splitlines()[0])
            self.assertIn("eval_nll", row)
            self.assertGreater(row["eval_nll"], 0)

    def test_weights_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg,
                "B0",
                "cpu",
                steps=1,
                accum=1,
                save_dir=save,
                save_every=1,
                save_optim=False,
            ).run()
            ckpt = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
            self.assertIsNone(ckpt["optimizer"])
            self.assertTrue(ckpt["model"])

    def test_keep_last_prunes_step_ckpts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                self.cfg,
                "B0",
                "cpu",
                steps=3,
                accum=1,
                save_dir=save,
                save_every=1,
                save_keep=2,
            ).run()
            steps = sorted(save.glob("step_*.pt"))
            self.assertEqual([p.name for p in steps], ["step_2.pt", "step_3.pt"])
            self.assertTrue((save / "latest.pt").is_file())

    def test_c1_chain_continues_packed_cursor(self) -> None:
        from cat_yoko.trainer import run_c1_chain

        toks = list(range(96))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tok.bin"
            path.write_bytes(struct.pack("<" + "i" * len(toks), *toks))
            save = Path(td) / "c1"
            out = run_c1_chain(
                self.cfg,
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                data=path,
                save_dir=save,
            )
            self.assertEqual(out["B2"].stream["i"], 3)
            for phase in ("B0", "B1", "B2"):
                self.assertTrue((save / phase / "latest.pt").is_file(), msg=phase)

    def test_wsd_b1_offset_skips_warmup(self) -> None:
        lr = wsd_lr(8e9, self.cfg, "B1")
        self.assertAlmostEqual(lr, self.cfg.lr)

    def test_kd_weight_needs_more_than_one_step(self) -> None:
        self.assertEqual(kd_weight(0, 1, 0.5), 0.0)
        self.assertGreater(kd_weight(0, 8, 0.5), 0.3)
        self.assertEqual(kd_kl(torch.zeros(2, 4), torch.zeros(2, 4), 2.0).shape, ())
        identical = kd_kl(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4), 2.0)
        self.assertAlmostEqual(float(identical), 0.0)
        ignore = torch.full((2, 3), -100)
        self.assertEqual(float(kd_kl(torch.ones(2, 3, 4), torch.zeros(2, 3, 4), 2.0, ignore=ignore)), 0.0)

    def test_kd_weight_on_token_budget(self) -> None:
        self.assertEqual(kd_weight(0, None, 0.5), 0.0)
        self.assertGreater(
            kd_weight(0, None, 0.5, tokens_in_phase=0.0, phase_budget=15e9),
            0.4,
        )
        late = kd_weight(0, None, 0.5, tokens_in_phase=14e9, phase_budget=15e9)
        self.assertGreater(late, 0.0)
        self.assertLess(late, 0.1)

    def test_kd_w_logged_when_tokens_without_steps(self) -> None:
        teacher = DummyTeacher(self.cfg.vocab_size, self.cfg.hidden_size)
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            Trainer(
                self.cfg,
                "B0",
                "cpu",
                tokens=64,
                steps=None,
                accum=1,
                micro_batch=2,
                teacher=teacher,
                log_path=log,
            ).run()
            row = json.loads(log.read_text().splitlines()[0])
            self.assertGreater(row["kd_w"], 0)

    def test_upcycle_src_dropped_after_copy(self) -> None:
        src = dummy_minicpm_state(self.cfg)
        tr = Trainer(self.cfg, "B0", "cpu", steps=1, accum=1, upcycle_src=src)
        tr.run()
        self.assertFalse(hasattr(tr, "upcycle_src"))

    def test_safe_ppl_caps(self) -> None:
        from cat_yoko.loss import safe_ppl

        self.assertAlmostEqual(safe_ppl(0.0), 1.0)
        self.assertIsNone(safe_ppl(float("nan")))
        self.assertIsNone(safe_ppl(99.0))
        self.assertGreater(safe_ppl(2.0), 7.0)

    def test_chunked_ce_matches_full_logits(self) -> None:
        from cat_yoko.loss import linear_cross_entropy

        torch.manual_seed(0)
        b, s, d, v = 2, 7, 8, 11
        h1 = torch.randn(b, s, d, requires_grad=True)
        h2 = h1.detach().clone().requires_grad_(True)
        lm1 = torch.nn.Linear(d, v, bias=False)
        lm2 = torch.nn.Linear(d, v, bias=False)
        lm2.load_state_dict(lm1.state_dict())
        labels = torch.randint(0, v, (b, s))
        labels[0, 3] = -100
        nll, n_valid = linear_cross_entropy(h1, labels, lm1, chunk_tokens=3)
        logits = lm2(h2)
        ref = torch.nn.functional.cross_entropy(
            logits.reshape(-1, v),
            labels.reshape(-1),
            ignore_index=-100,
        )
        self.assertEqual(int(n_valid), int((labels != -100).sum()))
        self.assertTrue(torch.allclose(nll, ref, atol=1e-5, rtol=1e-5))
        nll.backward()
        ref.backward()
        self.assertTrue(torch.allclose(h1.grad, h2.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(lm1.weight.grad, lm2.weight.grad, atol=1e-5, rtol=1e-5))

    def test_ce_chunk_auto_cpu_stays_small(self) -> None:
        from cat_yoko.loss import _ce_chunk_tokens

        self.assertEqual(_ce_chunk_tokens(100, 130560, None, torch.device("cpu")), 100)
        self.assertEqual(_ce_chunk_tokens(4096, 130560, None, torch.device("cpu")), 512)
        self.assertEqual(_ce_chunk_tokens(4096, 130560, 3, torch.device("cpu")), 3)

    def test_host_step_stats_match_python_floats(self) -> None:
        from cat_yoko.trainer import _host_step_stats

        nll = torch.tensor(1.5)
        n_valid = torch.tensor(8.0)
        loss = torch.tensor(1.25)
        aux = torch.tensor(0.0)
        w, n, lo, a = _host_step_stats(nll, n_valid, loss, aux)
        self.assertAlmostEqual(w, 1.5)
        self.assertAlmostEqual(n, 8.0)
        self.assertAlmostEqual(lo, 1.25)
        self.assertAlmostEqual(a, 0.0)

    def test_configure_cuda_enables_flash_sdp(self) -> None:
        import inspect

        from cat_yoko.trainer import configure_cuda

        src = inspect.getsource(configure_cuda)
        self.assertIn("enable_flash_sdp", src)
        self.assertIn("enable_cudnn_sdp", src)

    def test_runtime_flags_drop_logits_without_teacher(self) -> None:
        from cat_yoko.model import CATYokoForCausalLM

        tr = Trainer(self.cfg, "B0", "cpu", steps=1)
        model = CATYokoForCausalLM(self.cfg)
        tr._apply_runtime_flags(model)
        self.assertFalse(model.return_logits)
        ids = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.seq_len))
        out = model(input_ids=ids, labels=ids)
        self.assertNotIn("logits", out)
        self.assertIn("nll", out)
        self.assertTrue(torch.isfinite(out["nll"]))
        out["loss"].backward()
        self.assertIsNotNone(model.cache_k.weight.grad)

    def test_runtime_flags_keep_logits_for_kd(self) -> None:
        from cat_yoko.model import CATYokoForCausalLM

        teacher = DummyTeacher(self.cfg.vocab_size, self.cfg.hidden_size)
        tr = Trainer(self.cfg, "B0", "cpu", steps=1, teacher=teacher)
        model = CATYokoForCausalLM(self.cfg)
        tr._apply_runtime_flags(model)
        self.assertTrue(model.return_logits)
        ids = torch.randint(0, self.cfg.vocab_size, (1, 4))
        out = model(input_ids=ids, labels=ids)
        self.assertIn("logits", out)

    def test_auto_accum_tiny(self) -> None:
        self.assertEqual(auto_accum(self.cfg, micro_batch=2, world=1), 4)

    def test_adamw_skips_norm_decay(self) -> None:
        from cat_yoko.model import CATYokoForCausalLM
        from cat_yoko.freeze import apply_freeze

        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B0")
        groups = adamw_param_groups(model, 0.1)
        self.assertEqual(len(groups), 2)

    def test_adamw_router_nodecay_not_swiglu_gate(self) -> None:
        from cat_yoko.freeze import apply_freeze
        from cat_yoko.model import CATYokoForCausalLM

        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B1")
        groups = adamw_param_groups(model, 0.1)
        nodecay = {id(p) for p in groups[1]["params"]}
        decay = {id(p) for p in groups[0]["params"]}
        self.assertIn(id(model.decoder[0].mlp.router.weight), nodecay)
        self.assertIn(id(model.decoder[0].mlp.experts[0].gate_proj.weight), decay)

    def test_dummy_stream_ignores_packed_kind(self) -> None:
        from cat_yoko.data import DummyStream

        stream = DummyStream(self.cfg.vocab_size, self.cfg.seq_len, seed=0)
        before = stream.gen.get_state().clone()
        stream.load_state_dict({"kind": "packed", "i": 9})
        self.assertTrue(torch.equal(stream.gen.get_state(), before))

    def test_two_rank_gloo_one_step(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_ddp

        row = run_gloo_ddp(device="cpu", world=2, steps=1)
        self.assertTrue(row["ok"], msg=row)
        self.assertEqual(row["step"], 1)
        self.assertEqual(row["world"], 2)
        self.assertGreater(row["nll"], 0)

    def test_two_rank_gloo_accum(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_ddp

        row = run_gloo_ddp(device="cpu", world=2, steps=1, accum=2)
        self.assertTrue(row["ok"], msg=row)
        self.assertEqual(row["accum"], 2)

    def test_two_rank_gloo_c1_chain(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_c1

        row = run_gloo_c1(device="cpu", world=2, steps=1)
        self.assertTrue(row["ok"], msg=row)
        self.assertEqual(row["phase"], "C1")
        self.assertGreater(row["b0"], 0)
        self.assertGreater(row["b1"], 0)
        self.assertGreater(row["b2"], 0)


class CliTests(unittest.TestCase):
    def test_twelve_b_cli_errors_on_32gib_card(self) -> None:
        from cat_yoko.train import twelve_b_cli_errors

        self.assertTrue(
            any("seq-len 64" in e for e in twelve_b_cli_errors(seq_len=None, teacher_hf=False, gpu_gib=31.48))
        )
        self.assertEqual(twelve_b_cli_errors(seq_len=64, teacher_hf=False, gpu_gib=31.48), [])
        te = twelve_b_cli_errors(seq_len=64, teacher_hf=True, gpu_gib=31.48)
        self.assertTrue(any("teacher" in e for e in te))
        self.assertEqual(twelve_b_cli_errors(seq_len=None, teacher_hf=True, gpu_gib=80.0), [])

    def test_12b_cuda_device_without_runtime_errors_before_build(self) -> None:
        with patch("cat_yoko.train.cuda_runtime_available", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                main(
                    [
                        "--config",
                        "12b",
                        "--phase",
                        "B0",
                        "--steps",
                        "1",
                        "--device",
                        "cuda:0",
                        "--seq-len",
                        "64",
                    ]
                )
        self.assertEqual(cm.exception.code, 2)

    def test_expandable_segments_env_is_set(self) -> None:
        import os

        from cat_yoko.trainer import enable_expandable_segments

        got = enable_expandable_segments()
        self.assertIn("expandable_segments:True", got)
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), got)

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

    def test_cli_resume_from_dir_when_latest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            run = td / "run"
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
                    str(run),
                    "--save-every",
                    "1",
                ]
            )
            self.assertEqual(code, 0)
            (run / "latest.pt").unlink()
            code = main(
                [
                    "--config",
                    "tiny",
                    "--phase",
                    "B0",
                    "--steps",
                    "2",
                    "--accum",
                    "1",
                    "--resume",
                    str(run),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((run / "step_1.pt").is_file())

    def test_12b_refuses_full_envelope(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "12b", "--phase", "B0"])

    def test_12b_cpu_steps_need_cuda(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "12b", "--phase", "B0", "--steps", "1"])

    def test_12b_b2_tokens_forces_accum_one(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=15e9, offload_blocks=None, phase="B2", c1=False),
            1,
        )

    def test_12b_b2_tokens_no_offload_keeps_auto(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=15e9, offload_blocks=False, phase="B2", c1=False),
            0,
        )

    def test_12b_b2_explicit_accum_without_no_offload_errors(self) -> None:
        with self.assertRaises(ValueError):
            resolve_12b_accum(1024, tokens=15e9, offload_blocks=None, phase="B2", c1=False)

    def test_12b_b2_explicit_accum_with_no_offload_ok(self) -> None:
        self.assertEqual(
            resolve_12b_accum(1024, tokens=15e9, offload_blocks=False, phase="B2", c1=False),
            1024,
        )

    def test_12b_b0_tokens_keeps_auto(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=8e9, offload_blocks=None, phase="B0", c1=False),
            0,
        )

    def test_12b_smoke_steps_accum_one(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=None, offload_blocks=None, phase="B0", c1=False),
            1,
        )

    def test_12b_c1_forces_accum_one(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=None, offload_blocks=None, phase="B0", c1=True),
            1,
        )

    def test_12b_explicit_offload_blocks_on_b0_forces_accum_one(self) -> None:
        self.assertEqual(
            resolve_12b_accum(0, tokens=8e9, offload_blocks=True, phase="B0", c1=False),
            1,
        )

    def test_12b_cli_b2_accum_without_no_offload_errors(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "--config",
                    "12b",
                    "--phase",
                    "B2",
                    "--tokens",
                    "15e9",
                    "--accum",
                    "4",
                    "--device",
                    "cuda",
                ]
            )


if __name__ == "__main__":
    unittest.main()
