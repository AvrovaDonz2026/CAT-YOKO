#!/usr/bin/env python3
"""B0/B1/B2 phase CLIs, trainable LFS checkpoints, sharded full graphs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.checkpoint import (
    is_trainable_ckpt,
    load_checkpoint,
    load_trainable_state,
    newest_trainable_checkpoint,
    prune_step_checkpoints,
    resolve_resume_path,
    save_sharded_checkpoint,
    save_trainable_checkpoint,
    shard_state_dict,
    trainable_state_dict,
)
from cat_yoko.config import CATYokoConfig, C1_SPLIT
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.phases import PHASES, TRY_STEPS
from cat_yoko.trainer import Trainer


class PhaseSpecTests(unittest.TestCase):
    def test_published_envelopes(self) -> None:
        self.assertEqual(PHASES["B0"].tokens, C1_SPLIT["B0"])
        self.assertEqual(PHASES["B1"].tokens, C1_SPLIT["B1"])
        self.assertEqual(PHASES["B2"].tokens, C1_SPLIT["B2"])
        self.assertEqual(PHASES["B0"].student, "bf16")
        self.assertEqual(PHASES["B1"].student, "fp8_moe")
        self.assertTrue(PHASES["B2"].offload_blocks)
        self.assertFalse(PHASES["B0"].offload_blocks)

    def test_b0_try_argv(self) -> None:
        argv = build_phase_argv("B0", ["--try", "--save-dir", "/tmp/b0"])
        self.assertIn("--phase", argv)
        self.assertEqual(argv[argv.index("--phase") + 1], "B0")
        self.assertIn("--dummy-upcycle", argv)
        self.assertIn("--no-save-full", argv)
        self.assertIn("--save-trainable", argv)
        self.assertIn("--no-save-optim", argv)
        self.assertIn("--offload-encoder", argv)
        self.assertNotIn("--optim-cpu", argv)
        self.assertIn("--seq-len", argv)
        self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
        self.assertEqual(argv[argv.index("--steps") + 1], str(TRY_STEPS))
        self.assertNotIn("--tokens", argv)

    def test_b0_try_keeps_explicit_upcycle(self) -> None:
        argv = build_phase_argv(
            "B0", ["--try", "--upcycle-hf", "openbmb/MiniCPM5-2B-Base"]
        )
        self.assertIn("--upcycle-hf", argv)
        self.assertNotIn("--dummy-upcycle", argv)

    def test_tight_gpu_refuses_default_envelope(self) -> None:
        from unittest.mock import patch

        with patch("cat_yoko.phase_train._tight_gpu", return_value=True):
            with self.assertRaises(SystemExit):
                build_phase_argv("B0", ["--save-dir", "/tmp/b0"])
            argv = build_phase_argv("B0", ["--try", "--save-dir", "/tmp/b0"])
            self.assertEqual(argv[argv.index("--steps") + 1], str(TRY_STEPS))

    def test_b1_envelope_argv(self) -> None:
        argv = build_phase_argv("B1", ["--save-dir", "/tmp/b1"])
        self.assertEqual(argv[argv.index("--phase") + 1], "B1")
        self.assertIn("--offload-encoder", argv)
        self.assertIn("--optim-cpu", argv)
        self.assertNotIn("--offload-blocks", argv)
        self.assertEqual(argv[argv.index("--tokens") + 1], str(C1_SPLIT["B1"]))

    def test_b2_envelope_argv(self) -> None:
        argv = build_phase_argv("B2", ["--save-dir", "/tmp/b2"])
        self.assertIn("--offload-blocks", argv)
        self.assertIn("--optim-cpu", argv)
        self.assertNotIn("--offload-encoder", argv)
        self.assertEqual(argv[argv.index("--tokens") + 1], str(C1_SPLIT["B2"]))
        self.assertIn("--no-save-full", argv)


class TrainableCkptTests(unittest.TestCase):
    def test_b0_trainable_roundtrip(self) -> None:
        cfg = CATYokoConfig.tiny()
        src = CATYokoForCausalLM(cfg)
        apply_freeze(src, "B0")
        names = set(trainable_state_dict(src))
        self.assertTrue(any("cross_attn" in n for n in names))
        self.assertFalse(any(n.startswith("encoder.") for n in names))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trainable.pt"
            save_trainable_checkpoint(path, model=src, extra={"phase": "B0", "step": 3})
            ckpt = load_checkpoint(path)
            self.assertTrue(is_trainable_ckpt(ckpt))
            dst = CATYokoForCausalLM(cfg)
            apply_freeze(dst, "B0")
            load_trainable_state(dst, ckpt["trainable"])
            for n, p in src.named_parameters():
                if p.requires_grad:
                    self.assertTrue(torch.equal(p.cpu(), dict(dst.named_parameters())[n].cpu()), n)
            self.assertEqual(resolve_resume_path(Path(td)).name, "trainable.pt")
            self.assertEqual(newest_trainable_checkpoint(Path(td)), None)

    def test_resume_prefers_full_latest_over_trainable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "trainable.pt").write_bytes(b"x")
            (td / "latest.pt").write_bytes(b"y")
            self.assertEqual(resolve_resume_path(td).name, "latest.pt")

    def test_prune_trainable_keeps_hardlinked_pointer(self) -> None:
        from cat_yoko.checkpoint import publish_latest

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = td / "trainable_step_1.pt"
            b = td / "trainable_step_2.pt"
            torch.save({"kind": "trainable", "trainable": {}}, a)
            torch.save({"kind": "trainable", "trainable": {}}, b)
            publish_latest(a, td / "trainable.pt")
            prune_step_checkpoints(td, keep=1)
            self.assertTrue(a.is_file())
            self.assertTrue(b.is_file())
            self.assertTrue((td / "trainable.pt").samefile(a))

    def test_shard_splits_under_cap(self) -> None:
        state = {
            "a": torch.zeros(8, dtype=torch.float32),
            "b": torch.zeros(8, dtype=torch.float32),
        }
        shards = shard_state_dict(state, max_bytes=20)
        self.assertGreaterEqual(len(shards), 2)
        keys = [k for s in shards for k in s]
        self.assertCountEqual(keys, ["a", "b"])

    def test_sharded_checkpoint_writes_manifest(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as td:
            man = save_sharded_checkpoint(
                Path(td), model=model, extra={"phase": "B0"}, max_bytes=50_000
            )
            self.assertTrue(man.is_file())
            text = man.read_text()
            self.assertIn("sharded", text)
            self.assertGreaterEqual(len(list(Path(td).glob("shard-*.pt"))), 1)

    def test_trainer_writes_trainable_not_full(self) -> None:
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                cfg,
                "B0",
                "cpu",
                steps=1,
                accum=1,
                save_dir=save,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
            ).run()
            self.assertTrue((save / "trainable.pt").is_file())
            self.assertTrue((save / "trainable_step_1.pt").is_file())
            self.assertFalse((save / "latest.pt").is_file())
            self.assertFalse((save / "step_1.pt").is_file())
            ckpt = load_checkpoint(save / "trainable.pt")
            self.assertTrue(is_trainable_ckpt(ckpt))
            self.assertEqual(ckpt["extra"]["step"], 1)


if __name__ == "__main__":
    unittest.main()
