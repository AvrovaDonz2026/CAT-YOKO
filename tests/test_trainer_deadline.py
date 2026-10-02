"""A wall-clock stop must save the next unread batch, including its RNG."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import DummyStream
from cat_yoko.trainer import Trainer


class TrainerDeadlineTests(unittest.TestCase):
    def test_deadline_after_update_restores_unread_prefetch(self):
        cfg = CATYokoConfig.tiny()
        seed = 37
        expected = DummyStream(cfg.vocab_size, cfg.seq_len, seed)
        expected.batch(1, "cpu")
        expected_next = expected.batch(1, "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            trainer = Trainer(
                cfg, "B0", "cpu", run_steps=10, tokens=1e6,
                micro_batch=1, accum=1, seed=seed, max_train_seconds=1,
                save_dir=dest, save_full=False, save_trainable=True,
                save_optim=True,
            )
            with patch("cat_yoko.trainer.time.monotonic", side_effect=[0.0, 0.0, 2.0]):
                result = trainer.run()
            self.assertEqual(result.step, 1)
            self.assertEqual(result.stop_reason, "deadline")
            checkpoint = torch.load(dest / "trainable.pt", weights_only=False)
            self.assertEqual(checkpoint["extra"]["step"], 1)
            actual = DummyStream(cfg.vocab_size, cfg.seq_len, seed)
            actual.load_state_dict(checkpoint["extra"]["stream"])
            actual_next = actual.batch(1, "cpu")
            for key in expected_next:
                torch.testing.assert_close(actual_next[key], expected_next[key], rtol=0, atol=0)
            self.assertIn("optimizer", checkpoint)

    def test_deadline_before_first_update_preserves_first_batch(self):
        cfg = CATYokoConfig.tiny()
        seed = 38
        expected = DummyStream(cfg.vocab_size, cfg.seq_len, seed).batch(1, "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                cfg, "B0", "cpu", run_steps=10, micro_batch=1, accum=1,
                seed=seed, max_train_seconds=1, save_dir=Path(tmp),
                save_full=False, save_trainable=True,
            )
            with patch("cat_yoko.trainer.time.monotonic", side_effect=[0.0, 2.0]):
                result = trainer.run()
            self.assertEqual(result.step, 0)
            self.assertEqual(result.stop_reason, "deadline")
            actual = DummyStream(cfg.vocab_size, cfg.seq_len, seed)
            actual.load_state_dict(result.stream)
            batch = actual.batch(1, "cpu")
            for key in expected:
                torch.testing.assert_close(batch[key], expected[key], rtol=0, atol=0)

    def test_deadline_rejects_invalid_duration(self):
        for seconds in [0, -1, math.inf, math.nan]:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                Trainer(CATYokoConfig.tiny(), "B0", "cpu", max_train_seconds=seconds)


if __name__ == "__main__":
    unittest.main()
