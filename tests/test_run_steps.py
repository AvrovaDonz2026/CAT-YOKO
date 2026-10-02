"""Bounded continuation must count new updates without shortening the curriculum."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from cat_yoko.config import CATYokoConfig, C1_SPLIT
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.train import main
from cat_yoko.trainer import Trainer


class RunStepsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = replace(CATYokoConfig.tiny(), warmup_tokens=1024)

    def _resumable_checkpoint(self, path: Path) -> Path:
        Trainer(
            self.cfg, "B0", "cpu", steps=1, accum=1, micro_batch=1,
            save_dir=path,
        ).run()
        checkpoint = path / "latest.pt"
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        blob["extra"].update(step=40, tokens_in_phase=160, tokens_seen=160)
        torch.save(blob, checkpoint)
        return checkpoint

    def test_resume_beyond_try_limit_preserves_token_schedule_and_final_save(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            save = root / "continued"
            log = root / "metrics.jsonl"
            result = Trainer(
                self.cfg, "B0", "cpu", run_steps=2, tokens=4096,
                accum=1, micro_batch=1, resume=resume, save_dir=save,
                log_every=1000, log_path=log,
            ).run()
            self.assertEqual(result.step, 42)
            self.assertEqual(result.tokens_seen, 192)
            blob = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(blob["extra"]["step"], 42)
            self.assertEqual(blob["extra"]["tokens_in_phase"], 192)
            self.assertAlmostEqual(blob["extra"]["gate"], 0.3 * 177 / 4096, places=7)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [42])
            self.assertAlmostEqual(rows[0]["lr"], self.cfg.lr * 176 / 1024)
            self.assertEqual(rows[0]["steps_or_inf"], 42)

    def test_phase_token_budget_can_stop_before_run_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            log = root / "metrics.jsonl"
            result = Trainer(
                self.cfg, "B0", "cpu", run_steps=5, tokens=176,
                accum=1, micro_batch=1, resume=resume,
                log_every=1000, log_path=log,
            ).run()
            self.assertEqual(result.step, 41)
            self.assertEqual(result.tokens_seen, 176)
            self.assertEqual(json.loads(log.read_text())["step"], 41)

    def test_absolute_step_budget_still_applies(self) -> None:
        result = Trainer(
            self.cfg, "B0", "cpu", steps=1, run_steps=3,
            accum=1, micro_batch=1,
        ).run()
        self.assertEqual(result.step, 1)

    def test_cli_runs_requested_updates_without_tiny_default_step_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            self.assertEqual(main([
                "--config", "tiny", "--phase", "B0", "--device", "cpu",
                "--run-steps", "4", "--accum", "1", "--micro-batch", "1",
                "--save-dir", str(save),
            ]), 0)
            blob = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(blob["extra"]["step"], 4)

    def test_tight_gpu_phase_cli_retains_published_token_schedule(self) -> None:
        with patch("cat_yoko.phase_train._tight_gpu", return_value=True):
            argv = build_phase_argv("B0", ["--run-steps", "2", "--resume", "/tmp/b0-full"])
        self.assertEqual(argv[argv.index("--run-steps") + 1], "2")
        self.assertEqual(float(argv[argv.index("--tokens") + 1]), C1_SPLIT["B0"])
        self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
        self.assertNotIn("--steps", argv)

    def test_run_steps_requires_positive_update_count(self) -> None:
        for count in (0, -1):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "positive"):
                    Trainer(self.cfg, "B0", "cpu", run_steps=count)
                with self.assertRaises(SystemExit):
                    main(["--run-steps", str(count)])
                with self.assertRaises(SystemExit):
                    build_phase_argv("B0", ["--run-steps", str(count)])


if __name__ == "__main__":
    unittest.main()
