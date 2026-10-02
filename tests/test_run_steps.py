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
from cat_yoko.data import DummyStream
from cat_yoko.moe import arm_moe_load_tracking
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
        blob["extra"].update(step=100, tokens_in_phase=160, tokens_seen=160)
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
            self.assertEqual(result.step, 102)
            self.assertEqual(result.tokens_seen, 192)
            blob = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(blob["extra"]["step"], 102)
            self.assertEqual(blob["extra"]["tokens_in_phase"], 192)
            self.assertAlmostEqual(blob["extra"]["gate"], 0.3 * 177 / 4096, places=7)
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [102])
            self.assertAlmostEqual(rows[0]["lr"], self.cfg.lr * 176 / 1024)
            self.assertEqual(rows[0]["steps_or_inf"], 102)

    def _assert_stream_cursor(self, source: dict, actual: dict, batches: int) -> None:
        expected = DummyStream(self.cfg.vocab_size, self.cfg.seq_len)
        expected.load_state_dict(source)
        for _ in range(batches):
            expected.batch(1, "cpu")
        self.assertTrue(torch.equal(actual["gen"], expected.state_dict()["gen"]))
        restored = DummyStream(self.cfg.vocab_size, self.cfg.seq_len)
        restored.load_state_dict(actual)
        expected_next = expected.batch(1, "cpu")
        actual_next = restored.batch(1, "cpu")
        for key in expected_next:
            self.assertTrue(torch.equal(actual_next[key], expected_next[key]), key)

    def test_relative_bounds_do_not_prefetch_past_final_update(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            source = torch.load(resume, map_location="cpu", weights_only=False)["extra"]["stream"]
            for option in ("run_steps", "more_steps"):
                for updates in (1, 2, 3):
                    with self.subTest(option=option, updates=updates):
                        save = root / f"{option}_{updates}"
                        log = save / "metrics.jsonl"
                        result = Trainer(
                            self.cfg, "B0", "cpu", **{option: updates}, tokens=4096,
                            accum=2, micro_batch=1, resume=resume, save_dir=save,
                            save_every=0, log_every=1000, log_path=log,
                        ).run()
                        self.assertEqual(result.step, 100 + updates)
                        self.assertEqual(result.tokens_seen, 160 + 32 * updates)
                        final = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
                        self.assertEqual(final["extra"]["step"], 100 + updates)
                        self._assert_stream_cursor(source, result.stream, updates * 2)
                        self._assert_stream_cursor(source, final["extra"]["stream"], updates * 2)
                        rows = [json.loads(line) for line in log.read_text().splitlines()]
                        self.assertEqual([row["step"] for row in rows], [100 + updates])
                        self.assertGreater(rows[0]["nll"], 0)
                        self.assertAlmostEqual(rows[0]["gate"], 0.3 * (161 + 32 * (updates - 1)) / 4096)
                        self.assertAlmostEqual(rows[0]["lr"], self.cfg.lr * (160 + 32 * (updates - 1)) / 1024)
                        self.assertEqual(rows[0]["steps_or_inf"], 100 + updates)

    def test_periodic_checkpoint_cursor_excludes_prefetched_next_batch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            source = torch.load(resume, map_location="cpu", weights_only=False)["extra"]["stream"]
            save = root / "continued"
            result = Trainer(
                self.cfg, "B0", "cpu", more_steps=3, tokens=4096,
                accum=1, micro_batch=1, resume=resume, save_dir=save,
                save_every=101, log_every=1000,
            ).run()
            periodic = torch.load(save / "step_101.pt", map_location="cpu", weights_only=False)
            self._assert_stream_cursor(source, periodic["extra"]["stream"], 1)
            self._assert_stream_cursor(source, result.stream, 3)

    def test_phase_token_budget_can_stop_before_run_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            log = root / "metrics.jsonl"
            source = torch.load(resume, map_location="cpu", weights_only=False)["extra"]["stream"]
            with patch("cat_yoko.trainer.arm_moe_load_tracking", wraps=arm_moe_load_tracking) as tracking:
                result = Trainer(
                    self.cfg, "B0", "cpu", run_steps=5, tokens=176,
                    accum=1, micro_batch=1, resume=resume,
                    log_every=1000, log_path=log,
                ).run()
            self.assertEqual(result.step, 101)
            self.assertEqual(result.tokens_seen, 176)
            self.assertEqual(json.loads(log.read_text())["step"], 101)
            self.assertTrue(tracking.call_args.kwargs["log_step"])
            self._assert_stream_cursor(source, result.stream, 1)

    def test_reached_absolute_cap_does_not_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            resume = self._resumable_checkpoint(root / "source")
            source = torch.load(resume, map_location="cpu", weights_only=False)["extra"]["stream"]
            result = Trainer(
                self.cfg, "B0", "cpu", steps=99, more_steps=3, tokens=4096,
                accum=1, micro_batch=1, resume=resume,
            ).run()
            self.assertEqual(result.step, 100)
            self.assertEqual(result.tokens_seen, 160)
            self._assert_stream_cursor(source, result.stream, 0)

    def test_absolute_step_budget_still_applies(self) -> None:
        result = Trainer(
            self.cfg, "B0", "cpu", steps=1, run_steps=3,
            accum=1, micro_batch=1,
        ).run()
        self.assertEqual(result.step, 1)

    def test_cli_runs_requested_updates_without_tiny_default_step_limit(self) -> None:
        for option in ("--run-steps", "--more-steps"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as td:
                save = Path(td)
                self.assertEqual(main([
                    "--config", "tiny", "--phase", "B0", "--device", "cpu",
                    option, "4", "--accum", "1", "--micro-batch", "1",
                    "--save-dir", str(save),
                ]), 0)
                blob = torch.load(save / "latest.pt", map_location="cpu", weights_only=False)
                self.assertEqual(blob["extra"]["step"], 4)

    def test_tight_gpu_phase_cli_retains_published_token_schedule(self) -> None:
        for option in ("--run-steps", "--more-steps"):
            with self.subTest(option=option), patch("cat_yoko.phase_train._tight_gpu", return_value=True):
                argv = build_phase_argv("B0", [option, "2", "--resume", "/tmp/b0-full"])
            self.assertEqual(argv[argv.index(option) + 1], "2")
            self.assertEqual(float(argv[argv.index("--tokens") + 1]), C1_SPLIT["B0"])
            self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
            self.assertNotIn("--steps", argv)

    def test_run_steps_requires_positive_update_count(self) -> None:
        for option in ("run_steps", "more_steps"):
            for count in (0, -1):
                with self.subTest(option=option, count=count):
                    with self.assertRaisesRegex(ValueError, "positive"):
                        Trainer(self.cfg, "B0", "cpu", **{option: count})
                    flag = "--" + option.replace("_", "-")
                    with self.assertRaises(SystemExit):
                        main([flag, str(count)])
                    with self.assertRaises(SystemExit):
                        build_phase_argv("B0", [flag, str(count)])

    def test_relative_options_reject_conflicting_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            Trainer(self.cfg, "B0", "cpu", more_steps=1, run_steps=2)
        flags = ["--more-steps", "1", "--run-steps", "2"]
        with self.assertRaises(SystemExit):
            main(flags)
        with self.assertRaises(SystemExit):
            build_phase_argv("B0", flags)

    def test_relative_options_allow_matching_counts(self) -> None:
        result = Trainer(
            self.cfg, "B0", "cpu", more_steps=2, run_steps=2,
            accum=1, micro_batch=1,
        ).run()
        self.assertEqual(result.step, 2)
        argv = build_phase_argv("B0", ["--more-steps", "2", "--run-steps", "2"])
        self.assertEqual(argv.count("--run-steps"), 1)
        self.assertEqual(argv[argv.index("--run-steps") + 1], "2")


if __name__ == "__main__":
    unittest.main()
