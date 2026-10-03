"""The experiment queue must wait for final, recoverable source state."""

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from operators.rocm.run_operator_queue import steady_metrics, supervise, wait_for_training


class OperatorQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "training"
        (self.run / "long/train").mkdir(parents=True)

    def write_source(self, supervisor, **report):
        (self.run / "status.json").write_text(json.dumps(supervisor))
        (self.run / "long/parity.json").write_text(json.dumps({
            "status": "training_complete", "source_optimizer_present": True,
            "save_optimizer": True, **report}))
        (self.run / "long/train/trainable.pt").write_bytes(b"checkpoint-placeholder")

    def test_waits_for_verified_completion_without_using_partial_checkpoint(self):
        self.write_source({"status": "long_training", "child_pid": 123})
        events = []

        def finish(seconds):
            self.assertGreater(seconds, 0)
            self.write_source({"status": "complete", "final_step": 45000})

        result = wait_for_training(self.run, events.append, max_seconds=60,
                                   sleep=finish, clock=lambda: 0)
        self.assertEqual(result["final_step"], 45000)
        self.assertEqual(events[0]["source_child_pid"], 123)

    def test_failed_incomplete_and_missing_adam_source_are_rejected(self):
        self.write_source({"status": "failed", "error": "disk full"})
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            wait_for_training(self.run, lambda _: None, max_seconds=60)
        for override in ({"status": "parity_pass"}, {"source_optimizer_present": False},
                         {"save_optimizer": False}):
            self.write_source({"status": "complete"}, **override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                wait_for_training(self.run, lambda _: None, max_seconds=60)

    def test_busy_source_wait_is_bounded(self):
        self.write_source({"status": "long_training"})
        times = iter([0, 61])
        with self.assertRaises(TimeoutError):
            wait_for_training(self.run, lambda _: None, max_seconds=60,
                              sleep=lambda _: self.fail("must not sleep after deadline"),
                              clock=lambda: next(times))

    def test_steady_metrics_discard_warmup_and_validate_throughput(self):
        path = self.root / "metrics.jsonl"
        rows = [{"tok_s": value, "mem_mib": 123} for value in (1, 2, 3, 4, 5, 500, 600, 700)]
        rows.append({"eval_nll": 8.2})
        path.write_text("\n".join(json.dumps(row) for row in rows))
        result = steady_metrics(path)
        self.assertEqual(result["median_loop_tokens_per_second"], 600)
        self.assertEqual(result["measured_updates"], 3)
        rows[-2]["tok_s"] = 0
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(ValueError):
            steady_metrics(path)

    def test_queue_cannot_write_inside_active_source(self):
        args = Namespace(training_run=self.run, source_dir=self.root / "source",
                         base=self.root / "base", data=self.root / "data/train.bin",
                         eval_data=self.root / "data/eval.bin", out=self.run / "operator-results",
                         python="python", max_wait_hours=36)
        with self.assertRaisesRegex(ValueError, "outside training"):
            supervise(args)
        self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
