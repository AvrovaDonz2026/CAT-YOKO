"""CPU lifecycle checks for verified continuation and bounded native fallback."""
from argparse import Namespace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from operators.rocm import continue_operator_switch as continuation
from operators.rocm import run_operator_switch as helpers
from operators.rocm import test_run_operator_switch as fixtures


class ContinueOperatorSwitchTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.OperatorSwitchTests("test_selection_requires_five_percent_against_faster_baseline")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        fixture.metadata["checkpoint_verified"] = True
        self.comparison = self.root / "comparison"
        self.comparison.mkdir()
        native_args = fixture.args()
        self.baselines, self.candidates = [], []
        for name, variant, speed, group in (("baseline_before", "native", 100, self.baselines),
                                           ("baseline_after", "native", 120, self.baselines),
                                           ("attention_steps", "attention", 140, self.candidates)):
            directory = self.root / "trials" / name
            fixture.write_run(directory, variant, speed)
            result = helpers.validate_run(directory, source=fixture.resume, source_step=42100,
                steps=20, variant=variant, source_dir=fixture.source,
                expected_names=fixture.metadata["trainable_names"], args=native_args)
            group.append(result)
            (self.comparison / (name + ".validation.json")).write_text(json.dumps(result))
        self.decision = {**helpers.choose_variant(self.baselines, self.candidates),
                         "baselines": self.baselines, "candidates": self.candidates,
                         "source_checkpoint": str(fixture.resume), "source_sha256": helpers.digest_file(fixture.resume),
                         "benchmark_steps": 20}
        (self.comparison / "decision.json").write_text(json.dumps(self.decision))
        (self.comparison / "status.json").write_text(json.dumps({"status": "evaluated_only", "child_pid": None,
            "source_metadata": fixture.metadata}))
        (self.comparison / "source_preflight.log").write_text("CPU diagnostic\n" + json.dumps(fixture.metadata) + "\nAllocator warning on stderr\n")
        self.args = Namespace(comparison=self.comparison, source_dir=fixture.source,
            out=self.root / "continued", python="python", base=fixture.base, data=fixture.data,
            eval_data=fixture.eval_data, deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            gpu_idle_max_wait=3600, stage_timeout_seconds=3600)

    def test_prepare_rejects_comparison_output_choice_and_source_mutation(self):
        self.assertEqual(continuation.prepare(self.args)[0]["selected"], "attention")
        self.args.out = self.comparison / "child"
        with self.assertRaisesRegex(ValueError, "separate"):
            continuation.prepare(self.args)
        self.args.out = self.root / "continued"
        self.decision["selected"] = "native"
        (self.comparison / "decision.json").write_text(json.dumps(self.decision))
        with self.assertRaisesRegex(ValueError, "choice"):
            continuation.prepare(self.args)
        self.decision["selected"] = "attention"
        (self.comparison / "decision.json").write_text(json.dumps(self.decision))
        self.fixture.resume.write_bytes(b"changed fixed source")
        with self.assertRaises(helpers.SourceIntegrityError):
            continuation.prepare(self.args)

    def test_fallback_requires_no_checkpoint_no_metrics_updates_and_no_saved_progress(self):
        directory = self.root / "failed"
        self.assertTrue(continuation.safe_native_fallback(directory, 42100))
        train = directory / "train"
        train.mkdir(parents=True)
        metrics = train / "metrics.jsonl"
        metrics.write_text(json.dumps({"eval_nll": 8.5, "step": 42100}) + "\n")
        self.assertTrue(continuation.safe_native_fallback(directory, 42100))
        metrics.write_text(json.dumps({"step": 42101, "tok_s": 123}) + "\n")
        self.assertFalse(continuation.safe_native_fallback(directory, 42100))
        metrics.unlink()
        checkpoint = train / "trainable_step_42101.pt.tmp"
        checkpoint.write_bytes(b"possible saved progress")
        self.assertFalse(continuation.safe_native_fallback(directory, 42100))
        checkpoint.unlink()
        (directory / "parity.json").write_text(json.dumps({"updates": 1}))
        self.assertFalse(continuation.safe_native_fallback(directory, 42100))
        (directory / "parity.json").unlink()
        metrics.write_text("truncated invalid metric")
        self.assertFalse(continuation.safe_native_fallback(directory, 42100))

    def run_mocked(self, *, candidate_fails=False, records_progress=False):
        commands = []
        fixture = self.fixture

        def child(command, **kwargs):
            commands.append(command)
            flags = command[3:]
            directory = Path(flags[flags.index("--out") + 1])
            if len(commands) == 1 and candidate_fails:
                if records_progress:
                    (directory / "train").mkdir(parents=True)
                    (directory / "train/metrics.jsonl").write_text(json.dumps({"step": 42101, "tok_s": 123}))
                code = 1
            else:
                variant = "attention" if "--packed-attention" in flags else "native"
                report, _ = fixture.write_run(directory, variant, 140, steps=3)
                report["eval_batches"] = 32
                (directory / "parity.json").write_text(json.dumps(report))
                code = 0

            class Child:
                pid = 987654
                def wait(self, timeout):
                    return code
                def kill(self):
                    self.killed = True
            return Child()

        def cpu_check(args, checkpoint, updates, log):
            log.write_text("verified fixture")
            metadata = dict(fixture.metadata)
            if checkpoint.resolve() != fixture.resume:
                for key in ("source_step", "source_adam_step", "source_stream_i"):
                    metadata[key] += updates
            return metadata

        with patch.object(helpers, "cpu_check", side_effect=cpu_check) as checks, \
                patch.object(continuation, "wait_for_gpu_idle", return_value={}), \
                patch.object(continuation.subprocess, "Popen", side_effect=child), patch("builtins.print"):
            if candidate_fails and records_progress:
                with self.assertRaisesRegex(RuntimeError, "manual recovery"):
                    continuation.supervise(self.args)
                result = json.loads((self.args.out / "status.json").read_text())
            else:
                result = continuation.supervise(self.args)
        return result, commands, checks.call_args_list

    def test_continuation_uses_original_source_full_remaining_rows_and_verifies_actual_delta(self):
        before = self.fixture.resume.read_bytes()
        result, commands, checks = self.run_mocked()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(commands), 1)
        command = commands[0]
        for flag, expected in (("--resume", str(self.fixture.resume)), ("--run-steps", "90"),
                               ("--save-every-seconds", "300"), ("--keep-last", "3"),
                               ("--eval-every", "250"), ("--eval-batches", "32")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--packed-attention", command)
        self.assertIn("--save-optim", command)
        self.assertEqual(checks[-1].args[2], 3)
        self.assertEqual(self.fixture.resume.read_bytes(), before)
        self.assertTrue((self.args.out / "continuation.receipt.json").is_file())

    def test_setup_failure_falls_back_once_but_recorded_update_stops(self):
        result, commands, _ = self.run_mocked(candidate_fails=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(commands), 2)
        self.assertIn("--packed-attention", commands[0])
        self.assertNotIn("--packed-attention", commands[1])
        self.assertEqual(result["continuation"]["variant"], "native")
        self.args.out = self.root / "progress_failure"
        result, commands, _ = self.run_mocked(candidate_fails=True, records_progress=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(commands), 1)
        self.assertTrue((self.args.out / "continuation/train/metrics.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
