"""CPU-only lifecycle, numerical admission and throughput selection tests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from operators.rocm import run_next_operator_round as controller
from operators.rocm import test_run_operator_switch as fixtures


class NextOperatorRoundTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.OperatorSwitchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.temp.cleanup)
        self.root, self.metadata = self.fixture.root, self.fixture.metadata
        self.args = controller.build_parser().parse_args([
            "--source-dir", str(self.fixture.source), "--resume", str(self.fixture.resume),
            "--base", str(self.fixture.base), "--data", str(self.fixture.data),
            "--eval-data", str(self.fixture.eval_data), "--micro-dir", str(self.root / "micro"),
            "--out", str(self.root / "comparison"), "--deadline-utc", "2099-10-03T14:40:00Z"])
        for name in ("next_candidate_bench.py", "gpu_adam.py", "gpu_adam_bench.py", "bucketed_attention.py",
                     "bucketed_attention_bench.py", "update_timing.py"):
            (self.fixture.source / "operators/rocm" / name).write_text("independent operator fixture\n")
        self.sha = controller.helpers.digest_file(self.fixture.resume)
        self.args.micro_dir.mkdir()
        self.make_micro()
        controller.prepare(self.args)

    def save(self, path, value):
        path.write_text(json.dumps(value) + "\n")

    def make_micro(self):
        numerical = {"pass": True, "parameters": 132, "all_weights_bitwise": True,
                     "gate": {"fp32_moment_atol": 1e-8, "fp32_moment_rtol": 1e-6}}
        adam = {"status": "complete", "device": "cuda", "source_sha256": self.sha,
                "source_step": self.metadata["source_step"], "source_checkpoint": str(self.fixture.resume),
                "checkpoint_moments_cpu_fp32": True, "persistent_fp32_master": False,
                "parity": [{**numerical, "invocation_step": i + 1} for i in range(5)],
                "terminal": numerical, "installation": {"state_exports": 1}}
        status = {"status": "complete", "source_sha256": self.sha, "source_step": self.metadata["source_step"],
                  "source_checkpoint": str(self.fixture.resume),
                  "file_sha256": {"operators/rocm/" + name: controller.helpers.digest_file(self.fixture.source / "operators/rocm" / name)
                                  for name in ("gpu_adam.py", "gpu_adam_bench.py", "bucketed_attention.py",
                                               "bucketed_attention_bench.py", "packed_attention.py")},
                  "results": [{"stage": stage, "exit_code": 0} for stage in ("gpu_adam", "bucketed")]}
        self.save(self.args.micro_dir / "gpu_adam.json", adam)
        self.save(self.args.micro_dir / "status.json", status)
        cases = [{"event": "candidate", "case": "real", "layout": layout, "candidate": candidate,
                  "status": "pass", "dtype": "bf16", "output": {"pass": True},
                  "gradients": {key: {"pass": True} for key in ("dq", "dk", "dv")},
                  "documents": {"path": str(self.fixture.data), "row": self.metadata["source_stream_i"],
                                "eos_id": 1, "source_pack_width": 4096}}
                 for layout in ("packed_qkv", "cross_cache") for candidate in ("packed", "bucketed")]
        rows = [{"event": "environment", "window": 8192, "shape": [1, 16, 4096, 128],
                 "kv_heads": 2, "merge_heads_included": True}, *cases,
                {"event": "summary", "passed": 4, "failed": 0, "errors": 0, "bucketed_exercised": 1}]
        (self.args.micro_dir / "bucketed.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return adam, status, rows

    def timing(self, speed=120, updates=20):
        samples = [{"invocation_update": i + 1, "tokens": 4096, "elapsed_s": 4096 / (1 if i < 5 else speed),
                    "completed_update_tokens_s": 1 if i < 5 else speed} for i in range(updates)]
        return {"status": "complete", "incomplete_update": False, "warmup_updates": 5,
                "started_updates": updates, "completed_updates": updates, "measured_updates": updates - 5,
                "discarded_updates": 5, "samples": samples,
                "steady_tokens": 4096 * (updates - 5), "sum_steady_elapsed_s": (updates - 5) * 4096 / speed,
                "median_completed_update_tokens_s": speed, "aggregate_completed_update_tokens_s": speed}

    def write_trial(self, directory, variant, speed, *, updates=20, continuation=False):
        report, _ = self.fixture.write_run(directory, "attention", speed, steps=updates)
        if continuation:
            report["eval_batches"] = 32
            self.save(directory / "parity.json", report)
        gpu, bucket = "--gpu-fp32-adam" in controller.FLAGS[variant], "--bucketed-attention" in controller.FLAGS[variant]
        files = ["next_candidate_bench.py", "candidate_bench.py"]
        files += [name for enabled, name in ((gpu, "gpu_adam.py"), (bucket, "bucketed_attention.py"),
                                           (not continuation, "update_timing.py")) if enabled]
        next_report = {"status": "completed", "patches_installed_after_native_reference": True,
                       "gpu_fp32_adam": gpu, "bucketed_attention": bucket, "sync_update_timing": not continuation,
                       "file_sha256": {"operators/rocm/" + name: controller.helpers.digest_file(
                           self.fixture.source / "operators/rocm" / name) for name in files},
                       "gpu_fp32_adam_calls": {"state_loads": 1, "state_exports": 1,
                                               "parameter_updates": 132 * updates, "optimized_calls": 132 * updates,
                                               "persistent_fp32_master": False},
                       "bucketed_attention_parity_calls": {"bucketed_calls": 1},
                       "bucketed_attention_calls": {"bucketed_calls": 21}}
        self.save(directory / "next_operators.json", next_report)
        self.save(directory / "update_timing.json", self.timing(speed, updates))
        metrics = directory / "train/metrics.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        for row in rows:
            row.update(actual_optimizer_device="cuda" if gpu else "cpu", bucketed_attention=bucket)
        metrics.write_text("\n".join(json.dumps(row) for row in rows))
        return next_report

    def test_selection_requires_all_three_measures_against_faster_baselines(self):
        row = lambda variant, a, b, c: dict(variant=variant, validated=True, **dict(zip(controller.SPEEDS, (a, b, c))))
        baselines = [row("baseline", 100, 120, 100), row("baseline", 120, 100, 110)]
        candidates = [row("gpu_adam", 150, 125.99, 140), row("bucketed", 140, 140, 115.49),
                      row("combined", 126, 126, 115.5)]
        self.assertEqual(controller.choose_variant(baselines, candidates)["selected"], "combined")
        candidates[-1]["validated"] = False
        self.assertEqual(controller.choose_variant(baselines, candidates)["selected"], "baseline")
        candidates[0][controller.SPEEDS[1]] = math.nan
        self.assertEqual(controller.choose_variant(baselines, candidates)["selected"], "baseline")
        with self.assertRaisesRegex(ValueError, "both"):
            controller.choose_variant(baselines[:1], candidates)

    def test_micro_admission_rejects_failed_byteexact_or_source_and_unexercised_bucket(self):
        gates = lambda: controller.micro_gates(self.args, self.metadata, self.sha)
        self.assertTrue(all(row["pass"] for row in gates().values()))
        for mutation in (lambda r: r["parity"][0].update(all_weights_bitwise=False),
                         lambda r: r.update(source_sha256="another checkpoint"),
                         lambda r: r["terminal"].update(parameters=131),
                         lambda r: r.update(checkpoint_moments_cpu_fp32=False),
                         lambda r: r["parity"][0]["gate"].update(fp32_moment_atol=1e-7)):
            adam, _, _ = self.make_micro()
            mutation(adam)
            self.save(self.args.micro_dir / "gpu_adam.json", adam)
            self.assertFalse(gates()["gpu_adam"]["pass"])
            self.assertTrue(gates()["bucketed"]["pass"])
        _, status, rows = self.make_micro()
        status["results"][0]["exit_code"] = 1
        self.save(self.args.micro_dir / "status.json", status)
        self.assertFalse(gates()["gpu_adam"]["pass"])
        rows[-1]["bucketed_exercised"] = 0
        (self.args.micro_dir / "bucketed.jsonl").write_text("\n".join(map(json.dumps, rows)))
        self.assertFalse(gates()["bucketed"]["pass"])
        _, status, _ = self.make_micro()
        status["file_sha256"]["operators/rocm/bucketed_attention.py"] = "unrelated-version"
        self.save(self.args.micro_dir / "status.json", status)
        self.assertFalse(gates()["bucketed"]["pass"])

    def test_timing_recomputes_raw_samples_and_rejects_incomplete_or_forged_summary(self):
        path = self.root / "timing.json"
        self.save(path, self.timing())
        self.assertEqual(controller.validate_timing(path, 20)[controller.SPEEDS[0]], 120)
        for mutate in (lambda r: r.update(completed_updates=19), lambda r: r.update(median_completed_update_tokens_s=999),
                       lambda r: r["samples"][6].update(tokens=4095), lambda r: r["samples"].pop()):
            report = self.timing()
            mutate(report)
            self.save(path, report)
            with self.assertRaises(ValueError):
                controller.validate_timing(path, 20)

    def test_trial_rejects_changed_parameter_gate_counters_or_source_hash(self):
        directory = self.root / "gpu_adam_steps"
        self.args.out.mkdir()
        check = lambda: controller.validate_trial(self.args, directory, self.metadata, "gpu_adam", steps=20)
        with patch.object(controller.helpers, "cpu_check", return_value=self.metadata) as cpu:
            self.write_trial(directory, "gpu_adam", 130)
            self.assertTrue(check()["validated"])
            self.assertEqual(cpu.call_args.args[2], 20)
            for mutate in (lambda r: r["gpu_fp32_adam_calls"].update(parameter_updates=2639),
                           lambda r: r["gpu_fp32_adam_calls"].update(state_exports=0),
                           lambda r: r.update(gpu_fp32_adam=False)):
                report = self.write_trial(directory, "gpu_adam", 130)
                mutate(report)
                self.save(directory / "next_operators.json", report)
                with self.assertRaises(ValueError):
                    check()
            self.write_trial(directory, "gpu_adam", 130)
            report = json.loads((directory / "parity.json").read_text())
            gradients = report["events"][1]["gradients"]
            gradients["unexpected_parameter"] = gradients.pop("p131")
            self.save(directory / "parity.json", report)
            with self.assertRaisesRegex(ValueError, "names"):
                check()
            self.write_trial(directory, "gpu_adam", 130)
            (self.fixture.source / "operators/rocm/gpu_adam.py").write_text("changed after snapshot\n")
            with self.assertRaises(controller.helpers.SourceIntegrityError):
                check()

    def test_output_isolation_and_live_command_use_original_source_without_sync(self):
        for out in (self.args.micro_dir / "child", self.fixture.resume.parent / "child", self.root):
            changed = copy.copy(self.args)
            changed.out = out
            with self.assertRaises((ValueError, FileExistsError)):
                controller.prepare(changed)
        flags = controller.trial_args(self.args, self.root / "live", 90, "gpu_adam", hours=0.5)
        self.assertNotIn("--sync-update-timing", flags)
        for key, value in (("--resume", str(self.fixture.resume)), ("--run-steps", "90"),
                           ("--save-every-seconds", "300"), ("--keep-last", "3"),
                           ("--eval-every", "250"), ("--eval-batches", "32")):
            self.assertEqual(flags[flags.index(key) + 1], value)
        self.assertIn("--packed-attention", flags)
        self.assertIn("--gpu-adam-gate", flags)

    def test_reuse_then_candidate_failure_still_retests_baseline_and_continues_fixed_source(self):
        self.args.profile_current = True
        self.args.baseline_run = self.root / "external_baseline"
        self.write_trial(self.args.baseline_run, "baseline", 100)
        commands = []
        source_bytes = self.fixture.resume.read_bytes()

        def child(command, **kwargs):
            commands.append(command)
            flags = command[3:]
            directory = Path(flags[flags.index("--out") + 1])
            variant = "combined" if "--gpu-fp32-adam" in flags and "--bucketed-attention" in flags else (
                "gpu_adam" if "--gpu-fp32-adam" in flags else "bucketed" if "--bucketed-attention" in flags else "baseline")
            code = 1 if variant == "gpu_adam" or directory.name == "profile_current" else 0
            if not code:
                self.write_trial(directory, variant, {"baseline": 100, "bucketed": 130, "combined": 120}[variant],
                                 updates=3 if directory.name == "continuation" else 20,
                                 continuation=directory.name == "continuation")
            class Child:
                pid = 987654
                def wait(self, timeout):
                    return code
                def kill(self):
                    raise AssertionError("successful mock children must not be signaled")
            return Child()

        with patch.object(controller.helpers, "cpu_check", return_value=self.metadata) as checks, \
                patch.object(controller, "wait_for_gpu_idle", return_value={}), \
                patch.object(controller.subprocess, "Popen", side_effect=child), patch("builtins.print"):
            result = controller.supervise(self.args)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["decision"]["selected"], "bucketed")
        self.assertEqual(result["decision"]["baselines"][0]["reused_from"], str(self.args.baseline_run))
        self.assertEqual([Path(command[command.index("--out") + 1]).name for command in commands],
                         ["gpu_adam_steps", "bucketed_steps", "combined_steps", "baseline_after", "profile_current", "continuation"])
        self.assertFalse(result["profile_current"]["validated"])
        profile_command = commands[-2]
        self.assertEqual(Path(profile_command[2]).name, "profile_training.py")
        self.assertEqual(profile_command[profile_command.index("--run-steps") + 1], "6")
        self.assertNotIn("--sync-update-timing", profile_command)
        self.assertTrue(all(command[command.index("--resume") + 1] == str(self.fixture.resume) for command in commands))
        self.assertEqual(checks.call_args.args[2], 3)
        self.assertEqual(commands[-1][commands[-1].index("--run-steps") + 1], "90")
        self.assertNotIn("--sync-update-timing", commands[-1])
        self.assertEqual(self.fixture.resume.read_bytes(), source_bytes)
        self.assertTrue((self.args.out / "gpu_adam_steps.validation.json").is_file())

    def test_profile_requires_cross_thread_backward_attribution_and_verified_six_update_checkpoint(self):
        directory = self.root / "profile_current"
        self.args.out.mkdir()
        self.fixture.write_run(directory, "attention", 100, steps=6)
        (directory / "profile").mkdir()
        summary = {"status": "complete", "recorded_updates": 4, "requested_active_updates": 4,
                   "completed_updates": 6, "warmup_updates": 2, "phases": {"backward": {"device_work_us": 1000}},
                   "top_steady_operators_by_device_work": [{"name": "aten::bmm"}],
                   "top_steady_operators_by_self_cpu": [{"name": "aten::copy_"}]}
        self.save(directory / "profile/summary.json", summary)
        with patch.object(controller.helpers, "cpu_check", return_value=self.metadata) as check:
            self.assertFalse(controller.validate_profile(self.args, directory, self.metadata)["participates_in_selection"])
            self.assertEqual(check.call_args.args[2], 6)
            summary["phases"]["backward"]["device_work_us"] = 0
            self.save(directory / "profile/summary.json", summary)
            with self.assertRaisesRegex(ValueError, "attribution"):
                controller.validate_profile(self.args, directory, self.metadata)

    def test_source_mutation_stops_before_another_child_and_baseline_failure_aborts(self):
        for mutate in (True, False):
            self.args.out = self.root / ("source_mutation" if mutate else "baseline_failed")
            commands = []
            original = self.fixture.resume.read_bytes()

            def child(command, **kwargs):
                commands.append(command)
                if mutate:
                    self.fixture.resume.write_bytes(b"source checkpoint replaced during owned trial")
                class Child:
                    pid = 987654
                    def wait(self, timeout):
                        return 0 if mutate else 1
                return Child()

            with patch.object(controller.helpers, "cpu_check", return_value=self.metadata), \
                    patch.object(controller, "wait_for_gpu_idle", return_value={}), \
                    patch.object(controller.subprocess, "Popen", side_effect=child), patch("builtins.print"):
                with self.assertRaises(controller.helpers.SourceIntegrityError if mutate else RuntimeError):
                    controller.supervise(self.args)
            self.assertEqual(len(commands), 1)
            self.assertEqual(json.loads((self.args.out / "status.json").read_text())["status"], "failed")
            self.assertTrue((self.args.out / "baseline_before.receipt.json").is_file())
            self.fixture.resume.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
