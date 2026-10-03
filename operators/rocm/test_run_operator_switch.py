"""Standard-library tests for bounded, isolated operator switching."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from operators.rocm import run_operator_switch as switch


class OperatorSwitchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        for relative in (*("operators/rocm/" + name for name in switch.SCRIPTS),
                         "operators/rocm/model_bench.py", "operators/rocm/shared_storage_moe.py",
                         "operators/rocm/packed_attention.py", "operators/rocm/grad_norm.py",
                         *("cat_yoko/" + name for name in ("moe.py", "trainer.py", "checkpoint.py", "optim.py", "data.py", "attention.py", "loss.py"))):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("isolated source fixture\n")
        self.resume = self.root / "checkpoint/trainable.pt"
        self.resume.parent.mkdir()
        self.resume.write_bytes(b"immutable source")
        self.base = self.root / "base"
        self.base.mkdir()
        self.data = self.root / "data/train.bin"
        self.data.parent.mkdir()
        self.data.write_bytes(b"\0" * (4096 * 4 * 100))
        self.eval_data = self.data.with_name("eval.bin")
        self.eval_data.write_bytes(b"\0" * 16384)
        self.metadata = {"source_checkpoint": str(self.resume), "source_step": 42100,
                         "source_adam_step": 7298, "source_stream_i": 10,
                         "source_data_rows": 100, "optimizer_states": 132,
                         "trainable_names": [f"p{index}" for index in range(132)]}

    def args(self, *extra):
        return switch.build_parser().parse_args([
            "--source-dir", str(self.source), "--resume", str(self.resume), "--base", str(self.base),
            "--data", str(self.data), "--eval-data", str(self.eval_data), "--out", str(self.root / "comparison"),
            "--deadline-utc", (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(), *extra])

    def write_run(self, directory, variant="native", speed=100, steps=20):
        directory.mkdir(parents=True, exist_ok=True)
        gradients = {f"p{index}": {"pass": True, "finite": True, "relative_l2": 0.0} for index in range(132)}
        comparison = {"seq_len": 4096, "pass": True, "gradients": gradients, "failed_gradients": [],
                      "loss_abs": 0.0, "global_gradient_relative_l2": 0.0,
                      "selected_logits": {"finite": True, "relative_l2": 0.0},
                      "final_hidden": {"finite": True, "relative_l2": 0.0}}
        hashes = {str(path.relative_to(self.source)): switch.digest_file(path) for path in (
            self.source / "operators/rocm/model_bench.py", self.source / "operators/rocm/shared_storage_moe.py",
            *(self.source / "cat_yoko" / name for name in ("moe.py", "trainer.py", "checkpoint.py", "optim.py", "data.py", "attention.py", "loss.py")))}
        report = {"status": "training_complete", "source_checkpoint": str(self.resume), "source_step": 42100,
                  "data": str(self.data), "eval_data": str(self.eval_data), "eos_id": 1, "eval_eos_id": 1,
                  "dtype": "bf16", "moe_layout": "shared-storage", "parity_sequences": [4096],
                  "source_stream_kind": "packed", "eval_batches": 2,
                  "source_optimizer_present": True, "save_optimizer": True, "deterministic_parity": True,
                  "reference_repeat": True, "updates": steps, "initial_eval": {"pass": True, "eval_nll": 7.5},
                  "final_eval": {"pass": True, "eval_nll": 7.4}, "source_code_version": {"file_sha256": hashes},
                  "thresholds": {"loss_atol": 0.02, "gradient_relative_l2_per_tensor_and_global": 0.05,
                                 "selected_output_relative_l2": 0.05},
                  "events": [{**comparison, "event": "reference_repeat"}, {**comparison, "event": "parity"},
                             {"event": "training_complete", "step": 42100 + steps, "updates": steps,
                              "optimizer_restored": True, "source_optimizer_present": True, "save_optimizer": True}]}
        operators = {"status": "completed", "patches_installed_after_native_reference": True,
                     "packed_attention": "--packed-attention" in switch.FLAGS[variant],
                     "batched_grad_norm": "--batched-grad-norm" in switch.FLAGS[variant],
                     "packed_attention_parity_calls": {"optimized_calls": 1},
                     "packed_attention_calls": {"optimized_calls": 21}}
        (directory / "parity.json").write_text(json.dumps(report))
        (directory / "operators.json").write_text(json.dumps(operators))
        train = directory / "train"
        train.mkdir(exist_ok=True)
        (train / "trainable.pt").write_bytes(b"experimental checkpoint")
        (train / "metrics.jsonl").write_text("\n".join(json.dumps({"step": 42101 + index,
            "tok_s": 1 if index < 5 else speed, "mem_mib": 100}) for index in range(steps)))
        return report, operators

    def test_selection_requires_five_percent_against_faster_baseline(self):
        baselines = [{"validated": True, "median_tokens_per_second": speed} for speed in (100, 120)]
        rows = [{"variant": "attention", "validated": True, "median_tokens_per_second": 125.99},
                {"variant": "norm", "validated": True, "median_tokens_per_second": 126},
                {"variant": "combined", "validated": False, "median_tokens_per_second": 1000}]
        result = switch.choose_variant(baselines, rows)
        self.assertEqual(result["selected"], "norm")
        self.assertEqual(result["conservative_baseline_median"], 120)
        self.assertEqual(switch.choose_variant(baselines[:1], rows)["selected"], "native")
        rows[1]["median_tokens_per_second"] = float("nan")
        self.assertEqual(switch.choose_variant(baselines, rows)["selected"], "native")

    def test_report_rejects_missing_adam_unexercised_operator_threshold_or_wrong_source(self):
        directory = self.root / "trial"
        report, operators = self.write_run(directory, "attention", 130)
        valid = lambda: switch.validate_run(directory, source=self.resume, source_step=42100, steps=20,
                                            variant="attention", source_dir=self.source)
        self.assertEqual(valid()["median_tokens_per_second"], 130)
        for change in (lambda r, o: r["events"][-1].update(optimizer_restored=False),
                       lambda r, o: o["packed_attention_parity_calls"].update(optimized_calls=0),
                       lambda r, o: r["thresholds"].update(selected_output_relative_l2=0.051),
                       lambda r, o: r.update(source_step=42099),
                       lambda r, o: r["events"][1]["gradients"].pop("p131"),
                       lambda r, o: r["final_eval"].update(eval_nll=float("nan"))):
            report, operators = self.write_run(directory, "attention", 130)
            change(report, operators)
            (directory / "parity.json").write_text(json.dumps(report))
            (directory / "operators.json").write_text(json.dumps(operators))
            with self.assertRaises(ValueError):
                valid()
        self.write_run(directory, "attention", 130)
        (self.source / "cat_yoko/attention.py").write_text("changed\n")
        with self.assertRaisesRegex(switch.SourceIntegrityError, "hashes"):
            valid()

    def test_deadline_and_continuation_preserve_fixed_source_and_single_pass(self):
        deadline = switch.parse_deadline("2026-10-03T14:40:00Z")
        self.assertEqual(switch.remaining_hours(deadline, datetime(2026, 10, 3, 14, 10, tzinfo=timezone.utc)), 0.5)
        self.assertEqual(switch.remaining_hours(deadline, deadline + timedelta(seconds=1)), 0)
        with self.assertRaises(ValueError):
            switch.parse_deadline("2026-10-03T14:40:00")
        args = self.args()
        flags = switch.continuation_args(args, self.root / "continued", 90, 0.5, "combined")
        for name, value in (("--resume", str(self.resume)), ("--run-steps", "90"),
                            ("--save-every", "0"), ("--save-every-seconds", "300"),
                            ("--keep-last", "3"), ("--eval-every", "250"), ("--eval-batches", "32")):
            self.assertEqual(flags[flags.index(name) + 1], value)
        self.assertIn("--save-optim", flags)
        self.assertIn("--packed-attention", flags)
        self.assertIn("--batched-grad-norm", flags)
        with self.assertRaises(ValueError):
            switch.continuation_args(args, self.root / "continued", 90, 0, "native")

    def test_preflight_reuse_checks_exact_source_and_real_packed_row(self):
        directory = self.root / "preflight"
        directory.mkdir()
        (directory / "status.json").write_text(json.dumps({"status": "complete", "results": [
            {"stage": "packed", "exit_code": 0}, {"stage": "norm", "exit_code": 0}]}))
        sha = switch.digest_file(self.resume)
        norm = {"status": "complete", "source_sha256": sha, "source_step": 42100,
                "source_checkpoint": str(self.resume), "cases": {key: {"parity": {"pass": True}}
                for key in ("clipped", "unclipped")}}
        (directory / "norm.json").write_text(json.dumps(norm))
        rows = [{"event": "candidate", "seq": 4096, "window": 8192, "dtype": "bf16", "status": "pass",
                 "layout": layout, "candidate": candidate, "correctness_path_counts": {"optimized_calls": 1},
                 "documents": {"path": str(self.data), "row": 10, "eos_id": 1, "source_pack_width": 4096}}
                for layout in ("packed_qkv", "cross_cache") for candidate in ("native", "packed_attention")]
        rows.append({"event": "summary", "passed": 4, "failed": 0, "errors": 0})
        packed = directory / "packed.jsonl"
        packed.write_text("\n".join(json.dumps(row) for row in rows))
        self.assertEqual(switch.reusable_preflight(directory, self.args(), self.metadata, sha), (True, True))
        rows[0]["documents"]["row"] = 11
        packed.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "corpus row"):
            switch.reusable_preflight(directory, self.args(), self.metadata, sha)
        norm["source_sha256"] = "wrong"
        (directory / "norm.json").write_text(json.dumps(norm))
        with self.assertRaisesRegex(ValueError, "different source"):
            switch.reusable_preflight(directory, self.args(), self.metadata, sha)

    def fake_child(self, command, **kwargs):
        self.commands.append(command)
        script = Path(command[2]).name
        args = command[3:]
        if script == "packed_attention_bench.py":
            Path(args[args.index("--output") + 1]).write_text(json.dumps({"event": "summary", "passed": 4, "failed": 0, "errors": 0}))
        elif script == "grad_norm_bench.py":
            Path(args[args.index("--json") + 1]).write_text(json.dumps({"status": "complete", "cases": {
                "clipped": {"parity": {"pass": True}}, "unclipped": {"parity": {"pass": True}}}}))
        else:
            directory = Path(args[args.index("--out") + 1])
            variant = "combined" if "--packed-attention" in args and "--batched-grad-norm" in args else (
                "attention" if "--packed-attention" in args else "norm" if "--batched-grad-norm" in args else "native")
            speed = {"attention": 125, "norm": 130, "combined": 150, "native": 100 if directory.name == "baseline_before" else 120}[variant]
            self.write_run(directory, variant, speed)
        class Child:
            pid = 987654
            def wait(self, timeout):
                return 0
        return Child()

    def test_evaluate_only_runs_ordered_trials_same_source_and_never_continues(self):
        self.commands = []
        args = self.args("--evaluate-only")
        original = self.resume.read_bytes()
        with patch.object(switch, "cpu_check", return_value=self.metadata), \
                patch.object(switch, "wait_for_gpu_idle", return_value={}), \
                patch.object(switch.subprocess, "Popen", side_effect=self.fake_child), patch("builtins.print"):
            result = switch.supervise(args)
        self.assertEqual(result["status"], "evaluated_only")
        self.assertEqual(result["decision"]["selected"], "combined")
        self.assertEqual([record["name"] for record in result["stages"]], ["packed_operator", "norm_operator", "baseline_before",
            "attention_steps", "norm_steps", "combined_steps", "baseline_after"])
        for command in self.commands[2:]:
            self.assertEqual(command[command.index("--resume") + 1], str(self.resume))
        self.assertEqual(self.resume.read_bytes(), original)
        self.assertFalse((args.out / "continuation").exists())
        self.assertTrue((args.out / "decision.json").is_file())

    def test_invalid_output_and_insufficient_warmup_fail_before_creating_files(self):
        args = self.args("--out", str(self.resume.parent / "bad"))
        with self.assertRaisesRegex(ValueError, "separate"):
            switch.validate_args(args)
        self.assertFalse(args.out.exists())
        with self.assertRaisesRegex(ValueError, "warmup"):
            switch.validate_args(self.args("--benchmark-steps", "5"))


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "CPU serialization tests require Torch")
class EmbeddedCheckpointCheckTests(unittest.TestCase):
    """Run the actual embedded verifier against real, small Torch archives."""

    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source_path, self.target_path = self.root / "source.pt", self.root / "target.pt"
        self.data = self.root / "train.bin"
        self.data.write_bytes(b"\0" * (100 * 4096 * 4))
        torch = self.torch
        weights = {}
        for index in range(66):
            # Interleave decay and no-decay names as a real model state does.
            weights[f"layer{index}.weight"] = torch.full((2, 3), index / 128, dtype=torch.bfloat16, device="cpu")
            weights[f"layer{index}.norm.weight"] = torch.ones(3, dtype=torch.bfloat16, device="cpu")
        ordered_parameters = [value for name, value in weights.items() if ".norm." not in name]
        ordered_parameters += [value for name, value in weights.items() if ".norm." in name]
        self.source = {
            "kind": "trainable", "trainable": weights,
            "extra": {"phase": "B0", "name": "CAT-YOKO-12B", "step": 100,
                      "tokens_in_phase": 40000.0, "tokens_seen": 200000.0,
                      "seq_len": 4096, "seed": 123, "use_kda": False, "sparse": False,
                      "stream": {"kind": "packed", "stride": 1, "nseq": 100, "i": 10},
                      "cfg": {"vocab_size": 130560, "hidden_size": 2048,
                              "encoder_layers": 16, "decoder_layers": 26,
                              "n_routed_enc": 20, "n_routed_dec": 20, "top_k_dec": 10,
                              "use_nvfp4": False, "use_fp8": False}},
            "optimizer": {
                "param_groups": [{"params": list(range(66)), "lr": 0.001, "betas": (0.9, 0.95),
                                  "eps": 1e-8, "weight_decay": 0.1},
                                 {"params": list(range(66, 132)), "lr": 0.001, "betas": (0.9, 0.95),
                                  "eps": 1e-8, "weight_decay": 0.0}],
                "state": {parameter: {"step": 12,
                                      "exp_avg": torch.full_like(value, 0.125, dtype=torch.float32),
                                      "exp_avg_sq": torch.full_like(value, 0.25, dtype=torch.float32)}
                          for parameter, value in enumerate(ordered_parameters)},
            },
        }
        self.target = copy.deepcopy(self.source)
        self.target["extra"]["step"] += 3
        self.target["extra"]["tokens_in_phase"] += 3 * 4096
        self.target["extra"]["tokens_seen"] += 3 * 4096
        self.target["extra"]["stream"]["i"] += 3
        for value in self.target["trainable"].values():
            value.add_(0.03125)
        for state in self.target["optimizer"]["state"].values():
            state["step"] += 3
            state["exp_avg"].add_(0.0625)
            state["exp_avg_sq"].add_(0.125)
        # A schedule updates LR; all other group options and IDs stay fixed.
        for group in self.target["optimizer"]["param_groups"]:
            group["lr"] = 0.0009
        torch.save(self.source, self.source_path)

    def run_verifier(self, target=None, *, same_source=False, window=0):
        if not same_source:
            self.torch.save(self.target if target is None else target, self.target_path)
        return subprocess.run([
            sys.executable, "-c", switch.CHECKPOINT_CHECK, str(self.source_path),
            str(self.source_path if same_source else self.target_path), str(self.data),
            "0" if same_source else "3", str(window)],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONOPTIMIZE="1"))

    def test_actual_save_load_preserves_source_and_verifies_full_resume_delta(self):
        before = self.source_path.read_bytes()
        source_result = self.run_verifier(same_source=True, window=3)
        self.assertEqual(source_result.returncode, 0, source_result.stderr)
        result = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(metadata["checkpoint_verified"])
        self.assertEqual(metadata["source_step"], 103)
        self.assertEqual(metadata["source_adam_step"], 15)
        self.assertEqual(metadata["source_stream_i"], 13)
        self.assertEqual((metadata["optimizer_states"], metadata["optimizer_moments"]), (132, 264))
        self.assertEqual(metadata["trainable_names"], list(self.source["trainable"]))
        self.assertEqual(metadata["trainable_shapes"]["layer0.weight"], [2, 3])
        self.assertEqual(self.source_path.read_bytes(), before)
        target = self.torch.load(self.target_path, map_location="cpu", weights_only=False)
        self.assertEqual(target["optimizer"]["state"][0]["exp_avg"].dtype, self.torch.float32)
        self.assertEqual(target["optimizer"]["state"][0]["exp_avg"].device.type, "cpu")

    def test_serialized_corruption_is_rejected_with_python_optimization_enabled(self):
        torch = self.torch

        def negative_squared_moment(target):
            target["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1

        def permute_parameter_mapping(target):
            # All moments still have valid shapes and ID coverage; detect the
            # semantic remapping rather than relying on missing state alone.
            parameters = target["optimizer"]["param_groups"][0]["params"]
            parameters[0], parameters[1] = parameters[1], parameters[0]

        faults = [
            ("moment shape", lambda target: target["optimizer"]["state"][0].update(
                exp_avg=torch.zeros(1, dtype=torch.float32)), "Adam moment shape mismatch"),
            ("negative variance", negative_squared_moment, "negative squared Adam moment"),
            ("lossy moment dtype", lambda target: target["optimizer"]["state"][0].update(
                exp_avg=target["optimizer"]["state"][0]["exp_avg"].to(torch.bfloat16)), "CPU FP32"),
            ("nonfinite moment", lambda target: target["optimizer"]["state"][0]["exp_avg"].fill_(float("nan")), "non-finite Adam moment"),
            ("cursor drift", lambda target: target["extra"]["stream"].update(i=14), "packed cursor delta mismatch"),
            ("phase token drift", lambda target: target["extra"].update(tokens_in_phase=target["extra"]["tokens_in_phase"] + 1), "token counter delta mismatch"),
            ("global token drift", lambda target: target["extra"].update(tokens_seen=target["extra"]["tokens_seen"] + 1), "token counter delta mismatch"),
            ("parameter remapping", permute_parameter_mapping, "Adam parameter groups/IDs changed"),
            ("Adam step drift", lambda target: [state.update(step=16) for state in target["optimizer"]["state"].values()], "Adam counter delta mismatch"),
            ("nonfinite weight", lambda target: target["trainable"]["layer0.weight"].fill_(float("inf")), "non-finite weight"),
        ]
        for name, inject, message in faults:
            with self.subTest(fault=name):
                target = copy.deepcopy(self.target)
                inject(target)
                result = self.run_verifier(target)
                self.assertNotEqual(result.returncode, 0, "corruption was accepted")
                self.assertIn(message, result.stderr)

    def test_source_preflight_rejects_a_window_that_would_wrap(self):
        result = self.run_verifier(same_source=True, window=91)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrap", result.stderr)


if __name__ == "__main__":
    unittest.main()
