"""Exercise the real runner composition without importing a GPU runtime.

Only heavyweight model/context operations are mocked. The inherited parser,
argument validation, packed candidate wrapper and new entry are loaded from
their real sources, so these integration tests also run without local torch.
Numerical/full-model parity remains the existing model_bench GPU check.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager, redirect_stderr
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import operators.rocm as rocm


ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def runner_harness():
    """Retain real control flow, substituting contexts and expensive model work."""
    model = types.ModuleType("operators.rocm.model_bench")

    class TorchPolicy:
        deterministic = False
        warn_only = True

        def are_deterministic_algorithms_enabled(self):
            return self.deterministic

        def is_deterministic_algorithms_warn_only_enabled(self):
            return self.warn_only

        def use_deterministic_algorithms(self, enabled, *, warn_only=False):
            self.deterministic, self.warn_only = enabled, warn_only

    model.__dict__.update(argparse=argparse, Path=Path, math=math,
                          contextmanager=contextmanager, torch=TorchPolicy())
    source = ast.parse((ROOT / "operators/rocm/model_bench.py").read_text())
    selected = [node for node in source.body if isinstance(node, ast.FunctionDef)
                and node.name in {"build_parser", "validate_args", "parity_determinism"}]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / "operators/rocm/model_bench.py"),
                 "exec"), model.__dict__)
    state = types.SimpleNamespace(active=[], events=[], metrics=[], creations=[],
        packed={"optimized_calls": 0}, split={"split_optimized_calls": 0},
        cached={"state_loads": 0, "parameter_updates": 0, "gpu_parameter_updates": 0,
                "cache_reuses": 0, "fallback_parameter_updates": 0, "cached_parameters": 0},
        timing={"status": "incomplete"}, training_event=None, refuse_preflight=False,
        fail_after_install=False, run_args=None, parity_states=[], after_parity=None,
        trainer_determinism=[])

    class Installation:
        def __init__(self, counters):
            self.counters = counters

        def report(self):
            return dict(self.counters)

    def context(name, counters):
        @contextmanager
        def installed(*args, **kwargs):
            state.events.append(("enter", name, list(state.active), kwargs))
            state.active.append(name)
            try:
                yield Installation(counters)
            finally:
                state.events.append(("exit", name))
                assert state.active.pop() == name
                if name == "cached":
                    counters["cached_parameters"] = 0
        return installed

    packed = types.ModuleType("operators.rocm.packed_attention")
    packed.packed_attention_context = context("packed", state.packed)
    split = types.ModuleType("operators.rocm.split_attention")
    split.split_attention_context = context("split", state.split)
    cached = types.ModuleType("operators.rocm.cpu_adam_cached")
    cached.use_cached_cpu_adam = context("cached", state.cached)
    timing = types.ModuleType("operators.rocm.update_timing")
    timing.synchronized_update_timing = context("timing", state.timing)
    trainer = types.ModuleType("cat_yoko.trainer")

    class Trainer:
        def __init__(self):
            state.creations.append(list(state.active))
            state.trainer_determinism.append(model.torch.are_deterministic_algorithms_enabled())

        def _log(self, row):
            state.metrics.append(row)

    trainer.Trainer = Trainer

    def install(model_object, layout):
        state.events.append(("layout", layout, list(state.active)))
        return {"layers": 42}

    def emit(path, report, event):
        state.events.append(dict(event))
        report.setdefault("events", []).append(dict(event))

    model.install_moe_layout = install
    model.source_code_version = lambda layout: {"layout": layout, "native_source": True}
    model.emit = emit
    model.open_parity_stream = lambda *args, **kwargs: (None, {})

    def native_main(argv):
        args = model.build_parser().parse_args(argv)
        state.run_args = args
        model.validate_args(model.build_parser(), args)
        with model.parity_determinism(args.deterministic_parity):
            state.parity_states.append(model.torch.are_deterministic_algorithms_enabled())
            state.events.append(("reference", list(state.active)))
            if args.reference_repeat:
                state.events.append(("reference_repeat", list(state.active)))
            if state.refuse_preflight:
                raise RuntimeError("source/output isolation")
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / "parity.json"
            path.write_text("{}")
            full_report = {}
            model.install_moe_layout(object(), args.moe_layout)
            if state.fail_after_install:
                raise RuntimeError("injected failure")
            state.packed["optimized_calls"] += 1
            if "split" in state.active:
                state.split["split_optimized_calls"] += 1
            model.emit(path, full_report, {"event": "parity_complete", "pass": True,
                                         "gradient_tensors": 132})
        state.after_parity = model.torch.are_deterministic_algorithms_enabled()
        if not args.parity_only:
            owner = Trainer()
            owner._log({"event": "train", "step": 17815})
            if "cached" in state.active:
                state.cached.update(state_loads=1, parameter_updates=132 * args.run_steps,
                    gpu_parameter_updates=132 * args.run_steps, cache_reuses=132 * (args.run_steps - 1),
                    cached_parameters=132)
            event = {"event": "training_complete", "updates": args.run_steps,
                     "source_optimizer_present": True, "optimizer_restored": True}
            if state.training_event:
                event.update(state.training_event)
            model.emit(path, full_report, event)
        return 0

    model.main = native_main
    modules = {"operators.rocm.model_bench": model, "operators.rocm.packed_attention": packed,
               "operators.rocm.split_attention": split, "operators.rocm.cpu_adam_cached": cached,
               "operators.rocm.update_timing": timing, "cat_yoko.trainer": trainer}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(rocm, "model_bench", model, create=True))
        parent = load_file("_round3_test_candidate", ROOT / "operators/rocm/candidate_bench.py")
        stack.enter_context(patch.object(rocm, "candidate_bench", parent, create=True))
        stack.enter_context(patch.dict(sys.modules, {"operators.rocm.candidate_bench": parent}))
        entry = load_file("_round3_test_entry", ROOT / "operators/rocm/round3_candidate_bench.py")
        yield entry, parent, model, Trainer, state


class Round3CandidateTests(unittest.TestCase):
    def argv(self, root):
        train, heldout = root / "train.bin", root / "eval.bin"
        train.write_bytes(b"train")
        heldout.write_bytes(b"heldout")
        return ["--base", str(root / "base"), "--resume", str(root / "source.pt"),
                "--out", str(root / "out"), "--data", str(train), "--eval-data", str(heldout),
                "--moe-layout", "shared-storage", "--packed-attention", "--deterministic-parity",
                "--reference-repeat", "--save-optim", "--save-every-seconds", "300", "--keep-last", "3",
                "--run-steps", "2"]

    def test_forwarding_preserves_data_checkpoint_retention_and_native_gates(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, parent, model, _, _):
            argv = self.argv(Path(name))
            extra = ["--split-attention", "--cpu-adam-cached", "--sync-update-timing", "--deterministic-training"]
            self.assertEqual(entry.legacy_argv(argv + extra), argv)
            new = entry.build_parser().parse_args(argv + extra)
            native = model.build_parser().parse_args(parent.native_argv(entry.legacy_argv(argv + extra)))
            for field in ("resume", "data", "eval_data", "save_optim", "deterministic_parity",
                          "reference_repeat", "save_every_seconds", "keep_last", "parity_seqs",
                          "loss_atol", "grad_relative_l2", "output_relative_l2"):
                self.assertEqual(getattr(new, field), getattr(native, field), field)
            self.assertEqual(new.save_every_seconds, 300)
            self.assertEqual(new.keep_last, 3)

    def test_all_four_variants_keep_native_capture_before_install_and_trainer_after(self):
        for split_on, cached_on in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(split=split_on, cached=cached_on), tempfile.TemporaryDirectory() as name, \
                    runner_harness() as (entry, parent, model, Trainer, state):
                argv = self.argv(Path(name)) + ["--sync-update-timing"]
                argv += ["--split-attention"] if split_on else []
                argv += ["--cached-cpu-adam"] if cached_on else []
                originals = (model.install_moe_layout, model.emit, model.source_code_version,
                             model.open_parity_stream, Trainer._log, parent.candidate_context)
                self.assertEqual(entry.main(argv), 0)
                self.assertEqual(originals, (model.install_moe_layout, model.emit, model.source_code_version,
                                            model.open_parity_stream, Trainer._log, parent.candidate_context))
                self.assertEqual(state.active, [])
                self.assertIn(("reference", []), state.events)
                self.assertIn(("reference_repeat", []), state.events)
                expected = ["packed"] + (["split"] if split_on else []) + (["cached"] if cached_on else []) + ["timing"]
                self.assertEqual(state.creations, [expected])
                report = json.loads((Path(name) / "out/round3_operators.json").read_text())
                self.assertEqual(report["status"], "completed")
                self.assertTrue(report["patches_installed_after_native_reference"])
                self.assertTrue(report["full_model_parity_passed"])
                self.assertEqual(report["cached_cpu_adam_training_exercised"], cached_on)
                self.assertEqual(state.metrics[0]["actual_optimizer_device"], "cpu")
                self.assertEqual(state.metrics[0]["split_attention"], split_on)
                self.assertEqual(state.metrics[0]["cached_cpu_adam"], cached_on)
                if cached_on:
                    counts = report["cached_cpu_adam_calls"]
                    self.assertEqual((counts["state_loads"], counts["cache_reuses"], counts["parameter_updates"]),
                                     (1, 132, 264))
                    self.assertEqual(counts["cached_parameters"], 132)
                    self.assertEqual(state.cached["cached_parameters"], 0)
                if split_on:
                    self.assertEqual(report["split_attention_parity_calls"]["split_optimized_calls"], 1)
                self.assertEqual(next(event for event in state.events if isinstance(event, dict)
                                      and event.get("event") == "parity_complete")["gradient_tensors"], 132)
                timing_enter = next(event for event in state.events if isinstance(event, tuple)
                                    and event[:2] == ("enter", "timing"))
                self.assertEqual(timing_enter[3]["warmup"], 5)

    def test_new_retention_defaults_reach_actual_runner_and_bench_can_override(self):
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as name, \
                    runner_harness() as (entry, _, _, _, state):
                argv = self.argv(Path(name))
                position = argv.index("--save-every-seconds")
                del argv[position:position + 4]
                if override:
                    argv += ["--save-every-seconds=0", "--keep-last=1"]
                self.assertEqual(entry.main(argv), 0)
                self.assertEqual(state.run_args.save_every_seconds, 0 if override else 300)
                self.assertEqual(state.run_args.keep_last, 1 if override else 3)

    def test_opt_in_determinism_covers_training_after_inner_parity_and_restores_caller(self):
        for original in (False, True):
            for requested in (False, True):
                with self.subTest(original=original, requested=requested), tempfile.TemporaryDirectory() as name, \
                        runner_harness() as (entry, _, model, _, state):
                    model.torch.use_deterministic_algorithms(original, warn_only=True)
                    argv = self.argv(Path(name))
                    if requested:
                        argv += ["--deterministic-training"]
                    self.assertEqual(entry.main(argv), 0)
                    expected = requested or original
                    self.assertEqual(state.parity_states, [True])
                    self.assertEqual(state.after_parity, expected)
                    self.assertEqual(state.trainer_determinism, [expected])
                    self.assertTrue(model.torch.is_deterministic_algorithms_warn_only_enabled())
                    self.assertEqual(model.torch.are_deterministic_algorithms_enabled(), original)
                    report = json.loads((Path(name) / "out/round3_operators.json").read_text())
                    self.assertEqual(report["deterministic_training"], requested)
                    self.assertEqual(report["original_deterministic_algorithms"], original)
                    self.assertEqual(report["deterministic_algorithms"], expected)
                    self.assertEqual(report["restored_deterministic_algorithms"], original)
                    self.assertTrue(all(row["deterministic_algorithms"] == expected
                                        and row["deterministic_training"] == requested for row in state.metrics))
                    final = next(event for event in state.events if isinstance(event, dict)
                                 and event.get("event") == "training_complete")
                    self.assertEqual(final["deterministic_algorithms"], expected)

    def test_deterministic_training_restores_policy_on_nested_failure(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, _, model, _, state):
            state.fail_after_install = True
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                entry.main(self.argv(Path(name)) + ["--deterministic-training", "--split-attention"])
            self.assertFalse(model.torch.are_deterministic_algorithms_enabled())
            self.assertTrue(model.torch.is_deterministic_algorithms_warn_only_enabled())
            report = json.loads((Path(name) / "out/round3_operators.json").read_text())
            self.assertTrue(report["deterministic_algorithms"])
            self.assertFalse(report["restored_deterministic_algorithms"])

    def test_install_is_deferred_and_exception_restores_every_wrapper(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, parent, model, Trainer, state):
            args = entry.build_parser().parse_args(self.argv(Path(name)) + ["--split-attention", "--cached-cpu-adam"])
            report = {}
            original_context, original_log = parent.candidate_context, Trainer._log
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with entry.round3_context(args, report), parent.candidate_context(args):
                    self.assertEqual(state.active, [])
                    model.install_moe_layout(object(), "shared-storage")
                    self.assertEqual(state.active, ["packed", "split", "cached"])
                    raise RuntimeError("injected failure")
            self.assertEqual(state.active, [])
            self.assertIs(parent.candidate_context, original_context)
            self.assertIs(Trainer._log, original_log)
            self.assertIn("cached_cpu_adam_calls", report)
            self.assertIn("split_attention_calls", report)

    def test_unused_split_rejects_passing_parity_but_does_not_hide_failed_parity(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, parent, model, _, state):
            args = entry.build_parser().parse_args(self.argv(Path(name)) + ["--split-attention"])
            report, full = {}, {}
            with entry.round3_context(args, report), parent.candidate_context(args):
                model.install_moe_layout(object(), "shared-storage")
                model.emit(Path("unused"), full, {"event": "parity_complete", "pass": False})
                self.assertFalse(report["full_model_parity_passed"])
                with self.assertRaisesRegex(RuntimeError, "split attention was not exercised"):
                    model.emit(Path("unused"), full, {"event": "parity_complete", "pass": True})
                self.assertEqual(full["status"], "round3_candidate_not_exercised")
                self.assertEqual(state.creations, [])

    def test_optimizer_requires_all132_gpu_parameters_and_source_moment_restoration(self):
        mutations = [{"updates": 0}, {"gpu_parameter_updates": 131}, {"parameter_updates": 131},
                     {"fallback_parameter_updates": 1}, {"optimizer_restored": False}]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name, \
                    runner_harness() as (entry, parent, model, _, state):
                args = entry.build_parser().parse_args(self.argv(Path(name)) + ["--cached-cpu-adam"])
                report, full = {}, {}
                with entry.round3_context(args, report), parent.candidate_context(args):
                    model.install_moe_layout(object(), "shared-storage")
                    state.cached.update(parameter_updates=132, gpu_parameter_updates=132)
                    event = {"event": "training_complete", "updates": 1,
                             "source_optimizer_present": True, "optimizer_restored": True}
                    for key, value in mutation.items():
                        (state.cached if key in state.cached else event)[key] = value
                    with self.assertRaisesRegex(RuntimeError, "all 132 parameters or restore"):
                        model.emit(Path("unused"), full, event)
                    self.assertEqual(full["status"], "round3_optimizer_not_exercised_or_restored")

    def test_parity_only_never_claims_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, _, _, _, state):
            self.assertEqual(entry.main(self.argv(Path(name)) + ["--cached-cpu-adam", "--parity-only"]), 0)
            report = json.loads((Path(name) / "out/round3_operators.json").read_text())
            self.assertFalse(report["cached_cpu_adam_training_exercised"])
            self.assertEqual(report["cached_cpu_adam_calls"]["parameter_updates"], 0)
            self.assertEqual(state.creations, [])

    def test_strict_preconditions_and_numeric_thresholds(self):
        with tempfile.TemporaryDirectory() as name, runner_harness() as (entry, _, _, _, _):
            parser = entry.build_parser()
            args = parser.parse_args(self.argv(Path(name)))
            entry.validate_extra_args(parser, args)  # Comparable baseline is valid.
            mutations = [("packed_attention", False), ("moe_layout", "compact"),
                         ("deterministic_parity", False), ("reference_repeat", False),
                         ("parity_seqs", "64,256"), ("batched_grad_norm", True),
                         ("loss_atol", .02001), ("grad_relative_l2", .05001),
                         ("output_relative_l2", .05001), ("loss_atol", float("nan")),
                         ("grad_relative_l2", float("inf")), ("output_relative_l2", float("nan"))]
            for key, value in mutations:
                with self.subTest(key=key), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    entry.validate_extra_args(parser, argparse.Namespace(**{**vars(args), key: value}))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.validate_extra_args(parser, argparse.Namespace(**{**vars(args), "cached_cpu_adam": True,
                                                                       "save_optim": False}))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.validate_extra_args(parser, argparse.Namespace(**{**vars(args), "sync_update_timing": True,
                                                                       "parity_only": True}))

    def test_failed_preflight_creates_no_output_and_late_failure_is_reported(self):
        for late in (False, True):
            with self.subTest(late=late), tempfile.TemporaryDirectory() as name, \
                    runner_harness() as (entry, _, _, _, state):
                argv = self.argv(Path(name)) + ["--split-attention", "--cached-cpu-adam"]
                state.refuse_preflight = not late
                state.fail_after_install = late
                with self.assertRaises(RuntimeError):
                    entry.main(argv)
                self.assertEqual(state.active, [])
                output = Path(name) / "out"
                if late:
                    report = json.loads((output / "round3_operators.json").read_text())
                    self.assertEqual(report["status"], "failed")
                    self.assertIn("injected failure", report["error"])
                    self.assertIn("cached_cpu_adam_calls", report)
                else:
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
