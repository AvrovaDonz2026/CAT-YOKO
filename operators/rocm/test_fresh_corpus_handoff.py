"""Fail-closed corpus coordinates, native state reuse and held-out restoration."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operators.rocm import fresh_corpus_handoff as handoff
from operators.rocm import fresh_corpus_continuation as continuation
from operators.rocm import run_fresh_corpus_window as controller


def fixture_plan(root):
    root = root.resolve()
    plan = {"schema_version": 1, "mapping": handoff.MAPPING, "no_wrap": True, "seq_len": 4096,
            "eos_id": 1, "old_nseq": 19531, "new_nseq": 19531, "fresh_eval_nseq": 244,
            "absolute_cursor_origin": 47062, "logical_row_origin": 0, "source_step": 81864,
            "source_phase_tokens": 355010560, "source_tokens_seen": 355010560,
            "source_real_tokens": 192765952, "source_unique_tokens": 79998976, "updates": 4000,
            "target_step": 85864, "target_cursor": 51062, "target_logical_row": 4000,
            "target_phase_tokens": 371394560, "target_tokens_seen": 371394560,
            "target_real_tokens": 209149952, "new_unique_tokens": 79998976, "total_unique_tokens": 159997952}
    for prefix in ("source_checkpoint", "old_train", "new_train", "fresh_eval", "corpus_manifest"):
        plan[prefix + "_path"] = str(root / (prefix + ".bin"))
        plan[prefix + "_sha256"] = handoff.SOURCE_SHA if prefix == "source_checkpoint" else "a" * 64
    return plan


class CoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/fresh-coordinates-test")
        self.plan = fixture_plan(self.root)

    def test_exact_source_and_endpoint_preserve_learning_clocks(self):
        handoff.validate_handoff(self.plan)
        old = {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531}
        state = handoff.check_stream_state(old, self.plan, allow_source=True)
        self.assertEqual(state["i"], 47062)
        self.assertEqual(state["corpus_row"], 0)
        final = handoff.stream_state(self.plan, 51062)
        self.assertEqual(final["corpus_row"], 4000)
        window = controller.fresh_plan(self.plan)
        self.assertEqual(window["target_step"], 85864)
        self.assertEqual(window["target_adam_step"], 51062)
        self.assertEqual(window["target_tokens_in_phase"], 371394560)
        self.assertEqual(self.plan["target_real_tokens"] - self.plan["source_real_tokens"], 4000 * 4096)

    def test_legacy_restore_requires_explicit_exact_source(self):
        old = {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531}
        with self.assertRaises(RuntimeError):
            handoff.check_stream_state(old, self.plan)
        for change in ({"i": 47061}, {"nseq": 19530}, {"stride": 2}, {"kind": "dummy"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                handoff.check_stream_state(dict(old, **change), self.plan, allow_source=True)

    def test_wrong_corpus_hash_offset_counter_and_extra_fields_fail_closed(self):
        valid = handoff.stream_state(self.plan, 47562)
        for change in ({"corpus_sha256": "b" * 64}, {"absolute_cursor_origin": 47061},
                       {"corpus_row": 499}, {"i": 47561}, {"nseq": 19530}, {"no_wrap": False},
                       {"data_handoff_sha256": "b" * 64}, {"stride": 2}, {"unknown": 1}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                handoff.check_stream_state(dict(valid, **change), self.plan)

    def test_invalid_authorization_or_wrapping_plans_reject(self):
        for change in ({"updates": 4001}, {"target_cursor": 51063}, {"source_step": 81863},
                       {"source_phase_tokens": 355010561}, {"new_nseq": 3999}, {"fresh_eval_nseq": 31},
                       {"no_wrap": False}, {"absolute_cursor_origin": True}, {"new_unique_tokens": 1}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                handoff.validate_handoff(dict(self.plan, **change))
        for cursor in (47061, 47062 + 19532, True, 47062.0):
            with self.subTest(cursor=cursor), self.assertRaises(RuntimeError):
                handoff.stream_state(self.plan, cursor)

    def test_canonical_json_identity_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "handoff.json"
            handoff.write_handoff(path, self.plan)
            self.assertEqual(handoff.load_handoff(path, verify_files=False), self.plan)
            path.write_text(json.dumps(self.plan, indent=2))
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                handoff.load_handoff(path, verify_files=False)

    def test_worker_command_has_no_time_cap_and_restores_correct_source(self):
        args = SimpleNamespace(python="python", source_dir=self.root / "source", resume=self.root / "source.pt",
                               base=self.root / "base", data=self.root / "train.bin", eval_data=self.root / "old-eval.bin",
                               fresh_eval_data=self.root / "fresh-eval.bin", out=self.root / "run",
                               accepted_run=self.root / "accepted", handoff_manifest=self.root / "handoff.json")
        command = controller.worker_command(args, self.plan)
        for flag, value in (("--run-steps", "4000"), ("--save-every-seconds", "300"), ("--keep-last", "3"),
                            ("--eval-every", "250"), ("--eval-batches", "32"), ("--extra-heldout-start-row", "32"),
                            ("--data-handoff", str(args.handoff_manifest)), ("--fresh-eval-data", str(args.fresh_eval_data))):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--save-optim", command)
        self.assertIn("--split-attention", command)
        self.assertIn("--cached-cpu-adam", command)
        self.assertNotIn("--max-hours", command)
        self.assertNotIn("--deterministic-training", command)
        fresh_args, native_argv = continuation.parse_fresh_argv(command[3:])
        expected_native = command[3:]
        for flag in ("--data-handoff", "--fresh-eval-data"):
            index = expected_native.index(flag)
            expected_native = expected_native[:index] + expected_native[index + 2:]
        self.assertEqual(native_argv, expected_native)
        self.assertEqual(native_argv[native_argv.index("--data") + 1], str(args.data))
        self.assertEqual(fresh_args.data_handoff, args.handoff_manifest)
        self.assertEqual(fresh_args.fresh_eval_data, args.fresh_eval_data)

    def test_fresh_parser_preserves_native_data_with_either_option_order(self):
        for argv in (("--data", "train.bin", "--data-handoff", "plan.json", "--fresh-eval-data", "fresh.bin"),
                     ("--data-handoff", "plan.json", "--fresh-eval-data", "fresh.bin", "--data", "train.bin")):
            with self.subTest(argv=argv):
                fresh, native = continuation.parse_fresh_argv(argv)
                self.assertEqual(native, ["--data", "train.bin"])
                self.assertEqual(fresh.data_handoff, Path("plan.json"))

    def test_missing_native_paths_fail_closed_before_resolve(self):
        with self.assertRaisesRegex(RuntimeError, "explicit resume"):
            continuation.validate_worker_args(SimpleNamespace(resume=None, data=None, out=None, eval_data=None), self.plan)


class FakePacked:
    def __init__(self, path, seq_len, **options):
        self.nseq = 19531
        self._i = 0

    def batch(self, micro_batch, device):
        rows = [(self._i + j) % self.nseq for j in range(micro_batch)]
        self._i += micro_batch
        return {"rows": rows}


class AdapterLogicTests(unittest.TestCase):
    def setUp(self):
        self.plan = fixture_plan(Path("/tmp/fresh-adapter"))
        data_module = ModuleType("cat_yoko.data")
        data_module.PackedBinStream = FakePacked
        source = Path(__file__).with_name("handoff_stream.py")
        spec = importlib.util.spec_from_file_location("fresh_adapter_fixture", source)
        self.adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"cat_yoko.data": data_module}):
            spec.loader.exec_module(self.adapter)

    def stream(self):
        return self.adapter.FreshPackedBinStream(self.plan["new_train_path"], 4096, handoff=self.plan, eos_id=1)

    def test_reads_actual_row_zero_not_old_cursor_modulo(self):
        stream = self.stream()
        stream.load_state_dict({"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531})
        self.assertEqual(stream.batch(2, "cpu")["rows"], [0, 1])
        self.assertEqual(stream.state_dict()["i"], 47064)
        self.assertEqual(stream.state_dict()["corpus_row"], 2)
        self.assertEqual(stream.first_batch_row, 0)

    def test_saved_offset_resume_continues_without_skip(self):
        initial = self.stream()
        initial.batch(3, "cpu")
        resumed = self.stream()
        resumed.load_state_dict(initial.state_dict())
        self.assertEqual(resumed.batch(1, "cpu")["rows"], [3])
        with self.assertRaises(RuntimeError):
            resumed.load_state_dict({"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531})

    def test_end_of_corpus_refuses_wrap_and_preserves_cursor(self):
        stream = self.stream()
        stream.load_state_dict(handoff.stream_state(self.plan, 47062 + 19530))
        self.assertEqual(stream.batch(1, "cpu")["rows"], [19530])
        before = stream.state_dict()
        with self.assertRaisesRegex(RuntimeError, "wrap"):
            stream.batch(1, "cpu")
        self.assertEqual(stream.state_dict(), before)

    def test_native_failure_does_not_consume_row(self):
        stream = self.stream()
        with patch.object(FakePacked, "batch", side_effect=OSError("transfer failure")):
            with self.assertRaises(OSError):
                stream.batch(1, "cpu")
        self.assertEqual(stream.state_dict()["corpus_row"], 0)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return FakeTensor(self.value)


class FreshQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.plan = fixture_plan(self.root)
        fresh = Path(self.plan["fresh_eval_path"])
        fresh.write_bytes(b"fresh held-out identity")
        self.plan["fresh_eval_sha256"] = handoff.digest(fresh)
        self.state = FakeTensor(7)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                get_rng_state=lambda: self.state.clone(),
                                set_rng_state=lambda state: setattr(self, "state", state.clone()),
                                equal=lambda a, b: a.value == b.value)
        self.model = SimpleNamespace(training=True)
        self.model.train = lambda mode: setattr(self.model, "training", mode)

        class EvaluationStream:
            def __init__(inner):
                inner.i = 0

            def state_dict(inner):
                return {"kind": "packed", "i": inner.i, "stride": 1, "nseq": 244}

        self.original_open = lambda path, seed: EvaluationStream()
        self.trainer = SimpleNamespace(seq_len=4096, micro_batch=1, world=1, eval_data=self.root / "old-eval.bin",
                                       eval_batches=32, _open=self.original_open)
        self.fail_fresh = False

        def evaluate(trainer, model, *, event, step):
            if event.startswith("fresh_"):
                model.train(False)
                self.state.value += 1
                random.random()
                stream = trainer._open(trainer.eval_data, 0)
                stream.i += trainer.eval_batches
                if self.fail_fresh:
                    raise OSError("injected fresh evaluation failure")
            return {"event": event, "step": step, "pass": True, "eval_nll": 7.1,
                    "eval_valid_tokens": 32 * 4095, "eval_batches": trainer.eval_batches}

        self.bench = SimpleNamespace(torch=torch, evaluate_heldout=evaluate, clear_moe_statistics=Mock())

    def test_fresh_pair_restores_rng_eval_path_mode_and_open(self):
        old_rng, old_py = self.state.value, random.getstate()
        old_path, old_evaluate = self.trainer.eval_data, self.bench.evaluate_heldout
        with continuation.fresh_quality_context(self.bench, self.root, self.plan) as report:
            self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=81864)
            self.bench.evaluate_heldout(self.trainer, self.model, event="final_eval", step=85864)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(self.state.value, old_rng)
        self.assertEqual(random.getstate(), old_py)
        self.assertEqual(self.trainer.eval_data, old_path)
        self.assertEqual(self.trainer._open, self.original_open)
        self.assertIs(self.bench.evaluate_heldout, old_evaluate)
        self.assertTrue(self.model.training)
        validated = handoff.validate_fresh_quality_report(self.root / "fresh_quality.json",
                    source_dir=Path(__file__).resolve().parents[2], handoff=self.plan)
        self.assertEqual(validated["initial_eval"]["stream_after"]["i"], 32)

    def test_fresh_failure_restores_every_training_state_and_marks_failed(self):
        self.fail_fresh = True
        old_rng, old_py, old_path = self.state.value, random.getstate(), self.trainer.eval_data
        with self.assertRaises(OSError):
            with continuation.fresh_quality_context(self.bench, self.root, self.plan):
                self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=81864)
        self.assertEqual(self.state.value, old_rng)
        self.assertEqual(random.getstate(), old_py)
        self.assertEqual(self.trainer.eval_data, old_path)
        self.assertEqual(self.trainer._open, self.original_open)
        self.assertTrue(self.model.training)
        self.assertEqual(json.loads((self.root / "fresh_quality.json").read_text())["status"], "failed")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch CPU runtime required")
class NativeCpuTests(unittest.TestCase):
    def test_actual_production_parser_round_trips_supervisor_worker_command(self):
        from operators.rocm import round3_candidate_bench
        root = Path("/tmp/fresh-production-parser").resolve()
        plan = fixture_plan(root)
        args = SimpleNamespace(python="python", source_dir=root / "source", resume=Path(plan["source_checkpoint_path"]),
                               base=root / "base", data=Path(plan["new_train_path"]), eval_data=root / "old-eval.bin",
                               fresh_eval_data=Path(plan["fresh_eval_path"]), out=root / "run",
                               accepted_run=root / "accepted", handoff_manifest=root / "handoff.json")
        command = controller.worker_command(args, plan)
        fresh, native_argv = continuation.parse_fresh_argv(command[3:])
        parser = round3_candidate_bench.build_parser()
        parser.allow_abbrev = False
        inherited, production_extra = parser.parse_known_args(native_argv)
        continuation.validate_worker_args(inherited, plan)
        self.assertEqual(inherited.data, args.data)
        self.assertEqual(inherited.resume, args.resume)
        self.assertEqual(inherited.eval_data, args.eval_data)
        self.assertEqual(inherited.out, args.out / "continuation")
        self.assertEqual(fresh.data_handoff, args.handoff_manifest)
        self.assertEqual(production_extra, ["--accepted-run", str(args.accepted_run),
                                           "--extra-heldout-start-row", "32", "--extra-heldout-batches", "32"])

    def test_actual_cpu_adapter_reads_zero_and_resumes_saved_row(self):
        import torch
        from operators.rocm.handoff_stream import FreshPackedBinStream
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = fixture_plan(root)
            plan.update(new_nseq=4000, new_unique_tokens=4000 * 4096,
                        total_unique_tokens=79998976 + 4000 * 4096)
            path = Path(plan["new_train_path"])
            with path.open("wb") as output:
                output.truncate(4000 * 4096 * 4)
                output.seek(0)
                output.write(torch.full((4096,), 7, dtype=torch.int32).numpy().tobytes())
                output.seek((47062 % 4000) * 4096 * 4)
                output.write(torch.full((4096,), 9, dtype=torch.int32).numpy().tobytes())
            first = FreshPackedBinStream(path, 4096, handoff=plan, eos_id=1)
            first.load_state_dict({"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531})
            batch = first.batch(1, "cpu")
            self.assertTrue(torch.equal(batch["input_ids"], torch.full((1, 4096), 7, dtype=torch.long)))
            saved = first.state_dict()
            resumed = FreshPackedBinStream(path, 4096, handoff=plan, eos_id=1)
            resumed.load_state_dict(saved)
            resumed.batch(1, "cpu")
            self.assertEqual(resumed.state_dict()["corpus_row"], 2)

    def test_native_132_weight_264_moment_audit_allows_only_explicit_stream_mapping(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = fixture_plan(root)
            plan.update(new_nseq=4000, new_unique_tokens=4000 * 4096,
                        total_unique_tokens=79998976 + 4000 * 4096)
            with Path(plan["old_train_path"]).open("wb") as output:
                output.truncate(19531 * 4096 * 4)
            source = {"kind": "trainable", "trainable": {f"weight_{i}": torch.ones((2, 2), dtype=torch.bfloat16) for i in range(132)},
                      "extra": {"phase": "B0", "name": "CAT-YOKO-12B", "seq_len": 4096, "seed": 0,
                                "step": 81864, "tokens_in_phase": 355010560, "tokens_seen": 355010560,
                                "cfg": {"vocab_size": 130560, "hidden_size": 2048, "encoder_layers": 16,
                                        "decoder_layers": 26, "n_routed_enc": 20, "n_routed_dec": 20, "top_k_dec": 10,
                                        "use_nvfp4": False, "use_fp8": False}, "use_kda": False, "sparse": "window",
                                "stream": {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531},
                                "rng_py": random.getstate(), "rng_torch": torch.get_rng_state(),
                                "rng_cuda": [torch.zeros(4, dtype=torch.uint8)]},
                      "optimizer": {"param_groups": [{"params": list(range(132)), "lr": 7.1e-5}, {"params": [], "lr": 7.1e-5}],
                                    "state": {i: {"step": 47062, "exp_avg": torch.ones((2, 2)), "exp_avg_sq": torch.ones((2, 2))} for i in range(132)}}}
            target = deepcopy(source)
            target["extra"].update(step=81865, tokens_in_phase=355014656, tokens_seen=355014656,
                                   stream=handoff.stream_state(plan, 47063))
            for state in target["optimizer"]["state"].values():
                state["step"] = 47063
            source_path, target_path = Path(plan["source_checkpoint_path"]), root / "target.pt"
            torch.save(source, source_path)
            torch.save(target, target_path)
            validate = handoff.validate_handoff
            with patch.object(handoff, "validate_handoff", side_effect=lambda p, **kw: validate(p, verify_files=False)):
                metadata = handoff.audit_checkpoint(source_path, target_path, plan, 1)
                self.assertEqual(metadata["source_stream_i"], 47063)
                self.assertEqual(metadata["source_logical_row"], 1)
                self.assertEqual(metadata["source_adam_step"], 47063)
                self.assertEqual(metadata["optimizer_moments"], 264)
                target["optimizer"]["state"][0]["exp_avg_sq"][0, 0] = -1
                torch.save(target, target_path)
                with self.assertRaisesRegex(ValueError, "negative"):
                    handoff.audit_checkpoint(source_path, target_path, plan, 1)


if __name__ == "__main__":
    unittest.main()
