"""Standard-library fakes prove slice isolation and failure restoration."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from operators.rocm.continuation_quality import quality_context, validate_quality_report


class FakeCuda:
    def __init__(self):
        self.states = [17, 29]

    def is_available(self):
        return True

    def get_rng_state_all(self):
        return self.states[:]

    def set_rng_state_all(self, states):
        self.states = states[:]


class FakeTorch:
    def __init__(self):
        self.state = 11
        self.cuda = FakeCuda()

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.state = state

    @staticmethod
    def equal(a, b):
        return a == b


class FakeStream:
    def __init__(self, nseq=64, stride=1):
        self.state = dict(kind="packed", i=0, stride=stride, nseq=nseq)

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state.update(state)


class FakeModel:
    def __init__(self):
        self.training = True
        self.stats = "training"

    def train(self, value=True):
        self.training = value


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.eval_path = self.root / "eval.bin"
        self.eval_path.write_bytes(bytes(64 * 4096 * 4))
        self.nseq, self.stride, self.fail, self.underconsume = 64, 1, False, False
        self.bad_nll = None
        self.trainer = SimpleNamespace(eval_data=self.eval_path, data=self.root / "train.bin",
                                       seq_len=4096, micro_batch=1, world=1, eval_batches=32,
                                       resume=self.root / "checkpoint.pt")
        self.open_calls, self.rows = [], []

        def stream_open(path, seed):
            self.open_calls.append((path, seed))
            return FakeStream(self.nseq, self.stride)

        self.trainer._open = stream_open
        self.original_open = stream_open
        self.model = FakeModel()
        self.torch = FakeTorch()

        def evaluate(trainer, model, *, event, step):
            stream = trainer._open(trainer.eval_data, 7)
            self.rows.append((event, stream.state_dict()["i"], trainer.eval_batches))
            if event.startswith("independent_"):
                model.train(False)
                model.stats = "eval"
                self.torch.state = 99
                self.torch.cuda.states = [101, 102]
                random.random()
                if self.fail:
                    raise RuntimeError("fake evaluation failure")
            stream.state["i"] += trainer.eval_batches - int(self.underconsume and event.startswith("independent_"))
            return dict(event=event, step=step, **{"pass": True},
                        eval_nll=self.bad_nll if event.startswith("independent_") and self.bad_nll is not None else 8.0,
                        eval_valid_tokens=131000, eval_batches=trainer.eval_batches)

        self.original_evaluate = evaluate
        self.bench = SimpleNamespace(evaluate_heldout=evaluate, torch=self.torch,
                                     clear_moe_statistics=lambda model: setattr(model, "stats", None))

    def run_pair(self):
        with quality_context(self.bench, self.root):
            before = self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=73864)
            after = self.bench.evaluate_heldout(self.trainer, self.model, event="final_eval", step=77864)
        return before, after

    def validate(self):
        return validate_quality_report(self.root / "quality.json", source_dir=Path(__file__).resolve().parents[2])

    def test_pair_disjoint_primary_unchanged_and_strict_receipt(self):
        py_state = random.getstate()
        before, after = self.run_pair()
        self.assertEqual(before["event"], "initial_eval")
        self.assertEqual(after["event"], "final_eval")
        self.assertNotIn("row_start", before)
        self.assertEqual(self.rows, [("initial_eval", 0, 32), ("independent_initial_eval", 32, 32),
                                     ("final_eval", 0, 32), ("independent_final_eval", 32, 32)])
        self.assertEqual(self.torch.state, 11)
        self.assertEqual(self.torch.cuda.states, [17, 29])
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(self.model.training)
        self.assertIsNone(self.model.stats)
        self.assertIs(self.trainer._open, self.original_open)
        self.assertIs(self.bench.evaluate_heldout, self.original_evaluate)
        report = self.validate()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["final_eval"]["stream_after"]["i"], 64)

    def test_periodic_eval_does_not_trigger_extra(self):
        with quality_context(self.bench, self.root):
            self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=1)
            self.bench.evaluate_heldout(self.trainer, self.model, event="periodic_eval", step=250)
            self.bench.evaluate_heldout(self.trainer, self.model, event="final_eval", step=4001)
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(self.rows[2], ("periodic_eval", 0, 32))

    def test_eval_failure_restores_rng_model_open_and_raises(self):
        py_state = random.getstate()
        self.fail = True
        with self.assertRaisesRegex(RuntimeError, "fake evaluation failure"):
            self.run_pair()
        self.assertEqual(self.torch.state, 11)
        self.assertEqual(self.torch.cuda.states, [17, 29])
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(self.model.training)
        self.assertIsNone(self.model.stats)
        self.assertIs(self.trainer._open, self.original_open)
        self.assertIs(self.bench.evaluate_heldout, self.original_evaluate)
        self.assertEqual(json.loads((self.root / "quality.json").read_text())["status"], "failed")

    def test_insufficient_rows_and_ddp_stride_rejected(self):
        for nseq, stride in ((63, 1), (64, 2)):
            with self.subTest(nseq=nseq, stride=stride):
                target = self.root / f"case-{nseq}-{stride}"
                self.nseq, self.stride = nseq, stride
                with self.assertRaises(ValueError), quality_context(self.bench, target):
                    self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=1)
                self.assertIs(self.trainer._open, self.original_open)

    def test_underconsumed_or_nan_result_is_not_success(self):
        self.underconsume = True
        with self.assertRaisesRegex(ValueError, "cursor"):
            self.run_pair()
        self.assertEqual(json.loads((self.root / "quality.json").read_text())["status"], "failed")

    def test_nonfinite_independent_nll_fails_without_invalid_json(self):
        self.bad_nll = float("nan")
        with self.assertRaisesRegex(ValueError, "NLL"):
            self.run_pair()
        report = json.loads((self.root / "quality.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("initial_eval", report)

    def test_caught_extra_failure_cannot_be_hidden(self):
        self.fail = True
        with self.assertRaisesRegex(ValueError, "failed inside"), quality_context(self.bench, self.root):
            with self.assertRaises(RuntimeError):
                self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=1)
            self.assertEqual(json.loads((self.root / "quality.json").read_text())["status"], "failed")

    def test_pair_must_use_same_model_and_increasing_step(self):
        for changed_model in (True, False):
            with self.subTest(changed_model=changed_model):
                target = self.root / ("other-model" if changed_model else "no-updates")
                with self.assertRaises(ValueError), quality_context(self.bench, target):
                    self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=1)
                    self.bench.evaluate_heldout(self.trainer, FakeModel() if changed_model else self.model,
                                               event="final_eval", step=2 if changed_model else 1)

    def test_missing_pair_and_reused_receipt_fail(self):
        with self.assertRaisesRegex(ValueError, "both evaluations"), quality_context(self.bench, self.root):
            self.bench.evaluate_heldout(self.trainer, self.model, event="initial_eval", step=1)
        with self.assertRaises(FileExistsError), quality_context(self.bench, self.root):
            pass

    def test_completed_receipt_tampering_fails(self):
        self.run_pair()
        path = self.root / "quality.json"
        original = json.loads(path.read_text())
        for update in ({"status": "initial_checked"}, {"file_sha256": {}}, {"eval_row_start": 0}):
            path.write_text(json.dumps(dict(original, **update)))
            with self.assertRaises(ValueError):
                self.validate()
        path.write_text(json.dumps(original))
        self.eval_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "corpus changed"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
