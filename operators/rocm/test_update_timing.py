"""CPU-only timing lifecycle checks with a mocked GPU completion boundary."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cat_yoko.optim import CPUOffloadAdamW
from cat_yoko.trainer import Trainer
from operators.rocm.update_timing import synchronized_update_timing


class UpdateTimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "timing.json"
        self.model = nn.Linear(2, 2, bias=False)
        self.optimizer = CPUOffloadAdamW(self.model.parameters())
        self.owner = object.__new__(Trainer)
        self.owner.device = "cpu"
        self.batch = {"input_ids": torch.ones(1, 16, dtype=torch.long)}

    def _report(self):
        return json.loads(self.output.read_text())

    def _patch_methods(self, run, forward=None, step=None):
        stack = ExitStack()
        stack.enter_context(patch.object(Trainer, "run", run))
        stack.enter_context(patch.object(Trainer, "_forward_loss", forward or (lambda *args: "loss")))
        stack.enter_context(patch.object(CPUOffloadAdamW, "step", step or (lambda *args, **kwargs: None)))
        return stack

    def test_records_every_completed_update_and_discards_five_warmups(self):
        def run(owner):
            for _ in range(6):
                owner._forward_loss(self.model, self.batch)
                self.optimizer.step()
            return "result"

        with self._patch_methods(run), patch(
            "operators.rocm.update_timing.time.perf_counter",
            side_effect=[0, 2, 10, 13, 20, 24, 30, 35, 40, 46, 50, 58],
        ), patch("operators.rocm.update_timing.torch.cuda.synchronize") as sync:
            with synchronized_update_timing(self.output) as timing:
                self.assertEqual(self.owner.run(), "result")
            self.assertEqual(len(timing.samples), 6)
            sync.assert_not_called()
        report = self._report()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_updates"], 6)
        self.assertEqual(report["measured_updates"], 1)
        self.assertEqual(report["discarded_updates"], 5)
        self.assertEqual(report["median_completed_update_tokens_s"], 2)
        self.assertEqual(report["aggregate_completed_update_tokens_s"], 2)
        self.assertEqual(report["sum_steady_elapsed_s"], 8)
        self.assertEqual(report["steady_tokens"], 16)
        self.assertEqual([row["invocation_update"] for row in report["samples"]], list(range(1, 7)))

    def test_gpu_sync_follows_then_installed_candidate_step(self):
        self.owner.device = "cuda:0"
        order = []

        def run(owner):
            owner._forward_loss(self.model, self.batch)
            return self.optimizer.step()

        def forward(*args):
            order.append("forward")

        def candidate_step(*args, **kwargs):
            order.append("candidate queued GPU update")
            return "candidate loss"

        with self._patch_methods(run, forward, candidate_step), patch(
            "operators.rocm.update_timing.torch.cuda.synchronize",
            side_effect=lambda device: order.append("sync " + str(device)),
        ), patch("operators.rocm.update_timing.time.perf_counter", side_effect=[10, 14]):
            installed = CPUOffloadAdamW.step
            with synchronized_update_timing(self.output, warmup=0):
                self.assertEqual(self.owner.run(), "candidate loss")
            self.assertIs(CPUOffloadAdamW.step, installed)
        self.assertEqual(order, ["sync cuda:0", "forward", "candidate queued GPU update", "sync cuda:0"])
        self.assertEqual(self._report()["samples"][0]["elapsed_s"], 4)

    def test_accumulation_has_one_sample_and_foreign_optimizer_cannot_end_it(self):
        foreign = CPUOffloadAdamW(nn.Linear(2, 2).parameters())

        def run(owner):
            owner._forward_loss(self.model, self.batch)
            foreign.step()
            owner._forward_loss(self.model, self.batch)
            self.optimizer.step()

        with self._patch_methods(run), patch("operators.rocm.update_timing.time.perf_counter", side_effect=[1, 3]):
            with synchronized_update_timing(self.output, warmup=0):
                # A forward outside the bound Trainer.run is an initial eval
                # or another caller and must not start a measured update.
                self.owner._forward_loss(self.model, self.batch)
                self.owner.run()
        report = self._report()
        self.assertEqual(report["completed_updates"], 1)
        self.assertEqual(report["samples"][0]["tokens"], 32)
        self.assertEqual(report["median_completed_update_tokens_s"], 16)

    def test_failed_update_is_not_a_completed_sample_and_methods_restore(self):
        def run(owner):
            owner._forward_loss(self.model, self.batch)
            self.optimizer.step()
            owner._forward_loss(self.model, self.batch)
            self.optimizer.step()

        calls = []

        def step(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("failed update")

        with self._patch_methods(run, step=step), patch(
            "operators.rocm.update_timing.time.perf_counter", side_effect=[1, 3, 4]
        ):
            originals = (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step)
            with self.assertRaisesRegex(RuntimeError, "failed update"):
                with synchronized_update_timing(self.output, warmup=0):
                    self.owner.run()
            self.assertEqual(originals, (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step))
        report = self._report()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["started_updates"], 2)
        self.assertEqual(report["completed_updates"], 1)
        self.assertTrue(report["incomplete_update"])
        self.assertEqual(report["samples"][0]["elapsed_s"], 2)

    def test_nested_profiler_wrapper_preserves_completed_update_boundary(self):
        order = []

        def run(owner):
            owner._forward_loss(self.model, self.batch)
            self.optimizer.step()

        with self._patch_methods(run, step=lambda *args: order.append("candidate step")), patch(
            "operators.rocm.update_timing.time.perf_counter", side_effect=[1, 2]
        ):
            native = CPUOffloadAdamW.step
            with synchronized_update_timing(self.output, warmup=0):
                measured_step = CPUOffloadAdamW.step

                def profiler_step(*args, **kwargs):
                    result = measured_step(*args, **kwargs)
                    order.append("profiler after_update")
                    return result

                with patch.object(CPUOffloadAdamW, "step", profiler_step):
                    self.owner.run()
                self.assertIs(CPUOffloadAdamW.step, measured_step)
            self.assertIs(CPUOffloadAdamW.step, native)
        self.assertEqual(order, ["candidate step", "profiler after_update"])
        self.assertEqual(self._report()["completed_updates"], 1)

    def test_default_tokens_and_no_complete_samples_have_no_fabricated_speed(self):
        def run(owner):
            owner._forward_loss(self.model, {})
            self.optimizer.step()

        with self._patch_methods(run), patch("operators.rocm.update_timing.time.perf_counter", side_effect=[1, 3]):
            with synchronized_update_timing(self.output):
                self.owner.run()
        report = self._report()
        self.assertEqual(report["samples"][0]["tokens"], 4096)
        self.assertIsNone(report["median_completed_update_tokens_s"])
        self.assertEqual(report["measured_updates"], 0)
        with synchronized_update_timing(self.output):
            pass
        report = self._report()
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["completed_updates"], 0)
        self.assertIsNone(report["aggregate_completed_update_tokens_s"])

    def test_forward_exception_restores_nested_contexts_and_marks_failed(self):
        def run(owner):
            owner._forward_loss(self.model, self.batch)

        def forward(*args):
            raise ValueError("forward failure")

        second_output = self.output.with_name("inner.json")
        with self._patch_methods(run, forward=forward), patch(
            "operators.rocm.update_timing.time.perf_counter", side_effect=[1, 2]
        ):
            originals = (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step)
            with self.assertRaisesRegex(ValueError, "forward failure"):
                with synchronized_update_timing(self.output):
                    with synchronized_update_timing(second_output):
                        self.owner.run()
            self.assertEqual(originals, (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step))
        for output in (self.output, second_output):
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["completed_updates"], 0)
            self.assertTrue(report["incomplete_update"])

    def test_invalid_warmup_rejected_before_any_patches(self):
        originals = (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step)
        for warmup in (-1, 1.5, True):
            with self.subTest(warmup=warmup), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                with synchronized_update_timing(self.output, warmup=warmup):
                    pass
        self.assertEqual(originals, (Trainer.run, Trainer._forward_loss, CPUOffloadAdamW.step))


if __name__ == "__main__":
    unittest.main()
