"""CPU regressions for profiler boundaries, state restoration, and accounting."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.optim import CPUOffloadAdamW
from cat_yoko.trainer import Trainer
from operators.rocm.profile_training import (
    PREFIX, TrainingProfile, benchmark_arguments, build_parser, check_resume,
    event_phase, summarize_events, validate_profile_args,
)


def event(name, cpu, device, parent=None, device_type="CPU"):
    return SimpleNamespace(name=name, self_cpu_time_total=cpu,
                           self_device_time_total=device, cpu_parent=parent,
                           device_type=device_type)


class FakeProfiler:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls = []

    def start(self):
        self.calls.append("start")

    def step(self):
        self.calls.append("step")

    def stop(self):
        self.calls.append("stop")

    def events(self):
        return []

    def export_chrome_trace(self, path):
        Path(path).write_text("{}")

    def key_averages(self, **kwargs):
        return SimpleNamespace(table=lambda **kwargs: "fake profiler table")


class ProfileTrainingTests(unittest.TestCase):
    def test_nested_ce_and_transfer_account_once_excluding_eval_and_save(self):
        forward = event(PREFIX + "forward", 1, 0)
        ce = event(PREFIX + "ce", 1, 0, forward)
        optimizer = event(PREFIX + "optimizer", 1, 0)
        evaluation = event(PREFIX + "eval", 1, 0)
        eval_ce = event(PREFIX + "ce", 1, 0, evaluation)
        save = event(PREFIX + "save", 1, 0)
        events = [forward, ce, optimizer, evaluation, eval_ce, save,
                  event("aten::mm", 10, 20, forward),
                  event("aten::cross_entropy", 20, 30, ce),
                  event("aten::copy_", 30, 40, optimizer),
                  event("aten::cross_entropy", 1000, 2000, eval_ce),
                  event("aten::copy_", 2000, 3000, save),
                  event("same linked HIP kernel", 0, 20, device_type="CUDA")]
        summary = summarize_events(events)
        self.assertEqual(event_phase(events[-3]), "eval")
        self.assertEqual(summary["phases"]["forward"]["device_work_us"], 20)
        self.assertEqual(summary["phases"]["ce"]["device_work_us"], 30)
        self.assertEqual(summary["steady_device_work_us"], 90)
        self.assertEqual(summary["transfer_overlay"]["device_work_us"], 40)
        self.assertIsNone(summary["phases"]["eval"]["steady_device_work_fraction"])

    def test_window_counts_real_updates_and_restores_wrappers_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []

            def factory(**kwargs):
                profiler = FakeProfiler(**kwargs)
                captured.append(profiler)
                return profiler

            session = TrainingProfile(Path(tmp), device="cpu", profiler_factory=factory)
            originals = (Trainer._forward_loss, torch.autograd.backward, CPUOffloadAdamW.step,
                         torch.nn.functional.linear)
            owner = object.__new__(Trainer)
            with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                try:
                    with session.instrument(owner):
                        for _ in range(2):
                            session.begin_forward()
                            self.assertIsNone(session.profiler)
                            session.after_update()
                        session.begin_forward()
                        session.after_update()
                        raise RuntimeError("controlled failure")
                except RuntimeError as error:
                    session.finish(error)
                    raise
            self.assertEqual(originals, (Trainer._forward_loss, torch.autograd.backward, CPUOffloadAdamW.step,
                                         torch.nn.functional.linear))
            self.assertEqual(captured[0].calls, ["start", "stop"])
            self.assertTrue(captured[0].options["record_shapes"])
            self.assertEqual(len(captured[0].options["activities"]), 1)
            report = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["steps"][0]["invocation_update"], 3)
            self.assertEqual(report["recorded_updates"], 1)

    def test_real_cpu_training_profile_excludes_warmups_and_keeps_adam_checkpoint(self):
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, heldout = root / "train.bin", root / "eval.bin"
            rows = 16
            ids = [3 + index % (cfg.vocab_size - 3) for index in range(rows * cfg.seq_len)]
            train.write_bytes(struct.pack("<" + "i" * len(ids), *ids))
            heldout.write_bytes(struct.pack("<" + "i" * len(ids), *ids[::-1]))
            trainer = Trainer(cfg, "B0", "cpu", run_steps=6, tokens=1e7,
                              data=train, eval_data=heldout, eos_id=1, seq_len=cfg.seq_len,
                              micro_batch=1, accum=1, optim_cpu=True, grad_ckpt=True,
                              log_every=1, eval_every=3, eval_batches=1,
                              save_dir=root / "checkpoint", save_every=3,
                              save_full=False, save_trainable=True, save_optim=True)
            session = TrainingProfile(root / "profile", device="cpu")
            original_run = Trainer.run
            with session.wrap_trainer_run():
                result = trainer.run()
            self.assertIs(Trainer.run, original_run)
            self.assertEqual(result.step, 6)
            self.assertEqual(session.updates, 6)
            report = json.loads((root / "profile/summary.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual([row["invocation_update"] for row in report["steps"]], [3, 4, 5, 6])
            self.assertTrue((root / "profile/trace.json").is_file())
            self.assertGreater(report["phases"]["ce"]["self_cpu_us"], 0)
            self.assertGreater(report["phases"]["backward"]["self_cpu_us"], 0)
            self.assertGreater(report["phases"]["optimizer"]["self_cpu_us"], 0)
            self.assertGreater(report["phases"]["save"]["self_cpu_us"], 0)
            self.assertGreater(report["phases"]["eval"]["self_cpu_us"], 0)
            checkpoint = torch.load(root / "checkpoint/trainable.pt", weights_only=False)
            self.assertNotIn("model", checkpoint)
            self.assertTrue(checkpoint["optimizer"]["state"])
            self.assertEqual(checkpoint["extra"]["stream"]["i"], 6)
            for state in checkpoint["optimizer"]["state"].values():
                self.assertEqual(state["step"], 6)
                self.assertEqual(state["exp_avg"].dtype, torch.float32)
                self.assertEqual(state["exp_avg"].device.type, "cpu")

    def test_cli_requires_bounded_real_data_and_independent_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train.bin", "eval.bin", "source.pt"):
                (root / name).write_bytes(b"fixture")
            base = ["--base", str(root), "--resume", str(root / "source.pt"),
                    "--data", str(root / "train.bin"), "--eval-data", str(root / "eval.bin"),
                    "--out", str(root / "new"), "--eos-id", "1"]
            parser = build_parser()
            args = parser.parse_args(base)
            resume = validate_profile_args(parser, args)
            self.assertEqual(args.run_steps, 6)
            command = benchmark_arguments(args, resume)
            self.assertIn("--save-optim", command)
            self.assertIn("--deterministic-parity", command)
            self.assertIn("--reference-repeat", command)
            for options in (["--profile-warmup", "0"], ["--profile-active", "9"],
                            ["--run-steps", "7"], ["--max-hours", "1"],
                            ["--eos-id", "2"], ["--moe-layout", "compact"], ["--parity-only"]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    validate_profile_args(parser, parser.parse_args(base + options))
            (root / "new").mkdir()
            (root / "new/previous.json").write_text("{}")
            with self.assertRaises(SystemExit):
                validate_profile_args(parser, parser.parse_args(base))

    def test_resume_refuses_missing_or_lossy_adam_and_packed_wrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "train.bin"
            data.write_bytes(b"\0" * 4096 * 4 * 12)
            checkpoint = {
                "kind": "trainable", "trainable": {"adapter.weight": torch.ones(2, 2).bfloat16()},
                "extra": {"phase": "B0", "step": 34900,
                          "stream": {"kind": "packed", "stride": 1, "i": 5, "nseq": 12}},
                "optimizer": {"param_groups": [{"params": [0]}],
                              "state": {0: {"step": 5, "exp_avg": torch.ones(2, 2),
                                            "exp_avg_sq": torch.ones(2, 2)}}},
            }
            path = root / "source.pt"
            torch.save(checkpoint, path)
            self.assertEqual(check_resume(path, data, 4096, 6)["source_adam_step"], 5)
            with self.assertRaisesRegex(ValueError, "wrap"):
                check_resume(path, data, 4096, 8)
            checkpoint["optimizer"]["state"][0]["exp_avg"] = torch.ones(2, 2).bfloat16()
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "CPU FP32"):
                check_resume(path, data, 4096, 6)
            checkpoint.pop("optimizer")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "include Adam"):
                check_resume(path, data, 4096, 6)


if __name__ == "__main__":
    unittest.main()
