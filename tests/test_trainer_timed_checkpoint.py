"""Time-triggered saves preserve completed updates, optimizer state and cursor."""

from __future__ import annotations

import hashlib
import math
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import PackedBinStream
from cat_yoko.trainer import Trainer


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class _ClockedTrainer(Trainer):
    """A real tiny model; only elapsed step/save time is synthetic."""

    def __init__(self, *args, clock, step_seconds, save_seconds=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = clock
        self.step_seconds = iter(step_seconds)
        self.save_seconds = save_seconds
        self.periodic_steps = []

    def _forward_loss(self, model, batch):
        result = super()._forward_loss(model, batch)
        self.clock.now += next(self.step_seconds)
        return result

    def _maybe_save(self, model, opt, extra, tag):
        super()._maybe_save(model, opt, extra, tag)
        if tag.startswith("step_"):
            self.periodic_steps.append(extra["step"])
            self.clock.now += self.save_seconds


class TrainerTimedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(CATYokoConfig.tiny(), warmup_tokens=64)

    def _data(self, root):
        path = root / "tokens.bin"
        tokens = [2 + i % (self.cfg.vocab_size - 2) for i in range(self.cfg.seq_len * 12)]
        path.write_bytes(struct.pack("<" + "i" * len(tokens), *tokens))
        return path

    def _args(self, data, output, **overrides):
        options = dict(
            run_steps=3, tokens=4096, data=data, seed=39, micro_batch=1,
            accum=1, optim_cpu=True, grad_ckpt=True, save_dir=output,
            save_trainable=True, save_full=False, save_optim=True,
            log_every=1000,
        )
        options.update(overrides)
        return options

    def _load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def _check_adam(self, checkpoint, expected_steps):
        optimizer = checkpoint["optimizer"]
        parameters = [pid for group in optimizer["param_groups"] for pid in group["params"]]
        self.assertEqual(len(parameters), len(set(parameters)))
        self.assertEqual(set(parameters), set(optimizer["state"]))
        self.assertEqual(len(parameters), len(checkpoint["trainable"]))
        for state in optimizer["state"].values():
            self.assertEqual(int(state["step"]), expected_steps)
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(value.device.type, "cpu")
                self.assertTrue(torch.isfinite(value).all())

    def _assert_equal_state(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_equal_state(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self._assert_equal_state(a, b)
        else:
            self.assertEqual(left, right)

    def test_timed_checkpoint_resumes_exact_next_update_with_fp32_adam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self._data(root)
            clock = _Clock()
            continuous = root / "continuous"
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[6, 5, 6],
                save_every_seconds=10, **self._args(data, continuous),
            )
            with patch("cat_yoko.trainer.time.monotonic", clock):
                result = trainer.run()
            # The first save takes 20s. Its completion, rather than start,
            # anchors the next interval, so step 3 does not immediately save.
            self.assertEqual(trainer.periodic_steps, [2])
            self.assertEqual(trainer._last_checkpoint_time, 131.0)
            self.assertTrue((continuous / "trainable.pt").samefile(continuous / "trainable_step_3.pt"))
            source = continuous / "trainable_step_2.pt"
            periodic = self._load(source)
            self.assertEqual(periodic["extra"]["stream"]["i"], 2)
            self.assertEqual(periodic["extra"]["tokens_in_phase"], self.cfg.seq_len * 2)
            self._check_adam(periodic, 2)
            expected = PackedBinStream(data, self.cfg.seq_len)
            expected.batch(1, "cpu")
            expected.batch(1, "cpu")
            restored = PackedBinStream(data, self.cfg.seq_len)
            restored.load_state_dict(periodic["extra"]["stream"])
            self._assert_equal_state(expected.batch(1, "cpu"), restored.batch(1, "cpu"))

            resumed = root / "resumed"
            resumed_trainer = Trainer(
                self.cfg, "B0", "cpu",
                **self._args(data, resumed, run_steps=1, resume=source),
            )
            resumed_result = resumed_trainer.run()
            self.assertTrue(resumed_trainer.optimizer_restored)
            self.assertEqual(resumed_result.step, result.step)
            uninterrupted = self._load(continuous / "trainable.pt")
            restarted = self._load(resumed / "trainable.pt")
            self._check_adam(restarted, 3)
            for key in ("trainable", "optimizer"):
                self._assert_equal_state(uninterrupted[key], restarted[key])
            for key in ("step", "tokens_seen", "tokens_in_phase", "stream", "gate"):
                self._assert_equal_state(uninterrupted["extra"][key], restarted["extra"][key])

    def test_step_cadence_also_resets_successful_time_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clock = _Clock()
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[3, 3, 8, 3],
                save_every_seconds=10, save_every=2,
                **self._args(self._data(root), root / "save", run_steps=4),
            )
            with patch("cat_yoko.trainer.time.monotonic", clock):
                trainer.run()
            self.assertEqual(trainer.periodic_steps, [2, 4])
            self.assertEqual(trainer._last_checkpoint_time, 157.0)

    def test_failed_timed_save_keeps_prior_checkpoint_and_timer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = self._data(root), root / "save"
            Trainer(
                self.cfg, "B0", "cpu",
                **self._args(data, output, run_steps=1, save_every=1, save_keep=1),
            ).run()
            old_path = output / "trainable_step_1.pt"
            old_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
            latest = output / "trainable.pt"
            clock = _Clock()
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[11],
                save_every_seconds=10,
                **self._args(data, output, run_steps=1, resume=old_path, save_keep=1),
            )
            original_save = torch.save

            def fail_new_checkpoint(payload, path, *args, **kwargs):
                if Path(path).name == "trainable_step_2.pt.tmp":
                    Path(path).write_bytes(b"interrupted serialization")
                    raise OSError("injected save failure")
                return original_save(payload, path, *args, **kwargs)

            with patch("cat_yoko.trainer.time.monotonic", clock), patch(
                "cat_yoko.checkpoint.torch.save", side_effect=fail_new_checkpoint
            ):
                with self.assertRaisesRegex(OSError, "injected save failure"):
                    trainer.run()
            self.assertEqual(trainer._last_checkpoint_time, 100.0)
            self.assertEqual(trainer.periodic_steps, [])
            self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), old_hash)
            self.assertTrue(latest.samefile(old_path))
            self.assertFalse((output / "trainable_step_2.pt").exists())
            self._check_adam(self._load(latest), 1)

    def test_time_mode_final_save_prunes_only_after_new_checkpoint_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = self._data(root), root / "save"
            Trainer(
                self.cfg, "B0", "cpu",
                **self._args(data, output, run_steps=2, save_every=1),
            ).run()
            source = output / "trainable_step_2.pt"
            old_paths = [output / "trainable_step_1.pt", source]
            clock = _Clock()
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[1],
                save_every_seconds=10,
                **self._args(data, output, run_steps=1, resume=source, save_keep=1),
            )
            from cat_yoko.checkpoint import publish_latest

            def observed_publish(step_path, latest_path=None):
                self.assertTrue(all(path.is_file() for path in old_paths))
                new_state = self._load(step_path)
                self.assertEqual(new_state["extra"]["step"], 3)
                self.assertEqual(new_state["extra"]["stream"]["i"], 3)
                self._check_adam(new_state, 3)
                return publish_latest(step_path, latest_path)

            with patch("cat_yoko.trainer.time.monotonic", clock), patch(
                "cat_yoko.trainer.publish_latest", side_effect=observed_publish
            ) as publication:
                trainer.run()
            self.assertEqual(publication.call_count, 1)
            self.assertEqual(
                [path.name for path in output.glob("trainable_step_*.pt")],
                ["trainable_step_3.pt"],
            )
            self.assertTrue((output / "trainable.pt").samefile(output / "trainable_step_3.pt"))
            self.assertFalse(any(path.exists() for path in old_paths))

    def test_failed_time_mode_final_save_retains_resumed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = self._data(root), root / "save"
            Trainer(
                self.cfg, "B0", "cpu",
                **self._args(data, output, run_steps=2, save_every=1),
            ).run()
            source = output / "trainable_step_2.pt"
            original_bytes = source.read_bytes()
            clock = _Clock()
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[1],
                save_every_seconds=10,
                **self._args(data, output, run_steps=1, resume=source, save_keep=1),
            )
            with patch("cat_yoko.trainer.time.monotonic", clock), patch(
                "cat_yoko.checkpoint.torch.save", side_effect=OSError("final save failure")
            ):
                with self.assertRaisesRegex(OSError, "final save failure"):
                    trainer.run()
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertTrue((output / "trainable.pt").samefile(source))
            self.assertTrue((output / "trainable_step_1.pt").is_file())
            self.assertFalse((output / "trainable_step_3.pt").exists())
            self.assertEqual(trainer._last_checkpoint_time, 100.0)

    def test_timed_full_only_publishes_before_pruning_and_directory_resume_is_current(self):
        from cat_yoko.checkpoint import publish_latest, resolve_resume_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = self._data(root), root / "full"
            Trainer(
                self.cfg, "B0", "cpu",
                **self._args(
                    data, output, run_steps=1, save_every=1, save_keep=1,
                    save_trainable=False, save_full=True,
                ),
            ).run()
            latest = output / "latest.pt"
            previous = output / "step_1.pt"
            self.assertTrue(latest.samefile(previous))
            clock = _Clock()
            trainer = _ClockedTrainer(
                self.cfg, "B0", "cpu", clock=clock, step_seconds=[6, 5, 6],
                save_every_seconds=10,
                **self._args(
                    data, output, run_steps=3, resume=output, save_keep=1,
                    save_trainable=False, save_full=True,
                ),
            )
            published_steps = []

            def observed_publish(step_path, latest_path=None):
                nonlocal previous
                self.assertEqual(Path(latest_path), latest)
                self.assertTrue(previous.is_file())
                self.assertTrue(latest.samefile(previous))
                checkpoint = self._load(step_path)
                step = checkpoint["extra"]["step"]
                self.assertEqual(checkpoint["extra"]["stream"]["i"], step)
                self.assertIn("model", checkpoint)
                self.assertNotIn("trainable", checkpoint)
                self.assertTrue(checkpoint["optimizer"])
                for state in checkpoint["optimizer"]["state"].values():
                    self.assertEqual(int(state["step"]), step)
                result = publish_latest(step_path, latest_path)
                self.assertTrue(latest.samefile(step_path))
                self.assertEqual(self._load(resolve_resume_path(output))["extra"]["step"], step)
                published_steps.append(step)
                previous = Path(step_path)
                return result

            with patch("cat_yoko.trainer.time.monotonic", clock), patch(
                "cat_yoko.trainer.publish_latest", side_effect=observed_publish
            ):
                result = trainer.run()
            self.assertEqual(result.step, 4)
            self.assertEqual(trainer.periodic_steps, [3])
            self.assertEqual(published_steps, [3, 4])
            self.assertEqual([path.name for path in output.glob("step_*.pt")], ["step_4.pt"])
            self.assertFalse(list(output.glob("trainable*.pt")))
            self.assertTrue(latest.samefile(output / "step_4.pt"))
            self.assertEqual(self._load(resolve_resume_path(output))["extra"]["step"], 4)

    def test_disabled_timer_preserves_step_cadence_without_extra_clock_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = Trainer(
                self.cfg, "B0", "cpu", save_every_seconds=0, save_every=2,
                **self._args(self._data(root), root / "save"),
            )
            with patch("cat_yoko.trainer.time.monotonic", return_value=100.0) as clock:
                trainer.run()
            self.assertEqual(clock.call_count, 1)
            self.assertIsNone(trainer._last_checkpoint_time)
            self.assertEqual(
                [path.name for path in (root / "save").glob("trainable_step_*.pt")],
                ["trainable_step_2.pt"],
            )

    def test_timer_rejects_invalid_values_and_distributed_clock(self):
        for seconds in (-1.0, math.nan, math.inf, -math.inf):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                Trainer(self.cfg, "B0", "cpu", save_every_seconds=seconds)
        with patch("cat_yoko.trainer.init_distributed", return_value=("cpu", 0, 2)):
            with self.assertRaisesRegex(ValueError, "single process"):
                Trainer(self.cfg, "B0", "cpu", save_every_seconds=300)


if __name__ == "__main__":
    unittest.main()
