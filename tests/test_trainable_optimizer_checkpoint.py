"""Trainable overlays resume CPU Adam moments and the exact next update."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cat_yoko.checkpoint import (
    is_trainable_ckpt, load_checkpoint, load_optimizer_state, load_trainable_state,
    save_trainable_checkpoint,
)
from cat_yoko.optim import CPUOffloadAdamW, adamw_param_groups
from cat_yoko.config import CATYokoConfig
from cat_yoko.trainer import Trainer


class OverlayModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.backbone = nn.Linear(6, 8, bias=False, dtype=dtype)
        self.adapter = nn.Linear(8, 3, dtype=dtype)
        self.backbone.requires_grad_(False)

    def forward(self, x):
        return self.adapter(self.backbone(x).tanh()).float()


def optimizer_for(model):
    return CPUOffloadAdamW(adamw_param_groups(model, 0.03), lr=0.02,
                          betas=(0.8, 0.95), state_dtype=torch.float32)


def update(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    x, target = batch
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


class TrainableOptimizerCheckpointTests(unittest.TestCase):
    def test_requested_optimizer_resume_fails_on_incompatible_state(self):
        model = OverlayModel()
        opt = optimizer_for(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainable.pt"
            save_trainable_checkpoint(path, model=model, extra={"phase": "B0"})
            checkpoint = load_checkpoint(path)
            checkpoint["optimizer"] = {"state": {}, "param_groups": []}
            torch.save(checkpoint, path)
            trainer = Trainer(CATYokoConfig.tiny(), "B0", "cpu", resume=path, save_optim=True)
            trainer.save_optim = True
            with self.assertRaisesRegex(RuntimeError, "could not be restored"):
                trainer._load_resume(model, opt, None)

    def assert_host_moments(self, optimizer):
        self.assertTrue(optimizer.state)
        for state in optimizer.state.values():
            self.assertIsInstance(state["step"], int)
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(state[name].device.type, "cpu")
                self.assertEqual(state[name].dtype, torch.float32)

    def assert_same_training_state(self, expected, expected_opt, actual, actual_opt):
        expected_params = dict(expected.named_parameters())
        actual_params = dict(actual.named_parameters())
        self.assertEqual(set(expected_params), set(actual_params))
        for name, parameter in expected_params.items():
            torch.testing.assert_close(actual_params[name], parameter, atol=0, rtol=0, msg=name)
            if not parameter.requires_grad:
                self.assertNotIn(actual_params[name], actual_opt.state)
                continue
            reference, restored = expected_opt.state[parameter], actual_opt.state[actual_params[name]]
            self.assertEqual(restored["step"], reference["step"])
            for moment in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(restored[moment], reference[moment], atol=0, rtol=0,
                                           msg=f"{name}.{moment}")
        self.assert_host_moments(actual_opt)

    def test_overlay_adam_resume_matches_continuous_fp32_and_bf16_training(self):
        # BF16 parameters expose the base Optimizer loader's lossy BF16 cast
        # of FP32 moments, even though both the checkpoint and Adam live on CPU.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(613)
                continuous = OverlayModel(dtype)
                initial = copy.deepcopy(continuous)
                resumed = copy.deepcopy(initial)
                opt = optimizer_for(continuous)
                batches = [(torch.randn(5, 6).to(dtype), torch.randn(5, 3)) for _ in range(5)]
                for batch in batches[:3]:
                    update(continuous, opt, batch)
                self.assert_host_moments(opt)
                live_moments = [state[name] for state in opt.state.values()
                                for name in ("exp_avg", "exp_avg_sq")]
                pointers = [moment.data_ptr() for moment in live_moments]
                before = [moment.clone() for moment in live_moments]
                path = Path(directory) / "trainable_step_3.pt"
                # Fail immediately if the overlay code accidentally requests
                # the full frozen backbone or serializes its model state.
                with patch("cat_yoko.checkpoint.model_state_dict", side_effect=AssertionError("full model")):
                    save_trainable_checkpoint(path, model=continuous, optimizer=opt,
                                              save_optimizer=True, extra={"step": 3, "phase": "B0"})
                for moment, pointer, value in zip(live_moments, pointers, before):
                    self.assertEqual(moment.data_ptr(), pointer)
                    torch.testing.assert_close(moment, value, atol=0, rtol=0)
                checkpoint = load_checkpoint(path)
                self.assertTrue(is_trainable_ckpt(checkpoint))
                self.assertNotIn("model", checkpoint)
                self.assertEqual(set(checkpoint["trainable"]), {"adapter.weight", "adapter.bias"})
                self.assertEqual(checkpoint["nbytes"], sum(p.numel() * p.element_size()
                                                         for p in continuous.adapter.parameters()))
                self.assertIsNotNone(checkpoint["optimizer"])
                frozen_before = resumed.backbone.weight.detach().clone()
                load_trainable_state(resumed, checkpoint["trainable"])
                restored_opt = optimizer_for(resumed)
                load_optimizer_state(restored_opt, checkpoint["optimizer"])
                torch.testing.assert_close(resumed.backbone.weight, frozen_before, atol=0, rtol=0)
                self.assert_same_training_state(continuous, opt, resumed, restored_opt)
                for batch in batches[3:]:
                    reference_loss = update(continuous, opt, batch)
                    resumed_loss = update(resumed, restored_opt, batch)
                    torch.testing.assert_close(resumed_loss, reference_loss, atol=0, rtol=0)
                    self.assert_same_training_state(continuous, opt, resumed, restored_opt)

    def test_old_overlay_without_optimizer_and_disabled_save_remain_compatible(self):
        torch.manual_seed(73)
        source = OverlayModel()
        opt = optimizer_for(source)
        update(source, opt, (torch.randn(4, 6), torch.randn(4, 3)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trainable.pt"
            # The original call signature remains valid and excludes moments.
            save_trainable_checkpoint(path, model=source, extra={"step": 1})
            checkpoint = load_checkpoint(path)
            self.assertIsNone(checkpoint.get("optimizer"))
            checkpoint.pop("optimizer", None)
            torch.save(checkpoint, path)
            legacy = load_checkpoint(path)
            restored = copy.deepcopy(source)
            restored_opt = optimizer_for(restored)
            load_trainable_state(restored, legacy["trainable"])
            load_optimizer_state(restored_opt, legacy.get("optimizer"))
            self.assertFalse(restored_opt.state)
            update(restored, restored_opt, (torch.randn(4, 6), torch.randn(4, 3)))
            self.assert_host_moments(restored_opt)
            with patch.object(opt, "state_dict", side_effect=AssertionError("disabled optimizer save")):
                save_trainable_checkpoint(path, model=source, optimizer=opt,
                                          save_optimizer=False, extra={"step": 1})
            self.assertIsNone(load_checkpoint(path)["optimizer"])

    def test_trainer_overlay_saves_adam_for_step_and_direct_latest(self):
        model = OverlayModel()
        opt = optimizer_for(model)
        update(model, opt, (torch.randn(4, 6), torch.randn(4, 3)))
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(Trainer)
            trainer.save_dir = Path(directory)
            trainer.save_trainable = True
            trainer.save_full = False
            trainer.save_optim = True
            trainer.save_keep = 2
            trainer.save_every_seconds = 0.0
            trainer.rank = 0
            trainer._trainable_names = None
            trainer._maybe_save(model, opt, {"step": 4}, "step_4.pt")
            step = trainer.save_dir / "trainable_step_4.pt"
            latest = trainer.save_dir / "trainable.pt"
            self.assertTrue(step.samefile(latest))
            self.assertIsNotNone(load_checkpoint(step)["optimizer"])
            # Exercise latest when no step file exists (not just the hardlink).
            trainer._maybe_save(model, opt, {"step": 5}, "latest.pt")
            self.assertEqual(load_checkpoint(latest)["extra"]["step"], 5)
            self.assertIsNotNone(load_checkpoint(latest)["optimizer"])
            trainer.save_optim = False
            trainer._maybe_save(model, opt, {"step": 6}, "step_6.pt")
            self.assertIsNone(load_checkpoint(trainer.save_dir / "trainable_step_6.pt")["optimizer"])

    def test_cpu_loader_preserves_hooks_group_mapping_and_legacy_tensor_step(self):
        model = OverlayModel(torch.bfloat16)
        opt = optimizer_for(model)
        update(model, opt, (torch.randn(4, 6).bfloat16(), torch.randn(4, 3)))
        state = copy.deepcopy(opt.state_dict())
        for value in state["state"].values():
            value["step"] = torch.tensor(value["step"], dtype=torch.float32)
        restored = copy.deepcopy(model)
        restored_opt = optimizer_for(restored)
        events = []
        restored_opt.register_load_state_dict_pre_hook(lambda *args: events.append("pre"))
        restored_opt.register_load_state_dict_post_hook(lambda *args: events.append("post"))
        load_optimizer_state(restored_opt, state)
        self.assertEqual(events, ["pre", "post"])
        self.assert_same_training_state(model, opt, restored, restored_opt)
        bad_groups = copy.deepcopy(state)
        bad_groups["param_groups"].pop()
        with self.assertRaisesRegex(ValueError, "different number"):
            restored_opt.load_state_dict(bad_groups)
        bad_size = copy.deepcopy(state)
        bad_size["param_groups"][0]["params"].append(1234567)
        with self.assertRaisesRegex(ValueError, "doesn't match"):
            restored_opt.load_state_dict(bad_size)


if __name__ == "__main__":
    unittest.main()
