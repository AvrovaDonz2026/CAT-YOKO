"""CPU tests for native arithmetic, cache coherence and checkpoint lifetimes."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from cat_yoko.optim import CPUOffloadAdamW
    from operators.rocm.cpu_adam_cached import use_cached_cpu_adam
    from operators.rocm.gpu_adam_bench import compare_optimizers


class CachedSourceTests(unittest.TestCase):
    def test_candidate_module_parses(self):
        ast.parse(Path(__file__).with_name("cpu_adam_cached.py").read_text())


@unittest.skipUnless(HAS_TORCH, "cached CPU Adam arithmetic requires Torch")
class CachedCPUAdamTests(unittest.TestCase):
    def parameters(self, device="cpu"):
        generator = torch.Generator().manual_seed(902)
        # Include a noncontiguous matrix to catch shape/stride changes.
        values = [torch.randn(512, 37, generator=generator).t().to(torch.bfloat16),
                  torch.ones(512, dtype=torch.bfloat16)]
        return [nn.Parameter(value.to(device=device)) for value in values]

    def optimizer(self, parameters, **kwargs):
        return CPUOffloadAdamW([dict(params=parameters[:1], weight_decay=0.1),
                               dict(params=parameters[1:], weight_decay=0.0)], lr=3e-4,
                              state_dtype=kwargs.get("state_dtype", torch.float32),
                              retain_state=kwargs.get("retain_state", True))

    def assign(self, parameters, seed, skip_last=False):
        generator = torch.Generator().manual_seed(seed)
        for index, parameter in enumerate(parameters):
            parameter.grad = None if skip_last and index == 1 else torch.randn(
                parameter.shape, generator=generator).to(device=parameter.device, dtype=torch.bfloat16)

    def assert_match(self, native, cached):
        comparison = compare_optimizers(native, cached, ["matrix", "norm"])
        self.assertTrue(comparison["pass"], comparison["failed_names"])
        self.assertTrue(comparison["all_weights_bitwise"])
        self.assertTrue(comparison["all_moments_bitwise"])

    def test_multistep_native_weights_and_moments_are_byteequal_with_skipped_gradient_and_lr_schedule(self):
        left, right = self.parameters(), self.parameters()
        native, cached = self.optimizer(left), self.optimizer(right)
        identities = [id(p) for p in right]
        with use_cached_cpu_adam(allow_cpu=True, optimizers=[cached]) as installation:
            for step in range(7):
                self.assign(left, 713 + step, skip_last=step == 3)
                self.assign(right, 713 + step, skip_last=step == 3)
                for optimizer in (native, cached):
                    for group in optimizer.param_groups:
                        group["lr"] = 3e-4 / (step + 1)
                    optimizer.step()
                self.assert_match(native, cached)
                self.assertEqual([id(p) for p in right], identities)
                self.assertTrue(all(p.grad is None for p in right))
                self.assertTrue(all(entry[1].device.type == "cpu" and entry[1].dtype == torch.bfloat16
                                    for entry in installation.cache.values()))
            self.assertEqual(installation.parameter_updates, 13)
            self.assertEqual(installation.cache_rebuilds, 2)
            self.assertEqual(installation.cache_reuses, 11)
            self.assertFalse(installation.report()["persistent_fp32_master"])
        self.assertEqual(installation.report()["cache_bytes"], 0)

    def test_external_version_change_and_explicit_alias_invalidation_rebuild_latest_bf16(self):
        left, right = self.parameters(), self.parameters()
        native, cached = self.optimizer(left), self.optimizer(right)
        with use_cached_cpu_adam(allow_cpu=True, optimizers=[cached]) as installation:
            for step in range(3):
                if step == 1:
                    with torch.no_grad():
                        left[0].add_(0.125)
                        right[0].add_(0.125)
                elif step == 2:
                    left[1].data.mul_(0.5)
                    right[1].data.mul_(0.5)
                    installation.invalidate(right[1])
                self.assign(left, 200 + step)
                self.assign(right, 200 + step)
                native.step()
                cached.step()
                self.assert_match(native, cached)
            self.assertEqual(installation.cache_rebuilds, 4)
            self.assertEqual(installation.cache_reuses, 2)

    def test_storage_replacement_is_detected_even_when_parameter_version_does_not_change(self):
        left, right = self.parameters(), self.parameters()
        native, cached = self.optimizer(left), self.optimizer(right)
        with use_cached_cpu_adam(allow_cpu=True, optimizers=[cached]) as installation:
            for step in range(2):
                if step:
                    old_version = right[0]._version
                    replacement = torch.full_like(right[0], 0.25)
                    left[0].data = replacement.clone()
                    right[0].data = replacement.clone()
                    self.assertEqual(right[0]._version, old_version)
                self.assign(left, 400 + step)
                self.assign(right, 400 + step)
                native.step()
                cached.step()
                self.assert_match(native, cached)
            self.assertEqual(installation.cache_rebuilds, 3)

    def test_checkpoint_resume_load_invalidates_cache_without_new_state_keys_or_group_mapping(self):
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        with tempfile.TemporaryDirectory() as directory, use_cached_cpu_adam(allow_cpu=True, optimizers=[optimizer]) as installation:
            self.assign(parameters, 110)
            optimizer.step()
            saved = {"parameters": [p.detach().clone() for p in parameters], "optimizer": optimizer.state_dict()}
            path = Path(directory) / "checkpoint.pt"
            torch.save(saved, path)
            restored = torch.load(path, map_location="cpu", weights_only=False)
            self.assign(parameters, 111)
            optimizer.step()
            for parameter, value in zip(parameters, restored["parameters"]):
                with torch.no_grad():
                    parameter.copy_(value)
            optimizer.load_state_dict(copy.deepcopy(restored["optimizer"]))
            self.assertEqual(installation.report()["cached_parameters"], 0)
            reference_params = [nn.Parameter(value.clone()) for value in restored["parameters"]]
            reference = self.optimizer(reference_params)
            reference.load_state_dict(copy.deepcopy(restored["optimizer"]))
            for params, opt in ((parameters, optimizer), (reference_params, reference)):
                self.assign(params, 111)
                opt.step()
            self.assert_match(reference, optimizer)
            self.assertEqual(installation.state_loads, 1)
            state = optimizer.state_dict()
            self.assertEqual(state["param_groups"], reference.state_dict()["param_groups"])
            self.assertEqual(set(state["state"]), {0, 1})
            for value in state["state"].values():
                self.assertEqual(set(value), {"step", "exp_avg", "exp_avg_sq"})
                self.assertTrue(all(value[key].dtype == torch.float32 and value[key].device.type == "cpu"
                                    for key in ("exp_avg", "exp_avg_sq")))

    def test_step_params_uses_native_fallback_and_later_regular_step_rebuilds(self):
        left, right = self.parameters(), self.parameters()
        native, cached = self.optimizer(left), self.optimizer(right)
        with use_cached_cpu_adam(allow_cpu=True, optimizers=[cached]) as installation:
            for step in range(3):
                self.assign(left, 310 + step)
                self.assign(right, 310 + step)
                if step == 1:
                    native.step_params(iter(left))
                    cached.step_params(iter(right))
                    self.assertEqual(installation.report()["cached_parameters"], 0)
                else:
                    native.step()
                    cached.step()
                self.assert_match(native, cached)
            self.assertEqual(installation.parameter_updates, 4)
            self.assertEqual(installation.fallback_parameter_updates, 2)
            self.assertEqual(installation.cache_rebuilds, 4)

    def test_per_step_bf16_rounding_uses_no_fp32_master(self):
        parameter = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        optimizer = CPUOffloadAdamW([parameter], lr=1e-4, betas=(0, 0), weight_decay=0)
        with use_cached_cpu_adam(allow_cpu=True, optimizers=[optimizer]) as installation:
            for _ in range(64):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
            self.assertEqual(float(parameter), 1.0)
            self.assertEqual(installation.cache_rebuilds, 1)
            self.assertEqual(installation.report()["cache_dtype"], "bfloat16")

    def test_guards_leave_bad_gradients_unconsumed_and_exception_restores_methods_and_clears_cache(self):
        original = (CPUOffloadAdamW._update_one, CPUOffloadAdamW.load_state_dict, CPUOffloadAdamW.step_params,
                    CPUOffloadAdamW.state_dict)
        for policy in ({"state_dtype": torch.float16}, {"retain_state": False}):
            parameters = self.parameters()
            optimizer = self.optimizer(parameters, **policy)
            self.assign(parameters, 19)
            with use_cached_cpu_adam(allow_cpu=True, optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "retained FP32"):
                optimizer.step()
            self.assertTrue(all(p.grad is not None for p in parameters))
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        self.assign(parameters, 22)
        with use_cached_cpu_adam(optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "CUDA/HIP"):
            optimizer.step()
        self.assertTrue(all(p.grad is not None for p in parameters))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with use_cached_cpu_adam(allow_cpu=True, optimizers=[optimizer]) as installation:
                optimizer.step()
                raise RuntimeError("injected")
        self.assertEqual(installation.report()["cache_bytes"], 0)
        self.assertEqual(original, (CPUOffloadAdamW._update_one, CPUOffloadAdamW.load_state_dict,
                                   CPUOffloadAdamW.step_params, CPUOffloadAdamW.state_dict))
        self.assign(parameters, 23)
        optimizer.step()
        self.assertEqual(optimizer.state[parameters[0]]["step"], 2)


if __name__ == "__main__":
    unittest.main()
