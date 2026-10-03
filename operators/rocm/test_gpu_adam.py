"""Native-order Adam arithmetic, checkpoint snapshots and context lifetimes."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from cat_yoko.optim import CPUOffloadAdamW
    from operators.rocm.gpu_adam import use_gpu_fp32_adam
    from operators.rocm.gpu_adam_bench import compare_optimizers, main as benchmark_main


@unittest.skipUnless(HAS_TORCH, "Adam arithmetic tests require Torch")
class GPUAdamTests(unittest.TestCase):
    def parameters(self, device="cpu"):
        generator = torch.Generator(device="cpu").manual_seed(739)
        return [nn.Parameter(torch.randn(7, 5, generator=generator).to(device=device, dtype=torch.bfloat16)),
                nn.Parameter(torch.ones(5, device=device, dtype=torch.bfloat16))]

    def optimizer(self, parameters, **kwargs):
        return CPUOffloadAdamW([{"params": parameters[:1], "weight_decay": 0.1},
                               {"params": parameters[1:], "weight_decay": 0.0}],
                              lr=3e-4, state_dtype=kwargs.get("state_dtype", torch.float32),
                              retain_state=kwargs.get("retain_state", True))

    def assign(self, parameters, seed, skip_last=False):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for index, parameter in enumerate(parameters):
            parameter.grad = None if skip_last and index == len(parameters) - 1 else (
                torch.randn(parameter.shape, generator=generator).to(device=parameter.device, dtype=torch.bfloat16))

    def pair(self, device="cpu", steps=7):
        left, right = self.parameters(device), self.parameters(device)
        native, candidate = self.optimizer(left), self.optimizer(right)
        identities = [id(parameter) for parameter in right]
        history = []
        with use_gpu_fp32_adam(allow_cpu=device == "cpu", optimizers=[candidate]) as installation:
            for step in range(steps):
                self.assign(left, 800 + step, skip_last=step == 2)
                self.assign(right, 800 + step, skip_last=step == 2)
                for optimizer in (native, candidate):
                    for group in optimizer.param_groups:
                        group["lr"] = 3e-4 * (1 - step / (steps + 1))
                    optimizer.step()
                comparison = compare_optimizers(native, candidate, ["matrix", "norm"])
                history.append(comparison)
                self.assertEqual([id(parameter) for parameter in right], identities)
                self.assertTrue(all(parameter.grad is None for parameter in right))
                self.assertTrue(all(state[key].dtype == torch.float32 and state[key].device.type == torch.device(device).type
                    for state in candidate.state.values() for key in ("exp_avg", "exp_avg_sq")))
            self.assertEqual(installation.parameter_updates, steps * 2 - 1)
        self.assertTrue(all(state[key].device.type == "cpu" for state in candidate.state.values()
                            for key in ("exp_avg", "exp_avg_sq")))
        return history

    def test_cpu_matches_all_parameter_and_moment_bytes_across_multi_step_schedule(self):
        for row in self.pair():
            self.assertTrue(row["pass"])
            self.assertTrue(row["all_weights_bitwise"])
            self.assertTrue(row["all_moments_bitwise"])

    def test_bf16_rounding_occurs_each_step_without_persistent_master(self):
        parameter = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        optimizer = CPUOffloadAdamW([parameter], lr=1e-4, betas=(0.0, 0.0), weight_decay=0.0)
        with use_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]):
            for _ in range(64):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
        self.assertEqual(float(parameter), 1.0)
        self.assertEqual(optimizer.state[parameter]["step"], 64)
        self.assertEqual(set(optimizer.state[parameter]), {"step", "exp_avg", "exp_avg_sq"})

    def test_export_is_independent_cpu_fp32_and_resume_matches_continuous_training(self):
        original = self.parameters()
        continuous = self.optimizer(original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.pt"
            with use_gpu_fp32_adam(allow_cpu=True, optimizers=[continuous]):
                for step in range(3):
                    self.assign(original, 51 + step)
                    continuous.step()
                snapshot = continuous.state_dict()
                torch.save({"parameters": [p.detach().clone() for p in original], "optimizer": snapshot}, path)
                snapshot["state"][0]["exp_avg"].zero_()
                self.assertFalse(torch.equal(snapshot["state"][0]["exp_avg"], continuous.state[original[0]]["exp_avg"]))
            saved = torch.load(path, map_location="cpu", weights_only=False)
            resumed_params = [nn.Parameter(value.clone()) for value in saved["parameters"]]
            resumed = self.optimizer(resumed_params)
            with use_gpu_fp32_adam(allow_cpu=True, optimizers=[continuous, resumed]) as installation:
                resumed.load_state_dict(saved["optimizer"])
                for step in range(3, 6):
                    for parameters, optimizer in ((original, continuous), (resumed_params, resumed)):
                        self.assign(parameters, 51 + step)
                        optimizer.step()
                    self.assertTrue(compare_optimizers(continuous, resumed, ["matrix", "norm"])["pass"])
                self.assertEqual(installation.state_loads, 1)
                self.assertEqual(resumed.param_groups[0]["weight_decay"], 0.1)
                self.assertEqual(resumed.param_groups[1]["weight_decay"], 0.0)

    def test_invalid_policy_parameter_or_loaded_moments_reject_without_consuming_gradient(self):
        for options in ({"state_dtype": torch.float16}, {"retain_state": False}):
            parameters = self.parameters()
            optimizer = self.optimizer(parameters, **options)
            self.assign(parameters, 83)
            before = [p.clone() for p in parameters]
            with use_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "retained FP32"):
                optimizer.step()
            self.assertTrue(all(p.grad is not None and torch.equal(p, old) for p,old in zip(parameters,before)))
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        self.assign(parameters, 93)
        with use_gpu_fp32_adam(optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "CUDA/HIP"):
            optimizer.step()
        self.assertTrue(all(p.grad is not None for p in parameters))
        optimizer.step()
        valid = optimizer.state_dict()
        for change in (lambda state: state["state"][0].update(exp_avg=torch.zeros(7, 5, dtype=torch.bfloat16)),
                       lambda state: state["state"][0].update(exp_avg=torch.zeros(1, dtype=torch.float32))):
            bad = copy.deepcopy(valid)
            change(bad)
            with use_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "FP32 with the parameter shape"):
                optimizer.load_state_dict(bad)

    def test_context_restores_methods_on_exception_and_native_can_continue_after_exit(self):
        original_methods = (CPUOffloadAdamW._update_one, CPUOffloadAdamW.state_dict, CPUOffloadAdamW.load_state_dict)
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with use_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]):
                with use_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]):
                    self.assign(parameters, 80)
                    optimizer.step()
                raise RuntimeError("injected")
        self.assertEqual(original_methods, (CPUOffloadAdamW._update_one, CPUOffloadAdamW.state_dict, CPUOffloadAdamW.load_state_dict))
        self.assign(parameters, 81)
        optimizer.step()
        self.assertEqual(optimizer.state[parameters[0]]["step"], 2)

    def test_cpu_benchmark_generates_real_numeric_and_timing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adam.json"
            self.assertEqual(benchmark_main(["--device", "cpu", "--json", str(path), "--steps", "3",
                                              "--warmup", "1", "--repeats", "2"]), 0)
            report = json.loads(path.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["whole_training_speedup_measured"])
            self.assertFalse(report["persistent_fp32_master"])
            self.assertTrue(report["checkpoint_moments_cpu_fp32"])
            self.assertEqual(len(report["parity"]), 3)
            self.assertTrue(all(row["all_moments_bitwise"] for row in report["parity"]))
            self.assertGreater(report["candidate"]["wall_ms_median"], 0)

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), "CUDA/HIP runtime required")
    def test_gpu_multi_step_numeric_gate_and_state_cleanup(self):
        history = self.pair("cuda")
        # Strict gate intentionally rejects GPU arithmetic if weights differ.
        for row in history:
            self.assertTrue(row["pass"], row["failed_names"])


if __name__ == "__main__":
    unittest.main()
