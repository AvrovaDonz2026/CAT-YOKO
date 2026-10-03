"""CPU arithmetic, unchanged gates and ordered-context lifecycle checks."""
from __future__ import annotations

import ast
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
    from operators.rocm import gpu_adam
    from operators.rocm.gpu_adam_bench import compare_optimizers
    from operators.rocm.gpu_adam_ordered import use_ordered_gpu_fp32_adam
    from operators.rocm.gpu_adam_ordered_bench import main as benchmark_main


class OrderedSourceTests(unittest.TestCase):
    def test_new_modules_parse_without_changing_the_original_implementation(self):
        directory = Path(__file__).parent
        for name in ("gpu_adam_ordered.py", "gpu_adam_ordered_bench.py"):
            ast.parse((directory / name).read_text(), filename=name)


@unittest.skipUnless(HAS_TORCH, "ordered arithmetic requires Torch")
class OrderedAdamTests(unittest.TestCase):
    def parameters(self):
        generator = torch.Generator().manual_seed(943)
        return [nn.Parameter(torch.randn(shape, generator=generator).to(torch.bfloat16))
                for shape in ((512,), (37, 512))]

    def optimizer(self, parameters):
        return CPUOffloadAdamW([{"params": parameters[:1], "weight_decay": 0.0},
                               {"params": parameters[1:], "weight_decay": 0.1}], lr=3e-4,
                              state_dtype=torch.float32, retain_state=True)

    def assign(self, parameters, step):
        generator = torch.Generator().manual_seed(741 + step)
        for parameter in parameters:
            parameter.grad = torch.randn(parameter.shape, generator=generator).to(torch.bfloat16)

    def test_explicit_cpu_formula_matches_native_gate_on_vector_and_matrix_multistep(self):
        left, right = self.parameters(), self.parameters()
        native, ordered = self.optimizer(left), self.optimizer(right)
        identities = [id(parameter) for parameter in right]
        with use_ordered_gpu_fp32_adam(allow_cpu=True, optimizers=[ordered]) as installation:
            for step in range(7):
                self.assign(left, step)
                self.assign(right, step)
                for optimizer in (native, ordered):
                    for group in optimizer.param_groups:
                        group["lr"] = 3e-4 / (step + 1)
                    optimizer.step()
                result = compare_optimizers(native, ordered, ["vector", "matrix"])
                self.assertTrue(result["pass"], result["failed_names"])
                self.assertTrue(result["all_weights_bitwise"])
                self.assertEqual([id(parameter) for parameter in right], identities)
                self.assertTrue(all(parameter.grad is None for parameter in right))
            self.assertEqual(installation.parameter_updates, 14)

    def test_checkpoint_roundtrip_and_exception_restore_original_context(self):
        parameters = self.parameters()
        optimizer = self.optimizer(parameters)
        original = (gpu_adam.update_one_on_parameter_device, CPUOffloadAdamW._update_one,
                    CPUOffloadAdamW.state_dict, CPUOffloadAdamW.load_state_dict)
        with use_ordered_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]):
            self.assign(parameters, 0)
            optimizer.step()
            snapshot = optimizer.state_dict()
            saved_parameters = [parameter.detach().clone() for parameter in parameters]
            independent = copy.deepcopy(snapshot)
            snapshot["state"][0]["exp_avg"].zero_()
            self.assertFalse(torch.equal(snapshot["state"][0]["exp_avg"], optimizer.state[parameters[0]]["exp_avg"]))
        resumed_parameters = [nn.Parameter(value) for value in saved_parameters]
        resumed = self.optimizer(resumed_parameters)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with use_ordered_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer, resumed]) as installation:
                resumed.load_state_dict(independent)
                self.assign(parameters, 1)
                self.assign(resumed_parameters, 1)
                optimizer.step()
                resumed.step()
                self.assertTrue(compare_optimizers(optimizer, resumed, ["vector", "matrix"])["pass"])
                self.assertEqual(installation.state_loads, 1)
                raise RuntimeError("injected")
        self.assertEqual(original, (gpu_adam.update_one_on_parameter_device, CPUOffloadAdamW._update_one,
                                   CPUOffloadAdamW.state_dict, CPUOffloadAdamW.load_state_dict))
        for values in resumed.state.values():
            self.assertEqual(set(values), {"step", "exp_avg", "exp_avg_sq"})
            self.assertTrue(all(values[key].device.type == "cpu" and values[key].dtype == torch.float32
                                for key in ("exp_avg", "exp_avg_sq")))

    def test_each_step_bf16_rounding_and_existing_policy_guard_remain(self):
        parameter = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        optimizer = CPUOffloadAdamW([parameter], lr=1e-4, betas=(0.0, 0.0), weight_decay=0)
        with use_ordered_gpu_fp32_adam(allow_cpu=True, optimizers=[optimizer]):
            for _ in range(64):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
        self.assertEqual(float(parameter.detach()), 1.0)
        parameter.grad = torch.ones_like(parameter)
        with use_ordered_gpu_fp32_adam(optimizers=[optimizer]), self.assertRaisesRegex(ValueError, "CUDA/HIP"):
            optimizer.step()
        self.assertIsNotNone(parameter.grad)

    def test_benchmark_preserves_numeric_gate_and_records_ordered_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordered.json"
            code = benchmark_main(["--device", "cpu", "--json", str(path), "--steps", "5", "--warmup", "1", "--repeats", "2"])
            report = json.loads(path.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["arithmetic_variant"], "ordered-fp32")
            self.assertEqual(set(report["ordered_source_sha256"]), {
                "operators/rocm/gpu_adam_ordered.py", "operators/rocm/gpu_adam_ordered_bench.py",
                "operators/rocm/gpu_adam.py", "operators/rocm/gpu_adam_bench.py", "cat_yoko/optim.py"})
            self.assertEqual(report["optimizer_source_sha256"], report["ordered_source_sha256"])
            self.assertTrue(report["checkpoint_moments_cpu_fp32"])
            self.assertFalse(report["persistent_fp32_master"])
            for row in report["parity"]:
                self.assertTrue(row["all_weights_bitwise"])
                self.assertEqual(row["gate"]["fp32_moment_atol"], 1e-8)
                self.assertEqual(row["gate"]["fp32_moment_rtol"], 1e-6)


if __name__ == "__main__":
    unittest.main()
