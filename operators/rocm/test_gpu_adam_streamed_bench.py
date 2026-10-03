"""Small CPU sources exercise streamed arithmetic and unchanged source state."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from operators.rocm import gpu_adam_streamed_bench as benchmark


class StreamedSourceTests(unittest.TestCase):
    def test_standalone_source_parses(self):
        ast.parse(Path(__file__).with_name("gpu_adam_streamed_bench.py").read_text())


@unittest.skipUnless(HAS_TORCH, "streamed arithmetic requires Torch")
class StreamedAdamTests(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(317)
        weights = {"projection.weight": torch.randn(13, 17, generator=generator).to(torch.bfloat16),
                   "norm.weight": torch.ones(17, dtype=torch.bfloat16)}
        states = {i: {"step": 8689, "exp_avg": torch.randn(value.shape, generator=generator).mul_(1e-5),
                      "exp_avg_sq": torch.rand(value.shape, generator=generator).mul_(1e-5)}
                  for i, value in enumerate(weights.values())}
        groups = [dict(params=[i], lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=decay)
                  for i, decay in enumerate((0.1, 0.0))]
        return {"kind": "trainable", "trainable": weights, "optimizer": {"state": states, "param_groups": groups},
                "extra": {"step": 43491, "phase": "B0", "name": "CAT-YOKO-12B"}}

    def test_all_five_steps_cover_every_parameter_and_source_moments_remain_unchanged(self):
        source = self.fixture()
        before = copy.deepcopy(source)
        report = benchmark.probe(source, device="cpu", expected_parameters=2)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["parameters_checked"], 2)
        self.assertEqual(report["exported_moments_checked"], 4)
        self.assertFalse(report["timing_measured"])
        self.assertFalse(report["whole_training_speedup_measured"])
        self.assertTrue(report["checkpoint_moments_cpu_fp32"])
        for row in report["parity"]:
            self.assertEqual(row["parameters"], 2)
            self.assertEqual(set(row["tensors"]), set(source["trainable"]))
            self.assertTrue(row["pass"])
            self.assertTrue(row["all_weights_bitwise"])
            self.assertTrue(all(value["steps_match"] for value in row["tensors"].values()))
        self.assertEqual(source["optimizer"]["param_groups"], before["optimizer"]["param_groups"])
        for name, parameter in source["trainable"].items():
            self.assertTrue(torch.equal(parameter, before["trainable"][name]))
        for identifier, state in source["optimizer"]["state"].items():
            self.assertEqual(state["step"], 8689)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(state[key], before["optimizer"]["state"][identifier][key]))

    def test_source_validation_rejects_wrong_shape_dtype_counter_mapping_or_square(self):
        for mutate in (lambda r: r["optimizer"]["state"][0].update(exp_avg=torch.zeros(1)),
                       lambda r: r["optimizer"]["state"][0].update(exp_avg=torch.zeros(13, 17, dtype=torch.bfloat16)),
                       lambda r: r["optimizer"]["state"][1].update(step=8690),
                       lambda r: r["optimizer"]["param_groups"][1].update(params=[0]),
                       lambda r: r["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1)):
            source = self.fixture()
            mutate(source)
            with self.assertRaises(ValueError):
                benchmark.validate_source(source, expected_parameters=2)

    def test_first_mismatch_is_incomplete_and_cannot_be_mistaken_for_full_gate(self):
        compare = benchmark.compare_optimizers
        def reject(*args, **kwargs):
            result = compare(*args, **kwargs)
            result.update({"pass": False, "all_weights_bitwise": False, "failed_names": ["projection.weight"]})
            return result
        with patch.object(benchmark, "compare_optimizers", side_effect=reject):
            report = benchmark.probe(self.fixture(), device="cpu", expected_parameters=2)
        self.assertEqual(report["status"], "parity_failure")
        self.assertEqual(report["failed_invocation_step"], 1)
        self.assertTrue(report["incomplete_parameter"])
        self.assertEqual(report["parameters_checked"], 0)
        self.assertEqual(report["parity"][0]["parameters"], 1)
        self.assertEqual(report["parity"][1]["parameters"], 0)


if __name__ == "__main__":
    unittest.main()
