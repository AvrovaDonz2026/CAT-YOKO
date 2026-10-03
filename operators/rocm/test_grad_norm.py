"""Compare candidate clipping against native rounding and every gradient byte."""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from cat_yoko.offload import clip_grad_norm_mixed as native_clip
from operators.rocm.grad_norm import clip_grad_norm_mixed, use_batched_grad_norm


def parameters(rows):
    result = []
    for grad in rows:
        if grad is None:
            result.append(nn.Parameter(torch.ones(1)))
            continue
        parameter = nn.Parameter(torch.zeros_like(grad))
        parameter.grad = grad.clone()
        result.append(parameter)
    return result


class BatchedGradNormTests(unittest.TestCase):
    def assert_matches_native(self, rows, max_norm):
        reference, candidate = parameters(rows), parameters(rows)
        expected = native_clip(iter(reference), max_norm)
        actual = clip_grad_norm_mixed(iter(candidate), max_norm)
        if math.isnan(expected):
            self.assertTrue(math.isnan(actual))
        else:
            self.assertEqual(actual, expected)
        self.assertIsInstance(actual, float)
        for index, (left, right) in enumerate(zip(reference, candidate)):
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                # Compare raw storage, including NaN payloads and signed zero.
                self.assertTrue(torch.equal(left.grad.contiguous().view(torch.uint8),
                                            right.grad.contiguous().view(torch.uint8)), index)
        return actual

    def test_mixed_dtypes_and_clipping_boundaries_match_every_byte(self):
        generator = torch.Generator().manual_seed(876)
        rows = [torch.randn(7, 5, generator=generator).to(dtype)
                for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64)]
        rows.insert(1, None)
        total = native_clip(parameters(rows), float("inf"))
        for threshold in (0.0, 0.25, total - 1e-6, total, total + 1e-6, total + 1.0, float("inf")):
            with self.subTest(threshold=threshold):
                self.assert_matches_native(rows, threshold)

    def test_bf16_norm_is_not_recomputed_in_fp32(self):
        grad = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        bf16_norm = float(grad.norm(2))
        self.assertNotEqual(bf16_norm, float(grad.float().norm(2)))
        self.assertEqual(self.assert_matches_native([grad], 0.5), bf16_norm)

    def test_python_parameter_order_is_preserved_after_device_grouping(self):
        # Adding many unit squares after a 1e16 square loses them; reversing
        # that order keeps them. A device sum or reordered CPU sum is different.
        rows = [torch.tensor([1e8], dtype=torch.float64)]
        rows.extend(torch.ones(1, dtype=torch.float64) for _ in range(32))
        first = self.assert_matches_native(rows, 0.1)
        last = self.assert_matches_native(list(reversed(rows)), 0.1)
        self.assertNotEqual(first, last)

    def test_empty_none_zero_and_noncontiguous_gradients(self):
        for rows in ([], [None, None], [torch.empty(0)], [torch.zeros(2, dtype=torch.bfloat16)],
                     [torch.arange(12.0).reshape(3, 4).t()]):
            with self.subTest(rows=len(rows)):
                self.assert_matches_native(rows, 0.0)

    def test_nonfinite_and_negative_limits_keep_native_behavior(self):
        for rows in ([torch.tensor([float("nan"), 1.0])],
                     [torch.tensor([float("inf"), -1.0])],
                     [torch.tensor([float("-inf"), 0.0])],
                     [torch.tensor([3.0, 4.0], dtype=torch.bfloat16)]):
            for threshold in (1.0, -1.0, float("nan"), float("inf"), float("-inf")):
                with self.subTest(rows=rows, threshold=threshold):
                    self.assert_matches_native(rows, threshold)

    def test_python_overflow_matches_native_exception_without_mutation(self):
        rows = [torch.tensor([1e200], dtype=torch.float64)]
        for function in (native_clip, clip_grad_norm_mixed):
            values = parameters(rows)
            with self.assertRaises(OverflowError):
                function(values, 1.0)
            self.assertTrue(torch.equal(values[0].grad, rows[0]))

    def test_one_cpu_readback_for_all_same_device_scalar_norms(self):
        values = parameters([torch.ones(3, dtype=dtype)
                             for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64)])
        original_cpu = torch.Tensor.cpu
        copied_shapes = []

        def counted_cpu(tensor, *args, **kwargs):
            copied_shapes.append(tuple(tensor.shape))
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", counted_cpu):
            clip_grad_norm_mixed(values, 1.0)
        self.assertEqual(copied_shapes, [(4,)])

    def test_context_restores_both_references_after_exception_and_nesting(self):
        import cat_yoko.offload as offload
        import cat_yoko.trainer as trainer

        old_offload, old_trainer = offload.clip_grad_norm_mixed, trainer.clip_grad_norm_mixed
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with use_batched_grad_norm():
                self.assertIs(offload.clip_grad_norm_mixed, clip_grad_norm_mixed)
                self.assertIs(trainer.clip_grad_norm_mixed, clip_grad_norm_mixed)
                with use_batched_grad_norm():
                    self.assertIs(trainer.clip_grad_norm_mixed, clip_grad_norm_mixed)
                self.assertIs(trainer.clip_grad_norm_mixed, clip_grad_norm_mixed)
                raise RuntimeError("injected")
        self.assertIs(offload.clip_grad_norm_mixed, old_offload)
        self.assertIs(trainer.clip_grad_norm_mixed, old_trainer)

    def test_benchmark_restoration_prevents_cumulative_clipping(self):
        from operators.rocm.grad_norm_bench import parity, reset_gradients

        templates = [torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
                     torch.tensor([5.0, 6.0], dtype=torch.bfloat16)]
        values = parameters(templates)
        equality = parity(values, templates, ["a", "b"], 0.25)
        self.assertTrue(equality["pass"])
        norms = []
        for _ in range(3):
            reset_gradients(values, templates)
            self.assertTrue(torch.equal(values[0].grad, templates[0]))
            norms.append(clip_grad_norm_mixed(values, 0.25))
        self.assertEqual(norms, [equality["native_norm"]] * 3)


if __name__ == "__main__":
    unittest.main()
