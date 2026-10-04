"""Independent attention oracle, exact packed parity, and fallback contracts."""

from dataclasses import replace
import math
import unittest
from unittest.mock import Mock, patch

import torch

import cat_yoko.attention as native
from cat_yoko.config import CATYokoConfig
from operators.rocm import packed_attention as packed
from operators.rocm import split_attention as split


def documents(rows):
    result = []
    for lengths in rows:
        result.append(torch.cat([torch.full((length,), index, dtype=torch.long)
                                 for index, length in enumerate(lengths)]))
    return torch.stack(result)


def values(batch, seq, dtype, layout):
    generator = torch.Generator().manual_seed(720 + seq)
    heads, kv_heads, dim = 4, 2, 8
    q, k, v = [torch.randn(batch, count, seq, dim, dtype=dtype, generator=generator)
               for count in (heads, kv_heads, kv_heads)]
    if layout == "packed_qkv":
        fused = torch.cat([value.transpose(1, 2).reshape(batch, seq, -1)
                           for value in (q, k, v)], dim=-1)
        q, k, v = [piece.reshape(batch, seq, count, dim).transpose(1, 2)
                   for piece, count in zip(fused.split((32, 16, 16), dim=-1),
                                           (heads, kv_heads, kv_heads))]
    elif layout == "cross_cache":
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        cache = torch.cat([value.transpose(1, 2).reshape(batch, seq, -1)
                           for value in (k, v)], dim=-1)
        ks, vs = cache.split((16, 16), dim=-1)
        k = ks.reshape(batch, seq, kv_heads, dim).contiguous().transpose(1, 2)
        v = vs.reshape(batch, seq, kv_heads, dim).transpose(1, 2)
    else:
        raise ValueError(layout)
    return tuple(value.detach().requires_grad_() for value in (q, k, v))


def dense_oracle(q, k, v, docs, window):
    """Explicit dense keep-set; no native/packed mask or plan helpers."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        repeats = q.size(1) // k.size(1)
        qf = q.float()
        kf, vf = (value.float().repeat_interleave(repeats, dim=1) for value in (k, v))
        scores = (qf @ kf.transpose(-1, -2)) / math.sqrt(q.size(-1))
        query = torch.arange(q.size(-2))[:, None]
        key = torch.arange(k.size(-2))[None, :]
        keep = ((key <= query) & (query - key < window)).unsqueeze(0)
        if docs is not None:
            keep = keep & (docs[:, :, None] == docs[:, None, :])
        probabilities = scores.masked_fill(~keep[:, None], float("-inf")).softmax(-1)
        return (probabilities @ vf).to(q.dtype)


def output_and_gradients(function, qkv, dy, *args, **kwargs):
    output = function(*qkv, *args, **kwargs)
    return (output, *torch.autograd.grad(output, qkv, dy))


class SplitAttentionTests(unittest.TestCase):
    def setUp(self):
        packed.clear_doc_plan_cache()

    def assert_oracle_close(self, actual, expected, dtype):
        atol, rtol, relative = ((3e-5, 3e-5, 3e-5) if dtype == torch.float32
                                else (0.02, 0.02, 0.005))
        for label, left, right in zip(("output", "dq", "dk", "dv"), actual, expected):
            self.assertTrue(torch.isfinite(left).all(), label)
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=label)
            error = (left.float() - right.float()).norm() / right.float().norm().clamp_min(1e-20)
            self.assertLessEqual(float(error.detach()), relative, label)

    def test_independent_dense_oracle_and_exact_packed_all_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("packed_qkv", "cross_cache"):
                for lengths in ([1, 7, 13, 20], [32, 3, 3, 3], [1] * 41):
                    with self.subTest(dtype=dtype, layout=layout, lengths=lengths):
                        seq = sum(lengths)
                        docs = documents([lengths])
                        qkv = values(1, seq, dtype, layout)
                        self.assertFalse(qkv[0].is_contiguous())
                        self.assertFalse(qkv[2].is_contiguous())
                        dy = torch.randn(qkv[0].shape, dtype=dtype,
                                         generator=torch.Generator().manual_seed(819))
                        dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
                        reference = output_and_gradients(packed.packed_window_sdpa, qkv, dy,
                                                         8192, docs)
                        expected = output_and_gradients(
                            lambda q, k, v: dense_oracle(q, k, v, docs, 8192), qkv, dy)
                        stats = split.new_stats()
                        actual = output_and_gradients(split.split_window_sdpa, qkv, dy,
                                                      8192, docs, stats=stats)
                        self.assert_oracle_close(actual, expected, dtype)
                        for label, left, right in zip(("output", "dq", "dk", "dv"), actual, reference):
                            torch.testing.assert_close(left, right, atol=0, rtol=0, msg=label)
                        self.assertEqual(stats["split_optimized_calls"], 1)
                        self.assertEqual(stats["split_documents"], len(lengths))
                        self.assertEqual(stats["doc_fragment_calls"], len(lengths))
                        self.assertEqual(stats["fragment_score_elements"], 4 * sum(n * n for n in lengths))

    def test_multi_batch_and_single_document_preserve_packed_exactly(self):
        for rows, reason in (([[3, 9, 17], [6, 8, 15]], "batch_not_one"),
                             ([[29]], "fewer_than_two_documents")):
            docs = documents(rows)
            qkv = values(len(rows), 29, torch.float32, "cross_cache")
            dy = torch.randn(qkv[0].shape, generator=torch.Generator().manual_seed(401))
            expected = output_and_gradients(packed.packed_window_sdpa, qkv, dy, 8192, docs)
            stats = split.new_stats()
            actual = output_and_gradients(split.split_window_sdpa, qkv, dy, 8192, docs, stats=stats)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(stats["split_optimized_calls"], 0)
            self.assertEqual(stats["split_fallback_reasons"], {reason: 1})

    def test_repeated_and_invalid_plan_ids_keep_native_links_and_all_gradients(self):
        docs = documents([[8, 7, 14]])
        docs[:, 15:] = 0  # A later run may attend to the earlier same-ID run.
        for supplied in (docs, documents([[8, 7, 14]]).float()):
            qkv = values(1, 29, torch.float32, "packed_qkv")
            dy = torch.randn(qkv[0].shape, generator=torch.Generator().manual_seed(335))
            expected = output_and_gradients(
                lambda q, k, v: dense_oracle(q, k, v, supplied, 8192), qkv, dy)
            reference = output_and_gradients(packed.packed_window_sdpa, qkv, dy, 8192, supplied)
            stats = split.new_stats()
            actual = output_and_gradients(split.split_window_sdpa, qkv, dy, 8192, supplied, stats=stats)
            self.assert_oracle_close(actual, expected, torch.float32)
            for left, right in zip(actual, reference):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(stats["split_optimized_calls"], 0)
            self.assertEqual(stats["split_fallback_reasons"],
                             {"invalid_or_noncontiguous_document_ids": 1})

    def test_non_dense_attention_modes_preserve_packed_output_and_gradients(self):
        docs = documents([[3, 5, 21]])
        for window, supplied, bias in ((8192, None, None), (7, docs, None),
                                      (8192, docs, torch.zeros(29, 29))):
            qkv = values(1, 29, torch.float32, "cross_cache")
            dy = torch.randn(qkv[0].shape, generator=torch.Generator().manual_seed(442))
            expected = output_and_gradients(packed.packed_window_sdpa, qkv, dy,
                                            window, supplied, bias)
            stats = split.new_stats()
            actual = output_and_gradients(split.split_window_sdpa, qkv, dy,
                                          window, supplied, bias, stats=stats)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertEqual(stats["split_optimized_calls"], 0)
            self.assertEqual(stats["split_fallback_reasons"], {"non_dense_or_extra_bias": 1})

    def test_causal_document_isolation_and_mutated_plan(self):
        docs = documents([[7, 11, 23]])
        q, k, v = values(1, 41, torch.float32, "cross_cache")
        stats = split.new_stats()
        rng = torch.get_rng_state().clone()
        before = split.split_window_sdpa(q, k, v, 8192, docs, stats=stats)
        split.split_window_sdpa(q, k, v, 8192, docs, stats=stats)
        self.assertEqual(stats["host_plan_builds"], 1)
        kk, vv = k.detach().clone(), v.detach().clone()
        kk[:, :, :7], vv[:, :, :7] = 80, -80
        after = split.split_window_sdpa(q, kk, vv, 8192, docs)
        torch.testing.assert_close(before[:, :, 7:], after[:, :, 7:], atol=0, rtol=0)
        kk, vv = k.detach().clone(), v.detach().clone()
        kk[:, :, 5:7], vv[:, :, 5:7] = -80, 80
        after = split.split_window_sdpa(q, kk, vv, 8192, docs)
        torch.testing.assert_close(before[:, :, :5], after[:, :, :5], atol=0, rtol=0)
        docs[:, :2] = -1
        dy = torch.randn(q.shape, generator=torch.Generator().manual_seed(834))
        actual = output_and_gradients(split.split_window_sdpa, (q, k, v), dy, 8192, docs, stats=stats)
        expected = output_and_gradients(packed.packed_window_sdpa, (q, k, v), dy, 8192, docs)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        self.assertEqual(stats["host_plan_builds"], 2)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_cross_attention_input_cache_and_projection_gradients_exact_with_context(self):
        cfg = replace(CATYokoConfig.tiny(), qk_norm=True)
        docs = documents([[7, 11, 23]])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                module = native.CrossAttention(cfg).to(dtype=dtype)
                generator = torch.Generator().manual_seed(188)
                x = torch.randn(1, 41, cfg.hidden_size, dtype=dtype, generator=generator).requires_grad_()
                cache = torch.randn(1, 41, 2 * cfg.kv_dim, dtype=dtype, generator=generator)
                k, v = (part.detach().requires_grad_() for part in cache.split(cfg.kv_dim, -1))
                inputs = (x, k, v, *module.parameters())
                dy = torch.randn(x.shape, dtype=dtype, generator=generator)
                with packed.packed_attention_context(min_seq_len=0):
                    reference = module(x, k, v, docs)
                    reference_grads = torch.autograd.grad(reference, inputs, dy)
                    with split.split_attention_context() as installation:
                        actual = module(x, k, v, docs)
                        gradients = torch.autograd.grad(actual, inputs, dy)
                    self.assertEqual(installation.report()["split_optimized_calls"], 1)
                    self.assertEqual(installation.report()["split_documents"], 3)
                for left, right in zip((actual, *gradients), (reference, *reference_grads)):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_installed_fallback_calls_previous_helper_and_restores_after_exception(self):
        original = packed._dense_document_sdpa
        native_window = native._window_sdpa
        sentinel = object()
        previous = Mock(return_value=sentinel)
        qkv = values(2, 29, torch.float32, "cross_cache")
        docs = documents([[3, 9, 17], [6, 8, 15]])
        stats = packed.new_stats()
        with patch.object(packed, "_dense_document_sdpa", previous):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with split.split_attention_context() as installation:
                    self.assertIs(native._window_sdpa, native_window)
                    self.assertIs(packed._dense_document_sdpa(*qkv, docs,
                        packed._ORIGINAL_WINDOW_SDPA, 8192, None, stats), sentinel)
                    previous.assert_called_once_with(*qkv, docs,
                        packed._ORIGINAL_WINDOW_SDPA, 8192, None, stats)
                    self.assertEqual(installation.report()["split_fallback_calls"], 1)
                    raise RuntimeError("injected")
            self.assertIs(packed._dense_document_sdpa, previous)
        self.assertIs(packed._dense_document_sdpa, original)

    def test_nested_installation_lifo_and_report_copy(self):
        original = packed._dense_document_sdpa
        first = split.install_split_attention()
        second = split.install_split_attention()
        try:
            with self.assertRaisesRegex(RuntimeError, "reverse order"):
                first.remove()
            report = second.report()
            report["split_fallback_reasons"]["external"] = 3
            self.assertEqual(second.report()["split_fallback_reasons"], {})
        finally:
            second.remove()
            first.remove()
        first.remove()
        self.assertIs(packed._dense_document_sdpa, original)

    def test_invalid_shapes_and_mixed_dtype_are_rejected(self):
        q, k, v = values(1, 29, torch.float32, "packed_qkv")
        docs = documents([[3, 5, 21]])
        for args in ((q[0], k, v, 8192, docs), (q, k, v.double(), 8192, docs),
                     (q, k, v, 8192, docs[:, :-1]),
                     (q[:, :3], k, v, 8192, docs)):
            with self.subTest(shapes=[getattr(item, "shape", None) for item in args]):
                with self.assertRaises(ValueError):
                    split.split_window_sdpa(*args)


if __name__ == "__main__":
    unittest.main()
