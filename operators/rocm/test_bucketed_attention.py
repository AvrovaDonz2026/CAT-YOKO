"""CPU checks for bucket budgets, packed semantics, gradients and restoration."""

from __future__ import annotations

from dataclasses import replace
import gc
import math
import unittest
import weakref

import torch

import cat_yoko.attention as native
from cat_yoko.config import CATYokoConfig
from operators.rocm import bucketed_attention as bucketed
from operators.rocm import packed_attention as packed
from operators.rocm.bucketed_attention_bench import case_lengths


def documents(lengths):
    return torch.stack([torch.repeat_interleave(torch.arange(len(row)), torch.tensor(row))
                        for row in lengths])


def values(batch, seq, dtype, layout, *, heads=8, kv_heads=2, dim=16):
    generator = torch.Generator().manual_seed(188)
    q, k, v = [torch.randn(batch, h, seq, dim, dtype=dtype, generator=generator)
               for h in (heads, kv_heads, kv_heads)]
    if layout == "packed_qkv":
        joined = torch.cat([value.transpose(1, 2).reshape(batch, seq, -1) for value in (q, k, v)], -1)
        q, k, v = [part.reshape(batch, seq, h, dim).transpose(1, 2)
                   for part, h in zip(joined.split((heads * dim, kv_heads * dim, kv_heads * dim), -1),
                                      (heads, kv_heads, kv_heads))]
    elif layout == "cross_cache":
        q, k = [value.transpose(1, 2).contiguous().transpose(1, 2) for value in (q, k)]
        cache = torch.cat([value.transpose(1, 2).reshape(batch, seq, -1) for value in (k, v)], -1)
        v = cache[..., kv_heads * dim:].reshape(batch, seq, kv_heads, dim).transpose(1, 2)
    return tuple(value.detach().requires_grad_() for value in (q, k, v))


def oracle(q, k, v, docs):
    # Independent explicit formula, with the input/output BF16 cast boundary.
    with torch.autocast(device_type=q.device.type, enabled=False):
        repeats = q.size(1) // k.size(1)
        qf, kf, vf = q.float(), k.float().repeat_interleave(repeats, 1), v.float().repeat_interleave(repeats, 1)
        scores = (qf @ kf.transpose(-1, -2)) / math.sqrt(q.size(-1))
        pos = torch.arange(q.size(-2), device=q.device)
        keep = (pos[None, :] <= pos[:, None])[None] & (docs[:, :, None] == docs[:, None, :])
        probabilities = scores.masked_fill(~keep[:, None], torch.finfo(torch.float32).min).softmax(-1)
        return (probabilities @ vf).to(q.dtype)


class BucketedAttentionTests(unittest.TestCase):
    def setUp(self):
        packed.clear_doc_plan_cache()
        bucketed.clear_bucket_plan_cache()

    def check_close(self, actual, reference, dtype):
        atol, rtol, relative_limit = (3e-5, 3e-5, 3e-5) if dtype == torch.float32 else (0.02, 0.02, 0.01)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        relative = (actual.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-20)
        self.assertLessEqual(float(relative), relative_limit)

    def test_outputs_and_all_gradients_batch_layout_padding_and_original_order(self):
        docs = documents([[3, 4, 5, 6, 11, 17, 19], [2, 5, 5, 6, 12, 16, 19]])
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("packed_qkv", "cross_cache"):
                with self.subTest(dtype=dtype, layout=layout):
                    qkv = values(2, 65, dtype, layout)
                    dy = torch.randn(qkv[0].shape, dtype=dtype, generator=torch.Generator().manual_seed(44))
                    dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
                    expected = oracle(*qkv, docs)
                    expected_grads = torch.autograd.grad(expected, qkv, dy)
                    stats = bucketed.new_stats()
                    actual = bucketed.bucketed_window_sdpa(*qkv, 8192, docs, stats=stats)
                    actual_grads = torch.autograd.grad(actual, qkv, dy)
                    for a, b in zip((actual, *actual_grads), (expected, *expected_grads)):
                        self.check_close(a, b, dtype)
                    self.assertEqual(stats["bucketed_calls"], 1)
                    self.assertGreater(stats["batched_sdpa_calls"], 0)
                    self.assertLess(stats["bucket_sdpa_calls"], stats["bucket_fragments"])
                    self.assertGreater(stats["padded_score_elements"], stats["raw_score_elements"])
                    self.assertLessEqual(stats["max_padding_ratio"], 1.25)

    def test_budget_is_per_bucket_and_no_batchable_or_long_documents_fall_back(self):
        config = bucketed.BucketConfig(max_bucket_size=3, max_bucket_score_elements=400)
        plan = (((0, 3), (3, 7), (7, 12), (12, 18), (18, 29), (29, 46), (46, 65)),)
        records, buckets, reason = bucketed.build_bucket_plan(plan, 65, config)
        self.assertIsNone(reason)
        for group in buckets:
            lengths = [records[i][2] - records[i][1] for i in group]
            self.assertLessEqual(len(group), 3)
            padded = len(group) * max(lengths) ** 2
            self.assertLessEqual(padded, 1.25 * sum(length ** 2 for length in lengths))
            if len(group) > 1:
                self.assertLessEqual(padded, 400)
        for lengths, expected_reason in (([65], "single_document"), ([50, 5, 5, 5], "long_document"),
                                         ([4, 9, 17, 35], "no_batchable_fragments")):
            docs = documents([lengths])
            qkv = values(1, 65, torch.float32, "packed_qkv")
            config = bucketed.BucketConfig(max_padding_ratio=1.0, long_document_fraction=1.0)
            if expected_reason == "long_document":
                config = bucketed.BucketConfig()
            stats = bucketed.new_stats()
            actual = bucketed.bucketed_window_sdpa(*qkv, 8192, docs, config=config, stats=stats)
            expected = packed.packed_window_sdpa(*qkv, 8192, docs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(stats["bucketed_calls"], 0)
            self.assertEqual(stats["bucket_fallback_reasons"][expected_reason], 1)

    def test_causal_and_cross_document_isolation_after_padding(self):
        docs = documents([[7, 8, 8, 9]])
        q, k, v = values(1, 32, torch.float32, "cross_cache")
        before = bucketed.bucketed_window_sdpa(q, k, v, 8192, docs)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[:, :, :7], changed_v[:, :, :7] = 70, -70
        after = bucketed.bucketed_window_sdpa(q, changed_k, changed_v, 8192, docs)
        torch.testing.assert_close(before[:, :, 7:], after[:, :, 7:], atol=0, rtol=0)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[:, :, 5:7], changed_v[:, :, 5:7] = -80, 80
        after = bucketed.bucketed_window_sdpa(q, changed_k, changed_v, 8192, docs)
        torch.testing.assert_close(before[:, :, :5], after[:, :, :5], atol=0, rtol=0)

    def test_repeated_ids_and_non_dense_cases_preserve_existing_fallback(self):
        docs = documents([[8, 8, 8, 8]])
        docs[:, 16:24] = 0
        qkv = values(1, 32, torch.float32, "packed_qkv")
        for window, supplied_docs, bias in ((8192, docs, None), (7, docs, None),
                                             (8192, None, None), (8192, docs, torch.zeros(32, 32))):
            stats = bucketed.new_stats()
            expected = packed.packed_window_sdpa(*qkv, window, supplied_docs, bias)
            actual = bucketed.bucketed_window_sdpa(*qkv, window, supplied_docs, bias, stats=stats)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(stats["bucketed_calls"], 0)
            self.assertEqual(stats["bucketed_fallback_calls"], 1)

    def test_plan_cache_mutation_config_weak_lifetime_and_rng(self):
        docs = documents([[8, 8, 8, 8]])
        qkv = values(1, 32, torch.float32, "packed_qkv")
        stats = bucketed.new_stats()
        rng = torch.get_rng_state().clone()
        for _ in range(2):
            bucketed.bucketed_window_sdpa(*qkv, 8192, docs, stats=stats)
        self.assertEqual(stats["host_plan_builds"], 1)
        self.assertEqual(stats["bucket_plan_builds"], 1)
        docs[:, :3] = -1
        actual = bucketed.bucketed_window_sdpa(*qkv, 8192, docs, stats=stats)
        self.check_close(actual, oracle(*qkv, docs), torch.float32)
        self.assertEqual(stats["host_plan_builds"], 2)
        self.assertEqual(stats["bucket_plan_builds"], 2)
        bucketed.bucketed_window_sdpa(*qkv, 8192, docs, stats=stats,
            config=bucketed.BucketConfig(max_bucket_size=2))
        self.assertEqual(stats["bucket_plan_builds"], 3)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        reference = weakref.ref(docs)
        del docs
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(bucketed._BUCKET_CACHE)

    def test_cross_projection_and_input_gradients_with_nested_context(self):
        docs = documents([[5, 6, 6, 7, 8, 9]])
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                cfg = replace(CATYokoConfig.tiny(), qk_norm=True)
                module = native.CrossAttention(cfg).to(dtype=dtype)
                generator = torch.Generator().manual_seed(113)
                x = torch.randn(1, 41, cfg.hidden_size, dtype=dtype, generator=generator).requires_grad_()
                kv = torch.randn(1, 41, 2 * cfg.kv_dim, dtype=dtype, generator=generator)
                k, v = [value.detach().requires_grad_() for value in kv.split(cfg.kv_dim, -1)]
                inputs = (x, k, v, *module.parameters())
                dy = torch.randn(x.shape, dtype=dtype, generator=generator)
                with packed.packed_attention_context(min_seq_len=0):
                    expected = module(x, k, v, docs)
                    expected_grads = torch.autograd.grad(expected, inputs, dy)
                    with bucketed.bucketed_attention_context() as installation:
                        actual = module(x, k, v, docs)
                        grads = torch.autograd.grad(actual, inputs, dy)
                    self.assertEqual(installation.report()["bucketed_calls"], 1)
                for a, b in zip((actual, *grads), (expected, *expected_grads)):
                    self.check_close(a, b, dtype)

    def test_context_restoration_exception_nested_order_and_owned_stats(self):
        original = packed._dense_document_sdpa
        window, cross = native._window_sdpa, native.CrossAttention.forward
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with packed.packed_attention_context(min_seq_len=0) as parent:
                packed_window = native._window_sdpa
                with bucketed.bucketed_attention_context() as installation:
                    native._window_sdpa(*values(1, 32, torch.float32, "packed_qkv"),
                                        8192, documents([[8, 8, 8, 8]]))
                    self.assertIs(native._window_sdpa, packed_window)
                    self.assertEqual(parent.report()["optimized_calls"], 1)
                    self.assertEqual(parent.report()["bucketed_calls"], 1)
                    self.assertEqual(installation.report()["bucketed_calls"], 1)
                    raise RuntimeError("injected")
        self.assertIs(packed._dense_document_sdpa, original)
        self.assertIs(native._window_sdpa, window)
        self.assertIs(native.CrossAttention.forward, cross)
        first = bucketed.install_bucketed_attention()
        second = bucketed.install_bucketed_attention()
        try:
            with self.assertRaisesRegex(RuntimeError, "reverse"):
                first.remove()
        finally:
            second.remove()
            first.remove()
        first.remove()
        self.assertIs(packed._dense_document_sdpa, original)

    def test_seq4096_low_heads_and_fixture_cases(self):
        for case in ("one_long", "six_medium", "many_short", "uneven"):
            lengths = case_lengths(case, 4096)
            self.assertEqual(sum(lengths), 4096)
            self.assertGreater(min(lengths), 0)
        docs = documents([case_lengths("six_medium", 4096)])
        qkv = values(1, 4096, torch.bfloat16, "cross_cache", heads=2, kv_heads=1, dim=8)
        dy = torch.randn(qkv[0].shape, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(41))
        expected = packed.packed_window_sdpa(*qkv, 8192, docs)
        expected_grads = torch.autograd.grad(expected, qkv, dy)
        stats = bucketed.new_stats()
        actual = bucketed.bucketed_window_sdpa(*qkv, 8192, docs, stats=stats)
        grads = torch.autograd.grad(actual, qkv, dy)
        for a, b in zip((actual, *grads), (expected, *expected_grads)):
            self.check_close(a, b, torch.bfloat16)
        self.assertEqual(stats["bucketed_calls"], 1)
        self.assertEqual(stats["bucket_sdpa_calls"], 1)
        self.assertEqual(stats["bucket_fragments"], 6)

    def test_invalid_config_and_shapes_fail(self):
        for kwargs in ({"max_padding_ratio": float("nan")}, {"max_padding_ratio": 0.9},
                       {"max_bucket_size": 1}, {"max_bucket_score_elements": 0},
                       {"long_document_fraction": 0}, {"long_document_fraction": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bucketed.BucketConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, "shape"):
            bucketed.bucketed_window_sdpa(*values(1, 32, torch.float32, "packed_qkv"),
                                          8192, torch.zeros(32, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
