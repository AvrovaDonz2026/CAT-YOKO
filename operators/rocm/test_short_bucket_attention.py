"""Short-only dispatch, mixed long docs, gradients and isolated installation."""
from dataclasses import replace
import gc
import unittest
import weakref

import torch

import cat_yoko.attention as native
from cat_yoko.config import CATYokoConfig
from operators.rocm import bucketed_attention as wide
from operators.rocm import packed_attention as packed
from operators.rocm import short_bucket_attention as short
from operators.rocm.test_bucketed_attention import documents, values, oracle


class ShortBucketAttentionTests(unittest.TestCase):
    def setUp(self):
        packed.clear_doc_plan_cache()
        short.clear_short_plan_cache()

    def close(self, actual, expected, dtype):
        atol, rtol, relative = (3e-5, 3e-5, 3e-5) if dtype == torch.float32 else (0.02, 0.02, 0.01)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
        self.assertLessEqual(float(error.detach()), relative)

    def test_mixed_long_batch_layouts_output_and_all_gradients(self):
        docs = documents([[110, 8, 9, 9, 9], [109, 9, 9, 9, 9]])
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("packed_qkv", "cross_cache"):
                with self.subTest(dtype=dtype, layout=layout):
                    qkv = values(2, 145, dtype, layout, heads=4, kv_heads=1, dim=8)
                    dy = torch.randn(qkv[0].shape, dtype=dtype, generator=torch.Generator().manual_seed(903))
                    dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
                    expected = oracle(*qkv, docs)
                    expected_grads = torch.autograd.grad(expected, qkv, dy)
                    stats = short.new_stats()
                    actual = short.short_bucket_window_sdpa(*qkv, 8192, docs, stats=stats)
                    grads = torch.autograd.grad(actual, qkv, dy)
                    for a, b in zip((actual, *grads), (expected, *expected_grads)):
                        self.close(a, b, dtype)
                    self.assertEqual(stats["short_bucketed_calls"], 1)
                    # The early [8,9] pair exceeds 1.10 padding, so the greedy
                    # plan leaves 8 alone and batches the seven equal nines.
                    self.assertEqual(stats["short_batched_fragments"], 7)
                    self.assertEqual(stats["short_singleton_sdpa_calls"], 3)
                    self.assertLessEqual(stats["short_max_padding_ratio"], 1.10)

    def test_dispatch_limits_and_real_row_8701_plan(self):
        lengths = [44,146,143,152,100,332,227,360,138,105,62,104,804,215,140,61,133,442,146,242]
        pos, fragments = 0, []
        for length in lengths:
            fragments.append((pos, pos + length)); pos += length
        records, groups, reason = short.build_short_bucket_plan((tuple(fragments),), 4096, short.ShortBucketConfig())
        self.assertIsNone(reason)
        combined = [g for g in groups if len(g) > 1]
        self.assertEqual(len(combined), 1)
        self.assertEqual([records[i][2] - records[i][1] for i in combined[0]], [133,138,140,143,146,146])
        self.assertEqual(sorted(i for g in groups for i in g), list(range(len(lengths))))
        for size in (64, 128, 256):
            for count in (4, 8, 16):
                plan = (tuple((i * size, (i + 1) * size) for i in range(count)),)
                _, groups, reason = short.build_short_bucket_plan(plan, size * count, short.ShortBucketConfig())
                self.assertIsNone(reason)
                self.assertEqual(list(map(len, groups)), [count])
        for lengths in ([256] * 3, [257] * 4, [7,8,9,10]):
            plan = (tuple((0, length) for length in lengths),)
            _, _, reason = short.build_short_bucket_plan(plan, sum(lengths), short.ShortBucketConfig())
            self.assertEqual(reason, "no_qualifying_short_bucket")

    def test_no_eligible_and_repeated_ids_preserve_fallback_exactly(self):
        for lengths in ([10, 20, 30, 40], [20] * 3):
            docs = documents([lengths]); qkv = values(1, sum(lengths), torch.float32, "cross_cache")
            expected = packed.packed_window_sdpa(*qkv, 8192, docs)
            stats = short.new_stats()
            actual = short.short_bucket_window_sdpa(*qkv, 8192, docs, stats=stats)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(stats["short_bucketed_calls"], 0)
        docs = documents([[8] * 4]); docs[:,16:24] = 0
        qkv = values(1, 32, torch.float32, "packed_qkv")
        for window, supplied, bias in ((8192,docs,None), (7,docs,None), (8192,None,None),
                                      (8192,docs,torch.zeros(32,32))):
            actual = short.short_bucket_window_sdpa(*qkv, window, supplied, bias)
            expected = packed.packed_window_sdpa(*qkv, window, supplied, bias)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_causal_and_document_isolation_with_right_padding(self):
        docs = documents([[12,13,13,13,94]])
        q,k,v = values(1,145,torch.float32,"cross_cache")
        stats = short.new_stats()
        before = short.short_bucket_window_sdpa(q,k,v,8192,docs,stats=stats)
        self.assertEqual(stats["short_bucketed_calls"],1)
        self.assertEqual(stats["short_batched_fragments"],4)
        self.assertGreater(stats["short_padded_score_elements"],stats["short_raw_score_elements"])
        kk,vv = k.detach().clone(),v.detach().clone();kk[:,:,:12]=75;vv[:,:,:12]=-75
        after = short.short_bucket_window_sdpa(q,kk,vv,8192,docs)
        torch.testing.assert_close(before[:,:,12:],after[:,:,12:],atol=0,rtol=0)
        kk,vv = k.detach().clone(),v.detach().clone();kk[:,:,10:12]=-70;vv[:,:,10:12]=70
        after = short.short_bucket_window_sdpa(q,kk,vv,8192,docs)
        torch.testing.assert_close(before[:,:,:10],after[:,:,:10],atol=0,rtol=0)

    def test_cache_version_configuration_lifetime_and_rng(self):
        docs = documents([[8] * 4]);qkv=values(1,32,torch.float32,"packed_qkv");stats=short.new_stats()
        rng=torch.get_rng_state().clone()
        for _ in range(2):short.short_bucket_window_sdpa(*qkv,8192,docs,stats=stats)
        self.assertEqual(stats["short_plan_builds"],1)
        docs[:,:3]=-1
        actual=short.short_bucket_window_sdpa(*qkv,8192,docs,stats=stats)
        self.close(actual,oracle(*qkv,docs),torch.float32)
        self.assertEqual(stats["short_plan_builds"],2)
        short.short_bucket_window_sdpa(*qkv,8192,docs,stats=stats,config=short.ShortBucketConfig(max_length=16))
        self.assertEqual(stats["short_plan_builds"],3)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        reference=weakref.ref(docs);del docs;gc.collect()
        self.assertIsNone(reference());self.assertFalse(short._PLAN_CACHE)

    def test_cross_input_and_projection_gradients_with_context(self):
        docs=documents([[12,13,13,13,94]])
        for dtype in (torch.float32,torch.bfloat16):
            cfg=replace(CATYokoConfig.tiny(),qk_norm=True)
            module=native.CrossAttention(cfg).to(dtype=dtype)
            g=torch.Generator().manual_seed(660)
            x=torch.randn(1,145,cfg.hidden_size,dtype=dtype,generator=g).requires_grad_()
            cache=torch.randn(1,145,2*cfg.kv_dim,dtype=dtype,generator=g)
            k,v=[part.detach().requires_grad_() for part in cache.split(cfg.kv_dim,-1)]
            inputs=(x,k,v,*module.parameters());dy=torch.randn(x.shape,dtype=dtype,generator=g)
            with packed.packed_attention_context(min_seq_len=0):
                expected=module(x,k,v,docs);expected_grads=torch.autograd.grad(expected,inputs,dy)
                with short.short_bucket_attention_context() as installation:
                    actual=module(x,k,v,docs);grads=torch.autograd.grad(actual,inputs,dy)
                self.assertEqual(installation.report()["short_bucketed_calls"],1)
            for a,b in zip((actual,*grads),(expected,*expected_grads)):self.close(a,b,dtype)

    def test_seq4096_mixed_long_document_keeps_short_group_and_bf16_gradients(self):
        docs=documents([[3840,64,64,64,64]])
        qkv=values(1,4096,torch.bfloat16,"cross_cache",heads=2,kv_heads=1,dim=8)
        dy=torch.randn(qkv[0].shape,dtype=torch.bfloat16,generator=torch.Generator().manual_seed(380))
        expected=packed.packed_window_sdpa(*qkv,8192,docs)
        expected_grads=torch.autograd.grad(expected,qkv,dy)
        stats=short.new_stats()
        actual=short.short_bucket_window_sdpa(*qkv,8192,docs,stats=stats)
        grads=torch.autograd.grad(actual,qkv,dy)
        for a,b in zip((actual,*grads),(expected,*expected_grads)):self.close(a,b,torch.bfloat16)
        self.assertEqual(stats["short_bucketed_calls"],1)
        self.assertEqual(stats["short_batched_fragments"],4)
        self.assertEqual(stats["short_singleton_sdpa_calls"],1)

    def test_context_restores_only_owned_helper_and_never_replaces_wide_planner(self):
        helper,planner,executor=packed._dense_document_sdpa,wide.build_bucket_plan,wide._execute
        window,cross=native._window_sdpa,native.CrossAttention.forward
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with packed.packed_attention_context(min_seq_len=0) as parent:
                with short.short_bucket_attention_context() as installation:
                    native._window_sdpa(*values(1,32,torch.float32,"packed_qkv"),8192,documents([[8]*4]))
                    self.assertEqual(parent.report()["short_bucketed_calls"],1)
                    self.assertEqual(installation.report()["short_bucketed_calls"],1)
                    self.assertIs(wide.build_bucket_plan,planner);self.assertIs(wide._execute,executor)
                    raise RuntimeError("injected")
        self.assertIs(packed._dense_document_sdpa,helper)
        self.assertIs(native._window_sdpa,window);self.assertIs(native.CrossAttention.forward,cross)
        first=short.install_short_bucket_attention();second=short.install_short_bucket_attention()
        try:
            with self.assertRaisesRegex(RuntimeError,"reverse"):first.remove()
        finally:second.remove();first.remove()
        first.remove();self.assertIs(packed._dense_document_sdpa,helper)

    def test_invalid_configuration_and_shapes(self):
        for kwargs in ({"max_length":0},{"min_members":1},{"max_bucket_size":3},
                       {"max_padding_ratio":float('nan')},{"max_padding_ratio":0.9},
                       {"max_bucket_score_elements":0}):
            with self.assertRaises(ValueError):short.ShortBucketConfig(**kwargs)
        with self.assertRaisesRegex(ValueError,"shape"):
            short.short_bucket_window_sdpa(*values(1,32,torch.float32,"packed_qkv"),8192,torch.zeros(32))


if __name__ == '__main__':
    unittest.main()
