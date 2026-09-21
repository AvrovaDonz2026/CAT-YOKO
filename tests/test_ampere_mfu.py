#!/usr/bin/env python3
"""Roofline MFU for Ampere BF16 operators."""

from __future__ import annotations

import inspect
import unittest

import torch

from cat_yoko.ampere_mfu import (
    RTX3090_BF16_PEAK,
    gemm_bytes_bf16,
    gemm_flops,
    roofline_mfu,
    theory_table,
)
from cat_yoko.moe import grouped_mm_available, reset_grouped_mm_cache
from cat_yoko.nvfp4_linear import fused_cat_linear


class RooflineTests(unittest.TestCase):
    def test_large_gemm_is_compute_bound(self) -> None:
        fl = gemm_flops(8192, 8192, 8192)
        nb = gemm_bytes_bf16(8192, 8192, 8192)
        self.assertGreaterEqual(roofline_mfu(fl, nb), 0.99)

    def test_tiny_gemm_is_below_peak(self) -> None:
        fl = gemm_flops(32, 32, 32)
        nb = gemm_bytes_bf16(32, 32, 32)
        self.assertLess(roofline_mfu(fl, nb), 0.5)

    def test_theory_table_has_probe_and_12b(self) -> None:
        blob = theory_table()
        self.assertAlmostEqual(blob["peak"]["bf16_dense_tflops"], RTX3090_BF16_PEAK / 1e12, places=2)
        names = {r["name"] for r in blob["bf16_probe"]}
        self.assertIn("fused_qkv", names)
        self.assertIn("dense_cross_flash", names)
        self.assertIn("masked_window", names)
        self.assertIn("moe_bmm", names)
        probe = {r["name"]: r["theo_mfu"] for r in blob["bf16_probe"]}
        pub = {r["name"]: r["theo_mfu"] for r in blob["published_12b"]}
        self.assertLess(probe["dense_cross_flash"], pub["dense_cross_flash"])
        self.assertGreaterEqual(pub["fused_qkv"], 0.99)
        probe_notes = {r["name"]: r["note"] for r in blob["bf16_probe"]}
        self.assertIn("fat tiles 256", probe_notes["masked_window"])
        self.assertIn("S×S", probe_notes["masked_window"])
        pub_notes = {r["name"]: r["note"] for r in blob["published_12b"]}
        self.assertIn("covers seq", pub_notes["masked_window"])
        self.assertIn("fused QK", pub_notes["indexer_fp32"])
        self.assertGreaterEqual(pub["indexer_fp32"], 0.99)


class BshdLayoutSourceTests(unittest.TestCase):
    def test_qk_norm_rope_stays_bshd(self) -> None:
        from cat_yoko.attention import CrossAttention, WindowAttention, rope_after_qk_norm

        self.assertIn("seq_dim=1", inspect.getsource(rope_after_qk_norm))
        self.assertNotIn("contiguous()", inspect.getsource(WindowAttention.forward))
        self.assertNotIn("contiguous()", inspect.getsource(CrossAttention.forward))
        self.assertIn("to_sdpa_layout", inspect.getsource(WindowAttention.forward))

    def test_rmsnorm_cuda_fused_cpu_fp32(self) -> None:
        from cat_yoko.rope import RMSNorm

        src = inspect.getsource(RMSNorm.forward)
        self.assertIn("is_cuda", src)
        self.assertIn("x.float()", src)


class GroupedMmGateTests(unittest.TestCase):
    def test_source_requires_sm90(self) -> None:
        src = inspect.getsource(grouped_mm_available)
        self.assertIn("major >= 9", src)
        self.assertIn("get_device_capability", src)

    def test_cpu_reports_unavailable(self) -> None:
        reset_grouped_mm_cache()
        if torch.cuda.is_available():
            self.skipTest("cuda present")
        self.assertFalse(grouped_mm_available())


class FrozenFusedCatTests(unittest.TestCase):
    def test_frozen_weights_are_concatenated_once(self) -> None:
        from unittest.mock import patch

        q = torch.nn.Linear(16, 16, bias=False)
        k = torch.nn.Linear(16, 8, bias=False)
        v = torch.nn.Linear(16, 8, bias=False)
        for p in list(q.parameters()) + list(k.parameters()) + list(v.parameters()):
            p.requires_grad_(False)
        x = torch.randn(2, 4, 16)
        y1 = fused_cat_linear([q, k, v], x)
        with patch("torch.cat", wraps=torch.cat) as spy:
            y2 = fused_cat_linear([q, k, v], x)
        self.assertEqual(spy.call_count, 0)
        self.assertTrue(torch.allclose(y1, y2))
        self.assertEqual(tuple(y1.shape), (2, 4, 32))

    def test_trainable_still_matches_three_linears(self) -> None:
        torch.manual_seed(0)
        q = torch.nn.Linear(16, 16, bias=False)
        k = torch.nn.Linear(16, 8, bias=False)
        v = torch.nn.Linear(16, 8, bias=False)
        x = torch.randn(2, 4, 16)
        fused = fused_cat_linear([q, k, v], x)
        ref = torch.cat((q(x), k(x), v(x)), dim=-1)
        self.assertTrue(torch.allclose(fused, ref, atol=1e-5, rtol=1e-5))


class FrozenMoEGuCacheTests(unittest.TestCase):
    def test_frozen_gate_up_cat_is_reused(self) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.moe import MoE, _swiglu_expert_weights

        cfg = CATYokoConfig.tiny()
        moe = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec)
        for p in moe.parameters():
            p.requires_grad_(False)
        _swiglu_expert_weights(moe.experts, moe)
        ptr = moe._swiglu_gu.data_ptr()
        _swiglu_expert_weights(moe.experts, moe)
        self.assertEqual(moe._swiglu_gu.data_ptr(), ptr)


if __name__ == "__main__":
    unittest.main()
