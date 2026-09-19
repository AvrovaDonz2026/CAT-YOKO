#!/usr/bin/env python3
"""SageBwd INT8 QK: shapes, CPU numeric, no CSA kernel, BSD notice."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cat_yoko.attention import WindowAttention, _sdpa
from cat_yoko.config import CATYokoConfig
from cat_yoko.int8_attn import (
    _Int8AttnFn,
    int8_qk_pv_forward,
    int8_shape_ok,
    int8_tc_ok,
    try_int8_attention,
)


class Int8ShapeTests(unittest.TestCase):
    def test_rejects_tiny_head_and_seq(self) -> None:
        q = torch.randn(1, 4, 16, 16)
        k = torch.randn(1, 2, 16, 16)
        v = torch.randn(1, 2, 16, 16)
        self.assertFalse(int8_shape_ok(q, k, v, causal=True))
        q = torch.randn(1, 4, 64, 16)
        k = torch.randn(1, 2, 64, 16)
        v = torch.randn(1, 2, 64, 16)
        self.assertFalse(int8_shape_ok(q, k, v, causal=True))

    def test_accepts_hd32_seq64(self) -> None:
        q = torch.randn(1, 4, 64, 32)
        k = torch.randn(1, 2, 64, 32)
        v = torch.randn(1, 2, 64, 32)
        self.assertTrue(int8_shape_ok(q, k, v, causal=True))
        self.assertFalse(int8_shape_ok(q, k, v, bias=torch.zeros(64, 64), causal=True))
        self.assertFalse(int8_shape_ok(q, k, v, causal=False))

    def test_tc_capability(self) -> None:
        self.assertTrue(int8_tc_ok((7, 5)))
        self.assertTrue(int8_tc_ok((8, 6)))
        self.assertTrue(int8_tc_ok((8, 0)))
        self.assertFalse(int8_tc_ok((7, 0)))
        self.assertFalse(int8_tc_ok(None))


class Int8NumericTests(unittest.TestCase):
    def test_int8_qk_close_to_fp32_sdpa(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(1, 2, 64, 32)
        k = torch.randn(1, 2, 64, 32)
        v = torch.randn(1, 2, 64, 32)
        hyp = int8_qk_pv_forward(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
        a = hyp.float().reshape(1, -1)
        b = ref.float().reshape(1, -1)
        cos = float(F.cosine_similarity(a, b).item())
        self.assertGreater(cos, 0.97)
        self.assertTrue(torch.isfinite(hyp).all())

    def test_int8_backward_finite(self) -> None:
        torch.manual_seed(1)
        q = torch.randn(1, 2, 64, 32, requires_grad=True)
        k = torch.randn(1, 2, 64, 32, requires_grad=True)
        v = torch.randn(1, 2, 64, 32, requires_grad=True)
        out = _Int8AttnFn.apply(q, k, v, True, 32 ** -0.5, "int_mm")
        out.sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(k.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())

    def test_try_int8_skips_cpu(self) -> None:
        q = torch.randn(1, 2, 64, 32)
        k = torch.randn(1, 2, 64, 32)
        v = torch.randn(1, 2, 64, 32)
        self.assertIsNone(try_int8_attention(q, k, v, causal=True))


class Int8ConfigAndLicenseTests(unittest.TestCase):
    def test_published_12b_keeps_int8_off(self) -> None:
        self.assertFalse(CATYokoConfig.middle_12b().use_int8)
        self.assertFalse(CATYokoConfig.tiny().use_int8)
        probe = CATYokoConfig.int8_probe()
        self.assertTrue(probe.use_int8)
        self.assertEqual(probe.head_dim, 32)
        self.assertEqual(probe.seq_len % 64, 0)
        self.assertGreaterEqual(probe.n_win, probe.seq_len)
        self.assertFalse(probe.use_kda)
        self.assertFalse(probe.use_nvfp4)

    def test_window_attention_stores_flag(self) -> None:
        off = WindowAttention(CATYokoConfig.tiny())
        self.assertFalse(off.use_int8)
        on = WindowAttention(CATYokoConfig.int8_probe())
        self.assertTrue(on.use_int8)

    def test_sdpa_source_keeps_flash_and_int8_hook(self) -> None:
        src = inspect.getsource(_sdpa)
        self.assertIn("scaled_dot_product_attention", src)
        self.assertIn("try_int8_attention", src)
        self.assertIn("use_int8", src)

    def test_not_a_csa_kernel(self) -> None:
        import cat_yoko.attention as attn_mod
        import cat_yoko.int8_attn as i8

        self.assertNotIn("class CSA", inspect.getsource(attn_mod))
        self.assertNotIn("class CSA", inspect.getsource(i8))
        kern = Path(__file__).resolve().parents[1] / "cat_yoko/kernels/attention_kernel_int8.py"
        text = kern.read_text()
        self.assertIn("Alyssa Vance", text)
        self.assertIn("SageBwd", text)
        self.assertNotIn("LightningIndexer", text)
        self.assertIn("Not a CSA CUDA kernel", text)

    def test_bsd_license_file(self) -> None:
        lic = (
            Path(__file__).resolve().parents[1] / "cat_yoko/kernels/LICENSE.flash-attn-triton"
        ).read_text()
        self.assertIn("BSD 3-Clause", lic)
        self.assertIn("Copyright (c) 2025, Alyssa Vance", lic)


if __name__ == "__main__":
    unittest.main()
