#!/usr/bin/env python3
"""Phase B attention must stay published YOCO: causal window, GQA, fp32 SDPA.

NVFP4 may wrap QKV/O Linears only. Do not implement CSA here.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import cat_yoko.attention as attn_mod
from cat_yoko.attention import CrossAttention, WindowAttention, _sdpa, _window_causal_bias
from cat_yoko.config import CATYokoConfig, KEEP_HIGH_PREC, NVFP4_GEMM_SLOTS
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.nvfp4_linear import Nvfp4Linear, apply_nvfp4
from cat_yoko.rope import RMSNorm


class PublishedPlanTests(unittest.TestCase):
    def test_12b_gqa_and_window_backend(self) -> None:
        c = CATYokoConfig.middle_12b()
        self.assertEqual(c.attention_backend, "window")
        self.assertEqual((c.num_heads, c.num_kv_heads), (16, 2))
        self.assertEqual(c.head_dim, 128)
        self.assertEqual(c.kv_dim, 256)
        self.assertTrue(c.qk_norm)
        self.assertEqual((c.encoder_layers, c.decoder_layers), (16, 26))

    def test_keep_high_prec_includes_softmax_and_qk_norm(self) -> None:
        self.assertIn("attn_softmax", KEEP_HIGH_PREC)
        self.assertIn("qk_norm", KEEP_HIGH_PREC)
        self.assertIn("attn_qkv", NVFP4_GEMM_SLOTS)
        self.assertIn("attn_o", NVFP4_GEMM_SLOTS)
        self.assertIn("cross_q", NVFP4_GEMM_SLOTS)
        self.assertIn("cross_o", NVFP4_GEMM_SLOTS)
        self.assertIn("cache_kv", NVFP4_GEMM_SLOTS)
        self.assertNotIn("lm_head", KEEP_HIGH_PREC)

    def test_sdpa_is_fp32_then_cast(self) -> None:
        src = inspect.getsource(_sdpa)
        self.assertIn(".float()", src)
        self.assertIn("is_causal", src)
        self.assertIn("scaled_dot_product_attention", src)
        self.assertIn("to(q.dtype)", src)

        seen: list[torch.dtype] = []
        orig = attn_mod.F.scaled_dot_product_attention

        def _spy(q, k, v, *args, **kwargs):
            seen.append(q.dtype)
            return orig(q, k, v, *args, **kwargs)

        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        with patch.object(attn_mod.F, "scaled_dot_product_attention", _spy):
            _sdpa(q, k, v, causal=True)
        self.assertEqual(seen, [torch.float32])

    def test_attention_module_has_no_csa_kernel(self) -> None:
        import cat_yoko.attention as attn_mod

        src = inspect.getsource(attn_mod)
        self.assertNotIn("class CSA", src)
        self.assertNotIn("LightningIndexer", src)
        self.assertIn("class WindowAttention", src)
        self.assertIn("class CrossAttention", src)


class WrapDoesNotChangeTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = replace(CATYokoConfig.tiny(), use_nvfp4=True)
        self.model = CATYokoForCausalLM(self.cfg)
        apply_freeze(self.model, "B1")
        apply_nvfp4(self.model, "B1", enabled=True)

    def test_encoder_and_decoder_still_window_cross(self) -> None:
        enc = self.model.encoder[0].attn
        dec = self.model.decoder[0].self_attn
        cross = self.model.decoder[0].cross_attn
        self.assertIsInstance(enc, WindowAttention)
        self.assertIsInstance(dec, WindowAttention)
        self.assertIsInstance(cross, CrossAttention)
        self.assertIsInstance(enc.q_proj, Nvfp4Linear)
        self.assertIsInstance(enc.q_norm, RMSNorm)
        self.assertIsInstance(enc.k_norm, RMSNorm)
        self.assertIsInstance(cross.q_norm, RMSNorm)
        self.assertNotIsInstance(enc.q_norm, Nvfp4Linear)
        for name, mod in self.model.named_modules():
            if name.endswith("q_norm") or name.endswith("k_norm"):
                self.assertIsInstance(mod, RMSNorm, msg=name)
                self.assertNotIsInstance(mod, Nvfp4Linear, msg=name)
        params = list(inspect.signature(CrossAttention.forward).parameters)
        self.assertIn("k", params)
        self.assertIn("v", params)

    def test_window_attention_is_causal_after_wrap(self) -> None:
        attn = self.model.encoder[0].attn
        torch.manual_seed(0)
        x = torch.randn(2, self.cfg.seq_len, self.cfg.hidden_size)
        x_fut = x.clone()
        x_fut[:, -1] = torch.randn_like(x_fut[:, -1])
        with torch.no_grad():
            y = attn(x)
            y_fut = attn(x_fut)
        self.assertTrue(torch.allclose(y[:, :-1], y_fut[:, :-1], atol=1e-5, rtol=1e-5))

    def test_cross_attention_is_causal_on_cache(self) -> None:
        cross = self.model.decoder[0].cross_attn
        b, s, d = 2, self.cfg.seq_len, self.cfg.hidden_size
        torch.manual_seed(1)
        x = torch.randn(b, s, d)
        k = torch.randn(b, s, self.cfg.kv_dim)
        v = torch.randn(b, s, self.cfg.kv_dim)
        k_fut = k.clone()
        k_fut[:, -1] = 0
        with torch.no_grad():
            y = cross(x, k, v)
            y_fut = cross(x, k_fut, v)
        self.assertTrue(torch.allclose(y[:, :-1], y_fut[:, :-1], atol=1e-5, rtol=1e-5))

    def test_full_lm_future_token_does_not_leak(self) -> None:
        self.model.eval()
        ids = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.seq_len))
        ids_fut = ids.clone()
        ids_fut[:, -1] = (ids_fut[:, -1] + 1) % self.cfg.vocab_size
        with torch.no_grad():
            a = self.model(input_ids=ids)["logits"]
            b = self.model(input_ids=ids_fut)["logits"]
        self.assertTrue(torch.allclose(a[:, :-1], b[:, :-1], atol=1e-4, rtol=1e-4))

    def test_window_bias_still_masks_future(self) -> None:
        bias = _window_causal_bias(4, 4, 8, torch.device("cpu"), torch.float32)
        self.assertLess(bias[1, 2].item(), -1e4)
        self.assertEqual(bias[2, 2].item(), 0.0)


class SdpaNumericTests(unittest.TestCase):
    def test_causal_fastpath_matches_mask(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(1, 2, 8, 8)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        bias = _window_causal_bias(8, 8, 8, q.device, torch.float32)
        masked = _sdpa(q, k, v, bias)
        fast = _sdpa(q, k, v, causal=True)
        self.assertTrue(torch.allclose(fast, masked, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
