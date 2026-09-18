#!/usr/bin/env python3
"""MoE permute + batched SwiGLU must match the serial expert loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import MoE, SwiGLU, _fused_gate_up
from cat_yoko.nvfp4_linear import Nvfp4Linear, apply_nvfp4


class FusedSwiGLUTests(unittest.TestCase):
    def test_fused_gate_up_matches_two_linears(self) -> None:
        torch.manual_seed(0)
        m = SwiGLU(16, 32)
        x = torch.randn(5, 16)
        fused = _fused_gate_up(m.gate_proj, m.up_proj, x)
        ref = torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x)
        self.assertTrue(torch.allclose(fused, ref, atol=1e-5, rtol=1e-5))
        y = m(x)
        y_ref = m.down_proj(ref)
        self.assertTrue(torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5))


class BatchedMoETests(unittest.TestCase):
    def _compare(self, hash_route: bool) -> None:
        cfg = CATYokoConfig.tiny()
        torch.manual_seed(1)
        a = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec, hash_route=hash_route)
        b = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec, hash_route=hash_route)
        b.load_state_dict(a.state_dict())
        a.batched_experts = True
        b.batched_experts = False
        x = torch.randn(2, cfg.seq_len, cfg.hidden_size)
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        ya = a(x, token_ids=ids if hash_route else None)
        yb = b(x, token_ids=ids if hash_route else None)
        self.assertTrue(torch.allclose(ya, yb, atol=1e-4, rtol=1e-4))
        ya.sum().backward()
        yb.sum().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.grad is None and pb.grad is None:
                continue
            self.assertIsNotNone(pa.grad)
            self.assertIsNotNone(pb.grad)
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=1e-4, rtol=1e-4), msg=str(pa.shape))

    def test_routed_batched_matches_serial(self) -> None:
        self._compare(hash_route=False)

    def test_hash_batched_matches_serial(self) -> None:
        self._compare(hash_route=True)


class FrozenNvfp4CacheTests(unittest.TestCase):
    def test_frozen_weight_cache_reuses_tensor(self) -> None:
        lin = torch.nn.Linear(16, 16, bias=False)
        lin.weight.requires_grad_(False)
        wrapped = Nvfp4Linear.from_linear(lin)
        x = torch.randn(4, 16)
        y1 = wrapped(x)
        ptr = wrapped._wq_cache.data_ptr()
        y2 = wrapped(x)
        self.assertEqual(wrapped._wq_cache.data_ptr(), ptr)
        self.assertTrue(torch.equal(y1, y2))

    def test_trainable_ste_still_has_weight_grad(self) -> None:
        lin = torch.nn.Linear(8, 8, bias=False)
        wrapped = Nvfp4Linear.from_linear(lin)
        x = torch.randn(2, 8, requires_grad=True)
        wrapped(x).sum().backward()
        self.assertIsNotNone(wrapped.weight.grad)
        self.assertTrue(wrapped.weight.requires_grad)
        self.assertIsNone(wrapped._wq_cache)

    def test_b0_encoder_wrap_still_finite_with_batched_moe(self) -> None:
        from dataclasses import replace

        cfg = replace(CATYokoConfig.tiny(), use_nvfp4=True)
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        apply_nvfp4(model, "B0", enabled=True)
        self.assertTrue(model.encoder[0].mlp.batched_experts)
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
        out = model(input_ids=ids, labels=ids)
        self.assertTrue(torch.isfinite(out["nll"]))
        out["loss"].backward()
        self.assertIsNotNone(model.cache_k.weight.grad)


if __name__ == "__main__":
    unittest.main()
