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
from cat_yoko.moe import MoE, SwiGLU, _fused_gate_up, _grouped_linear, _swiglu_experts_grouped, _swiglu_experts_serial
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

    def test_silu_mul_matches_mul_and_grads(self) -> None:
        from cat_yoko.moe import _silu_mul

        torch.manual_seed(4)
        g = torch.randn(6, 8, requires_grad=True)
        u = torch.randn(6, 8, requires_grad=True)
        y = _silu_mul(g, u)
        ref = torch.nn.functional.silu(g) * u
        self.assertTrue(torch.allclose(y, ref, atol=1e-6, rtol=1e-6))
        y.sum().backward()
        g2 = g.detach().clone().requires_grad_(True)
        u2 = u.detach().clone().requires_grad_(True)
        (torch.nn.functional.silu(g2) * u2).sum().backward()
        self.assertTrue(torch.allclose(g.grad, g2.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(u.grad, u2.grad, atol=1e-5, rtol=1e-5))

    def test_silu_mul_not_inplace_on_chunk_views(self) -> None:
        import inspect

        from cat_yoko.moe import _SiluMulFn, _fused_gate_up, _silu_mul

        self.assertIn("mul_(up)", inspect.getsource(_SiluMulFn.forward))
        self.assertNotIn("inplace=True", inspect.getsource(_SiluMulFn.forward))
        self.assertIn("_silu_mul", inspect.getsource(_fused_gate_up))
        gu = torch.randn(4, 8)
        g, u = gu.chunk(2, dim=-1)
        before = gu.clone()
        y = _silu_mul(g, u)
        self.assertTrue(torch.equal(gu, before))
        self.assertEqual(tuple(y.shape), tuple(g.shape))


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


def _grouped_mm_ref(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """CPU stand-in for jagged grouped GEMM: ``mat_a[T,K] @ mat_b[E,K,N]``."""
    parts: list[torch.Tensor] = []
    prev = 0
    for e, end in enumerate(offs.tolist()):
        end_i = int(end)
        sl = mat_a[prev:end_i]
        if sl.size(0):
            parts.append(sl @ mat_b[e])
        prev = end_i
    if not parts:
        return mat_a.new_zeros(mat_a.size(0), mat_b.size(-1))
    return torch.cat(parts, dim=0)


class GroupedMoETests(unittest.TestCase):
    def test_grouped_swiglu_matches_serial_with_empty_expert(self) -> None:
        from unittest.mock import patch

        from cat_yoko.moe import _swiglu_expert_weights

        cfg = CATYokoConfig.tiny()
        torch.manual_seed(2)
        moe = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec)
        x = torch.randn(7, cfg.hidden_size)
        counts = torch.tensor([2, 0, 3, 2])
        with patch("cat_yoko.moe._raw_grouped_mm", _grouped_mm_ref):
            y_g = _swiglu_experts_grouped(moe.experts, x, counts)
        y_s = _swiglu_experts_serial(moe.experts, x, counts)
        self.assertTrue(torch.allclose(y_g, y_s, atol=1e-5, rtol=1e-5))
        pack_a = _swiglu_expert_weights(moe.experts, moe)
        pack_b = _swiglu_expert_weights(moe.experts, moe)
        self.assertIsNot(pack_a, pack_b)
        for p in moe.parameters():
            p.requires_grad_(False)
        frozen_a = _swiglu_expert_weights(moe.experts, moe)
        frozen_b = _swiglu_expert_weights(moe.experts, moe)
        self.assertIs(frozen_a, frozen_b)

    def test_grouped_linear_dx_through_frozen_weights(self) -> None:
        from unittest.mock import patch

        torch.manual_seed(3)
        x = torch.randn(6, 8, requires_grad=True)
        w = torch.randn(2, 4, 8)
        w.requires_grad_(False)
        offs = torch.tensor([3, 6], dtype=torch.int32)
        with patch("cat_yoko.moe._raw_grouped_mm", _grouped_mm_ref):
            y = _grouped_linear(x, w, offs)
            y.sum().backward()
        dx_ref = _grouped_mm_ref(torch.ones(6, 4), w, offs)
        self.assertTrue(torch.allclose(x.grad, dx_ref, atol=1e-5, rtol=1e-5))

    def test_grouped_linear_dw_matches_serial(self) -> None:
        from unittest.mock import patch

        torch.manual_seed(4)
        x = torch.randn(5, 8, requires_grad=True)
        w = torch.randn(2, 4, 8, requires_grad=True)
        offs = torch.tensor([2, 5], dtype=torch.int32)
        with patch("cat_yoko.moe._raw_grouped_mm", _grouped_mm_ref):
            y = _grouped_linear(x, w, offs)
            y.sum().backward()
        dw_ref = torch.zeros_like(w)
        prev = 0
        ones = torch.ones(5, 4)
        for e, end in enumerate(offs.tolist()):
            end_i = int(end)
            if end_i > prev:
                dw_ref[e] = ones[prev:end_i].T @ x.detach()[prev:end_i]
            prev = end_i
        self.assertTrue(torch.allclose(w.grad, dw_ref, atol=1e-5, rtol=1e-5))


class MoeUtilizationTests(unittest.TestCase):
    def test_stacked_stats_match_per_layer_mean(self) -> None:
        from cat_yoko.moe import moe_utilization

        class Mlp:
            def __init__(self, load: torch.Tensor) -> None:
                self.last_load = load

            def mean_pending_load(self):
                return self.last_load

        class Blk:
            def __init__(self, load: torch.Tensor) -> None:
                self.mlp = Mlp(load)

        class Model:
            def __init__(self) -> None:
                self.encoder = [Blk(torch.tensor([0.2, 0.8])), Blk(torch.tensor([0.5, 0.5]))]
                self.decoder = []

        out = moe_utilization(Model())
        self.assertEqual(out["moe_layers"], 2)
        self.assertAlmostEqual(out["moe_max"], 0.8, places=5)
        self.assertAlmostEqual(out["moe_min"], 0.2, places=5)
        self.assertGreater(out["moe_cv"], 0.0)


class TrainableCacheTests(unittest.TestCase):
    def test_freeze_invalidates_and_encoder_is_frozen(self) -> None:
        from cat_yoko.moe import module_has_trainable

        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        self.assertTrue(module_has_trainable(model.encoder[0].mlp))
        apply_freeze(model, "B0")
        self.assertFalse(module_has_trainable(model.encoder[0].mlp))
        self.assertFalse(module_has_trainable(model.decoder[0].mlp))
        self.assertTrue(module_has_trainable(model.decoder[0].cross_attn))

    def test_frozen_moe_skips_z_loss_keeps_load(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        moe = model.encoder[0].mlp
        moe.train()
        x = torch.randn(2, cfg.seq_len, cfg.hidden_size)
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertIsNone(moe.last_aux)
        self.assertIsNotNone(moe.last_load)
        self.assertEqual(int(moe._load_n), 1)

    def test_repeat_by_counts_matches_repeat_interleave(self) -> None:
        from cat_yoko.moe import _repeat_by_counts

        counts = torch.tensor([2, 0, 3], dtype=torch.int64)
        ids = torch.arange(3)
        got = _repeat_by_counts(ids, counts, 5)
        ref = torch.repeat_interleave(ids, counts)
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(got.tolist(), [0, 0, 2, 2, 2])

    def test_kick_wait_max_count_cpu(self) -> None:
        from cat_yoko.moe import _kick_max_count, _wait_max_count

        counts = torch.tensor([2, 0, 5, 1], dtype=torch.int64)
        pinned, done = _kick_max_count(counts)
        self.assertIsNone(done)
        self.assertEqual(_wait_max_count(pinned, done), 5)


if __name__ == "__main__":
    unittest.main()
