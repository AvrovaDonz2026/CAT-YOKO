#!/usr/bin/env python3
"""B200 / SM100 NVFP4 hardware backend: recipe, wrap, no process-wide disable."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.nvfp4_hw import (
    compute_family,
    install_te_backend_for_tests,
    nvfp4_leading_ok,
    nvfp4_training_capable,
    prefer_te_linear,
    reset_te_probe,
    shape_fail_note,
    shape_failed,
    te_grouped_swiglu,
    te_nvfp4_recipe_kwargs,
    te_nvfp4_wrap_count,
)
from cat_yoko.nvfp4_linear import (
    Nvfp4Linear,
    TeNvfp4Linear,
    apply_nvfp4,
    fused_cat_linear,
    nvfp4_module_names,
    wrap_nvfp4_linears,
)


class FakeRecipe:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeAutocast:
    def __init__(self, enabled: bool = True, recipe=None) -> None:
        self.recipe = recipe

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


class FakeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, params_dtype=None, **kwargs):
        super().__init__()
        dt = params_dtype or torch.float32
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dt))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dt)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = int(x.reshape(-1, x.shape[-1]).size(0))
        if n == 32 and int(x.shape[-1]) == 16 and int(self.weight.shape[0]) == 16:
            raise RuntimeError("simulated TE miss")
        return F.linear(x, self.weight, self.bias)


class FakeGroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        bias: bool = False,
        params_dtype=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        dt = params_dtype or torch.float32
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        for i in range(num_gemms):
            self.register_parameter(
                f"weight{i}",
                nn.Parameter(torch.empty(out_features, in_features, dtype=dt, device=device)),
            )
            nn.init.kaiming_uniform_(getattr(self, f"weight{i}"), a=5**0.5)

    def forward(self, inp: torch.Tensor, m_splits) -> torch.Tensor:
        if not isinstance(m_splits, torch.Tensor):
            m_splits = torch.tensor(m_splits, dtype=torch.int64)
        parts = []
        prev = 0
        for i, n in enumerate(m_splits.tolist()):
            n_i = int(n)
            w = getattr(self, f"weight{i}")
            sl = inp[prev : prev + n_i]
            if sl.size(0):
                parts.append(F.linear(sl, w))
            prev += n_i
        if not parts:
            return inp.new_zeros(inp.size(0), self.out_features)
        return torch.cat(parts, dim=0)


class FakeTE:
    Linear = FakeLinear
    GroupedLinear = FakeGroupedLinear
    NVFP4BlockScaling = FakeRecipe
    autocast = FakeAutocast


class FamilyTests(unittest.TestCase):
    def test_sm100_is_training_capable(self) -> None:
        self.assertEqual(compute_family((10, 0)), "sm100")
        self.assertEqual(compute_family((10, 3)), "sm103")
        self.assertEqual(compute_family((12, 0)), "sm120")
        self.assertEqual(compute_family((8, 9)), "other")
        self.assertTrue(nvfp4_training_capable("sm100"))
        self.assertTrue(nvfp4_training_capable("sm103"))
        self.assertFalse(nvfp4_training_capable("sm120"))

    def test_sm100_recipe_is_full_default(self) -> None:
        self.assertEqual(te_nvfp4_recipe_kwargs("sm100"), {})
        self.assertEqual(te_nvfp4_recipe_kwargs("sm103"), {})
        kw = te_nvfp4_recipe_kwargs("sm120")
        self.assertTrue(kw.get("disable_rht"))
        self.assertTrue(kw.get("disable_stochastic_rounding"))

    def test_leading_dims(self) -> None:
        self.assertTrue(nvfp4_leading_ok(16, 128, 128))
        self.assertTrue(nvfp4_leading_ok(4096, 2048, 2048))
        self.assertFalse(nvfp4_leading_ok(8, 128, 128))
        from cat_yoko.nvfp4_hw import nvfp4_pad_tokens, pad_packed_counts

        self.assertEqual(nvfp4_pad_tokens(0), 0)
        self.assertEqual(nvfp4_pad_tokens(16), 16)
        self.assertEqual(nvfp4_pad_tokens(17), 32)
        x = torch.randn(12, 16)
        counts = torch.tensor([5, 7], dtype=torch.int64)
        xp, cp, keeps = pad_packed_counts(x, counts)
        self.assertEqual(cp.tolist(), [16, 16])
        self.assertEqual(int(xp.size(0)), 32)
        self.assertEqual(keeps, [(0, 5), (16, 7)])


class TeWrapTests(unittest.TestCase):
    def setUp(self) -> None:
        install_te_backend_for_tests(FakeTE(), force=True)

    def tearDown(self) -> None:
        install_te_backend_for_tests(None, force=False)

    def test_prefer_te_when_forced(self) -> None:
        self.assertTrue(prefer_te_linear())

    def test_from_linear_copy_once_preserves_state_dict_keys(self) -> None:
        torch.manual_seed(0)
        lin = nn.Linear(16, 32, bias=False)
        keys = set(lin.state_dict().keys())
        ptr_before = lin.weight.data_ptr()
        wrapped = TeNvfp4Linear.from_linear(lin)
        self.assertIsInstance(wrapped, Nvfp4Linear)
        self.assertEqual(set(wrapped.state_dict().keys()), keys)
        self.assertNotEqual(wrapped.weight.data_ptr(), ptr_before)
        self.assertTrue(torch.allclose(wrapped.weight.detach(), lin.weight.detach()))
        self.assertGreater(te_nvfp4_wrap_count(), 0)

    def test_trainable_backward_hits_te_weight(self) -> None:
        lin = nn.Linear(16, 16, bias=False)
        wrapped = TeNvfp4Linear.from_linear(lin)
        x = torch.randn(16, 16, requires_grad=True)
        y = wrapped(x)
        y.sum().backward()
        self.assertIsNotNone(wrapped.weight.grad)
        self.assertTrue(torch.isfinite(wrapped.weight.grad).all())

    def test_pad_leading_keeps_te_kernel(self) -> None:
        lin = nn.Linear(16, 16, bias=False)
        wrapped = TeNvfp4Linear.from_linear(lin)
        x = torch.randn(8, 16)
        y = wrapped(x)
        self.assertEqual(tuple(y.shape), (8, 16))
        self.assertTrue(torch.allclose(y, F.linear(x, wrapped.weight), atol=1e-5, rtol=1e-5))
        self.assertFalse(shape_failed(8, 16, 16))

    def test_frozen_fused_cat_is_one_gemm(self) -> None:
        torch.manual_seed(2)
        a0 = nn.Linear(16, 16, bias=False)
        b0 = nn.Linear(16, 16, bias=False)
        a0.weight.requires_grad_(False)
        b0.weight.requires_grad_(False)
        a = TeNvfp4Linear.from_linear(a0)
        b = TeNvfp4Linear.from_linear(b0)
        x = torch.randn(16, 16)
        y = fused_cat_linear([a, b], x)
        ref = torch.cat([F.linear(x, a.weight), F.linear(x, b.weight)], dim=-1)
        self.assertTrue(torch.allclose(y, ref, atol=1e-5, rtol=1e-5))
        self.assertIsNotNone(getattr(a, "_te_fused_cat", None))

    def test_illegal_shape_falls_back_without_process_wide_disable(self) -> None:
        reset_te_probe()
        lin = nn.Linear(16, 16, bias=False)
        wrapped = TeNvfp4Linear.from_linear(lin)
        x_bad = torch.randn(32, 16)  # legal %16, FakeLinear raises
        y = wrapped(x_bad)
        self.assertEqual(tuple(y.shape), (32, 16))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(prefer_te_linear())
        y2 = wrapped(torch.randn(16, 16))
        self.assertEqual(tuple(y2.shape), (16, 16))

    def test_shape_fail_note_is_per_shape(self) -> None:
        reset_te_probe()
        self.assertFalse(shape_failed(8, 128, 128))
        shape_fail_note(8, 128, 128)
        self.assertTrue(shape_failed(8, 128, 128))
        self.assertFalse(shape_failed(16, 128, 128))

    def test_b0_wrap_uses_te_linear(self) -> None:
        from dataclasses import replace

        from cat_yoko.config import CATYokoConfig
        from cat_yoko.freeze import apply_freeze
        from cat_yoko.model import CATYokoForCausalLM

        cfg = replace(CATYokoConfig.tiny(), use_nvfp4=True)
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        keys = set(model.state_dict().keys())
        n = apply_nvfp4(model, "B0", enabled=True)
        self.assertGreater(n, 0)
        self.assertEqual(set(model.state_dict().keys()), keys)
        self.assertIsInstance(model.encoder[0].attn.q_proj, TeNvfp4Linear)
        self.assertIsInstance(model.encoder[0].attn.q_proj, Nvfp4Linear)
        self.assertTrue(any(isinstance(m, TeNvfp4Linear) for m in model.modules()))
        self.assertGreater(len(nvfp4_module_names(model)), 0)

    def test_moe_te_experts_do_not_stack_bf16_grouped(self) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.moe import MoE, _stack_linear_weight

        cfg = CATYokoConfig.tiny()
        moe = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec)
        for e in moe.experts:
            e.gate_proj = TeNvfp4Linear.from_linear(e.gate_proj)
            e.up_proj = TeNvfp4Linear.from_linear(e.up_proj)
            e.down_proj = TeNvfp4Linear.from_linear(e.down_proj)
        with self.assertRaises(RuntimeError):
            _stack_linear_weight([e.gate_proj for e in moe.experts])
        x = torch.randn(2, cfg.seq_len, cfg.hidden_size)
        y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())

    def test_te_grouped_swiglu_matches_serial(self) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.moe import MoE, _swiglu_experts_serial

        cfg = CATYokoConfig.tiny()
        moe = MoE(cfg, 2, 1)
        for e in moe.experts:
            e.gate_proj = TeNvfp4Linear.from_linear(e.gate_proj)
            e.up_proj = TeNvfp4Linear.from_linear(e.up_proj)
            e.down_proj = TeNvfp4Linear.from_linear(e.down_proj)
        torch.manual_seed(0)
        x = torch.randn(32, cfg.hidden_size)
        counts = torch.tensor([16, 16], dtype=torch.int64)
        y_g = te_grouped_swiglu(moe.experts, x, counts, owner=moe)
        self.assertIsNotNone(y_g)
        y_s = _swiglu_experts_serial(moe.experts, x, counts)
        self.assertTrue(torch.allclose(y_g, y_s, atol=1e-4, rtol=1e-4))

    def test_te_grouped_swiglu_pads_and_fuses_frozen_gate_up(self) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.moe import MoE, _swiglu_experts_serial

        cfg = CATYokoConfig.tiny()
        moe = MoE(cfg, 2, 1)
        for e in moe.experts:
            e.gate_proj.weight.requires_grad_(False)
            e.up_proj.weight.requires_grad_(False)
            e.down_proj.weight.requires_grad_(False)
            e.gate_proj = TeNvfp4Linear.from_linear(e.gate_proj)
            e.up_proj = TeNvfp4Linear.from_linear(e.up_proj)
            e.down_proj = TeNvfp4Linear.from_linear(e.down_proj)
        torch.manual_seed(3)
        x = torch.randn(16, cfg.hidden_size)
        counts = torch.tensor([8, 8], dtype=torch.int64)
        y_g = te_grouped_swiglu(moe.experts, x, counts, owner=moe)
        self.assertIsNotNone(y_g)
        self.assertEqual(getattr(moe, "_te_grouped")[0], "fused_gu")
        y_s = _swiglu_experts_serial(moe.experts, x, counts)
        self.assertTrue(torch.allclose(y_g, y_s, atol=1e-4, rtol=1e-4))


class EnvOffTests(unittest.TestCase):
    def tearDown(self) -> None:
        install_te_backend_for_tests(None, force=False)

    def test_env_off_forces_emulation(self) -> None:
        install_te_backend_for_tests(FakeTE(), force=True)
        with patch.dict("os.environ", {"CAT_YOKO_TE_NVFP4": "0"}, clear=False):
            self.assertFalse(prefer_te_linear())
        wrap_nvfp4_linears  # keep import used
        from cat_yoko.nvfp4_linear import Nvfp4Linear as Emu

        lin = nn.Linear(4, 4, bias=False)
        wrapped = Emu.from_linear(lin)
        self.assertIsInstance(wrapped, Nvfp4Linear)
        self.assertNotIsInstance(wrapped, TeNvfp4Linear)


if __name__ == "__main__":
    unittest.main()
