#!/usr/bin/env python3
"""NVFP4 Linear wrap: B0 frozen GEMMs, B1 student GEMMs, shared Parameters."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.nvfp4_linear import (
    Nvfp4Linear,
    apply_nvfp4,
    nvfp4_linear,
    nvfp4_module_names,
    quantize_nvfp4,
    should_wrap_linear,
)


def _nv_tiny() -> CATYokoConfig:
    return replace(CATYokoConfig.tiny(), use_nvfp4=True)


class QuantizeTests(unittest.TestCase):
    def test_roundtrip_shape_and_finite(self) -> None:
        x = torch.randn(3, 17)
        y = quantize_nvfp4(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(torch.isfinite(y).all())

    def test_ste_linear_has_weight_grad(self) -> None:
        w = nn.Parameter(torch.randn(8, 4))
        x = torch.randn(2, 4, requires_grad=True)
        y = nvfp4_linear(x, w, None)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_from_linear_shares_data_ptr(self) -> None:
        lin = nn.Linear(4, 8, bias=False)
        ptr = lin.weight.data_ptr()
        wrapped = Nvfp4Linear.from_linear(lin)
        self.assertEqual(wrapped.weight.data_ptr(), ptr)
        self.assertIs(wrapped.weight, lin.weight)


class WrapPolicyTests(unittest.TestCase):
    def test_b0_wraps_frozen_skips_student_and_router(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        keys_before = set(model.state_dict().keys())
        n = apply_nvfp4(model, "B0", enabled=True)
        names = nvfp4_module_names(model)
        self.assertGreater(n, 0)
        self.assertEqual(set(model.state_dict().keys()), keys_before)
        self.assertIsInstance(model.encoder[0].attn.q_proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].attn.k_proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].attn.v_proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].attn.o_proj, Nvfp4Linear)
        self.assertIsInstance(model.lm_head, Nvfp4Linear)
        self.assertNotIsInstance(model.cache_k, Nvfp4Linear)
        self.assertNotIsInstance(model.cache_v, Nvfp4Linear)
        self.assertNotIsInstance(model.decoder[0].cross_attn.q_proj, Nvfp4Linear)
        self.assertNotIsInstance(model.decoder[0].cross_attn.o_proj, Nvfp4Linear)
        self.assertFalse(any(n.endswith("router") for n in names))
        self.assertFalse(should_wrap_linear("encoder.0.mlp.router", model.encoder[0].mlp.router, "B0"))
        self.assertEqual(apply_nvfp4(model, "B0", enabled=True), 0)

    def test_b1_wraps_lm_head_attn_moe_cache_cross(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B1")
        keys_before = set(model.state_dict().keys())
        n = apply_nvfp4(model, "B1", enabled=True)
        names = nvfp4_module_names(model)
        self.assertGreater(n, 0)
        self.assertEqual(set(model.state_dict().keys()), keys_before)
        self.assertIsInstance(model.lm_head, Nvfp4Linear)
        enc_attn = model.encoder[0].attn
        dec_attn = model.decoder[0].self_attn
        for proj in (enc_attn.q_proj, enc_attn.k_proj, enc_attn.v_proj, enc_attn.o_proj):
            self.assertIsInstance(proj, Nvfp4Linear)
        for proj in (dec_attn.q_proj, dec_attn.k_proj, dec_attn.v_proj, dec_attn.o_proj):
            self.assertIsInstance(proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].cross_attn.q_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].cross_attn.o_proj, Nvfp4Linear)
        self.assertIsInstance(model.cache_k, Nvfp4Linear)
        self.assertIsInstance(model.cache_v, Nvfp4Linear)
        expert0 = model.decoder[0].mlp.experts[0].gate_proj
        self.assertIsInstance(expert0, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.shared[0].down_proj, Nvfp4Linear)
        self.assertFalse(any(n.endswith("router") for n in names))
        self.assertNotIsInstance(model.encoder[0].mlp.router, Nvfp4Linear)
        self.assertNotIsInstance(model.decoder[0].mlp.router, Nvfp4Linear)

    def test_disabled_is_noop(self) -> None:
        model = CATYokoForCausalLM(_nv_tiny())
        apply_freeze(model, "B1")
        self.assertEqual(apply_nvfp4(model, "B1", enabled=False), 0)
        self.assertEqual(nvfp4_module_names(model), [])

    def test_tiny_default_flag_off(self) -> None:
        self.assertFalse(CATYokoConfig.tiny().use_nvfp4)
        self.assertTrue(CATYokoConfig.middle_12b().use_nvfp4)

    def test_finite_nll_after_b1_wrap(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B1")
        apply_nvfp4(model, "B1", enabled=True)
        torch.manual_seed(0)
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        out = model(input_ids=ids, labels=ids)
        self.assertTrue(torch.isfinite(out["nll"]))
        self.assertGreater(float(out["nll"].detach()), 0.0)
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_trainer_b1_hooks_wrap(self) -> None:
        from cat_yoko.trainer import Trainer

        cfg = _nv_tiny()
        tr = Trainer(cfg, "B1", "cpu", steps=1, accum=1, micro_batch=1)
        out = tr.run()
        self.assertGreater(tr.nvfp4_n, 0)
        self.assertTrue(out.nll == out.nll)
        self.assertGreater(out.nll, 0)


if __name__ == "__main__":
    unittest.main()
