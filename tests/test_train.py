#!/usr/bin/env python3
"""12B trainer tests: tiny graph, C1 freeze, meta param count."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig, encoder_layer_kind
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze, trainable_names, set_gate
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.train import build_model, train_loop
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm


class ConfigFrozenTests(unittest.TestCase):
    def test_12b_matches_frozen_spec(self) -> None:
        c = CATYokoConfig.middle_12b()
        self.assertEqual(c.hidden_size, 2304)
        self.assertEqual((c.encoder_layers, c.decoder_layers), (16, 24))
        self.assertEqual((c.n_routed_enc, c.n_routed_dec), (17, 17))
        self.assertEqual((c.top_k_enc, c.top_k_dec), (6, 8))
        self.assertFalse(c.first_dense)
        self.assertEqual(c.n_win, 8192)
        self.assertEqual(c.attention_backend, "window")
        self.assertFalse(c.use_muon)
        self.assertEqual(c.residual_scale, c.scale_depth / (40**0.5))
        self.assertEqual(c.logit_scale, 9)
        kinds = [encoder_layer_kind(i) for i in range(16)]
        self.assertEqual((kinds.count("sliding"), kinds.count("csa"), kinds.count("hca")), (2, 7, 7))


class TinyTrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)

    def test_forward_backward(self) -> None:
        nll = train_loop(self.cfg, "B0", steps=2, device="cpu")
        self.assertTrue(nll > 0)

    def test_b0_encoder_has_no_grad(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B0")
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        for p in model.encoder.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertIsNotNone(model.cache_k.weight.grad)
        names = trainable_names(model)
        self.assertTrue(any("cross_attn" in n for n in names))
        self.assertFalse(any("encoder" in n for n in names))

    def test_b1_decoder_trainable_encoder_frozen(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B1")
        self.assertTrue(model.detach_cache)
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertFalse(next(model.encoder.parameters()).requires_grad)
        self.assertTrue(next(model.decoder[0].self_attn.parameters()).requires_grad)
        self.assertTrue(next(model.decoder[0].mlp.parameters()).requires_grad)

    def test_detach_blocks_encoder_grad_even_if_unfrozen(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        model.set_detach(True)
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        for p in model.encoder.parameters():
            self.assertIsNone(p.grad)
        self.assertIsNotNone(model.cache_k.weight.grad)

    def test_fp8_policy_cpu_stays_off(self) -> None:
        self.assertFalse(should_autocast("B1", cuda=False, enabled=True))
        self.assertTrue(should_autocast("B1", cuda=True, enabled=True))
        self.assertFalse(should_autocast("B0", cuda=True, enabled=True))

    def test_document_mask_blocks_other_doc(self) -> None:
        from cat_yoko.attention import _window_causal_bias

        doc_ids = torch.tensor([[0, 0, 1, 1]])
        bias = _window_causal_bias(4, 4, 8, torch.device("cpu"), torch.float32, doc_ids)
        self.assertEqual(tuple(bias.shape), (1, 1, 4, 4))
        self.assertLess(bias[0, 0, 2, 0].item(), -1e4)  # other document
        self.assertEqual(bias[0, 0, 1, 0].item(), 0.0)  # same doc, causal

    def test_b2_unfreeze(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B2")
        self.assertFalse(model.detach_cache)
        self.assertTrue(model.embed.weight.requires_grad)
        self.assertTrue(next(model.encoder.parameters()).requires_grad)

    def test_gate_zero_nulls_cross_attn_delta(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B2")
        model.eval()
        set_gate(model, 0.0)
        ids = torch.randint(0, self.cfg.vocab_size, (1, self.cfg.seq_len))
        with torch.no_grad():
            a = model(ids)["logits"]
            set_gate(model, 0.0)
            # scramble cache projections; gate=0 must ignore them
            model.cache_k.weight.mul_(0)
            b = model(ids)["logits"]
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))

    def test_moe_cpu_bf16_autocast_index_put(self) -> None:
        """GPU B1+FP8 policy uses bf16 autocast; expert writes must match the buffer."""
        from cat_yoko.moe import MoE

        moe = MoE(self.cfg, self.cfg.n_routed_dec, self.cfg.top_k_dec, hash_route=True)
        x = torch.randn(2, self.cfg.seq_len, self.cfg.hidden_size)
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            hashed = moe(x, token_ids=ids)
            routed = moe(x, token_ids=None)
        self.assertEqual(tuple(hashed.shape), tuple(x.shape))
        self.assertEqual(tuple(routed.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(hashed).all())
        self.assertTrue(torch.isfinite(routed).all())

    def test_router_bias_updates_after_backward_not_in_forward(self) -> None:
        from cat_yoko.moe import MoE

        moe = MoE(self.cfg, self.cfg.n_routed_dec, self.cfg.top_k_dec)
        moe.train()
        x = torch.randn(2, self.cfg.seq_len, self.cfg.hidden_size, requires_grad=True)
        before = moe.e_score_correction_bias.clone()
        y = moe(x)
        self.assertTrue(torch.equal(before, moe.e_score_correction_bias))
        y.sum().backward()
        self.assertTrue(torch.equal(before, moe.e_score_correction_bias))
        moe.step_router_bias()
        self.assertFalse(torch.equal(before, moe.e_score_correction_bias))

    def test_upcycle_copies_embed(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        src = dummy_minicpm_state(self.cfg)
        upcycle_from_minicpm(model, src, self.cfg)
        self.assertTrue(torch.equal(model.embed.weight, src["model.embed_tokens.weight"]))

    def test_causal_mask_no_future(self) -> None:
        from cat_yoko.attention import _window_causal_bias

        bias = _window_causal_bias(4, 4, 8, torch.device("cpu"), torch.float32)
        # position 1 cannot see 2
        self.assertLess(bias[1, 2].item(), -1e4)
        self.assertEqual(bias[2, 2].item(), 0.0)


class MetaTwelveBTests(unittest.TestCase):
    def test_meta_param_count_near_12b(self) -> None:
        cfg = CATYokoConfig.middle_12b()
        model = build_model(cfg, "meta")
        n = sum(p.numel() for p in model.parameters())
        # YOCO shared top cache 2d² + per-layer Q/O cross-attn ≈ 11.59B, not 11.83B.
        self.assertGreater(n, 11.50e9)
        self.assertLess(n, 11.70e9)


if __name__ == "__main__":
    unittest.main()
