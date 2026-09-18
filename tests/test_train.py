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
        self.assertEqual(c.hidden_size, 2048)
        self.assertEqual((c.encoder_layers, c.decoder_layers), (16, 26))
        self.assertEqual((c.n_routed_enc, c.n_routed_dec), (20, 20))
        self.assertEqual((c.top_k_enc, c.top_k_dec), (7, 10))
        self.assertFalse(c.first_dense)
        self.assertEqual(c.n_win, 8192)
        self.assertEqual(c.attention_backend, "window")
        self.assertFalse(c.use_muon)
        self.assertFalse(c.use_mup)
        self.assertEqual(c.residual_scale, 1)
        self.assertEqual(c.logit_scale, 1)
        self.assertEqual(c.num_kv_heads, 2)
        self.assertFalse(c.tie_embeddings)
        self.assertEqual(c.vocab_size, 130560)
        self.assertEqual(c.dense_intermediate_size, 6144)
        self.assertEqual(c.rms_eps, 1e-6)
        self.assertEqual(c.rope_theta, 5_000_000.0)
        self.assertEqual(c.embed_scale, 1)
        self.assertEqual(c.base_layers, 42)
        kinds = [encoder_layer_kind(i) for i in range(16)]
        self.assertEqual((kinds.count("sliding"), kinds.count("csa"), kinds.count("hca")), (2, 7, 7))

    def test_mup_off_ignores_minicpm2b_dim_model_base(self) -> None:
        from dataclasses import replace

        c = replace(CATYokoConfig.middle_12b(), dim_model_base=256, scale_emb=12.0)
        self.assertFalse(c.use_mup)
        self.assertEqual(c.logit_scale, 1)
        self.assertEqual(c.embed_scale, 1)
        self.assertEqual(c.residual_scale, 1)

    def test_gqa_projections_use_kv_dim(self) -> None:
        c = CATYokoConfig.middle_12b()
        model = build_model(c, "meta")
        self.assertEqual(c.kv_dim, 256)
        self.assertEqual(model.encoder[0].attn.k_proj.out_features, c.kv_dim)
        self.assertEqual(model.encoder[0].attn.v_proj.out_features, c.kv_dim)
        self.assertEqual(model.cache_k.out_features, c.kv_dim)
        self.assertEqual(model.cache_v.out_features, c.kv_dim)
        self.assertEqual(model.decoder[0].self_attn.k_proj.out_features, c.kv_dim)


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
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertIsNotNone(model.cache_k.weight.grad)
        names = trainable_names(model)
        self.assertTrue(any("cross_attn" in n for n in names))
        self.assertFalse(any("encoder" in n for n in names))

    def test_b1_decoder_trainable_encoder_frozen(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B1")
        self.assertTrue(model.detach_cache)
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)
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

    def test_new_modules_small_init(self) -> None:
        from cat_yoko.upcycle import NEW_MODULE_INIT_STD

        torch.manual_seed(0)
        model = CATYokoForCausalLM(self.cfg)
        self.assertAlmostEqual(
            float(model.cache_k.weight.detach().std()), NEW_MODULE_INIT_STD, delta=0.008
        )
        self.assertAlmostEqual(
            float(model.cache_v.weight.detach().std()), NEW_MODULE_INIT_STD, delta=0.008
        )
        self.assertAlmostEqual(
            float(model.decoder[0].cross_attn.q_proj.weight.detach().std()),
            NEW_MODULE_INIT_STD,
            delta=0.008,
        )
        self.assertAlmostEqual(
            float(model.decoder[0].mlp.router.weight.detach().std()),
            NEW_MODULE_INIT_STD,
            delta=0.015,
        )

    def test_meta_skips_weight_init(self) -> None:
        model = build_model(self.cfg, "meta")
        self.assertEqual(next(model.parameters()).device.type, "meta")

    def test_rmsnorm_fp32_under_autocast(self) -> None:
        from cat_yoko.rope import RMSNorm

        torch.manual_seed(0)
        n = RMSNorm(16)
        x = torch.randn(4, 16)
        y32 = n(x)
        xbf = x.to(torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ybf = n(xbf)
        self.assertEqual(ybf.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(ybf.float(), y32, atol=2e-2, rtol=2e-2))

    def test_router_logits_fp32_under_autocast(self) -> None:
        from cat_yoko.moe import MoE

        moe = MoE(self.cfg, self.cfg.n_routed_dec, self.cfg.top_k_dec)
        moe.train()
        x = torch.randn(2, self.cfg.seq_len, self.cfg.hidden_size)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y = moe(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all())
        self.assertIsNotNone(moe.last_aux)

    def test_muon_switch_raises(self) -> None:
        from dataclasses import replace

        from cat_yoko.optim import build_optimizer

        cfg = replace(self.cfg, use_muon=True)
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        with self.assertRaises(NotImplementedError):
            build_optimizer(model, cfg)

    def test_upcycle_copies_embed(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        src = dummy_minicpm_state(self.cfg)
        upcycle_from_minicpm(model, src, self.cfg)
        self.assertTrue(torch.equal(model.embed.weight, src["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(model.lm_head.weight, src["lm_head.weight"]))

    def test_causal_mask_no_future(self) -> None:
        from cat_yoko.attention import _window_causal_bias

        bias = _window_causal_bias(4, 4, 8, torch.device("cpu"), torch.float32)
        # position 1 cannot see 2
        self.assertLess(bias[1, 2].item(), -1e4)
        self.assertEqual(bias[2, 2].item(), 0.0)

    def test_causal_fastpath_matches_mask(self) -> None:
        from cat_yoko.attention import _needs_explicit_mask, _sdpa, _window_causal_bias

        torch.manual_seed(0)
        q = torch.randn(1, 2, 8, 8)
        k = torch.randn(1, 2, 8, 8)
        v = torch.randn(1, 2, 8, 8)
        bias = _window_causal_bias(8, 8, 8, q.device, torch.float32)
        masked = _sdpa(q, k, v, bias)
        fast = _sdpa(q, k, v, causal=True)
        self.assertTrue(torch.allclose(fast, masked, atol=1e-4, rtol=1e-4))
        doc = torch.zeros(2, 8, dtype=torch.long)
        self.assertFalse(_needs_explicit_mask(8, 8, doc))
        mixed = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [0] * 8])
        self.assertTrue(_needs_explicit_mask(8, 8, mixed))

    def test_b0_frozen_moe_aux_is_zero(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B0")
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        out = model(input_ids=ids, labels=ids)
        self.assertEqual(float(out["aux"]), 0.0)
        self.assertGreater(int(out["n_valid"]), 0)

    def test_b0_does_not_step_frozen_router_bias(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B0")
        before_dec = model.decoder[-1].mlp.e_score_correction_bias.clone()
        before_enc = model.encoder[0].mlp.e_score_correction_bias.clone()
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        model.step_router_bias()
        self.assertTrue(torch.equal(before_dec, model.decoder[-1].mlp.e_score_correction_bias))
        self.assertTrue(torch.equal(before_enc, model.encoder[0].mlp.e_score_correction_bias))

    def test_b1_decoder_router_bias_updates(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        apply_freeze(model, "B1")
        before_dec = model.decoder[-1].mlp.e_score_correction_bias.clone()
        before_enc = model.encoder[0].mlp.e_score_correction_bias.clone()
        ids = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        model.step_router_bias()
        self.assertFalse(torch.equal(before_dec, model.decoder[-1].mlp.e_score_correction_bias))
        self.assertTrue(torch.equal(before_enc, model.encoder[0].mlp.e_score_correction_bias))

    def test_moe_load_accumulates_across_forwards(self) -> None:
        from cat_yoko.moe import MoE

        moe = MoE(self.cfg, self.cfg.n_routed_dec, self.cfg.top_k_dec)
        moe.train()
        x = torch.randn(2, self.cfg.seq_len, self.cfg.hidden_size)
        moe(x)
        self.assertEqual(moe._load_n, 1)
        first = moe.mean_pending_load().clone()
        moe(x)
        self.assertEqual(moe._load_n, 2)
        averaged = moe.mean_pending_load()
        self.assertEqual(tuple(averaged.shape), tuple(first.shape))
        moe.step_router_bias()
        self.assertEqual(moe._load_n, 0)
        self.assertIsNone(moe.last_load)


class MetaTwelveBTests(unittest.TestCase):
    def test_meta_param_count_near_12b(self) -> None:
        cfg = CATYokoConfig.middle_12b()
        model = build_model(cfg, "meta")
        n = sum(p.numel() for p in model.parameters())
        # MiniCPM5 GQA 12B stored params ~12.25B (not MiniCPM-2B 11.50–11.70B).
        self.assertGreater(n, 12.20e9)
        self.assertLess(n, 12.32e9)


if __name__ == "__main__":
    unittest.main()
