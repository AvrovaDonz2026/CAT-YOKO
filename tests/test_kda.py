#!/usr/bin/env python3
"""KDA gated-delta path: KV savings, causality, default-off schedule."""

from __future__ import annotations

import inspect
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.attention import WindowAttention
from cat_yoko.config import (
    CATYokoConfig,
    KEEP_HIGH_PREC,
    decoder_layer_kind,
    encoder_layer_kind,
)
from cat_yoko.indexer import set_sparse_mode
from cat_yoko.kda import KDAGates, gated_delta_scan, kda_attend, kv_ledger
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.phases import c_chain, resolve_phase_spec
from cat_yoko.trainer import Trainer


class ScheduleTests(unittest.TestCase):
    def test_default_12b_is_still_2_7_7(self) -> None:
        kinds = [encoder_layer_kind(i) for i in range(16)]
        self.assertEqual((kinds.count("sliding"), kinds.count("csa"), kinds.count("hca")), (2, 7, 7))
        self.assertEqual(kinds.count("kda"), 0)
        self.assertFalse(CATYokoConfig.middle_12b().use_kda)
        self.assertIn("kda_gate", KEEP_HIGH_PREC)

    def test_kda_encoder_is_three_to_one(self) -> None:
        kinds = [encoder_layer_kind(i, 16, use_kda=True) for i in range(16)]
        self.assertEqual(kinds[:2], ["sliding", "sliding"])
        self.assertEqual(kinds[5], "csa")
        self.assertEqual(kinds[9], "hca")
        n_kda = kinds.count("kda")
        n_sparse = kinds.count("csa") + kinds.count("hca")
        self.assertEqual((n_kda, n_sparse), (11, 3))

    def test_kda_decoder_keeps_sliding_anchors(self) -> None:
        kinds = [decoder_layer_kind(i, 26, use_kda=True) for i in range(26)]
        self.assertEqual(kinds.count("sliding"), 6)
        self.assertEqual(kinds.count("kda"), 20)
        self.assertEqual(kinds[3], "sliding")
        self.assertEqual(kinds[0], "kda")

    def test_tiny_keeps_csa_anchor(self) -> None:
        kinds = [encoder_layer_kind(i, 2, use_kda=True) for i in range(2)]
        self.assertEqual(kinds, ["sliding", "csa"])
        cfg = replace(CATYokoConfig.tiny(), use_kda=True)
        model = CATYokoForCausalLM(cfg)
        self.assertFalse(any(getattr(b, "kind", None) == "kda" for b in model.encoder))
        self.assertFalse(any("kda" in n for n, _ in model.named_modules()))


class ScanTests(unittest.TestCase):
    def test_stepwise_state_matches_full(self) -> None:
        torch.manual_seed(0)
        b, h, t, d = 2, 2, 6, 4
        q = torch.randn(b, h, t, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)
        alpha = torch.sigmoid(torch.randn(b, h, t, d))
        beta = torch.sigmoid(torch.randn(b, h, t))
        y_full, st = gated_delta_scan(q, k, v, alpha, beta, return_state=True)
        state = None
        chunks = []
        for i in range(t):
            y_i, state = gated_delta_scan(
                q[:, :, i : i + 1],
                k[:, :, i : i + 1],
                v[:, :, i : i + 1],
                alpha[:, :, i : i + 1],
                beta[:, :, i : i + 1],
                state=state,
                return_state=True,
            )
            chunks.append(y_i)
        y_step = torch.cat(chunks, dim=2)
        self.assertTrue(torch.allclose(y_full, y_step, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(st, state, atol=1e-4, rtol=1e-4))

    def test_kda_attend_is_causal(self) -> None:
        cfg = CATYokoConfig.tiny()
        attn = WindowAttention(cfg)
        gates = KDAGates(cfg)
        x = torch.randn(1, 8, cfg.hidden_size)
        y1 = kda_attend(attn, gates, x)
        x2 = x.clone()
        x2[:, -1] = torch.randn_like(x2[:, -1])
        y2 = kda_attend(attn, gates, x2)
        self.assertTrue(torch.allclose(y1[:, :-1], y2[:, :-1], atol=2e-4, rtol=2e-4))
        self.assertFalse(torch.allclose(y1[:, -1], y2[:, -1], atol=1e-3))

    def test_doc_boundary_resets_state(self) -> None:
        cfg = CATYokoConfig.tiny()
        attn = WindowAttention(cfg)
        gates = KDAGates(cfg)
        x = torch.randn(1, 6, cfg.hidden_size)
        docs = torch.tensor([[0, 0, 0, 1, 1, 1]])
        y_all = kda_attend(attn, gates, x, docs)
        y_b = kda_attend(attn, gates, x[:, 3:], None)
        self.assertTrue(torch.allclose(y_all[:, 3:], y_b, atol=2e-4, rtol=2e-4))


class KvLedgerTests(unittest.TestCase):
    def test_kda_does_not_shrink_yoco_cache(self) -> None:
        cfg = CATYokoConfig.middle_12b()
        off = kv_ledger(cfg, 131_072, use_kda=False)
        on = kv_ledger(cfg, 131_072, use_kda=True)
        self.assertEqual(off.yoco_cache_bytes, on.yoco_cache_bytes)
        self.assertEqual(on.decoder_kda_layers, 20)
        self.assertGreater(on.saved_vs_all_window, 0.0)
        self.assertLess(on.decode_bytes, off.decode_bytes)
        self.assertGreater(on.saved_vs_fullseq_decoder / on.decode_fullseq_decoder_bytes, 0.7)

    def test_kda_state_bytes_do_not_scale_with_seq(self) -> None:
        cfg = CATYokoConfig.middle_12b()
        from cat_yoko.kda import kda_state_bytes

        self.assertEqual(kda_state_bytes(cfg, 20), 20 * 16 * 128 * 128 * 2)
        a = kv_ledger(cfg, 8_192, use_kda=True)
        b = kv_ledger(cfg, 131_072, use_kda=True)
        self.assertEqual(a.decoder_kda_layers, b.decoder_kda_layers)
        self.assertEqual(a.decoder_kda_layers, 20)


class GraphTests(unittest.TestCase):
    def test_attention_module_has_no_kda_class(self) -> None:
        import cat_yoko.attention as attn_mod

        src = inspect.getsource(attn_mod)
        self.assertNotIn("class KDA", src)
        self.assertNotIn("gated_delta_scan", src)
        self.assertIn("class KDAGates", inspect.getsource(KDAGates))

    def test_opt_in_tiny_kda_step(self) -> None:
        cfg = replace(CATYokoConfig.tiny(), encoder_layers=4, decoder_layers=4, use_kda=True)
        model = CATYokoForCausalLM(cfg)
        self.assertTrue(any(b.kind == "kda" for b in model.encoder))
        self.assertTrue(any(b.kind == "kda" for b in model.decoder))
        set_sparse_mode(model, "kda")
        self.assertEqual(model.encoder[1].sparse_mode, "kda")
        csa = next(b for b in model.encoder if b.kind == "csa")
        self.assertEqual(csa.sparse_mode, "window")
        set_sparse_mode(model, "topk")
        self.assertEqual(model.encoder[1].sparse_mode, "kda")
        self.assertEqual(csa.sparse_mode, "topk")
        nll = Trainer(cfg, "C-topk", "cpu", steps=1, accum=1, micro_batch=1).run().nll
        self.assertTrue(math.isfinite(nll))

    def test_c_kda_does_not_light_csa(self) -> None:
        cfg = replace(CATYokoConfig.tiny(), encoder_layers=4, decoder_layers=4, use_kda=True)
        model = CATYokoForCausalLM(cfg)
        set_sparse_mode(model, "kda")
        kinds_modes = [(b.kind, b.sparse_mode) for b in model.encoder]
        self.assertIn(("kda", "kda"), kinds_modes)
        self.assertIn(("csa", "window"), kinds_modes)
        self.assertNotIn(("csa", "topk"), kinds_modes)
        nll = Trainer(cfg, "C-kda", "cpu", steps=1, accum=1, micro_batch=1).run().nll
        self.assertTrue(math.isfinite(nll))

    def test_kda_chain_puts_hca_last(self) -> None:
        chain = c_chain(use_kda=True)
        self.assertEqual(chain[0], "C-kda")
        self.assertEqual(chain[-1], "C-win")
        self.assertLess(chain.index("C-topk"), chain.index("C-hca"))
        self.assertEqual(c_chain(use_kda=False)[0], "C-index")
        self.assertEqual(resolve_phase_spec("C-index").sparse, "window")

    def test_default_tiny_has_no_kda_params(self) -> None:
        model = CATYokoForCausalLM(CATYokoConfig.tiny())
        self.assertFalse(any("kda" in n for n, _ in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
