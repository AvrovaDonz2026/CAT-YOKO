#!/usr/bin/env python3
"""Published base is MiniCPM5-2B only. MiniCPM-2B graphs and Hub ids are rejected."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.hf_minicpm import load_minicpm_state
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.recipe import (
    MINICPM5_HF,
    MINICPM5_TOKENIZER,
    MINICPM_HF,
    assert_minicpm5_hf_config,
    assert_minicpm5_id,
    is_minicpm5_id,
)
from cat_yoko.train import main as train_main
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm


class MiniCPM5IdTests(unittest.TestCase):
    def test_published_ids_are_minicpm5(self) -> None:
        self.assertEqual(MINICPM_HF, MINICPM5_HF)
        self.assertTrue(is_minicpm5_id(MINICPM5_HF))
        self.assertTrue(is_minicpm5_id(MINICPM5_TOKENIZER))
        self.assertEqual(assert_minicpm5_id(MINICPM5_HF), MINICPM5_HF)
        self.assertEqual(assert_minicpm5_id("/tmp/local-weights"), "/tmp/local-weights")
        self.assertEqual(assert_minicpm5_id("/tmp/minicpm.pt"), "/tmp/minicpm.pt")

    def test_reject_minicpm2b_and_siblings(self) -> None:
        for name in (
            "openbmb/MiniCPM-2B-sft-bf16",
            "openbmb/MiniCPM-2B",
            "openbmb/MiniCPM-2B-128k",
            "openbmb/MiniCPM3-4B",
            "openbmb/MiniCPM4-8B",
            "/data/MiniCPM-2B-sft-bf16",
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    assert_minicpm5_id(name)
                self.assertIn("MiniCPM5-2B", str(ctx.exception))

    def test_hf_config_rejects_minicpm2b_mha(self) -> None:
        bad = SimpleNamespace(
            hidden_size=2304,
            vocab_size=122753,
            num_key_value_heads=36,
            num_attention_heads=36,
            num_hidden_layers=40,
            intermediate_size=5760,
        )
        with self.assertRaises(ValueError) as ctx:
            assert_minicpm5_hf_config(bad)
        msg = str(ctx.exception)
        self.assertIn("2304", msg)
        self.assertIn("122753", msg)

    def test_hf_config_accepts_minicpm5(self) -> None:
        good = SimpleNamespace(
            hidden_size=2048,
            vocab_size=130560,
            num_key_value_heads=2,
            num_attention_heads=16,
            num_hidden_layers=42,
            intermediate_size=6144,
        )
        assert_minicpm5_hf_config(good)

    def test_load_state_rejects_minicpm2b_before_hub(self) -> None:
        with self.assertRaises(ValueError):
            load_minicpm_state("openbmb/MiniCPM-2B-sft-bf16")

    def test_cli_rejects_minicpm2b_upcycle_hf(self) -> None:
        with self.assertRaises(SystemExit):
            train_main(
                [
                    "--config",
                    "tiny",
                    "--upcycle-hf",
                    "openbmb/MiniCPM-2B-sft-bf16",
                    "--steps",
                    "1",
                ]
            )

    def test_load_causal_lm_cpu_requests_cpu_device_map(self) -> None:
        import sys
        from unittest.mock import MagicMock, patch

        from cat_yoko.hf_minicpm import load_causal_lm_cpu

        fake_mod = MagicMock()
        fake_model = MagicMock()
        param = MagicMock()
        param.device.type = "cpu"
        fake_model.parameters.return_value = iter([param])
        fake_mod.AutoModelForCausalLM.from_pretrained.return_value = fake_model
        with patch.dict(sys.modules, {"transformers": fake_mod}):
            load_causal_lm_cpu("openbmb/MiniCPM5-2B-Base")
        kwargs = fake_mod.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs.get("device_map"), "cpu")
        self.assertTrue(kwargs.get("low_cpu_mem_usage"))


class MiniCPM5UpcyleShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)

    def test_gqa_dummy_upcycle_ok(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        src = dummy_minicpm_state(self.cfg)
        upcycle_from_minicpm(model, src, self.cfg)
        self.assertTrue(torch.equal(model.embed.weight, src["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(model.norm.weight, src["model.norm.weight"]))

    def test_mha_k_proj_is_rejected(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        src = dummy_minicpm_state(self.cfg)
        d = self.cfg.hidden_size
        src["model.layers.0.self_attn.k_proj.weight"] = torch.randn(d, d)
        with self.assertRaises(ValueError) as ctx:
            upcycle_from_minicpm(model, src, self.cfg)
        self.assertIn("GQA", str(ctx.exception))

    def test_wrong_dense_ffn_is_rejected(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        src = dummy_minicpm_state(self.cfg)
        d = self.cfg.hidden_size
        src["model.layers.0.mlp.gate_proj.weight"] = torch.randn(self.cfg.dense_intermediate_size - 8, d)
        src["model.layers.0.mlp.up_proj.weight"] = torch.randn(self.cfg.dense_intermediate_size - 8, d)
        with self.assertRaises(ValueError) as ctx:
            upcycle_from_minicpm(model, src, self.cfg)
        self.assertIn("SwiGLU", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
