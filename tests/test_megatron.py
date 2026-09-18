#!/usr/bin/env python3
"""Megatron-LM adapter: mapping only, no megatron-core import required."""

from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.config import CATYokoConfig, encoder_layer_kind
from cat_yoko.megatron.mapping import (
    MEGATRON_LM,
    encoder_csa_compress_ratios,
    megatron_blueprint,
)
from cat_yoko.megatron.provider import MegatronNotInstalled, dump_mapping, import_megatron_core
from cat_yoko.parallel import ParallelPlan, legal_expert_parallel, legal_tensor_parallel, validate_parallel
from cat_yoko.train import main


class ParallelPlanTests(unittest.TestCase):
    def test_12b_legal_tp_and_prime_ep(self) -> None:
        cfg = CATYokoConfig.middle_12b()
        self.assertEqual(legal_tensor_parallel(cfg.num_heads, cfg.hidden_size), (1, 2, 4, 8, 16))
        self.assertEqual(legal_expert_parallel(cfg.n_routed_enc), (1, 2, 4, 5, 10, 20))
        validate_parallel(cfg, ParallelPlan())
        validate_parallel(cfg, ParallelPlan(tensor_parallel=4, expert_parallel=10))
        with self.assertRaises(ValueError):
            validate_parallel(cfg, ParallelPlan(expert_parallel=8))
        with self.assertRaises(ValueError):
            validate_parallel(cfg, ParallelPlan(tensor_parallel=5))
        with self.assertRaises(ValueError):
            validate_parallel(cfg, ParallelPlan(sequence_parallel=True))


class MappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.middle_12b()
        self.bp = megatron_blueprint(self.cfg, phase="B1")

    def test_upstream_url(self) -> None:
        self.assertEqual(self.bp["upstream"], MEGATRON_LM)
        self.assertTrue(MEGATRON_LM.startswith("https://github.com/NVIDIA/Megatron-LM"))

    def test_two_stacks_differ_on_topk_and_depth(self) -> None:
        enc, dec = self.bp["encoder"], self.bp["decoder"]
        self.assertEqual(enc["num_layers"], 16)
        self.assertEqual(dec["num_layers"], 26)
        self.assertEqual(enc["moe_router_topk"], 7)
        self.assertEqual(dec["moe_router_topk"], 10)
        self.assertEqual(enc["num_moe_experts"], 20)
        self.assertEqual(dec["num_moe_experts"], 20)
        self.assertEqual(enc["num_query_groups"], 2)
        self.assertEqual(dec["num_query_groups"], 2)
        self.assertEqual(dec["moe_ffn_hidden_size"], 2048)
        self.assertEqual(enc["moe_shared_expert_intermediate_size"], 2048)
        self.assertFalse(self.bp["yoco"]["tied_embeddings"])
        self.assertEqual(self.bp["yoco"]["scale_emb"], 1)
        self.assertEqual(self.bp["yoco"]["logit_scale"], 1)
        self.assertEqual(self.bp["yoco"]["residual_scale"], 1)
        self.assertTrue(self.bp["training"]["untie_embeddings_and_output_weights"])

    def test_router_matches_frozen_spec(self) -> None:
        enc = self.bp["encoder"]
        self.assertTrue(enc["moe_router_pre_softmax"])
        self.assertEqual(enc["moe_router_score_function"], "sqrtsoftplus")
        self.assertTrue(enc["moe_router_enable_expert_bias"])
        self.assertEqual(enc["moe_z_loss_coeff"], 1e-4)
        self.assertEqual(enc["normalization"], "RMSNorm")
        self.assertTrue(enc["qk_layernorm"])
        self.assertTrue(enc["gated_linear_unit"])
        self.assertEqual(enc["window_size"], [8192, 0])
        self.assertIsNone(enc["mtp_num_layers"])

    def test_b1_enables_fp8_hybrid_b0_does_not(self) -> None:
        b0 = megatron_blueprint(self.cfg, phase="B0")
        self.assertIsNone(b0["encoder"]["fp8"])
        self.assertEqual(self.bp["encoder"]["fp8"], "hybrid")
        self.assertTrue(self.bp["yoco"]["detach_cache"])
        b2 = megatron_blueprint(self.cfg, phase="B2")
        self.assertFalse(b2["yoco"]["detach_cache"])

    def test_csa_ratios_are_megatron_legal(self) -> None:
        ratios = encoder_csa_compress_ratios(self.cfg)
        self.assertEqual(len(ratios), 16)
        self.assertEqual(ratios[:2], [0, 0])
        self.assertEqual(set(ratios), {0, 4, 128})
        self.assertEqual(ratios.count(0), 2)
        self.assertEqual(ratios.count(4), 7)
        self.assertEqual(ratios.count(128), 7)
        self.assertEqual(self.bp["yoco"]["phase_c_csa"]["csa_compress_ratios"], ratios)
        kinds = [encoder_layer_kind(i) for i in range(16)]
        self.assertEqual(kinds.count("sliding"), 2)
        self.assertEqual(kinds.count("csa"), 7)
        self.assertEqual(kinds.count("hca"), 7)

    def test_forbids_gpt_model_in_custom_surface(self) -> None:
        blob = " ".join(self.bp["yoco"]["custom_surface"]).lower()
        self.assertIn("gptmodel", blob)

    def test_json_roundtrip(self) -> None:
        json.dumps(self.bp)

    def test_import_mapping_does_not_need_megatron(self) -> None:
        self.assertNotIn("megatron.core", sys.modules)


class ProviderStubTests(unittest.TestCase):
    def test_dump_cli(self) -> None:
        buf = io.StringIO()
        payload = dump_mapping(CATYokoConfig.tiny(), ParallelPlan(), "B0", file=buf)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["encoder"]["num_layers"], 2)
        self.assertEqual(payload["decoder"]["moe_router_topk"], 2)

    def test_import_megatron_core_raises_without_install(self) -> None:
        try:
            import megatron.core  # noqa: F401
        except ImportError:
            with self.assertRaises(MegatronNotInstalled) as ctx:
                import_megatron_core()
            self.assertIn("github.com/NVIDIA/Megatron-LM", str(ctx.exception))
        else:
            self.skipTest("megatron-core is installed in this env")

    def test_backend_megatron_exits_without_loop(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            code = main(["--backend", "megatron", "--config", "tiny", "--phase", "B0"])
        finally:
            sys.stderr = old
        self.assertIn(code, {2, 3})
        self.assertIn("Megatron", buf.getvalue())

    def test_dump_megatron_cli_zero(self) -> None:
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            self.assertEqual(main(["--config", "12b", "--dump-megatron", "--phase", "B1"]), 0)
        finally:
            sys.stdout = old
        self.assertIn("sqrtsoftplus", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
