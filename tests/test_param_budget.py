#!/usr/bin/env python3
"""Unit tests for the CAT-YOKO middle-tier theoretical budget."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import param_budget as pb  # noqa: E402


class MiddleTierPlaceholderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = pb.compute_budget(pb.TIERS["middle"])

    def test_published_targets(self) -> None:
        self.assertAlmostEqual(self.budget.total / 1e9, 12.05, places=2)
        self.assertAlmostEqual(self.budget.enc_active / 1e9, 2.29, places=2)
        self.assertAlmostEqual(self.budget.dec_active / 1e9, 4.49, places=2)

    def test_sparsity_is_top_k_over_experts_not_the_old_37_53(self) -> None:
        self.assertAlmostEqual(self.budget.sparsity_enc, 7 / 18)
        self.assertAlmostEqual(self.budget.sparsity_dec, 9 / 18)

    def test_three_tiers_conserve_expert_slots(self) -> None:
        slots = [pb.compute_budget(t).expert_slots for t in pb.TIERS.values()]
        self.assertEqual(slots, [720, 720, 720])
        totals = [pb.compute_budget(t).total for t in pb.TIERS.values()]
        self.assertTrue(all(t == totals[0] for t in totals))

    def test_attn_fraction_is_13pct_not_7pct(self) -> None:
        self.assertAlmostEqual(self.budget.attn_frac, 0.13, places=2)

    def test_claim_ledger_passes(self) -> None:
        failed = [c for c in pb.verify(self.budget) if not c.ok]
        self.assertEqual(failed, [], msg=[c.name for c in failed])

    def test_kv_1m_matches_plan_section_16(self) -> None:
        recipes = {r.name: r.bytes / 1e9 for r in pb.kv_table(1_000_000)}
        self.assertAlmostEqual(recipes["decoder-only MHA (40L)"], 368.64, places=2)
        self.assertAlmostEqual(recipes["decoder-only GQA-4 (40L)"], 40.96, places=2)
        self.assertAlmostEqual(recipes["decoder-only MLA-576 (40L)"], 46.08, places=2)
        self.assertAlmostEqual(recipes["YOCO + MLA-576 (1 global)"], 1.152, places=3)
        self.assertAlmostEqual(recipes["YOCO + MLA-576 + seq÷8"], 0.144, places=3)

    def test_training_hours_middle_50b(self) -> None:
        flops = pb.training_flops(self.budget.enc_active + self.budget.dec_active, 50e9)
        self.assertAlmostEqual(pb.gpu_hours(flops, pb.H100_BF16_EFF), 1413, delta=15)

    def test_csa_keys_window_dominated_at_long_context(self) -> None:
        n = 262_144
        self.assertEqual(pb.keys_csa(n), 256 + 8192)
        self.assertLess(pb.keys_csa(n) / n, 0.04)

    def test_csa_not_cheaper_than_dense_inside_the_window(self) -> None:
        n = 4096
        self.assertGreaterEqual(pb.keys_csa(n), pb.keys_dense(n))

    def test_mup_logit_scale(self) -> None:
        self.assertEqual(pb.D / pb.DIM_MODEL_BASE, 9)

    def test_first_dense_stays_near_middle_activations(self) -> None:
        dense = pb.compute_budget(pb.TIERS["middle"], first_dense=True)
        self.assertLess(abs(dense.enc_active - 2.3e9) / 2.3e9, 0.05)
        self.assertLess(abs(dense.dec_active - 4.5e9) / 4.5e9, 0.05)
        self.assertLess(dense.total, 12e9)

    def test_first_dense_retune_recovers_12b_without_touching_top_k(self) -> None:
        tier, budget = pb.retune_routed(
            first_dense=True, attn=pb.attn_accounting("placeholder")
        )
        self.assertEqual((tier.tk_e, tier.tk_d), (6, 8))
        self.assertAlmostEqual(budget.total / 1e9, 12.05, delta=0.05)
        self.assertEqual((tier.nr_e, tier.nr_d), (19, 17))


class StagedTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = pb.compute_budget(pb.TIERS["middle"])
        self.joint = pb.flops_joint(self.budget, 50e9)

    def test_freeze_encoder_saves_about_one_fifth(self) -> None:
        ratio = pb.flops_freeze_encoder(self.budget, 50e9) / self.joint
        self.assertGreater(ratio, 0.74)
        self.assertLess(ratio, 0.82)

    def test_independent_merge_costs_more_than_joint(self) -> None:
        rec = next(r for r in pb.staged_recipes(self.budget, 50e9) if r.name.startswith("independent"))
        self.assertGreater(rec.flops, self.joint)

    def test_unfreeze_curriculum_saves_without_extra_tokens(self) -> None:
        rec = next(r for r in pb.staged_recipes(self.budget, 50e9) if r.name.startswith("unfreeze"))
        self.assertAlmostEqual(rec.tokens, 50e9)
        self.assertLessEqual(rec.flops, 0.85 * self.joint)

    def test_new_modules_cheaper_than_freeze_enc(self) -> None:
        self.assertLess(
            pb.flops_new_modules(self.budget, 50e9),
            pb.flops_freeze_encoder(self.budget, 50e9),
        )


if __name__ == "__main__":
    unittest.main()
