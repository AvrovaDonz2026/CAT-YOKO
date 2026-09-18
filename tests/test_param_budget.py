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


class CurriculumTheoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = pb.compute_budget(pb.TIERS["middle"])
        self.joint = pb.flops_joint(self.budget, 50e9)

    def test_dense_encoder_cheaper_than_moe_encoder(self) -> None:
        dense = pb.encoder_active_no_emb(self.budget, dense=True)
        moe = pb.encoder_active_no_emb(self.budget, dense=False)
        self.assertLess(dense, moe)
        # 16 × MiniCPM SwiGLU vs 16 × 7 experts.
        self.assertAlmostEqual(dense, 16 * (self.budget.attn.self_attn + pb.DENSE_FFN))

    def test_delayed_curriculum_dominates_freeze_moe(self) -> None:
        c1 = pb.flops_curriculum(self.budget, 50e9, delayed=False)
        c2 = pb.flops_curriculum(self.budget, 50e9, delayed=True)
        self.assertLess(c2, c1)
        self.assertLessEqual(c2, 0.80 * self.joint)
        self.assertLessEqual(c1, 0.85 * self.joint)

    def test_freeze_enc_dense_cheaper_than_freeze_enc_moe(self) -> None:
        self.assertLess(
            pb.flops_freeze_encoder(self.budget, 50e9, encoder_dense=True),
            pb.flops_freeze_encoder(self.budget, 50e9, encoder_dense=False),
        )

    def test_detach_and_tied_emb_boundaries(self) -> None:
        b0, b1, b2 = pb.FREEZE_BOUNDARIES
        self.assertTrue(b0.detach_at_cache and b1.detach_at_cache)
        self.assertFalse(b2.detach_at_cache)
        self.assertEqual(b0.tied_emb, "freeze_tied")
        self.assertEqual(b1.tied_emb, "freeze_tied")
        self.assertIn("tied_emb", b0.frozen)
        self.assertIn("tied_emb", b1.frozen)
        self.assertIn("cache_proj", b0.trainable)
        self.assertIn("encoder", b0.frozen)
        self.assertNotIn("train_tied", (b0.tied_emb, b1.tied_emb))

    def test_b1_optimizer_tracks_decoder_share(self) -> None:
        tr, _fr = pb.phase_param_counts(self.budget, "B1", delayed=False)
        b2, _ = pb.phase_param_counts(self.budget, "B2", delayed=False)
        ratio = pb.optimizer_state_bytes(tr) / pb.optimizer_state_bytes(b2)
        self.assertGreater(ratio, 0.55)
        self.assertLess(ratio, 0.65)

    def test_c2_b0_stores_fewer_frozen_params(self) -> None:
        _tr1, fr_c1 = pb.phase_param_counts(self.budget, "B0", delayed=False)
        _tr2, fr_c2 = pb.phase_param_counts(self.budget, "B0", delayed=True)
        self.assertLess(fr_c2, fr_c1)

    def test_valid_splits_stay_under_85_percent(self) -> None:
        for delayed in (False, True):
            for label, _flops, ratio, valid in pb.curriculum_split_table(
                self.budget, delayed=delayed
            ):
                if valid:
                    self.assertLessEqual(ratio, 0.85, msg=label)
                else:
                    self.assertTrue(label.startswith("no B2"))

    def test_default_b2_at_least_10b(self) -> None:
        self.assertGreaterEqual(pb.DEFAULT_CURRICULUM_SPLIT[2], 10e9)
        self.assertAlmostEqual(sum(pb.DEFAULT_CURRICULUM_SPLIT), 50e9)

    def test_cache_proj_in_new_modules(self) -> None:
        n_new = pb.n_new_modules(self.budget)
        cross = self.budget.dec.layers * self.budget.attn.cross_attn
        self.assertEqual(n_new, cross + pb.CACHE_PROJ)
        self.assertGreater(pb.CACHE_PROJ, 0)

    def test_activation_keep_is_decoder_layers(self) -> None:
        self.assertEqual(pb.activation_keep_frac(), 24 / 40)

    def test_frozen_spec_is_c1(self) -> None:
        self.assertFalse(pb.DEFAULT_DELAYED_ENCODER_MOE)
        rec = next(r for r in pb.staged_recipes(self.budget, 50e9) if r.name.startswith("unfreeze"))
        self.assertAlmostEqual(rec.flops, pb.flops_curriculum(self.budget, 50e9, delayed=False))

    def test_curriculum_claims_pass(self) -> None:
        failed = [c for c in pb.claims_curriculum(self.budget) if not c.ok]
        self.assertEqual(failed, [], msg=[c.name for c in failed])


class Fp8TheoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.budget = pb.compute_budget(pb.TIERS["middle"])

    def test_fp8_does_not_change_six_nt(self) -> None:
        c1 = pb.flops_curriculum(self.budget, 50e9, delayed=False)
        self.assertEqual(c1, sum(pb.curriculum_phase_flops(self.budget)))
        self.assertAlmostEqual(
            pb.wallclock_h100_h(c1, speedup=1.5) * 1.5,
            pb.wallclock_h100_h(c1),
        )

    def test_conservative_speedup_is_1_5_not_peak(self) -> None:
        self.assertEqual(pb.FP8_SPEEDUP_CONSERVATIVE, 1.5)
        self.assertAlmostEqual(pb.FP8_SPEEDUP_PEAK, 2.0, places=2)
        self.assertLess(pb.FP8_SPEEDUP_CONSERVATIVE, pb.FP8_SPEEDUP_PEAK)

    def test_moe_is_the_considerable_portion_of_six_nt(self) -> None:
        self.assertGreaterEqual(pb.fp8_moe_active_frac(self.budget), 0.70)
        self.assertGreaterEqual(self.budget.moe_frac, 0.80)
        self.assertGreater(pb.fp8_gemm_frac(self.budget), pb.fp8_moe_active_frac(self.budget))

    def test_moe_only_amdahl_justifies_published_1_5(self) -> None:
        speedup = pb.fp8_speedup_from_gemm_frac(pb.fp8_moe_active_frac(self.budget))
        self.assertGreaterEqual(speedup, 1.50)
        self.assertLessEqual(speedup, 1.75)

    def test_phase_policy_keeps_b0_and_l0_bf16(self) -> None:
        pol = {p.phase: p for p in pb.FP8_PHASE_POLICY}
        self.assertEqual(pol["L0"].student, "bf16")
        self.assertEqual(pol["B0"].student, "bf16")
        self.assertEqual(pol["B0"].frozen_encoder_gemm, "fp8")
        self.assertEqual(pol["B1"].student, "fp8_moe")
        self.assertEqual(pol["B2"].student, "fp8_moe")
        self.assertEqual(pol["C"].student, "bf16")

    def test_keep_high_prec_covers_tied_router_norm(self) -> None:
        self.assertTrue(
            set(pb.FP8_KEEP_HIGH_PREC)
            >= {"tied_emb", "router", "rms_norm", "gate", "indexer", "attn_softmax"}
        )

    def test_mixed_policy_under_60pct_of_joint(self) -> None:
        joint_h = pb.wallclock_h100_h(pb.flops_joint(self.budget, 50e9))
        c1_h = pb.wallclock_h100_h(pb.flops_curriculum(self.budget, 50e9, delayed=False))
        mixed = pb.wallclock_c1_fp8_policy(self.budget)
        self.assertLess(mixed, c1_h)
        self.assertLessEqual(mixed, 0.60 * joint_h)
        self.assertAlmostEqual(mixed, 761, delta=15)

    def test_b0_is_a_minority_of_c1_flops(self) -> None:
        f0, f1, f2 = pb.curriculum_phase_flops(self.budget)
        self.assertLessEqual(f0 / (f0 + f1 + f2), 0.15)
        t0, t1, t2 = pb.DEFAULT_CURRICULUM_SPLIT
        self.assertGreaterEqual((t1 + t2) / (t0 + t1 + t2), 0.80)

    def test_fp8_claims_pass(self) -> None:
        failed = [c for c in pb.claims_fp8(self.budget) if not c.ok]
        self.assertEqual(failed, [], msg=[c.name for c in failed])

    def test_verify_includes_fp8(self) -> None:
        names = [c.name for c in pb.verify(self.budget)]
        self.assertTrue(any(n.startswith("FP8") for n in names))
        self.assertGreaterEqual(len(names), 46)


if __name__ == "__main__":
    unittest.main()
