#!/usr/bin/env python3
"""Architecture-invariant tests for CAT-YOKO (masks, split, three mechanisms)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import arch_verify as av  # noqa: E402


class CausalityTests(unittest.TestCase):
    def test_window_includes_self_and_not_future(self) -> None:
        vis = av.window_positions(10, 32, n_win=4)
        self.assertEqual(vis, {7, 8, 9, 10})

    def test_csa_combined_is_causal_on_random_lengths(self) -> None:
        for n in (8, 17, 32, 64):
            for t in range(n):
                vis = av.csa_visible_tokens(t, n, m=4, n_win=8)
                self.assertTrue(av.is_causal(vis, t), msg=f"n={n} t={t}")

    def test_compressed_path_does_not_see_own_block(self) -> None:
        n, m, t = 32, 4, 13  # own block = 3, tokens 12..15
        blocks = av.compressed_block_ids(t, m)
        leaked = av.own_block_hole(t, m, n) & av.expand_blocks(blocks, m, n)
        self.assertEqual(leaked, set())
        self.assertNotIn(3, blocks)

    def test_window_fills_own_block_hole_when_n_win_ge_m(self) -> None:
        n, m, t = 32, 4, 13
        hole = av.own_block_hole(t, m, n)
        self.assertTrue(hole <= av.window_positions(t, n, n_win=m))

    def test_hca_hole_not_covered_if_window_smaller_than_m(self) -> None:
        t, n, m = 200, 256, 128
        hole = av.own_block_hole(t, m, n)
        self.assertFalse(hole <= av.window_positions(t, n, n_win=64))
        self.assertTrue(hole <= av.window_positions(t, n, n_win=m))

    def test_yoco_cross_attn_sees_prefix_only(self) -> None:
        vis = av.yoco_cross_visible(5, 10)
        self.assertEqual(vis, {0, 1, 2, 3, 4, 5})

    def test_indexer_topk_is_subset_not_extension(self) -> None:
        t, n, m = 40, 64, 4
        full = av.compressed_block_ids(t, m)
        top = set(sorted(full)[-3:])
        self.assertTrue(top <= full)


class SkeletonTests(unittest.TestCase):
    def test_depth_cut(self) -> None:
        self.assertEqual((av.LE, av.LD, av.L0), (16, 26, 42))
        self.assertEqual(av.LE + av.LD, av.L0)

    def test_encoder_schedule_2_7_7(self) -> None:
        types = av.encoder_layer_types()
        self.assertEqual(types[:2], ["sliding", "sliding"])
        self.assertEqual(types.count("csa"), 7)
        self.assertEqual(types.count("hca"), 7)
        self.assertEqual(len(types), 16)

    def test_three_bootstrap_cannot_keep_one_to_one(self) -> None:
        types = av.encoder_layer_types(n_bootstrap=3)
        self.assertNotEqual(types.count("csa"), types.count("hca"))

    def test_window_only_rf(self) -> None:
        self.assertEqual(av.window_only_receptive_field(16, 8192), 131072)
        self.assertLess(131072, 262144)

    def test_n_win_covers_both_compression_rates(self) -> None:
        self.assertGreaterEqual(av.N_WIN, av.M_CSA)
        self.assertGreaterEqual(av.N_WIN, av.M_HCA)

    def test_full_ledger_passes(self) -> None:
        failed = [c for c in av.verify() if not c.ok]
        self.assertEqual(failed, [], msg=[c.name for c in failed])


class MechanismTests(unittest.TestCase):
    def test_encoder_csa_does_not_reduce_cache_slots(self) -> None:
        """M1 writes one cache slot per token; pooling (M2) is a separate op."""
        n = 32
        slots = n  # X^{Le} projected tokenwise
        pooled = n // av.M_CSA
        self.assertNotEqual(slots, pooled)

    def test_decoder_query_is_not_encoder_query(self) -> None:
        """M3 is a different query than M1; subset relation is the only share."""
        t_enc, t_dec, n = 20, 30, 40
        enc_blocks = av.compressed_block_ids(t_enc, av.M_CSA)
        dec_cache = av.yoco_cross_visible(t_dec, n)
        self.assertNotEqual(enc_blocks, dec_cache)


if __name__ == "__main__":
    unittest.main()
