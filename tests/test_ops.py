#!/usr/bin/env python3
"""Ampere BF16 operator snapshot helpers."""

from __future__ import annotations

import unittest

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.ops import PHASE_OPS, probe_sdpa_backends, snapshot_ops
from cat_yoko.plan_verify import PHASES_RUN


class PhaseOpsTests(unittest.TestCase):
    def test_every_mini_train_phase_has_ops(self) -> None:
        self.assertEqual(set(PHASE_OPS), set(PHASES_RUN))

    def test_cpu_probe_is_cpu(self) -> None:
        row = probe_sdpa_backends("cpu", torch.float32)
        self.assertEqual(row["dense_gqa"], "cpu")
        self.assertEqual(row["masked_equal"], "cpu")

    def test_snapshot_records_probe_shape(self) -> None:
        cfg = CATYokoConfig.bf16_probe()
        row = snapshot_ops(cfg, "cpu", phase="B0")
        self.assertEqual(row["head_dim"], 32)
        self.assertEqual(row["seq_len"], 128)
        self.assertFalse(row["use_fp8"])
        self.assertIn("dense_cross", row["intended"])
        self.assertIn("moe_bmm", PHASE_OPS["B2"])
        self.assertNotIn("grouped_mm", PHASE_OPS["B2"])


if __name__ == "__main__":
    unittest.main()
