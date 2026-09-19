#!/usr/bin/env python3
"""Dedicated-dir plan mini-train: attention, YOCO, PDSA-in-graph, C1 chain."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cat_yoko.config import CATYokoConfig, encoder_layer_kind
from cat_yoko.plan_verify import PHASES_RUN, run, static_ledger

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "137.175.",
    "westc.seetacloud",
    "weste.seetacloud",
    "westd.seetacloud",
    "BEGIN OPENSSH",
    "PRIVATE KEY",
    "dfghjkl",
    "31jEePeb",
    "vDw8xU9c",
    "XNn7kkf",
)


class PlanProbeConfigTests(unittest.TestCase):
    def test_probe_has_sliding_csa_hca_and_theorem_b_hole(self) -> None:
        cfg = CATYokoConfig.plan_probe()
        kinds = [encoder_layer_kind(i, cfg.encoder_layers, use_kda=cfg.use_kda) for i in range(cfg.encoder_layers)]
        self.assertEqual(kinds, ["sliding", "csa", "hca"])
        self.assertLess(cfg.n_win, cfg.seq_len)
        self.assertGreaterEqual(cfg.n_win, cfg.compress_m)
        self.assertGreaterEqual(cfg.n_win, cfg.compress_m_hca)
        self.assertFalse(cfg.use_kda)
        self.assertFalse(cfg.use_nvfp4)
        self.assertEqual(cfg.seq_len, 32)
        self.assertEqual(cfg.n_win, 8)

    def test_published_split_untouched(self) -> None:
        c12 = CATYokoConfig.middle_12b()
        self.assertEqual((c12.encoder_layers, c12.decoder_layers), (16, 26))
        kinds = [encoder_layer_kind(i, 16) for i in range(16)]
        self.assertEqual((kinds.count("sliding"), kinds.count("csa"), kinds.count("hca")), (2, 7, 7))


class StaticLedgerTests(unittest.TestCase):
    def test_static_claims_pass_except_deferred_pdsa(self) -> None:
        ledger = static_ledger()
        failed = [c for c in ledger if not c.ok]
        self.assertEqual(failed, [], msg=[(c.name, c.observed, c.expected) for c in failed])
        deferred = [c.name for c in ledger if c.deferred]
        self.assertIn("pdsa.tier1_calibrated_fallback", deferred)
        self.assertIn("pdsa.tier3_editable_memory", deferred)
        names = {c.name for c in ledger}
        self.assertIn("pdsa.hca_runtime_concat_slots", names)
        self.assertIn("yoco.cross_attn_q_o_only", names)
        self.assertIn("theorem_b.indexer_deletes_only", names)
        self.assertIn("attention.no_csa_cuda_kernel", names)


class MiniTrainTests(unittest.TestCase):
    def test_cpu_one_step_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan-verify"
            blob = run(device="cpu", out=out, steps=1)
            self.assertTrue((out / "ledger.json").is_file())
            disk = json.loads((out / "ledger.json").read_text())
            self.assertEqual(disk["ok"], blob["ok"])
        self.assertTrue(blob["ok"], msg=json.dumps(blob.get("failed"), indent=2)[:4000])
        self.assertEqual(blob["n_fail"], 0)
        self.assertEqual(blob["n_deferred"], 2)
        self.assertEqual(list(blob["phases"]), list(PHASES_RUN))
        for phase, st in blob["phases"].items():
            self.assertTrue(st["ok"], msg=phase)
            self.assertGreater(st["nll"], 0.0)
        self.assertNotIn("--save-full", json.dumps(blob["phases"]))


class SecretScanTests(unittest.TestCase):
    def test_suite_files_have_no_secrets_or_hosts(self) -> None:
        paths = [
            ROOT / "cat_yoko" / "plan_verify.py",
            ROOT / "cat_yoko" / "config.py",
            ROOT / "scripts" / "run_plan_verify.sh",
            ROOT / "docs" / "PLAN_VERIFY.md",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), msg=str(path))
            text = path.read_text(encoding="utf-8")
            for tok in FORBIDDEN:
                self.assertNotIn(tok, text, msg=f"{path} contains {tok}")

    def test_attention_modules_have_no_csa_class(self) -> None:
        for rel in ("cat_yoko/attention.py", "cat_yoko/sparse.py", "cat_yoko/indexer.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("class CSA", text)

    def test_shell_is_dedicated_dir_dummy_only(self) -> None:
        text = (ROOT / "scripts" / "run_plan_verify.sh").read_text(encoding="utf-8")
        self.assertIn("/root/autodl-tmp/plan-verify", text)
        self.assertIn("cat_yoko.plan_verify", text)
        self.assertIn("Does not download Ultra-FineWeb", text)
        self.assertNotRegex(text, r"plan_verify[^\n]*--save-full")
        self.assertTrue((ROOT / "scripts" / "run_plan_verify.sh").stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
