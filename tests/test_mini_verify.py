#!/usr/bin/env python3
"""Mini BF16/INT8 pipeline: recipes, chain, no secrets / 50B download."""

from __future__ import annotations

import json
import math
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.mini_verify import MINI_PHASES, RECIPES, main as mini_main, recipe_cfg, run_recipe
from cat_yoko.nvfp4 import low_prec_enabled, should_autocast


class MiniVerifyTests(unittest.TestCase):
    def test_phases_cover_b_through_f_not_128k(self) -> None:
        self.assertEqual(MINI_PHASES[0], "B0")
        self.assertEqual(MINI_PHASES[-1], "F")
        self.assertIn("C-index", MINI_PHASES)
        self.assertIn("D-8k", MINI_PHASES)
        self.assertNotIn("D-32k", MINI_PHASES)
        self.assertNotIn("D-128k", MINI_PHASES)

    def test_recipes_int8_not_nvfp4(self) -> None:
        bf = recipe_cfg("bf16", twelve=False)
        i8 = recipe_cfg("int8", twelve=False)
        self.assertEqual(RECIPES, ("bf16", "int8"))
        self.assertFalse(bf.use_fp8)
        self.assertFalse(bf.use_nvfp4)
        self.assertFalse(bf.use_int8)
        self.assertTrue(i8.use_int8)
        self.assertFalse(i8.use_nvfp4)
        self.assertFalse(i8.use_fp8)
        self.assertEqual(i8.head_dim, 32)
        self.assertEqual(i8.seq_len, 64)
        self.assertGreaterEqual(i8.n_win, i8.seq_len)
        self.assertFalse(low_prec_enabled(bf))
        self.assertFalse(low_prec_enabled(i8))
        self.assertFalse(should_autocast("B0", cuda=True, enabled=True))
        self.assertTrue(should_autocast("B1", cuda=True, enabled=True))
        t12 = recipe_cfg("int8", twelve=True)
        self.assertTrue(t12.use_int8)
        self.assertEqual(t12.head_dim, 128)
        self.assertFalse(t12.use_kda)

    def test_cpu_tiny_bf16_and_int8_chains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for name in ("bf16", "int8"):
                row = run_recipe(name, device="cpu", out=out, twelve=False, steps=1)
                self.assertTrue(row["ok"], msg=name)
                self.assertEqual(set(row["phases"]), set(MINI_PHASES))
                for phase, st in row["phases"].items():
                    self.assertTrue(math.isfinite(st["nll"]), msg=f"{name} {phase}")
                    self.assertGreater(st["nll"], 0.0, msg=f"{name} {phase}")
                self.assertTrue((out / name / "summary.json").is_file())
                self.assertFalse(row["use_nvfp4"])
                if name == "int8":
                    self.assertTrue(row["use_int8"])
                    self.assertEqual(row["seq_len"], 64)

    def test_cli_writes_root_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            rc = mini_main(["--out", str(out), "--device", "cpu", "--recipe", "bf16", "--steps", "1"])
            self.assertEqual(rc, 0)
            blob = json.loads((out / "summary.json").read_text())
            self.assertTrue(blob["ok"])
            self.assertEqual(blob["runs"][0]["recipe"], "bf16")

    def test_script_has_no_host_or_secret(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_mini_verify.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(script.stat().st_mode & stat.S_IXUSR)
        text = script.read_text()
        self.assertIn("cat_yoko.mini_verify", text)
        self.assertIn("mini-verify", text)
        self.assertNotIn("seetacloud", text)
        self.assertNotIn("137.175.", text)
        self.assertNotIn("westc", text)
        self.assertNotIn("westd", text)
        self.assertNotIn("XNn7", text)
        self.assertIn("Do not pass --save-full", text)
        self.assertNotRegex(text, r"mini_verify[^\n]*--save-full")
        self.assertNotIn("fp8", text.lower())
        self.assertIn("INT8", text)


if __name__ == "__main__":
    unittest.main()
