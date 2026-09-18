#!/usr/bin/env python3
"""Published B0 AutoDL launcher: 8e9 tokens, no --try, no secrets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_b0_full_autodl.sh"


class B0FullLauncherTests(unittest.TestCase):
    def test_script_exists_and_is_published_envelope(self) -> None:
        text = SCRIPT.read_text()
        self.assertTrue(SCRIPT.is_file())
        self.assertIn("cat_yoko.b0", text)
        self.assertIn("--no-offload-encoder", text)
        self.assertIn("--upcycle-hf", text)
        self.assertIn("8e9", text)
        self.assertIn("4096", text)
        self.assertNotRegex(text, r"cat_yoko\.b0.*--try")
        self.assertIn("Does not download Ultra-FineWeb", text)
        self.assertNotRegex(text, r"31jEePeb|vDw8xU9c|PRIVATE KEY")
        self.assertNotIn("westc.seetacloud", text)
        self.assertIn("b0-full", text)
        self.assertIn("runs/b0", text)

    def test_script_does_not_fetch_github(self) -> None:
        text = SCRIPT.read_text()
        self.assertNotIn("git fetch", text)
        self.assertNotIn("git pull", text)

    def test_hub_pointer(self) -> None:
        readme = ROOT / "checkpoints" / "b0-full" / "README.md"
        self.assertTrue(readme.is_file())
        body = readme.read_text()
        self.assertIn("huggingface.co/AvrovaDonz/CAT-YOKO", body)
        self.assertIn("8e9", body)
        self.assertIn("checkpoints/b0/", body)


if __name__ == "__main__":
    unittest.main()
