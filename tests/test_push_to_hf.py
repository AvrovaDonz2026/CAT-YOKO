"""push_to_hf.sh exists, is executable, and contains no private-key material."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "push_to_hf.sh"


class PushToHfTests(unittest.TestCase):
    def test_script_exists_and_targets_this_repo(self) -> None:
        self.assertTrue(SCRIPT.is_file(), f"missing {SCRIPT}")
        self.assertTrue(SCRIPT.stat().st_mode & 0o111, f"{SCRIPT} is not executable")
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("git@hf.co:AvrovaDonz/CAT-YOKO", text)
        self.assertIn("GIT_LFS_SKIP_SMUDGE", text)
        self.assertIn("checkpoints/b0-full/README.md", text)
        self.assertNotIn("BEGIN OPENSSH PRIVATE KEY", text)
        self.assertNotIn("huggingface/SSH.md", text)

    def test_repo_does_not_ship_deploy_key_docs(self) -> None:
        hf = ROOT / "huggingface"
        self.assertTrue((hf / "README.md").is_file())
        self.assertFalse((hf / "SSH.md").exists())
        self.assertFalse((hf / "SSH_PUBLIC_KEY.txt").exists())


if __name__ == "__main__":
    unittest.main()
