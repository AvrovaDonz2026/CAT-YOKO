"""GitHub must not track Git LFS weights."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NoGithubLfsTests(unittest.TestCase):
    def test_root_gitattributes_gone(self) -> None:
        self.assertFalse((ROOT / ".gitattributes").exists())

    def test_hub_gitattributes_stays_in_huggingface_dir(self) -> None:
        text = (ROOT / "huggingface" / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("filter=lfs", text)

    def test_overlay_not_in_tree(self) -> None:
        self.assertFalse((ROOT / "checkpoints" / "b0" / "trainable.pt").exists())

    def test_gitignore_covers_pt(self) -> None:
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.pt", text)
        self.assertNotIn("GitHub LFS", text)
