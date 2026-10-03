"""Latest publication preserves the last good checkpoint across failures."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from cat_yoko.checkpoint import publish_latest


class AtomicLatestTests(unittest.TestCase):
    def _files(self, root: Path):
        step = root / "trainable_step_2.pt"
        latest = root / "trainable.pt"
        step.write_bytes(b"new complete checkpoint" * 1000)
        latest.write_bytes(b"previous complete checkpoint")
        return step, latest, latest.with_name(latest.name + ".tmp")

    def test_hardlink_replaces_only_after_complete_temp_exists(self):
        real_replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            step, latest, tmp = self._files(Path(directory))
            previous = latest.read_bytes()

            def replace(source, destination):
                self.assertEqual(Path(source), tmp)
                self.assertEqual(Path(destination), latest)
                self.assertEqual(latest.read_bytes(), previous)
                self.assertEqual(tmp.read_bytes(), step.read_bytes())
                self.assertTrue(tmp.samefile(step))
                real_replace(source, destination)

            with patch("cat_yoko.checkpoint.os.replace", side_effect=replace), \
                    patch("cat_yoko.checkpoint.shutil.copy2", side_effect=AssertionError("unexpected copy")):
                self.assertEqual(publish_latest(step, latest), latest)
            self.assertTrue(latest.samefile(step))
            self.assertEqual(latest.read_bytes(), step.read_bytes())
            self.assertFalse(tmp.exists())

    def test_copy_fallback_never_exposes_partial_new_latest(self):
        real_copy, real_replace = shutil.copy2, os.replace
        with tempfile.TemporaryDirectory() as directory:
            step, latest, tmp = self._files(Path(directory))
            previous = latest.read_bytes()

            def copy(source, destination):
                self.assertEqual(Path(destination), tmp)
                tmp.write_bytes(b"part of new checkpoint")
                self.assertEqual(latest.read_bytes(), previous)
                return real_copy(source, destination)

            def replace(source, destination):
                self.assertEqual(latest.read_bytes(), previous)
                self.assertEqual(tmp.read_bytes(), step.read_bytes())
                real_replace(source, destination)

            with patch("cat_yoko.checkpoint.os.link", side_effect=OSError("hardlinks unavailable")), \
                    patch("cat_yoko.checkpoint.shutil.copy2", side_effect=copy), \
                    patch("cat_yoko.checkpoint.os.replace", side_effect=replace):
                publish_latest(step, latest)
            self.assertEqual(latest.read_bytes(), step.read_bytes())
            self.assertFalse(latest.samefile(step))
            self.assertFalse(tmp.exists())

    def test_failed_partial_copy_preserves_old_and_removes_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            step, latest, tmp = self._files(Path(directory))
            previous = latest.read_bytes()

            def copy(source, destination):
                Path(destination).write_bytes(b"partial checkpoint")
                self.assertEqual(latest.read_bytes(), previous)
                raise OSError("copy interrupted")

            with patch("cat_yoko.checkpoint.os.link", side_effect=OSError("link failed")), \
                    patch("cat_yoko.checkpoint.shutil.copy2", side_effect=copy), \
                    patch("cat_yoko.checkpoint.os.replace") as replace:
                with self.assertRaisesRegex(OSError, "copy interrupted"):
                    publish_latest(step, latest)
            replace.assert_not_called()
            self.assertEqual(latest.read_bytes(), previous)
            self.assertEqual(step.read_bytes(), b"new complete checkpoint" * 1000)
            self.assertFalse(tmp.exists())

    def test_replace_failure_preserves_old_in_both_link_and_copy_paths(self):
        for use_copy in (False, True):
            with self.subTest(copy=use_copy), tempfile.TemporaryDirectory() as directory:
                step, latest, tmp = self._files(Path(directory))
                previous = latest.read_bytes()
                with ExitStack() as stack:
                    if use_copy:
                        stack.enter_context(patch("cat_yoko.checkpoint.os.link", side_effect=OSError("link failed")))
                    stack.enter_context(patch("cat_yoko.checkpoint.os.replace", side_effect=OSError("replace failed")))
                    with self.assertRaisesRegex(OSError, "replace failed"):
                        publish_latest(step, latest)
                self.assertEqual(latest.read_bytes(), previous)
                self.assertFalse(tmp.exists())
                self.assertTrue(step.is_file())

    def test_link_and_disk_check_failures_do_not_remove_previous_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            step, latest, tmp = self._files(Path(directory))
            previous = latest.read_bytes()
            with patch("cat_yoko.checkpoint.os.link", side_effect=OSError("link failed")), \
                    patch("cat_yoko.checkpoint.require_free_bytes", side_effect=OSError("disk full")), \
                    patch("cat_yoko.checkpoint.shutil.copy2") as copy:
                with self.assertRaisesRegex(OSError, "disk full"):
                    publish_latest(step, latest)
            copy.assert_not_called()
            self.assertEqual(latest.read_bytes(), previous)
            self.assertFalse(tmp.exists())

    def test_samefile_returns_early_and_cleans_stale_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "step_2.pt"
            step.write_bytes(b"checkpoint")
            latest = root / "latest.pt"
            os.link(step, latest)
            tmp = root / "latest.pt.tmp"
            tmp.write_bytes(b"stale interrupted publication")
            with patch("cat_yoko.checkpoint.os.link") as link, \
                    patch("cat_yoko.checkpoint.shutil.copy2") as copy, \
                    patch("cat_yoko.checkpoint.os.replace") as replace:
                self.assertEqual(publish_latest(step), latest)
            link.assert_not_called()
            copy.assert_not_called()
            replace.assert_not_called()
            self.assertTrue(latest.samefile(step))
            self.assertFalse(tmp.exists())

    def test_first_publication_failure_leaves_no_partial_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step = root / "step_2.pt"
            step.write_bytes(b"new checkpoint")
            latest = root / "latest.pt"
            tmp = root / "latest.pt.tmp"

            def interrupted_copy(source, destination):
                Path(destination).write_bytes(b"partial")
                raise OSError("interrupted")

            with patch("cat_yoko.checkpoint.os.link", side_effect=OSError("link failed")), \
                    patch("cat_yoko.checkpoint.shutil.copy2", side_effect=interrupted_copy):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    publish_latest(step)
            self.assertFalse(latest.exists())
            self.assertFalse(tmp.exists())

    def test_dangling_latest_symlink_is_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            step, latest, tmp = self._files(Path(directory))
            latest.unlink()
            latest.symlink_to("missing_step.pt")
            publish_latest(step, latest)
            self.assertFalse(latest.is_symlink())
            self.assertTrue(latest.samefile(step))
            self.assertFalse(tmp.exists())


if __name__ == "__main__":
    unittest.main()
