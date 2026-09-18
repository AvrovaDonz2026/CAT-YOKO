"""download_minicpm5: hf-mirror first, ModelScope if Xet 403."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_minicpm5 as dl  # noqa: E402


class DownloadMinicpm5Tests(unittest.TestCase):
    def test_env_defaults_hf_mirror(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            dl._env()
            self.assertEqual(dl.os.environ["HF_ENDPOINT"], "https://hf-mirror.com")
            self.assertEqual(dl.os.environ["HF_HUB_DISABLE_XET"], "1")
            self.assertTrue(dl.os.environ["HF_HOME"].endswith("/hf"))

    def test_incomplete_size_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / dl.SAFETENSORS
            blob.write_bytes(b"incomplete")
            self.assertFalse(dl.safetensors_ok(root, check_hash=False))

    def test_complete_size_accepted_without_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / dl.SAFETENSORS
            payload = b"ok-weights"
            blob.write_bytes(payload)
            with patch.object(dl, "SAFETENSORS_BYTES", len(payload)):
                self.assertTrue(dl.safetensors_ok(root, check_hash=False))

    def test_auto_falls_back_to_modelscope(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"from-ms"

            def fake_hf(_local: Path) -> Path | None:
                return None

            def fake_ms(local: Path) -> Path:
                (local / dl.SAFETENSORS).write_bytes(payload)
                return local

            with (
                patch.object(dl, "SAFETENSORS_BYTES", len(payload)),
                patch.object(dl, "download_hf_mirror", fake_hf),
                patch.object(dl, "download_modelscope", fake_ms),
            ):
                self.assertEqual(dl.main(["--local-dir", str(root), "--source", "auto"]), 0)
            self.assertEqual((root / dl.SAFETENSORS).read_bytes(), payload)

    def test_hf_success_skips_modelscope(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"from-hf"
            called = {"ms": False}

            def fake_hf(local: Path) -> Path:
                (local / dl.SAFETENSORS).write_bytes(payload)
                return local

            def fake_ms(_local: Path) -> Path | None:
                called["ms"] = True
                return None

            with (
                patch.object(dl, "SAFETENSORS_BYTES", len(payload)),
                patch.object(dl, "download_hf_mirror", fake_hf),
                patch.object(dl, "download_modelscope", fake_ms),
            ):
                self.assertEqual(dl.main(["--local-dir", str(root)]), 0)
            self.assertFalse(called["ms"])


if __name__ == "__main__":
    unittest.main()
