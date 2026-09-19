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
        self.assertIn("trainable_step_", text)
        self.assertIn("has_overlay", text)
        self.assertIn("venv-nightly", text)
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

    def test_nightly_upgrade_script(self) -> None:
        up = ROOT / "scripts" / "upgrade_torch_te_nightly_autodl.sh"
        probe = ROOT / "scripts" / "probe_nvfp4_hw.py"
        self.assertTrue(up.is_file())
        self.assertTrue(probe.is_file())
        text = up.read_text()
        self.assertIn("nightly/cu130", text)
        self.assertIn("transformer_engine[pytorch,core-cu13]", text)
        self.assertIn("venv-nightly", text)
        self.assertIn("probe_nvfp4_hw.py", text)
        self.assertIn("transformers==5.17.0", text)
        self.assertNotIn("git fetch", text)
        self.assertNotRegex(text, r"31jEePeb|vDw8xU9c|PRIVATE KEY")
        self.assertNotIn("westc.seetacloud", text)
        probe_src = probe.read_text()
        self.assertIn("torch.randn(16, 128", probe_src)
        self.assertNotIn("torch.randn(4, 64", probe_src)
        self.assertNotIn("torch.randn(8, 128", probe_src)
        self.assertIn("copy_once", probe_src)
        self.assertIn("64, 2048", probe_src)
        self.assertIn("te_nvfp4_linear_fprop", probe_src)
        self.assertIn("te_nvfp4_linear_wgrad", probe_src)
        self.assertIn("te_nvfp4_linear_dx", probe_src)


class B200LauncherTests(unittest.TestCase):
    def test_b0_b200_script_is_published_envelope(self) -> None:
        script = ROOT / "scripts" / "run_b0_full_b200.sh"
        self.assertTrue(script.is_file())
        text = script.read_text()
        self.assertIn("cat_yoko.b0", text)
        self.assertIn("--no-offload-encoder", text)
        self.assertIn("--no-grad-ckpt", text)
        self.assertIn("--upcycle-hf", text)
        self.assertIn("8e9", text)
        self.assertIn("4096", text)
        self.assertIn("download_hub_overlay.py", text)
        self.assertIn("b0-full", text)
        self.assertIn("HF_HOME", text)
        self.assertIn("huggingface.co", text)
        self.assertIn("nvidia/cublas/lib", text)
        self.assertIn("cuda-12.9/lib64", text)
        self.assertIn("MICRO_BATCH", text)
        self.assertIn("--micro-batch", text)
        self.assertIn('MICRO="${MICRO_BATCH:-2}"', text)
        self.assertNotRegex(text, r"cat_yoko\.b0.*--try")
        self.assertIn("Does not download Ultra-FineWeb", text)
        self.assertNotIn("git fetch", text)
        self.assertNotRegex(text, r"31jEePeb|vDw8xU9c|PRIVATE KEY|dfghjkl")
        self.assertNotIn("westc.seetacloud", text)
        self.assertNotIn("137.175.", text)

    def test_upgrade_and_try_scripts(self) -> None:
        up = (ROOT / "scripts" / "upgrade_torch_te_b200.sh").read_text()
        self.assertIn("cu128", up)
        self.assertIn("transformer_engine[pytorch,core-cu12]", up)
        self.assertIn("core-cu13", up)
        self.assertIn("--no-build-isolation", up)
        self.assertIn("build_te_from_source.sh", up)
        self.assertIn("uv pip uninstall -y transformer-engine-cu13", up)
        self.assertIn("NVTE_CUDA_ARCHS", up)
        self.assertIn("probe_nvfp4_hw.py", up)
        boot = (ROOT / "scripts" / "run_b200.sh").read_text()
        self.assertIn("run_b0_full_b200.sh", boot)
        self.assertIn("download_minicpm5.py", boot)
        self.assertIn("HF_HOME", boot)
        self.assertIn("huggingface.co", boot)
        src = (ROOT / "scripts" / "build_te_from_source.sh").read_text()
        self.assertIn("NVTE_CUDA_ARCHS", src)
        self.assertIn("NVIDIA/TransformerEngine", src)
        self.assertIn("--no-build-isolation", src)
        self.assertIn("wheel_lib", src)
        self.assertIn("libtransformer_engine.so", src)
        self.assertIn("Root-Is-Purelib", src)
        self.assertIn("12.9", src)
        self.assertIn("STORE256", src)
        self.assertIn("CUDACXX", src)
        self.assertIn("CUDAToolkit_ROOT", src)
        self.assertIn("cublas_v2.h", src)
        self.assertIn("nproc", src)
        self.assertIn("MAX_JOBS", src)
        self.assertIn("nvtx3/nvToolsExt.h", src)
        self.assertIn("LD_LIBRARY_PATH", src)
        self.assertIn("cuda-12.9/lib64", src)
        self.assertIn("sysconfig.get_paths()", src)
        self.assertIn("cd /tmp", src)
        self.assertLess(src.find("cd /tmp"), src.find("import transformer_engine as te"))
        hw = (ROOT / "cat_yoko" / "nvfp4_hw.py").read_text()
        self.assertIn("torch.randn(4096, 2048", hw)
        self.assertNotIn("x = torch.randn(256, 2048", hw)
        b1 = (ROOT / "scripts" / "run_b1_try_b200.sh").read_text()
        self.assertIn("cat_yoko.b1", b1)
        self.assertIn("--try", b1)
        self.assertIn("--no-grad-ckpt", b1)
        b2 = (ROOT / "scripts" / "run_b2_try_b200.sh").read_text()
        self.assertIn("cat_yoko.b2", b2)
        overlay = ROOT / "scripts" / "download_hub_overlay.py"
        self.assertTrue(overlay.is_file())
        src = overlay.read_text()
        self.assertIn("efef3464730eaaee6b62a0e199437b3a06049812bfbc25e9a98f39467fdf0a2c", src)
        self.assertIn("AvrovaDonz/CAT-YOKO", src)

    def test_pull_vast_overlay_script(self) -> None:
        script = ROOT / "scripts" / "pull_vast_b0_overlay.sh"
        self.assertTrue(script.is_file())
        self.assertTrue(script.stat().st_mode & 0o111)
        text = script.read_text()
        self.assertIn("vast-b200", text)
        self.assertIn("trainable_step_", text)
        self.assertIn("checkpoints/b0-full/trainable.pt", text)
        self.assertIn("MIN_AGE", text)
        self.assertNotIn("137.175.", text)
        self.assertNotIn("git@hf.co", text)
        self.assertNotRegex(text, r"PRIVATE KEY|dfghjkl|31jEePeb|vDw8xU9c")
        self.assertNotIn("westc.seetacloud", text)


if __name__ == "__main__":
    unittest.main()
