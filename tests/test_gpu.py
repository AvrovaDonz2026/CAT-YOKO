#!/usr/bin/env python3
"""CUDA tests for the C1 trainer. Skip on CPU; never allocate 12B on GPU."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.fp8 import should_autocast
from cat_yoko.gpu_smoke import cuda_info, main as gpu_smoke_main, run_tiny_cuda
from cat_yoko.train import main as train_main
from cat_yoko.trainer import train_loop


class GpuSmokeSkipTests(unittest.TestCase):
    def test_main_skips_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA present; skip-path is for CPU CI")
        self.assertEqual(gpu_smoke_main([]), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class GpuTinyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_info_has_device_name(self) -> None:
        info = cuda_info()
        self.assertTrue(info["cuda"])
        self.assertTrue(info["device"])
        self.assertGreater(info["total_gib"], 0)

    def test_forward_backward_cuda(self) -> None:
        nll = train_loop(self.cfg, "B0", steps=2, device="cuda", accum=1)
        self.assertGreater(nll, 0)
        self.assertEqual(nll, nll)

    def test_bf16_one_step(self) -> None:
        if not torch.cuda.is_bf16_supported():
            self.skipTest("device has no bf16")
        from cat_yoko.trainer import Trainer

        out = Trainer(self.cfg, "B0", "cuda", steps=1, accum=1, dtype="bf16").run()
        self.assertGreater(out.nll, 0)
        self.assertEqual(out.nll, out.nll)

    def test_fp8_policy_on_cuda(self) -> None:
        self.assertFalse(should_autocast("B0", cuda=True, enabled=True))
        self.assertTrue(should_autocast("B1", cuda=True, enabled=True))
        self.assertTrue(should_autocast("B2", cuda=True, enabled=True))

    def test_12b_meta_does_not_allocate_cuda(self) -> None:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        code = train_main(["--config", "12b", "--meta"])
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        self.assertEqual(code, 0)
        self.assertEqual(after, before)

    def test_cli_tiny_cuda(self) -> None:
        code = train_main(
            ["--config", "tiny", "--phase", "B0", "--steps", "1", "--accum", "1", "--device", "cuda"]
        )
        self.assertEqual(code, 0)

    def test_full_tiny_bundle(self) -> None:
        result = run_tiny_cuda(steps=1, micro_batch=2)
        self.assertTrue(result["ok"], msg=result)


if __name__ == "__main__":
    unittest.main()
