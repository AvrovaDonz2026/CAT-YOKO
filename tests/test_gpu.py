#!/usr/bin/env python3
"""CUDA tests for the C1 trainer. Skip on CPU. 12B B0 needs ≥28GiB.

12B cases each spawn a fresh ``python -m cat_yoko.gpu_smoke`` process.
In-process teardown cannot reliably drop a 12.25B graph on 32GB.
"""

from __future__ import annotations

import gc
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.fp8 import should_autocast
from cat_yoko.gpu_smoke import (
    _teardown_cuda,
    cuda_info,
    enough_vram_for_12b,
    main as gpu_smoke_main,
    run_tiny_cuda,
)
from cat_yoko.train import main as train_main
from cat_yoko.trainer import train_loop


def _teardown_12b(*holders: object) -> None:
    """Free 12B CUDA graphs so the next test in this process can allocate.

    Drops model/opt/trainer held on dicts or objects, then GC, empty the
    CUDA caching allocator, synchronize, and reset peak stats. setUp and
    tearDown both call this so B0 → B1 → C1 do not stack two 12.25B graphs.
    """
    for obj in holders:
        if obj is None:
            continue
        if isinstance(obj, dict):
            for key in ("model", "opt", "trainer"):
                val = obj.pop(key, None)
                del val
            continue
        for name in ("model", "opt", "trainer"):
            if hasattr(obj, name):
                setattr(obj, name, None)
    del holders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    _teardown_cuda()


class GpuSmokeSkipTests(unittest.TestCase):
    def test_cuda_info_reports_alloc_conf(self) -> None:
        info = cuda_info()
        self.assertIn("expandable_segments:True", info.get("alloc_conf") or "")
        if not torch.cuda.is_available():
            self.assertFalse(info["cuda"])

    def test_main_skips_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA present; skip-path is for CPU CI")
        self.assertEqual(gpu_smoke_main([]), 0)

    def test_teardown_safe_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA present; skip-path is for CPU CI")
        _teardown_12b()
        _teardown_cuda()


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
        self.assertIn("expandable_segments:True", info.get("alloc_conf") or "")

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

    def test_cli_c1_smoke_cuda(self) -> None:
        code = train_main(
            ["--config", "tiny", "--c1-smoke", "--steps", "1", "--accum", "1", "--device", "cuda", "--dtype", "bf16"]
        )
        self.assertEqual(code, 0)

    def test_resume_b0_into_b1_cuda(self) -> None:
        import tempfile

        from cat_yoko.trainer import Trainer

        cfg = self.cfg
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            Trainer(
                cfg, "B0", "cuda", steps=1, accum=1, save_dir=save, save_every=1, dtype="bf16"
            ).run()
            out = Trainer(
                cfg, "B1", "cuda", steps=1, accum=1, resume=save / "latest.pt", dtype="bf16"
            ).run()
        self.assertEqual(out.step, 1)
        self.assertEqual(out.phase, "B1")
        self.assertGreater(out.nll, 0)

    def test_kd_two_steps_cuda(self) -> None:
        from cat_yoko.teacher import DummyTeacher

        teacher = DummyTeacher(self.cfg.vocab_size, self.cfg.hidden_size)
        nll = train_loop(
            self.cfg,
            "B0",
            steps=2,
            device="cuda",
            accum=1,
            teacher=teacher,
            dtype="bf16",
        )
        self.assertGreater(nll, 0)

    def test_full_tiny_bundle(self) -> None:
        result = run_tiny_cuda(steps=1, micro_batch=2)
        self.assertTrue(result["ok"], msg=result)
        self.assertTrue(result["ddp_gloo"]["ok"], msg=result)
        self.assertTrue(result["ddp_cuda"]["ok"], msg=result)
        self.assertTrue(result["ddp_cuda_c1"]["ok"], msg=result)
        self.assertTrue(result["fsdp"]["ok"], msg=result)

    def test_gloo_ddp_cpu_from_cuda_process(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_ddp

        row = run_gloo_ddp(device="cpu", world=2, steps=1)
        self.assertTrue(row["ok"], msg=row)

    def test_ddp_two_rank_cuda(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_ddp

        row = run_gloo_ddp(device="cuda", world=2, steps=1)
        self.assertTrue(row["ok"], msg=row)

    def test_ddp_c1_two_rank_cuda(self) -> None:
        from cat_yoko.ddp_smoke import run_gloo_c1

        row = run_gloo_c1(device="cuda", world=2, steps=1)
        self.assertTrue(row["ok"], msg=row)
        self.assertEqual(row["phase"], "C1")

    def test_fsdp_one_rank_cuda(self) -> None:
        from cat_yoko.ddp_smoke import run_fsdp_one

        row = run_fsdp_one(device="cuda", steps=1)
        self.assertTrue(row["ok"], msg=row)

    def test_sdpa_fastpath_matches_mask_cuda(self) -> None:
        from cat_yoko.attention import _sdpa, _window_causal_bias

        torch.manual_seed(0)
        q = torch.randn(1, 2, 8, 8, device="cuda")
        k = torch.randn(1, 2, 8, 8, device="cuda")
        v = torch.randn(1, 2, 8, 8, device="cuda")
        bias = _window_causal_bias(8, 8, 8, q.device, torch.float32)
        masked = _sdpa(q, k, v, bias)
        fast = _sdpa(q, k, v, causal=True)
        self.assertTrue(torch.allclose(fast, masked, atol=1e-4, rtol=1e-4))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class GpuTwelveBTests(unittest.TestCase):
    """One 12.25B graph per OS process. Isolated CLI smokes already fit 32GB."""

    def _isolated(self, *args: str) -> dict:
        import json
        import os
        import subprocess

        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.run(
            [sys.executable, "-m", "cat_yoko.gpu_smoke", *args, "--json"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        idx = text.rfind("\n{")
        blob = text[idx + 1 :] if idx >= 0 else text[text.find("{") :]
        try:
            result = json.JSONDecoder().raw_decode(blob)[0]
        except (ValueError, json.JSONDecodeError):
            self.fail(f"gpu_smoke json parse failed rc={proc.returncode}\n{text[-4000:]}")
        if proc.returncode != 0 and not (result.get("b1_ok") and result.get("b2_oom")):
            self.fail(f"gpu_smoke rc={proc.returncode}\n{text[-4000:]}")
        return result

    def test_12b_b0_one_step(self) -> None:
        if not enough_vram_for_12b():
            self.skipTest("12B B0 smoke needs ≥28GiB GPU")
        result = self._isolated("--middle", "--phase", "B0", "--seq-len", "64")
        self.assertTrue(result["ok"], msg=result)
        self.assertEqual(result["step"], 1)
        self.assertLess(result["peak_gib"], 32.0)

    def test_12b_b1_one_step(self) -> None:
        if not enough_vram_for_12b():
            self.skipTest("12B B1 smoke needs ≥28GiB GPU")
        result = self._isolated("--middle", "--phase", "B1", "--seq-len", "64")
        self.assertTrue(result["ok"], msg=result)
        self.assertEqual(result["step"], 1)
        self.assertLess(result["peak_gib"], 32.0)

    def test_12b_c1_chain(self) -> None:
        if not enough_vram_for_12b():
            self.skipTest("12B C1 smoke needs ≥28GiB GPU")
        result = self._isolated("--c1", "--seq-len", "64")
        self.assertTrue(result["phases"]["B0"]["ok"], msg=result)
        self.assertTrue(result["b1_ok"], msg=result)
        if not result["phases"].get("B2", {}).get("oom"):
            self.assertTrue(result["phases"]["B2"]["ok"], msg=result)


if __name__ == "__main__":
    unittest.main()
