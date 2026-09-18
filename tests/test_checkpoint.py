#!/usr/bin/env python3
"""Checkpoint tensors land on CPU so 12B save does not clone VRAM."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.checkpoint import (
    cleanup_save_tmp,
    load_checkpoint,
    newest_step_checkpoint,
    publish_latest,
    require_free_bytes,
    require_host_bytes,
    resolve_resume_path,
    save_checkpoint,
)
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import CPUOffloadAdamW, build_optimizer


def _assert_tensors_cpu(obj, *, prefix: str = "") -> None:
    if torch.is_tensor(obj):
        self_msg = prefix or "tensor"
        if obj.device.type != "cpu":
            raise AssertionError(f"{self_msg} on {obj.device}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_tensors_cpu(v, prefix=f"{prefix}.{k}" if prefix else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_tensors_cpu(v, prefix=f"{prefix}[{i}]")


class CheckpointCpuTests(unittest.TestCase):
    def test_saved_model_tensors_are_cpu(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            save_checkpoint(
                path, model=model, optimizer=None, extra={"phase": "B0"}, save_optimizer=False
            )
            ckpt = load_checkpoint(path, map_location="cpu")
            self.assertTrue(ckpt["model"])
            for name, t in ckpt["model"].items():
                self.assertTrue(torch.is_tensor(t), msg=name)
                self.assertEqual(t.device.type, "cpu", msg=name)

    def test_saved_optimizer_tensors_are_cpu_offload_untouched(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        opt = build_optimizer(model, cfg, cpu_offload=True)
        self.assertIsInstance(opt, CPUOffloadAdamW)
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        opt.step()
        live = [
            v
            for st in opt.state.values()
            for v in st.values()
            if torch.is_tensor(v)
        ]
        self.assertTrue(live)
        live_ptrs = [t.data_ptr() for t in live]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            save_checkpoint(path, model=model, optimizer=opt, extra={}, save_optimizer=True)
            ckpt = load_checkpoint(path, map_location="cpu")
            self.assertTrue(ckpt["model"])
            for name, t in ckpt["model"].items():
                self.assertEqual(t.device.type, "cpu", msg=name)
            self.assertIsNotNone(ckpt["optimizer"])
            _assert_tensors_cpu(ckpt["optimizer"])
        for t, ptr in zip(live, live_ptrs):
            self.assertEqual(t.device.type, "cpu")
            self.assertEqual(t.data_ptr(), ptr)

    def test_publish_latest_hardlinks_same_inode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            step = td / "step_1.pt"
            step.write_bytes(b"ckpt-bytes")
            latest = publish_latest(step)
            self.assertTrue(latest.is_file())
            self.assertTrue(latest.samefile(step))
            self.assertEqual(step.stat().st_nlink, 2)
            leftover = td / "latest.pt.tmp"
            leftover.write_bytes(b"stale")
            publish_latest(step)
            self.assertFalse(leftover.exists())
            self.assertTrue(latest.samefile(step))

    def test_resolve_resume_prefers_latest_then_step(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            step = td / "step_1.pt"
            step.write_bytes(b"a")
            self.assertEqual(resolve_resume_path(td), step)
            self.assertEqual(newest_step_checkpoint(td), step)
            latest = td / "latest.pt"
            latest.write_bytes(b"b")
            self.assertEqual(resolve_resume_path(td), latest)
            self.assertEqual(resolve_resume_path(latest), latest)
            with self.assertRaises(FileNotFoundError):
                resolve_resume_path(td / "missing")

    def test_cleanup_save_tmp_drops_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            stale = td / "latest.pt.tmp"
            stale.write_bytes(b"x")
            (td / "step_1.pt").write_bytes(b"y")
            cleanup_save_tmp(td)
            self.assertFalse(stale.exists())
            self.assertTrue((td / "step_1.pt").is_file())

    def test_require_free_bytes_errors_when_volume_is_tiny(self) -> None:
        from unittest.mock import patch

        fake = type("U", (), {"free": 128})()
        with patch("cat_yoko.checkpoint.shutil.disk_usage", return_value=fake):
            with self.assertRaises(OSError) as ctx:
                require_free_bytes(Path("/tmp"), 1 << 20, what="ckpt")
        self.assertIn("not enough disk", str(ctx.exception))
        self.assertIn("autodl-tmp", str(ctx.exception))

    def test_require_host_bytes_errors_when_cgroup_is_tiny(self) -> None:
        from unittest.mock import patch

        with patch("cat_yoko.optim.host_memory_limit_bytes", return_value=1 << 30):
            with patch("cat_yoko.optim.host_memory_used_bytes", return_value=0):
                with self.assertRaises(OSError) as ctx:
                    require_host_bytes(20 << 30, what="ckpt")
        self.assertIn("host RAM", str(ctx.exception))

    def test_save_checkpoint_refuses_full_disk(self) -> None:
        from unittest.mock import patch

        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        fake = type("U", (), {"free": 0})()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            with patch("cat_yoko.checkpoint.shutil.disk_usage", return_value=fake):
                with self.assertRaises(OSError) as ctx:
                    save_checkpoint(
                        path, model=model, optimizer=None, extra={}, save_optimizer=False
                    )
            self.assertIn("not enough disk", str(ctx.exception))
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("ckpt.pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
