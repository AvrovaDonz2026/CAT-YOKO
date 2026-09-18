#!/usr/bin/env python3
"""Checkpoint tensors land on CPU so 12B save does not clone VRAM."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.checkpoint import load_checkpoint, save_checkpoint
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


if __name__ == "__main__":
    unittest.main()
