#!/usr/bin/env python3
"""CPU tests for MiniCPM5 teacher placement (bf16 on CUDA, no KV cache)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cat_yoko.teacher import (
    DummyTeacher,
    load_teacher,
    place_teacher,
    teacher_param_dtype,
)


class _LogitsModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, use_cache=True):
        del use_cache
        b, s = input_ids.shape
        return SimpleNamespace(logits=input_ids.new_zeros(b, s, 4).float())


class TeacherPlacementTests(unittest.TestCase):
    def test_cpu_dtype_is_fp32_cuda_policy_is_bf16(self) -> None:
        self.assertEqual(teacher_param_dtype("cpu"), torch.float32)
        if torch.cuda.is_available():
            self.assertEqual(teacher_param_dtype("cuda"), torch.bfloat16)
        else:
            self.assertEqual(teacher_param_dtype("cuda"), torch.float32)

    def test_place_teacher_freezes_and_disables_cache(self) -> None:
        inner = _LogitsModule()
        teacher = place_teacher(inner, "cpu")
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertFalse(inner.config.use_cache)
        ids = torch.randint(0, 4, (2, 3))
        out = teacher(ids)
        self.assertEqual(tuple(out["logits"].shape), (2, 3, 4))

    def test_pickle_dummy_teacher_roundtrip(self) -> None:
        dummy = DummyTeacher(8, 4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.pt"
            torch.save(dummy, path)
            teacher = load_teacher(path, "cpu")
            ids = torch.randint(0, 8, (2, 3))
            logits = teacher(ids)["logits"]
            self.assertEqual(tuple(logits.shape), (2, 3, 8))
            self.assertEqual(next(teacher.parameters()).dtype, torch.float32)
            self.assertFalse(any(p.requires_grad for p in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
