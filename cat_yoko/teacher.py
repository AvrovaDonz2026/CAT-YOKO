"""Optional MiniCPM5-2B teacher for logit KD."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DummyTeacher(nn.Module):
    def __init__(self, vocab: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.head(self.embed(input_ids))}


def teacher_param_dtype(device: str) -> torch.dtype:
    """CUDA teacher stays bf16 (frozen spec); CPU tests stay fp32."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def place_teacher(model: nn.Module, device: str) -> nn.Module:
    """Move a teacher, freeze it, and wrap so `.forward` returns ``{\"logits\"}``."""
    dt = teacher_param_dtype(device)
    model.to(device=device, dtype=dt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    return _HfTeacher(model)


def load_teacher(source: str | Path, device: str) -> nn.Module:
    """Load a causal LM that returns `.logits` or `{\"logits\"}`.

    `.pt` pickle of an `nn.Module` is local. Hub ids / HF dirs need
    `pip install 'cat-yoko[data]'` (transformers; MiniCPM5 is Llama, no
    ``trust_remote_code``). MiniCPM-2B Hub ids are rejected. CUDA teachers
    are bf16; HF forward disables ``use_cache``.
    """
    path = Path(source)
    if path.is_file():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, nn.Module):
            return place_teacher(obj, device)
        raise RuntimeError(f"{path} is not a pickled nn.Module; use --teacher-hf for MiniCPM5")
    from cat_yoko.recipe import assert_minicpm5_hf_config, assert_minicpm5_id
    from cat_yoko.hf_minicpm import load_causal_lm_cpu

    assert_minicpm5_id(str(source), kind="teacher")
    dt = teacher_param_dtype(device)
    model = load_causal_lm_cpu(str(source), dt)
    assert_minicpm5_hf_config(model.config)
    return place_teacher(model, device)


class _HfTeacher(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        cfg = getattr(inner, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        try:
            out = self.inner(input_ids=input_ids, use_cache=False)
        except TypeError:
            out = self.inner(input_ids)
        if torch.is_tensor(out):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out["logits"]
        return {"logits": logits}
