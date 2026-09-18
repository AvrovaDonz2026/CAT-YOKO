"""Optional MiniCPM teacher for logit KD."""

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


def load_teacher(path: Path, device: str) -> nn.Module:
    """Load a causal LM that returns `.logits` or `{\"logits\"}`.

    Prefers HuggingFace `AutoModelForCausalLM` when `transformers` is installed;
    otherwise loads a pickled nn.Module state via torch.load.
    """
    path = Path(path)
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return _HfTeacher(model)
    except ImportError:
        pass
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, nn.Module):
        obj.to(device)
        obj.eval()
        for p in obj.parameters():
            p.requires_grad = False
        return obj
    raise RuntimeError(
        f"cannot load teacher from {path}: install transformers for MiniCPM "
        "or pass a pickled nn.Module / --dummy-teacher"
    )


class _HfTeacher(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.inner(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        return {"logits": logits}
