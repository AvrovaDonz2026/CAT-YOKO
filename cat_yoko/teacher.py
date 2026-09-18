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


def load_teacher(source: str | Path, device: str) -> nn.Module:
    """Load a causal LM that returns `.logits` or `{\"logits\"}`.

    `.pt` pickle of an `nn.Module` is local. Hub ids / HF dirs need
    `pip install 'cat-yoko[data]'` (transformers; MiniCPM5 is Llama, no
    ``trust_remote_code``). MiniCPM-2B Hub ids are rejected.
    """
    path = Path(source)
    if path.is_file():
        obj = torch.load(path, map_location=device, weights_only=False)
        if isinstance(obj, nn.Module):
            obj.to(device)
            obj.eval()
            for p in obj.parameters():
                p.requires_grad = False
            return obj
        raise RuntimeError(f"{path} is not a pickled nn.Module; use --teacher-hf for MiniCPM5")
    from cat_yoko.recipe import assert_minicpm5_hf_config, assert_minicpm5_id

    assert_minicpm5_id(str(source), kind="teacher")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "MiniCPM5 teacher needs transformers: pip install 'cat-yoko[data]'"
        ) from exc
    want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dt = torch.bfloat16 if want_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        torch_dtype=dt,
        low_cpu_mem_usage=True,
    )
    assert_minicpm5_hf_config(model.config)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return _HfTeacher(model)


class _HfTeacher(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.inner(input_ids=input_ids)
        logits = out.logits if hasattr(out, "logits") else out["logits"]
        return {"logits": logits}
