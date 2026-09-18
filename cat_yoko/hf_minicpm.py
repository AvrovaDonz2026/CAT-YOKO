"""Load MiniCPM-2B weights for upcycling / teacher KD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from cat_yoko.recipe import MINICPM_HF


def _unwrap_state(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, Mapping):
        if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            return dict(obj["state_dict"])
        if "model" in obj and isinstance(obj["model"], Mapping):
            inner = obj["model"]
            if any(k.endswith(".weight") for k in inner):
                return dict(inner)
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    raise TypeError("not a MiniCPM state_dict")


def load_minicpm_state(source: str | Path, *, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    """`.pt` file, local HF dir, or Hub id (default MiniCPM-2B-sft-bf16)."""
    path = Path(source)
    if path.is_file():
        obj = torch.load(path, map_location=map_location, weights_only=True)
        return _unwrap_state(obj)
    bin_path = path / "pytorch_model.bin"
    if path.is_dir() and bin_path.is_file():
        try:
            obj = torch.load(bin_path, map_location=map_location, weights_only=True)
            return _unwrap_state(obj)
        except (OSError, TypeError, RuntimeError, KeyError):
            pass
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "HuggingFace MiniCPM load needs transformers: pip install 'cat-yoko[data]'"
        ) from exc
    name = str(source) if str(source) else MINICPM_HF
    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    return sd
