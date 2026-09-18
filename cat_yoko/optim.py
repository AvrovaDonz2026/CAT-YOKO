"""AdamW groups and WSD (warmup-stable; decay is Phase E)."""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW

from cat_yoko.config import CATYokoConfig


def unwrap(module: nn.Module) -> nn.Module:
    while hasattr(module, "module"):
        module = module.module  # type: ignore[assignment]
    return module


def adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay: list = []
    nodecay: list = []
    for name, p in unwrap(model).named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(k in name for k in ("norm", "bias", "gate")):
            nodecay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: CATYokoConfig) -> AdamW:
    groups = adamw_param_groups(model, cfg.weight_decay)
    groups = [g for g in groups if g["params"]]
    return AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )


def wsd_lr(tokens_seen: float, cfg: CATYokoConfig, phase: str) -> float:
    base = cfg.lr_b2 if phase == "B2" else cfg.lr
    if tokens_seen < cfg.warmup_tokens:
        return base * max(tokens_seen / cfg.warmup_tokens, 1e-3)
    return base
