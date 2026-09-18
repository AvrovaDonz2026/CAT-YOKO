"""AdamW groups, optional CPU-offload moments, and WSD (warmup-stable; decay is Phase E)."""

from __future__ import annotations

import inspect
import math

import torch
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


class CPUOffloadAdamW(AdamW):
    """AdamW whose exp_avg / exp_avg_sq live on CPU.

    Parameters may sit on CUDA (B1 encoder-offload) or CPU (B2 block offload).
    GPU memory only holds one fp32 copy of the current tensor during the step.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group.get("eps", 1e-8))
            wd = float(group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach().to(device="cpu", dtype=torch.float32)
                p.grad = None
                p32 = p.detach().to(device="cpu", dtype=torch.float32)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p32)
                    state["exp_avg_sq"] = torch.zeros_like(p32)
                else:
                    if torch.is_tensor(state.get("step")):
                        state["step"] = int(state["step"].item())
                    state["exp_avg"] = state["exp_avg"].to(device="cpu", dtype=torch.float32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(device="cpu", dtype=torch.float32)
                state["step"] += 1
                t = int(state["step"])
                if wd != 0.0:
                    p32.mul_(1.0 - lr * wd)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**t)).add_(eps)
                p32.addcdiv_(exp_avg, denom, value=-lr / (1.0 - beta1**t))
                p.copy_(p32.to(device=p.device, dtype=p.dtype))
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for st in self.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.detach().cpu()


def _adamw_kwargs(cfg: CATYokoConfig) -> dict:
    kw: dict = {
        "lr": cfg.lr,
        "betas": (cfg.adam_beta1, cfg.adam_beta2),
        "weight_decay": cfg.weight_decay,
    }
    names = inspect.signature(AdamW.__init__).parameters
    if "foreach" in names:
        kw["foreach"] = False
    if "fused" in names:
        kw["fused"] = False
    return kw


def build_optimizer(model: nn.Module, cfg: CATYokoConfig, *, cpu_offload: bool = False) -> AdamW:
    groups = [g for g in adamw_param_groups(model, cfg.weight_decay) if g["params"]]
    cls = CPUOffloadAdamW if cpu_offload else AdamW
    return cls(groups, **_adamw_kwargs(cfg))


def wsd_lr(tokens_seen: float, cfg: CATYokoConfig, phase: str) -> float:
    base = cfg.lr_b2 if phase == "B2" else cfg.lr
    if tokens_seen < cfg.warmup_tokens:
        return base * max(tokens_seen / cfg.warmup_tokens, 1e-3)
    return base
