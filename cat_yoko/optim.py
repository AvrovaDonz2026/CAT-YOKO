"""AdamW groups, optional CPU-offload moments, and WSD (warmup-stable; decay is Phase E)."""

from __future__ import annotations

import gc
import inspect
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

def trim_host_allocator() -> None:
    """Return freed Python / glibc arenas to the cgroup (PyTorch CPU cache)."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError, ValueError):
        pass


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


def host_memory_limit_bytes() -> int | None:
    """Cgroup memory.max if it is a finite cap, else None."""
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if not path.is_file():
            continue
        raw = path.read_text().strip()
        if raw in {"max", ""}:
            return None
        try:
            n = int(raw)
        except ValueError:
            return None
        if n <= 0 or n >= (1 << 62):
            return None
        return n
    return None


def host_memory_used_bytes() -> int:
    path = Path("/sys/fs/cgroup/memory.current")
    if path.is_file():
        try:
            return int(path.read_text().strip())
        except ValueError:
            pass
    return 0


def plan_cpu_adam(
    n_params: int,
    *,
    steps: int | None,
) -> tuple[torch.dtype, bool, str]:
    """Pick moment dtype / whether to keep m,v under a RAM cgroup.

    fp32 moments are 8 bytes/param; fp16 are 4. A one-step run skips stored
    moments so a 62GiB cgroup can still host B2 grads after B1.
    """
    if steps is not None and int(steps) <= 1:
        return torch.float32, False, "ephemeral"
    limit = host_memory_limit_bytes()
    if limit is None:
        return torch.float32, True, "fp32"
    used = host_memory_used_bytes()
    slack = 6 * 1024**3
    room = max(limit - used - slack, 0)
    need_fp32 = n_params * 8
    need_fp16 = n_params * 4
    if need_fp32 <= room:
        return torch.float32, True, "fp32"
    if need_fp16 <= room:
        return torch.float16, True, "fp16"
    if steps is None or steps > 1:
        raise RuntimeError(
            f"CPU AdamW needs {need_fp16 / 1024**3:.1f}GiB fp16 moments "
            f"(or {need_fp32 / 1024**3:.1f}GiB fp32); cgroup room "
            f"{room / 1024**3:.1f}GiB. Use ZeRO / more host RAM, or --steps 1."
        )
    return torch.float32, False, "ephemeral"


class CPUOffloadAdamW(Optimizer):
    """AdamW whose exp_avg / exp_avg_sq live on CPU (optional fp16 / ephemeral).

    Subclass Optimizer, not AdamW — PyTorch 2.8 AdamW.__init__ can eagerly
    allocate fused/foreach state for 12B params and get the 62GiB cgroup killed.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        state_dtype: torch.dtype = torch.float32,
        retain_state: bool = True,
    ) -> None:
        self.state_dtype = state_dtype
        self.retain_state = retain_state
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    def _group_map(self) -> dict[int, dict]:
        return {id(p): g for g in self.param_groups for p in g["params"]}

    @torch.no_grad()
    def _update_one(self, p, group: dict) -> None:
        if p.grad is None:
            return
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group.get("eps", 1e-8))
        wd = float(group["weight_decay"])
        grad = p.grad.detach()
        p.grad = None
        if grad.device.type != "cpu" or grad.dtype != torch.float32:
            grad = grad.to(device="cpu", dtype=torch.float32)
        p32 = p.detach()
        if p32.device.type != "cpu" or p32.dtype != torch.float32:
            p32 = p32.to(device="cpu", dtype=torch.float32)
        state = self.state[p]
        if self.retain_state:
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros(
                    p32.shape, dtype=self.state_dtype, device="cpu"
                )
                state["exp_avg_sq"] = torch.zeros(
                    p32.shape, dtype=self.state_dtype, device="cpu"
                )
            elif torch.is_tensor(state.get("step")):
                state["step"] = int(state["step"].item())
            state["step"] = int(state.get("step", 0)) + 1
            t = int(state["step"])
            exp_avg = state["exp_avg"].float()
            exp_avg_sq = state["exp_avg_sq"].float()
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            state["exp_avg"] = exp_avg.to(dtype=self.state_dtype)
            state["exp_avg_sq"] = exp_avg_sq.to(dtype=self.state_dtype)
        else:
            t = 1
            exp_avg = grad * (1.0 - beta1)
            exp_avg_sq = grad * grad * (1.0 - beta2)
            state.clear()
        if wd != 0.0:
            p32.mul_(1.0 - lr * wd)
        denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**t)).add_(eps)
        p32.addcdiv_(exp_avg, denom, value=-lr / (1.0 - beta1**t))
        p.copy_(p32.to(device=p.device, dtype=p.dtype))
        del p32, grad, exp_avg, exp_avg_sq

    def step_params(self, params) -> None:
        gmap = self._group_map()
        default = self.param_groups[0]
        for p in params:
            self._update_one(p, gmap.get(id(p), default))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                self._update_one(p, group)
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for st in self.state.values():
            if torch.is_tensor(st.get("step")):
                st["step"] = int(st["step"].item())
            for k, v in list(st.items()):
                if k == "step" or not torch.is_tensor(v):
                    continue
                st[k] = v.detach().cpu().to(dtype=self.state_dtype)


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


def build_optimizer(
    model: nn.Module,
    cfg: CATYokoConfig,
    *,
    cpu_offload: bool = False,
    state_dtype: torch.dtype = torch.float32,
    retain_state: bool = True,
) -> Optimizer:
    groups = [g for g in adamw_param_groups(model, cfg.weight_decay) if g["params"]]
    if cpu_offload:
        return CPUOffloadAdamW(
            groups,
            lr=cfg.lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.weight_decay,
            state_dtype=state_dtype,
            retain_state=retain_state,
        )
    return AdamW(groups, **_adamw_kwargs(cfg))


def wsd_lr(tokens_seen: float, cfg: CATYokoConfig, phase: str) -> float:
    base = cfg.lr_b2 if phase == "B2" else cfg.lr
    if tokens_seen < cfg.warmup_tokens:
        return base * max(tokens_seen / cfg.warmup_tokens, 1e-3)
    return base
