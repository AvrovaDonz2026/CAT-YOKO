"""DeepSeek-style softmax-then-topK MoE with optional hash routing."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.config import CATYokoConfig


def _like(ref: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Autocast experts emit bf16 into an fp32 residual buffer; align dtypes."""
    if t.dtype != ref.dtype or t.device != ref.device:
        return t.to(dtype=ref.dtype, device=ref.device)
    return t


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(
        self,
        cfg: CATYokoConfig,
        n_routed: int,
        top_k: int,
        *,
        hash_route: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = cfg.hidden_size
        self.n_shared = cfg.n_shared
        self.n_routed = n_routed
        self.top_k = top_k
        self.hash_route = hash_route
        self.router_z_loss = cfg.router_z_loss
        self.seq_balance_loss = cfg.seq_balance_loss
        d, mid = cfg.hidden_size, cfg.moe_intermediate_size
        self.shared = nn.ModuleList(SwiGLU(d, mid) for _ in range(cfg.n_shared))
        self.experts = nn.ModuleList(SwiGLU(d, mid) for _ in range(n_routed))
        self.router = nn.Linear(d, n_routed, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed))
        self.last_aux: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None

    def step_router_bias(self) -> None:
        """Aux-loss-free bias. Call after backward so checkpoint recompute sees the same routing."""
        if self.last_load is None:
            return
        with torch.no_grad():
            target = 1.0 / self.n_routed
            # last_load is a Python attr; module.to("cpu") does not move it.
            load = self.last_load.to(
                device=self.e_score_correction_bias.device,
                dtype=self.e_score_correction_bias.dtype,
            )
            self.e_score_correction_bias += 1e-3 * (target - load)
        self.last_load = None

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, s, d = x.shape
        flat = x.reshape(b * s, d)
        shared_out = self.shared[0](flat)
        for m in self.shared[1:]:
            shared_out = shared_out + m(flat)
        if self.hash_route and token_ids is not None:
            ids = token_ids.reshape(b * s).to(torch.int64)
            expert_id = (ids * 2654435761).remainder(self.n_routed)
            routed = torch.zeros_like(flat)
            for e in range(self.n_routed):
                mask = expert_id == e
                if mask.any():
                    routed[mask] = _like(routed, self.experts[e](flat[mask]))
            self.last_aux = flat.new_zeros(())
            self.last_load = None
            return (shared_out + _like(shared_out, routed)).view(b, s, d)

        # Router logits + softmax stay fp32 (C1+FP8 whitelist).
        logits = F.linear(flat.float(), self.router.weight.float())
        affinity = torch.sqrt(F.softplus(logits))
        # softmax-then-topK (upcycling paper); bias is aux-loss-free, pre-softmax.
        probs = torch.softmax(affinity + self.e_score_correction_bias.float(), dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        gates = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        routed = torch.zeros_like(flat)
        for e in range(self.n_routed):
            hit = topi == e
            if not hit.any():
                continue
            tok_ix, slot = hit.nonzero(as_tuple=True)
            weight = gates[tok_ix, slot].unsqueeze(-1)
            contrib = weight * self.experts[e](flat[tok_ix])
            routed[tok_ix] = routed[tok_ix] + _like(routed, contrib)

        z_loss = logits.float().pow(2).mean()
        ones = torch.zeros(self.n_routed, device=x.device, dtype=x.dtype)
        ones.scatter_add_(0, topi.reshape(-1), _like(ones, gates.reshape(-1)))
        load = ones / (b * s)
        balance = self.n_routed * (load * load).sum()
        self.last_aux = self.router_z_loss * z_loss + self.seq_balance_loss * balance
        if self.training:
            self.last_load = load.detach()
        else:
            self.last_load = None
        return (shared_out + _like(shared_out, routed)).view(b, s, d)


def moe_utilization(model: nn.Module) -> dict[str, float]:
    """Layer-mean coefficient of variation of routed expert load (before bias step)."""
    cvs: list[float] = []
    maxs: list[float] = []
    mins: list[float] = []
    n_layers = 0
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    blocks = list(encoder or []) + list(decoder or [])
    for blk in blocks:
        mlp = getattr(blk, "mlp", None)
        load = getattr(mlp, "last_load", None)
        if load is None:
            continue
        p = load.detach().float().reshape(-1)
        if p.numel() == 0:
            continue
        n_layers += 1
        mean = float(p.mean().clamp_min(1e-12))
        cvs.append(float(p.std(unbiased=False) / mean))
        maxs.append(float(p.max()))
        mins.append(float(p.min()))
    if not cvs:
        return {"moe_cv": 0.0, "moe_max": 0.0, "moe_min": 0.0, "moe_layers": 0}
    return {
        "moe_cv": sum(cvs) / len(cvs),
        "moe_max": max(maxs),
        "moe_min": min(mins),
        "moe_layers": n_layers,
    }
