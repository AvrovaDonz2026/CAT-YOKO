"""DeepSeek-style softmax-then-topK MoE with optional hash routing.

Expert dispatch permutes tokens by expert id (one bincount sync per layer)
instead of 20× ``nonzero`` / ``.any()`` CUDA syncs. Routed SwiGLU prefers
``grouped_mm`` (jagged tokens, no padding). Padded batched GEMM is the
fallback; serial expert loop remains as a numeric reference. Routing math
is unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.config import CATYokoConfig


def grouped_mm_available() -> bool:
    """True when this PyTorch build exposes grouped GEMM (``F.grouped_mm`` / ``_grouped_mm``)."""
    return callable(getattr(F, "grouped_mm", None)) or callable(getattr(torch, "_grouped_mm", None))


def _raw_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``mat_a`` [T, K], ``mat_b`` [E, K, N], ``offs`` int32 cumulative token ends."""
    fn = getattr(F, "grouped_mm", None)
    if callable(fn):
        return fn(mat_a, mat_b, offs=offs)
    return torch._grouped_mm(mat_a, mat_b, offs=offs)


def _grouped_wgrad(x: torch.Tensor, dy: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """``dW[e] = dy_e.T @ x_e``. Prefer one padded bmm; serial slices if hugely imbalanced."""
    e, n_out, k = weight.shape
    n_tok = int(x.size(0))
    offs64 = offs.to(dtype=torch.int64, device=x.device)
    counts = torch.diff(offs64, prepend=offs64.new_zeros(1))
    max_n = int(counts.max().item()) if e else 0
    if max_n <= 0 or n_tok == 0:
        return torch.zeros_like(weight)
    if max_n * e > 8 * n_tok:
        dw = torch.zeros_like(weight)
        prev = 0
        for ei, end in enumerate(offs64.tolist()):
            end_i = int(end)
            if end_i > prev:
                dw[ei] = dy[prev:end_i].T @ x[prev:end_i]
            prev = end_i
        return dw
    starts = torch.zeros(e, dtype=torch.int64, device=x.device)
    if e > 1:
        starts[1:] = offs64[:-1]
    expert_sorted = torch.repeat_interleave(torch.arange(e, device=x.device), counts)
    local = torch.arange(n_tok, device=x.device) - torch.repeat_interleave(starts, counts)
    x_pad = x.new_zeros(e, max_n, k)
    dy_pad = dy.new_zeros(e, max_n, n_out)
    x_pad[expert_sorted, local] = x
    dy_pad[expert_sorted, local] = dy
    return torch.bmm(dy_pad.transpose(1, 2), x_pad)


class _GroupedLinear(torch.autograd.Function):
    """``y = grouped_mm(x, W.T, offs)`` with dX / optional dW. Native autograd is missing."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight, offs)
        ctx.weight_requires_grad = bool(weight.requires_grad)
        b = weight.transpose(-2, -1).contiguous()
        return _raw_grouped_mm(x.contiguous(), b, offs)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, weight, offs = ctx.saved_tensors
        dy = dy.contiguous()
        dx = _raw_grouped_mm(dy, weight.contiguous(), offs)
        dw = _grouped_wgrad(x, dy, weight, offs) if ctx.weight_requires_grad else None
        return dx, dw, None


def _grouped_linear(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return _GroupedLinear.apply(x, weight, offs)


def _like(ref: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Autocast experts emit bf16 into an fp32 residual buffer; align dtypes."""
    if t.dtype != ref.dtype or t.device != ref.device:
        return t.to(dtype=ref.dtype, device=ref.device)
    return t


def _fused_gate_up(gate: nn.Module, up: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """One GEMM for gate and up; SiLU(gate)*up. Same math as two Linears."""
    from cat_yoko.nvfp4_linear import fused_cat_linear

    if isinstance(gate, nn.Linear) and isinstance(up, nn.Linear):
        gu = fused_cat_linear([gate, up], x)
        g, u = gu.chunk(2, dim=-1)
        return F.silu(g) * u
    return F.silu(gate(x)) * up(x)


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_fused_gate_up(self.gate_proj, self.up_proj, x))


def _stack_linear_weight(linears: list[nn.Linear]) -> torch.Tensor:
    from cat_yoko.nvfp4_linear import Nvfp4Linear, TeNvfp4Linear

    if any(isinstance(lin, TeNvfp4Linear) for lin in linears):
        raise RuntimeError("TE NVFP4 experts must not stack into bf16 grouped_mm")
    if all(isinstance(lin, Nvfp4Linear) for lin in linears):
        return torch.stack([lin.quantized_weight() for lin in linears])
    return torch.stack([lin.weight for lin in linears])


def _swiglu_expert_weights(
    experts: nn.ModuleList, owner: nn.Module | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    from cat_yoko.nvfp4_linear import Nvfp4Linear

    nv = all(isinstance(e.gate_proj, Nvfp4Linear) for e in experts) and all(
        isinstance(e.up_proj, Nvfp4Linear) and isinstance(e.down_proj, Nvfp4Linear) for e in experts
    )
    trainable = any(e.gate_proj.weight.requires_grad for e in experts)
    if owner is not None and not trainable:
        hit = getattr(owner, "_swiglu_w", None)
        if hit is not None:
            return hit
    gate_w = _stack_linear_weight([e.gate_proj for e in experts])
    up_w = _stack_linear_weight([e.up_proj for e in experts])
    down_w = _stack_linear_weight([e.down_proj for e in experts])
    pack = (gate_w, up_w, down_w, nv)
    if owner is not None and not trainable:
        owner._swiglu_w = pack
    return pack


def _experts_are_te_nvfp4(experts: nn.ModuleList) -> bool:
    from cat_yoko.nvfp4_linear import TeNvfp4Linear

    if not experts:
        return False
    return all(
        isinstance(e.gate_proj, TeNvfp4Linear)
        and isinstance(e.up_proj, TeNvfp4Linear)
        and isinstance(e.down_proj, TeNvfp4Linear)
        for e in experts
    )


def _experts_are_swiglu(experts: nn.ModuleList) -> bool:
    if not experts:
        return False
    for e in experts:
        if not isinstance(e, SwiGLU):
            return False
        if not all(isinstance(getattr(e, n), nn.Linear) for n in ("gate_proj", "up_proj", "down_proj")):
            return False
    return True


def _swiglu_experts_serial(
    experts: nn.ModuleList, x_sorted: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    offset = 0
    for e, n in enumerate(counts.tolist()):
        n = int(n)
        if n == 0:
            continue
        parts.append(experts[e](x_sorted.narrow(0, offset, n)))
        offset += n
    if not parts:
        return x_sorted.new_zeros(x_sorted.shape)
    return torch.cat(parts, dim=0)


def _swiglu_experts_batched(
    experts: nn.ModuleList,
    x_sorted: torch.Tensor,
    expert_sorted: torch.Tensor,
    counts: torch.Tensor,
    owner: nn.Module | None = None,
) -> torch.Tensor:
    """Padded bmm over experts. Empty experts stay zero-padded rows."""
    from cat_yoko.nvfp4_linear import quantize_nvfp4

    n_tok = x_sorted.size(0)
    if n_tok == 0:
        return x_sorted
    max_n = int(counts.max().item())
    if max_n <= 0:
        return x_sorted.new_zeros(x_sorted.shape)
    e_count = len(experts)
    # Pathological imbalance: padding would do more GEMM than serial slices.
    if max_n * e_count > 8 * n_tok:
        return _swiglu_experts_serial(experts, x_sorted, counts)

    d = x_sorted.size(-1)
    x_pad = x_sorted.new_zeros(e_count, max_n, d)
    offs = torch.zeros(e_count, dtype=torch.int64, device=x_sorted.device)
    if e_count > 1:
        offs[1:] = torch.cumsum(counts[:-1], dim=0)
    local = torch.arange(n_tok, device=x_sorted.device) - offs[expert_sorted]
    x_pad[expert_sorted, local] = x_sorted

    gate_w, up_w, down_w, nv = _swiglu_expert_weights(experts, owner)
    if nv:
        ctx = torch.autocast(device_type="cuda", enabled=False) if x_sorted.is_cuda else nullcontext()
        with ctx:
            x_q = quantize_nvfp4(x_pad) + x_pad - x_pad.detach()
            gu_w = torch.cat((gate_w, up_w), dim=1)
            gu = torch.bmm(x_q, gu_w.transpose(1, 2))
            g, u = gu.chunk(2, dim=-1)
            hidden = F.silu(g) * u
            h_q = quantize_nvfp4(hidden) + hidden - hidden.detach()
            y_pad = torch.bmm(h_q, down_w.transpose(1, 2))
    else:
        gu_w = torch.cat((gate_w, up_w), dim=1).to(dtype=x_pad.dtype)
        down_w = down_w.to(dtype=x_pad.dtype)
        gu = torch.bmm(x_pad, gu_w.transpose(1, 2))
        g, u = gu.chunk(2, dim=-1)
        y_pad = torch.bmm(F.silu(g) * u, down_w.transpose(1, 2))
    return y_pad[expert_sorted, local]


def _swiglu_experts_grouped(
    experts: nn.ModuleList,
    x_sorted: torch.Tensor,
    counts: torch.Tensor,
    owner: nn.Module | None = None,
) -> torch.Tensor:
    """Jagged grouped GEMM. ``x_sorted`` is already packed by expert id."""
    from cat_yoko.nvfp4_linear import quantize_nvfp4

    n_tok = x_sorted.size(0)
    if n_tok == 0:
        return x_sorted
    if int(counts.max().item()) <= 0:
        return x_sorted.new_zeros(x_sorted.shape)
    offs = torch.cumsum(counts, dim=0).to(dtype=torch.int32)
    gate_w, up_w, down_w, nv = _swiglu_expert_weights(experts, owner)
    gu_w = torch.cat((gate_w, up_w), dim=1)
    if nv:
        ctx = torch.autocast(device_type="cuda", enabled=False) if x_sorted.is_cuda else nullcontext()
        with ctx:
            x_q = quantize_nvfp4(x_sorted) + x_sorted - x_sorted.detach()
            gu = _grouped_linear(x_q, gu_w, offs)
            g, u = gu.chunk(2, dim=-1)
            hidden = F.silu(g) * u
            h_q = quantize_nvfp4(hidden) + hidden - hidden.detach()
            return _grouped_linear(h_q, down_w, offs)
    gu_w = gu_w.to(dtype=x_sorted.dtype)
    down_w = down_w.to(dtype=x_sorted.dtype)
    gu = _grouped_linear(x_sorted, gu_w, offs)
    g, u = gu.chunk(2, dim=-1)
    return _grouped_linear(F.silu(g) * u, down_w, offs)


def _dispatch_experts(
    experts: nn.ModuleList,
    flat: torch.Tensor,
    expert_idx: torch.Tensor,
    token_idx: torch.Tensor,
    gates: torch.Tensor,
    n_routed: int,
    *,
    batched: bool,
    grouped: bool = True,
    owner: nn.Module | None = None,
) -> torch.Tensor:
    """``expert_idx`` / ``token_idx`` / ``gates`` are length ``T * k`` (or ``T``)."""
    order = expert_idx.argsort()
    expert_sorted = expert_idx.index_select(0, order)
    token_sorted = token_idx.index_select(0, order)
    gate_sorted = gates.index_select(0, order)
    x_sorted = flat.index_select(0, token_sorted)
    counts = torch.bincount(expert_sorted, minlength=n_routed)
    use_batched = batched and _experts_are_swiglu(experts)
    want_grouped = (
        use_batched
        and grouped
        and grouped_mm_available()
        and x_sorted.is_cuda
    )
    if _experts_are_te_nvfp4(experts):
        from cat_yoko.nvfp4_hw import te_grouped_swiglu

        y = te_grouped_swiglu(experts, x_sorted, counts, owner)
        if y is None:
            y = _swiglu_experts_serial(experts, x_sorted, counts)
    elif want_grouped:
        try:
            y = _swiglu_experts_grouped(experts, x_sorted, counts, owner=owner)
        except (RuntimeError, NotImplementedError):
            y = _swiglu_experts_batched(experts, x_sorted, expert_sorted, counts, owner=owner)
    elif use_batched:
        y = _swiglu_experts_batched(experts, x_sorted, expert_sorted, counts, owner=owner)
    else:
        y = _swiglu_experts_serial(experts, x_sorted, counts)
    y = y * gate_sorted.unsqueeze(-1).to(dtype=y.dtype)
    routed = torch.zeros_like(flat)
    routed.index_add_(0, token_sorted, _like(routed, y))
    return routed


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
        # Tests may flip these to compare against the serial expert loop.
        self.batched_experts = True
        self.grouped_experts = True
        d, mid = cfg.hidden_size, cfg.moe_intermediate_size
        self.shared = nn.ModuleList(SwiGLU(d, mid) for _ in range(cfg.n_shared))
        self.experts = nn.ModuleList(SwiGLU(d, mid) for _ in range(n_routed))
        self.router = nn.Linear(d, n_routed, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed))
        self.last_aux: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        self._load_n = 0

    def mean_pending_load(self) -> torch.Tensor | None:
        """Average expert load over micro-batches since the last bias step."""
        if self.last_load is None:
            return None
        n = max(int(self._load_n), 1)
        if n == 1:
            return self.last_load
        return self.last_load / n

    def step_router_bias(self) -> None:
        """Aux-loss-free bias. Call once per optimizer step, after DDP load sync."""
        load = self.mean_pending_load()
        if load is None:
            self._load_n = 0
            return
        with torch.no_grad():
            target = 1.0 / self.n_routed
            # last_load is a Python attr; module.to("cpu") does not move it.
            load = load.to(
                device=self.e_score_correction_bias.device,
                dtype=self.e_score_correction_bias.dtype,
            )
            self.e_score_correction_bias += 1e-3 * (target - load)
        self.last_load = None
        self._load_n = 0

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, s, d = x.shape
        flat = x.reshape(b * s, d)
        shared_out = self.shared[0](flat)
        for m in self.shared[1:]:
            shared_out = shared_out + m(flat)
        n_tok = b * s
        if self.hash_route and token_ids is not None:
            ids = token_ids.reshape(n_tok).to(torch.int64)
            expert_id = (ids * 2654435761).remainder(self.n_routed)
            token_idx = torch.arange(n_tok, device=flat.device)
            gates = torch.ones(n_tok, device=flat.device, dtype=flat.dtype)
            routed = _dispatch_experts(
                self.experts,
                flat,
                expert_id,
                token_idx,
                gates,
                self.n_routed,
                batched=self.batched_experts,
                grouped=self.grouped_experts,
                owner=self,
            )
            self.last_aux = flat.new_zeros(())
            self.last_load = None
            self._load_n = 0
            return (shared_out + _like(shared_out, routed)).view(b, s, d)

        # Router logits + softmax stay fp32 (must-high-prec; discrete top-k).
        logits = F.linear(flat.float(), self.router.weight.float())
        affinity = torch.sqrt(F.softplus(logits))
        # softmax-then-topK (upcycling paper); bias is aux-loss-free, pre-softmax.
        probs = torch.softmax(affinity + self.e_score_correction_bias.float(), dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        gates = topv / topv.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        token_idx = (
            torch.arange(n_tok, device=flat.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        routed = _dispatch_experts(
            self.experts,
            flat,
            topi.reshape(-1),
            token_idx,
            gates.reshape(-1),
            self.n_routed,
            batched=self.batched_experts,
            grouped=self.grouped_experts,
            owner=self,
        )

        z_loss = logits.float().pow(2).mean()
        ones = torch.zeros(self.n_routed, device=x.device, dtype=x.dtype)
        ones.scatter_add_(0, topi.reshape(-1), _like(ones, gates.reshape(-1)))
        load = ones / n_tok
        balance = self.n_routed * (load * load).sum()
        self.last_aux = self.router_z_loss * z_loss + self.seq_balance_loss * balance
        if self.training:
            ld = load.detach()
            if self.last_load is None:
                self.last_load = ld.clone()
                self._load_n = 1
            else:
                self.last_load = self.last_load + ld
                self._load_n += 1
        else:
            self.last_load = None
            self._load_n = 0
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
        load = None
        if mlp is not None and hasattr(mlp, "mean_pending_load"):
            load = mlp.mean_pending_load()
        else:
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
