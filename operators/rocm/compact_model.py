"""Experimental compact representation of identical, frozen B0 MoE experts.

Load/upcycle a native model and its trainable overlay, apply B0 freezing, then
call ``compact_frozen_moes(model, phase="B0")`` while it is on CPU. This removes
19 redundant routed experts per MoE before GPU transport. Shared experts keep
their native modules. It changes frozen full-state keys; B0 trainable overlay
keys and optimizer parameters are preserved. Reconstruct/upcycle the native
model before applying an overlay or entering B1/B2. This is an experimental
operator and does not install itself in production.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.moe import MoE


def _preflight(moe: MoE) -> None:
    if any(p.requires_grad for p in moe.parameters()):
        raise ValueError("compact MoE requires fully frozen B0 parameters")
    if len(moe.experts) != moe.n_routed or not moe.experts or not moe.shared:
        raise ValueError("invalid native shared/routed expert layout")
    for expert in (*moe.shared, *moe.experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(expert, name, None)
            if type(projection) is not nn.Linear or projection.bias is not None:
                raise ValueError("compact MoE requires native bias-free SwiGLU linears")
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight = getattr(moe.experts[0], name).weight
        raw = weight.detach().cpu().contiguous().view(torch.uint8)
        for expert in moe.experts[1:]:
            other = getattr(expert, name).weight
            if other.shape != weight.shape or other.dtype != weight.dtype:
                raise ValueError(f"routed {name} weights have different shape/dtype")
            if not torch.equal(raw, other.detach().cpu().contiguous().view(torch.uint8)):
                raise ValueError(f"routed {name} weights are not byte-identical")


def _drop_transient_caches(module: nn.Module) -> None:
    # Do not retain a native MoE's 20-expert packed GPU/CPU storage. Existing
    # autograd references remain untouched; callers install before forwarding.
    for child in module.modules():
        for name in ("_swiglu_w", "_swiglu_gu", "_wq_cache", "_wq_ver",
                     "_bf16_fused_cat", "_te_fused_cat", "_te_grouped"):
            if hasattr(child, name):
                setattr(child, name, None)


class CompactFrozenMoE(MoE):
    """Native MoE interface with one retained frozen routed SwiGLU.

    The MoE base class keeps block ``isinstance(..., MoE)`` checks and inherited
    load/bias helpers working. Its allocating initializer is intentionally
    skipped. Input gradients and cast normalized router gates are retained;
    BF16 GEMM/reduction reassociation means results are numerically approximate.
    Frozen weights must remain immutable; normal in-place updates are rejected.
    """

    def __init__(self, native: MoE, *, phase: str = "B0", _validated: bool = False):
        if phase != "B0":
            raise ValueError("compact identical experts are only supported in B0")
        if not _validated:
            _preflight(native)
        nn.Module.__init__(self)
        _drop_transient_caches(native)
        self.hidden = native.hidden
        self.n_shared = native.n_shared
        self.n_routed = native.n_routed
        self.top_k = native.top_k
        self.hash_route = native.hash_route
        self.router_z_loss = native.router_z_loss
        self.seq_balance_loss = native.seq_balance_loss
        self.batched_experts = False
        self.grouped_experts = False
        self.shared = native.shared
        # Keep expert 0's frozen key prefix compatible with native state.
        self.experts = nn.ModuleList([native.experts[0]])
        self.router = native.router
        self.register_buffer("e_score_correction_bias", native.e_score_correction_bias)
        self.last_aux = native.last_aux
        self.last_load = native.last_load
        self._load_n = native._load_n
        self._has_trainable_params = False
        self._frozen_parameters = tuple(self.parameters())
        self._frozen_versions = tuple(p._version for p in self._frozen_parameters)
        self._projection_signature = self._projections()
        self.train(native.training)

    def _projections(self) -> tuple:
        return tuple(
            (id(layer), type(layer))
            for expert in (*self.shared, *self.experts)
            for name in ("gate_proj", "up_proj", "down_proj")
            for layer in (getattr(expert, name),)
        )

    def _check_frozen(self) -> None:
        if self._projections() != self._projection_signature:
            raise RuntimeError("compact B0 MoE projection modules changed; reconstruct native MoE")
        current = tuple(self.parameters())
        if len(current) != len(self._frozen_parameters) or any(
            p is not saved or p.requires_grad or p._version != version
            for p, saved, version in zip(current, self._frozen_parameters, self._frozen_versions)
        ):
            raise RuntimeError("compact B0 MoE parameters changed or became trainable; reconstruct native MoE")

    def _apply(self, fn, recurse=True):
        # A device/dtype conversion may replace Parameter objects. Keep the
        # immutability guard aligned with supported nn.Module.to() conversions,
        # and avoid cached gate/up weights remaining on the previous device.
        self._check_frozen()
        _drop_transient_caches(self)
        result = super()._apply(fn, recurse=recurse)
        self._frozen_parameters = tuple(self.parameters())
        self._frozen_versions = tuple(p._version for p in self._frozen_parameters)
        return result

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None):
        self._check_frozen()
        flat = x.reshape(-1, self.hidden)
        shared = self.shared[0](flat)
        for expert in self.shared[1:]:
            shared = shared + expert(flat)
        if self.hash_route and token_ids is not None:
            # The selected identical expert and unit gate cannot change output.
            routed = self.experts[0](flat)
            self.last_aux = flat.new_zeros(())
            self.last_load = None
            self._load_n = 0
        else:
            logits = F.linear(flat.float(), self.router.weight.float())
            affinity = torch.sqrt(F.softplus(logits))
            probs = torch.softmax(affinity + self.e_score_correction_bias.float(), dim=-1)
            values, indices = torch.topk(probs, self.top_k, dim=-1)
            gates = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            expert_out = self.experts[0](flat)
            gate_sum = gates.to(expert_out.dtype).sum(dim=-1, keepdim=True, dtype=flat.dtype)
            routed = expert_out * gate_sum
            load = flat.new_zeros(self.n_routed)
            load.scatter_add_(0, indices.reshape(-1), gates.reshape(-1).to(load.dtype))
            load = load / flat.shape[0]
            self.last_aux = None
            if self.training:
                pending = load.detach()
                if self.last_load is None:
                    self.last_load = pending.clone()
                    self._load_n = 1
                else:
                    self.last_load = self.last_load + pending
                    self._load_n += 1
            else:
                self.last_load = None
                self._load_n = 0
        return (shared + routed.to(dtype=shared.dtype, device=shared.device)).reshape_as(x)


def compact_frozen_moes(model: nn.Module, *, phase: str = "B0") -> dict[str, int]:
    """Preflight every native MoE before replacing any module in the model.

    Call after checkpoint loading and B0 freezing. Dense blocks are retained.
    The returned storage counts cover model parameters, with aliases deduped.
    """
    if phase != "B0":
        raise ValueError("compact identical experts are only supported in B0")
    candidates = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, CompactFrozenMoE):
                continue
            if isinstance(child, MoE):
                _preflight(child)
                candidates.append((parent, name, child))
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_before = {name: id(p) for name, p in model.named_parameters() if p.requires_grad}
    for parent, name, native in candidates:
        setattr(parent, name, CompactFrozenMoE(native, phase=phase, _validated=True))
    trainable_after = {name: id(p) for name, p in model.named_parameters() if p.requires_grad}
    if trainable_after != trainable_before:
        raise AssertionError("compact replacement changed trainable overlay names or parameters")
    after = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"compacted_moes": len(candidates), "parameter_bytes_before": before,
            "parameter_bytes_after": after, "removed_parameter_bytes": before - after,
            "trainable_keys": len(trainable_after)}
