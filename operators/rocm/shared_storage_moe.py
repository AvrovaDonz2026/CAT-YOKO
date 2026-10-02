"""B0 identical expert storage sharing with the complete native MoE arithmetic.

Unlike compute collapse, this keeps all logical experts and native routing,
padded batched GEMM shapes, gate multiplication, and index_add reduction order.
Only byte-identical frozen expert storage is shared. Load/upcycle and apply B0
freezing before installing, with NVFP4/FP8 disabled. Native full-state expert
keys and trainable overlay names remain present. CPU and ROCm use batched GEMM;
a grouped GEMM backend must be disabled because zero expert stride is untested.
"""

from __future__ import annotations

import torch
from torch import nn

from cat_yoko.moe import MoE, grouped_mm_available
from operators.rocm.compact_model import _drop_transient_caches, _preflight


class SharedStorageFrozenMoE(MoE):
    """Native MoE.forward with aliased expert modules and stride-zero weights."""

    def __init__(self, native: MoE, *, phase: str = "B0", _validated: bool = False):
        if phase != "B0":
            raise ValueError("shared frozen expert storage is only supported in B0")
        if not _validated:
            _preflight(native)
        nn.Module.__init__(self)
        _drop_transient_caches(native)
        for name in ("hidden", "n_shared", "n_routed", "top_k", "hash_route",
                     "router_z_loss", "seq_balance_loss", "batched_experts", "grouped_experts"):
            setattr(self, name, getattr(native, name))
        self.shared = native.shared
        self.experts = nn.ModuleList([native.experts[0]] * native.n_routed)
        self.router = native.router
        self.register_buffer("e_score_correction_bias", native.e_score_correction_bias)
        self.last_aux = native.last_aux
        self.last_load = native.last_load
        self._load_n = native._load_n
        self.track_load = bool(getattr(native, "track_load", True))
        self._has_trainable_params = False
        self._swiglu_w = None
        self._swiglu_gu = None
        self._refresh_guards()
        self.train(native.training)

    def _projections(self) -> tuple:
        return tuple((id(layer), type(layer))
                     for expert in (*self.shared, *self.experts)
                     for name in ("gate_proj", "up_proj", "down_proj")
                     for layer in (getattr(expert, name),))

    def _refresh_guards(self) -> None:
        self._frozen_parameters = tuple(self.parameters())
        self._frozen_versions = tuple(p._version for p in self._frozen_parameters)
        self._projection_signature = self._projections()

    def _check_frozen(self) -> None:
        if len(self.experts) != self.n_routed or any(e is not self.experts[0] for e in self.experts):
            raise RuntimeError("logical expert aliases changed; reconstruct native MoE")
        if self._projections() != self._projection_signature:
            raise RuntimeError("shared B0 MoE projection modules changed; reconstruct native MoE")
        current = tuple(self.parameters())
        if len(current) != len(self._frozen_parameters) or any(
            p is not saved or p.requires_grad or p._version != version
            for p, saved, version in zip(current, self._frozen_parameters, self._frozen_versions)
        ):
            raise RuntimeError("shared B0 MoE parameters changed or became trainable; reconstruct native MoE")

    def _prepare_weights(self) -> None:
        if self._swiglu_w is not None and self._swiglu_gu is not None:
            return
        expert = self.experts[0]
        gate, up, down = (getattr(expert, name).weight
                          for name in ("gate_proj", "up_proj", "down_proj"))
        count = self.n_routed
        self._swiglu_w = tuple(w.unsqueeze(0).expand(count, *w.shape)
                              for w in (gate, up, down)) + (False,)
        # One gate/up copy instead of E copies; bmm consumes the expanded
        # [E,N,K] view with zero batch stride directly. Do not make contiguous.
        fused = torch.cat((gate, up), dim=0)
        self._swiglu_gu = fused.unsqueeze(0).expand(count, *fused.shape)

    def _apply(self, fn, recurse=True):
        self._check_frozen()
        _drop_transient_caches(self)
        result = super()._apply(fn, recurse=recurse)
        self._refresh_guards()
        return result

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor | None = None):
        self._check_frozen()
        if x.is_cuda and self.grouped_experts and grouped_mm_available():
            raise RuntimeError("shared-storage weights require native batched/serial GEMM, not grouped GEMM")
        if self.batched_experts and self.experts[0].gate_proj.weight.dtype != x.dtype:
            raise RuntimeError("shared-storage batched GEMM requires input/master dtypes to match")
        if x.is_cuda and self.batched_experts:
            from cat_yoko.offload import _read_autocast

            enabled, dtype = _read_autocast(x.device)
            if enabled and self.experts[0].gate_proj.weight.dtype != dtype:
                raise RuntimeError("GPU autocast must match frozen master dtype to preserve zero-stride storage")
        self._prepare_weights()
        return MoE.forward(self, x, token_ids)


def share_frozen_moe_storage(model: nn.Module, *, phase: str = "B0") -> dict[str, int]:
    """Preflight all native MoEs, then share storage without changing arithmetic."""
    if phase != "B0":
        raise ValueError("shared frozen expert storage is only supported in B0")
    cfg = getattr(model, "cfg", None)
    if cfg is not None and (getattr(cfg, "use_nvfp4", False) or getattr(cfg, "use_fp8", False)):
        raise ValueError("shared frozen expert storage requires NVFP4/FP8 disabled")
    candidates = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, SharedStorageFrozenMoE):
                continue
            if isinstance(child, MoE):
                _preflight(child)
                candidates.append((parent, name, child))
    before = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_before = {n: id(p) for n, p in model.named_parameters() if p.requires_grad}
    for parent, name, native in candidates:
        setattr(parent, name, SharedStorageFrozenMoE(native, phase=phase, _validated=True))
    trainable_after = {n: id(p) for n, p in model.named_parameters() if p.requires_grad}
    if trainable_after != trainable_before:
        raise AssertionError("shared storage changed trainable overlay names or parameters")
    after = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"shared_storage_moes": len(candidates), "parameter_bytes_before": before,
            "parameter_bytes_after": after, "removed_parameter_bytes": before - after,
            "trainable_keys": len(trainable_after)}


shared_storage_frozen_moes = share_frozen_moe_storage
