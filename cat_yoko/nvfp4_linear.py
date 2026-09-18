"""NVFP4 linear GEMM: E2M1 data + per-16-block scales.

Master weights stay the original ``nn.Linear`` Parameter (bf16). Allowed
slots (attn QKV/O, MoE experts, cross Q/O, cache KV, lm_head, encoder
linears) run this forward. Router / embed / RMSNorm stay out.
B0 wraps frozen encoder GEMMs only; B1/B2 wrap **all** allowed GEMMs
(including the unfrozen encoder in B2). Decoder self-attn stays
``WindowAttention`` either way.

On Blackwell, Transformer Engine ``NVFP4BlockScaling`` is preferred when
importable. Otherwise this module emulates E2M1 with 16-wide blocks
(fp32 block scale; E4M3 block scale on CUDA if ``float8_e4m3fn`` works).
Backward uses STE through the dequantized GEMM so Adam still sees bf16
master grads. True fused WGRAD/RHT needs TE.

Attention math is unchanged: wrap Q/K/V/O (and cross Q/O, cache KV in
B1/B2) only. Causal YOCO window / GQA / qk_norm / fp32 SDPA stay in
``attention.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.nvfp4 import POLICY

_BLOCK = 16
# E2M1 magnitudes (sign applied separately). Max abs is 6.
_E2M1_ABS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between consecutive E2M1 magnitudes (bucketize, not 8-wide argmin).
_E2M1_THRESH = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Leaf names that are NVFP4 GEMMs. Router / embed / RMSNorm are not Linear
# slots here (router is Linear but excluded in should_wrap_linear).
NVFP4_LINEAR_LEAVES = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "cache_k",
        "cache_v",
        "lm_head",
    }
)


def hardware_nvfp4() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 10  # Blackwell SM100 / sm_120 6000D


def te_nvfp4_recipe():
    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except Exception:
        return None
    return NVFP4BlockScaling()


def te_available() -> bool:
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: F401
    except Exception:
        return False
    return True


def hw_nvfp4_gemm_available() -> bool:
    """True when this PyTorch build can cast to float4 and run ``_scaled_mm``.

    torch 2.8.0+cu128 on sm_120 exposes ``torch.float4_e2m1fn_x2`` but
    ``copy_`` is NotImplemented, so the cuBLAS NVFP4 GEMM is not reachable.
    """
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        x = torch.zeros(32, device="cuda", dtype=torch.bfloat16)
        _ = x.to(torch.float4_e2m1fn_x2)
    except Exception:
        return False
    return True


def _e2m1_levels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(_E2M1_ABS, device=device, dtype=dtype)


def quantize_nvfp4(t: torch.Tensor, *, block: int = _BLOCK) -> torch.Tensor:
    """Quantize-dequantize ``t`` to emulated NVFP4. Same shape/dtype as input."""
    if t.numel() == 0:
        return t
    orig_shape = t.shape
    orig_dtype = t.dtype
    x = t.reshape(-1).to(dtype=torch.float32)
    n = x.numel()
    pad = (block - n % block) % block
    if pad:
        x = F.pad(x, (0, pad))
    chunks = x.view(-1, block)
    amax = chunks.abs().amax(dim=-1).clamp_min(1e-12)
    scale = amax / 6.0
    if t.is_cuda and hasattr(torch, "float8_e4m3fn"):
        tensor_scale = scale.max().clamp_min(1e-12)
        e4 = (scale / tensor_scale).clamp(max=448.0)
        try:
            e4 = e4.to(torch.float8_e4m3fn).to(dtype=torch.float32)
            scale = (e4 * tensor_scale).clamp_min(1e-12)
        except (TypeError, RuntimeError, NotImplementedError):
            pass
    y = chunks / scale.unsqueeze(-1)
    thresh = torch.tensor(_E2M1_THRESH, device=y.device, dtype=y.dtype)
    ax = y.abs()
    idx = torch.bucketize(ax.reshape(-1), thresh).view_as(ax)
    levels = _e2m1_levels(y.device, y.dtype)
    q = y.sign().where(y != 0, torch.ones_like(y)) * levels[idx]
    q = torch.where(ax < 1e-12, torch.zeros_like(q), q)
    recon = q * scale.unsqueeze(-1)
    recon = recon.reshape(-1)[:n].view(orig_shape)
    return recon.to(dtype=orig_dtype)


def nvfp4_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    x_q = quantize_nvfp4(x) + x - x.detach()
    w_q = quantize_nvfp4(weight) + weight - weight.detach()
    return F.linear(x_q, w_q, bias)


class Nvfp4Linear(nn.Linear):
    """``nn.Linear`` whose FPROP is NVFP4. Shares the original Parameter."""

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Nvfp4Linear":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.in_features = lin.in_features
        obj.out_features = lin.out_features
        obj.weight = lin.weight
        obj.bias = lin.bias
        return obj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize the bf16 master, not an autocast-promoted copy.
        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return nvfp4_linear(x, self.weight, self.bias)
        return nvfp4_linear(x, self.weight, self.bias)


def should_wrap_linear(name: str, lin: nn.Linear, phase: str) -> bool:
    """Router stays high-prec. B0 wraps frozen encoder GEMMs only.

    B1/B2 wrap every allowed GEMM leaf (attn QKV/O, MoE experts, cross Q/O,
    cache KV, lm_head, encoder linears). Embed / RMSNorm / QK-Norm are not
    ``nn.Linear`` and are never swapped.
    """
    if isinstance(lin, Nvfp4Linear):
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "router":
        return False
    pol = POLICY.get(phase)
    if pol is None:
        return False
    if phase in {"L0", "C"}:
        return False
    if phase == "B0":
        # Published: frozen encoder forward NVFP4. Student cache/cross, frozen
        # decoder self-attn/MoE, and frozen lm_head stay bf16 until B1.
        if not name.startswith("encoder."):
            return False
        return not any(p.requires_grad for p in lin.parameters())
    # B1/B2 student nvfp4. Allowlist so a new high-prec Linear is not wrapped.
    # Encoder GEMMs stay in the set (fresh B1 graph, and B2 unfrozen encoder).
    return leaf in NVFP4_LINEAR_LEAVES


def _parent_and_leaf(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    if "." not in name:
        return model, name
    parent_name, leaf = name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf


def wrap_nvfp4_linears(model: nn.Module, phase: str) -> int:
    """Replace allowed ``nn.Linear`` with ``Nvfp4Linear``. Idempotent."""
    swapped = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or isinstance(mod, Nvfp4Linear):
            continue
        if not name or not should_wrap_linear(name, mod, phase):
            continue
        parent, leaf = _parent_and_leaf(model, name)
        setattr(parent, leaf, Nvfp4Linear.from_linear(mod))
        swapped += 1
    return swapped


def nvfp4_module_names(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, Nvfp4Linear)]


def apply_nvfp4(model: nn.Module, phase: str, *, enabled: bool) -> int:
    if not enabled:
        return 0
    if phase not in POLICY:
        return 0
    return wrap_nvfp4_linears(model, phase)
