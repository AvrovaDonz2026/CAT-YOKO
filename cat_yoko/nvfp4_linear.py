"""NVFP4 linear GEMM: E2M1 data + per-16-block scales.

Master weights stay the original ``nn.Linear`` Parameter (bf16). Allowed
slots (attn QKV/O, MoE experts, cross Q/O, cache KV, lm_head, frozen
encoder linears) run this forward. Router / embed / RMSNorm stay out.

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
    levels = _e2m1_levels(y.device, y.dtype)
    idx = (y.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q = y.sign().where(y != 0, torch.ones_like(y)) * levels[idx]
    q = torch.where(y.abs() < 1e-12, torch.zeros_like(q), q)
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
    """Router stays high-prec. B0 student (trainable) stays bf16."""
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
        # Frozen encoder / frozen decoder GEMMs only. cache/cross stay bf16.
        return not any(p.requires_grad for p in lin.parameters())
    return True


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
