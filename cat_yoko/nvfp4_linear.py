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

from contextlib import nullcontext

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

    torch 2.8.0+cu128 and 2.15.0.dev20260918+cu130 on sm_120 both expose
    ``torch.float4_e2m1fn_x2`` but ``copy_`` still raises. Nightly still
    unlocks ``transformer_engine.pytorch`` (missing on 2.8). Probe with
    ``scripts/probe_nvfp4_hw.py`` / ``venv-nightly``.
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


_E2M1_CACHE: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _e2m1_tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached E2M1 midpoints and reconstruction levels (one tensor pair per device/dtype)."""
    key = (str(device), dtype)
    hit = _E2M1_CACHE.get(key)
    if hit is not None and hit[0].device == device and hit[0].dtype == dtype:
        return hit
    thresh = torch.tensor(_E2M1_THRESH, device=device, dtype=dtype)
    levels = torch.tensor(_E2M1_ABS, device=device, dtype=dtype)
    _E2M1_CACHE[key] = (thresh, levels)
    return thresh, levels


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
    thresh, levels = _e2m1_tables(y.device, y.dtype)
    ax = y.abs()
    idx = torch.bucketize(ax.reshape(-1), thresh).view_as(ax)
    q = y.sign().where(y != 0, torch.ones_like(y)) * levels[idx]
    q = torch.where(ax < 1e-12, torch.zeros_like(q), q)
    recon = q * scale.unsqueeze(-1)
    recon = recon.reshape(-1)[:n].view(orig_shape)
    return recon.to(dtype=orig_dtype)


def te_nvfp4_linear_enabled() -> bool:
    """True after a successful TE NVFP4 Linear probe/GEMM this process."""
    return getattr(_try_te_nvfp4_linear, "_state", None) is True


def fused_cat_linear(linears: list[nn.Linear], x: torch.Tensor) -> torch.Tensor:
    """One GEMM with weights concatenated on the out axis. Same math as sequential Linears.

    Used for SwiGLU gate+up and WindowAttention QKV. No bias (recipe Linears are bias-free).
    """
    if len(linears) == 1:
        return linears[0](x)
    if all(isinstance(lin, Nvfp4Linear) for lin in linears):
        ctx = torch.autocast(device_type="cuda", enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x_q = quantize_nvfp4(x) + x - x.detach()
            w = torch.cat([lin.quantized_weight() for lin in linears], dim=0)
            return F.linear(x_q, w, None)
    if all(isinstance(lin, nn.Linear) for lin in linears):
        w = torch.cat([lin.weight for lin in linears], dim=0)
        return F.linear(x, w, None)
    parts = [lin(x) for lin in linears]
    return torch.cat(parts, dim=-1)


def _probe_te_nvfp4_once() -> bool:
    """Throwaway 16×128 Linear. Never touches 12B masters.

    Share must keep the dummy Parameter identity and dtype. ``copy_`` into TE
    is not a fallback (bandwidth + sm_120 float4 ``copy_`` is NotImplemented).
    """
    state = getattr(_try_te_nvfp4_linear, "_state", None)
    if state is not None:
        return bool(state)
    if not torch.cuda.is_available():
        _try_te_nvfp4_linear._state = False
        return False
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except Exception:
        _try_te_nvfp4_linear._state = False
        return False
    try:
        recipe = NVFP4BlockScaling(disable_rht=True, disable_2d_quantization=True)
    except TypeError:
        try:
            recipe = NVFP4BlockScaling()
        except Exception:
            _try_te_nvfp4_linear._state = False
            return False
    try:
        device = torch.device("cuda")
        w = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
        x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        ptr = w.data_ptr()
        layer = te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16)
        layer = layer.to(device=device, dtype=torch.bfloat16)
        try:
            layer.weight = w
        except Exception:
            _try_te_nvfp4_linear._state = False
            return False
        if getattr(layer, "weight", None) is not w:
            _try_te_nvfp4_linear._state = False
            return False
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(x)
        if (
            getattr(layer, "weight", None) is not w
            or w.dtype != torch.bfloat16
            or w.data_ptr() != ptr
            or tuple(y.shape) != (16, 128)
            or not bool(torch.isfinite(y.float()).all())
        ):
            _try_te_nvfp4_linear._state = False
            return False
        _try_te_nvfp4_linear._state = True
        _try_te_nvfp4_linear._recipe = recipe
        _try_te_nvfp4_linear._te = te
        return True
    except Exception:
        _try_te_nvfp4_linear._state = False
        return False


def _try_te_nvfp4_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
    """Best-effort Transformer Engine NVFP4 GEMM sharing ``weight``. None = use emulation.

    Does not clone the 12B master. Probe uses a dummy 16×128 Linear first.
    If TE refuses to share a Parameter or the sm_120 kernel faults, disable
    for the rest of the process. Never ``copy_`` into TE.
    """
    if getattr(_try_te_nvfp4_linear, "_state", None) is False:
        return None
    if not x.is_cuda or not weight.is_cuda:
        return None
    if weight.requires_grad:
        # Trainable STE must hit the bf16 master; TE share is frozen-only.
        return None
    n = x.reshape(-1, x.shape[-1]).size(0)
    k = int(x.shape[-1])
    n_out = int(weight.shape[0])
    if n % 16 != 0 or k % 16 != 0 or n_out % 16 != 0:
        return None
    if not _probe_te_nvfp4_once():
        return None
    te = getattr(_try_te_nvfp4_linear, "_te", None)
    recipe = getattr(_try_te_nvfp4_linear, "_recipe", None)
    if te is None or recipe is None:
        _try_te_nvfp4_linear._state = False
        return None
    cache: dict = getattr(_try_te_nvfp4_linear, "_layers", None)
    if cache is None:
        cache = {}
        _try_te_nvfp4_linear._layers = cache
    key = (int(weight.shape[1]), int(weight.shape[0]), str(weight.device), str(weight.dtype), bias is not None)
    layer = cache.get(key)
    try:
        if layer is None:
            layer = te.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias is not None,
                params_dtype=weight.dtype,
            )
            layer = layer.to(device=weight.device, dtype=weight.dtype)
            cache[key] = layer
        if getattr(layer, "weight", None) is not weight:
            try:
                layer.weight = weight
            except Exception:
                _try_te_nvfp4_linear._state = False
                cache.pop(key, None)
                return None
        if getattr(layer, "weight", None) is not weight:
            _try_te_nvfp4_linear._state = False
            cache.pop(key, None)
            return None
        if bias is not None:
            try:
                layer.bias = bias
            except Exception:
                _try_te_nvfp4_linear._state = False
                cache.pop(key, None)
                return None
        ptr = weight.data_ptr()
        dt = weight.dtype
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(x)
        if getattr(layer, "weight", None) is not weight or weight.dtype != dt or weight.data_ptr() != ptr:
            _try_te_nvfp4_linear._state = False
            cache.pop(key, None)
            return None
        return y
    except Exception:
        _try_te_nvfp4_linear._state = False
        cache.pop(key, None)
        return None


def nvfp4_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    if not weight.requires_grad:
        hw = _try_te_nvfp4_linear(x, weight, bias)
        if hw is not None:
            return hw
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
        obj._wq_cache = None
        obj._wq_ver = None
        return obj

    def quantized_weight(self) -> torch.Tensor:
        """STE-quantized weight, or a cached dequant for frozen masters.

        B0 encoder GEMMs never receive Adam updates. Re-quantizing 1072
        frozen matrices every forward is wasted work; cache the dequant.
        Trainable B1/B2 slots still STE every call.
        """
        w = self.weight
        if not w.requires_grad:
            cache = getattr(self, "_wq_cache", None)
            ver = getattr(self, "_wq_ver", None)
            if (
                cache is not None
                and ver == w._version
                and cache.device == w.device
                and cache.dtype == w.dtype
                and cache.shape == w.shape
            ):
                return cache
            with torch.no_grad():
                cache = quantize_nvfp4(w)
            self._wq_cache = cache
            self._wq_ver = w._version
            return cache
        return quantize_nvfp4(w) + w - w.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize the bf16 master, not an autocast-promoted copy.
        # Frozen encoder: TE NVFP4 GEMM when the kernel accepts this shape.
        if not self.weight.requires_grad:
            hw = _try_te_nvfp4_linear(x, self.weight, self.bias)
            if hw is not None:
                return hw

        def _go() -> torch.Tensor:
            x_q = quantize_nvfp4(x) + x - x.detach()
            return F.linear(x_q, self.quantized_weight(), self.bias)

        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return _go()
        return _go()


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
