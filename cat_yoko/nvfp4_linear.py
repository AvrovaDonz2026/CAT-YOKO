"""NVFP4 linear GEMM: E2M1 data + per-16-block scales.

Master weights stay the original ``nn.Linear`` Parameter (bf16). Allowed
slots (attn QKV/O, MoE experts, cross Q/O, cache KV, lm_head, encoder
linears) run this forward. Router / embed / RMSNorm stay out.
B0 wraps frozen encoder GEMMs only; B1/B2 wrap **all** allowed GEMMs
(including the unfrozen encoder in B2). Decoder self-attn stays
``WindowAttention`` either way.

On B200 / SM 10.0 / 10.3, wrap prefers ``TeNvfp4Linear`` (copy-once
``te.Linear`` under default ``NVFP4BlockScaling()``: 2D + RHT + SR).
sm_120 stays on this module's E2M1/16 emulation. Backward on the
emulation path uses STE so Adam still sees bf16 master grads.

Attention math is unchanged: wrap Q/K/V/O (and cross Q/O, cache KV in
B1/B2) only. Causal YOCO window / GQA / qk_norm / fp32 SDPA stay in
``attention.py``.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.nvfp4 import POLICY, policy_key

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
    from cat_yoko.nvfp4_hw import te_nvfp4_recipe as _recipe

    return _recipe()


def te_available() -> bool:
    from cat_yoko.nvfp4_hw import te_pytorch_available

    return te_pytorch_available()


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
    """True after a successful TE NVFP4 Linear wrap or sm_120 share probe."""
    from cat_yoko.nvfp4_hw import te_nvfp4_wrap_enabled

    if te_nvfp4_wrap_enabled():
        return True
    return getattr(_try_te_nvfp4_linear, "_state", None) is True


def fused_cat_linear(linears: list[nn.Linear], x: torch.Tensor) -> torch.Tensor:
    """One GEMM with weights concatenated on the out axis. Same math as sequential Linears.

    Used for SwiGLU gate+up and WindowAttention QKV. No bias (recipe Linears are bias-free).
    Frozen ``TeNvfp4Linear`` on SM100: one copy-once fused ``te.Linear``.
    Trainable TE keeps sequential native GEMM so Adam still sees each leaf.
    Concatenating TE masters into an emulated GEMM would silently drop NVFP4.
    """
    if len(linears) == 1:
        return linears[0](x)
    if any(isinstance(lin, TeNvfp4Linear) for lin in linears):
        fused = _fused_frozen_te_cat(linears, x)
        if fused is not None:
            return fused
        parts = [lin(x) for lin in linears]
        return torch.cat(parts, dim=-1)
    if all(isinstance(lin, Nvfp4Linear) for lin in linears):
        ctx = torch.autocast(device_type="cuda", enabled=False) if x.is_cuda else nullcontext()
        with ctx:
            x_q = quantize_nvfp4(x) + x - x.detach()
            w = torch.cat([lin.quantized_weight() for lin in linears], dim=0)
            return F.linear(x_q, w, None)
    if all(isinstance(lin, nn.Linear) for lin in linears):
        return _fused_native_cat(linears, x)
    parts = [lin(x) for lin in linears]
    return torch.cat(parts, dim=-1)


def _fused_native_cat(linears: list[nn.Linear], x: torch.Tensor) -> torch.Tensor:
    """One GEMM. Frozen weights are concatenated once; trainable still cat each call."""
    frozen = all(not bool(lin.weight.requires_grad) for lin in linears)
    if frozen:
        owner = linears[0]
        sig = tuple((id(lin.weight), int(lin.weight._version)) for lin in linears)
        hit = getattr(owner, "_bf16_fused_cat", None)
        if isinstance(hit, tuple) and len(hit) == 2 and hit[0] == sig:
            w = hit[1]
        else:
            w = torch.cat([lin.weight for lin in linears], dim=0).contiguous()
            try:
                owner._bf16_fused_cat = (sig, w)
            except Exception:
                pass
        return F.linear(x, w, None)
    w = torch.cat([lin.weight for lin in linears], dim=0)
    return F.linear(x, w, None)


def _fused_frozen_te_cat(linears: list[nn.Linear], x: torch.Tensor) -> torch.Tensor | None:
    """One ``te.Linear`` for frozen QKV / gate+up. None → sequential TE."""
    from cat_yoko.nvfp4_hw import (
        nvfp4_leading_ok,
        pad_leading_to_block,
        shape_fail_note,
        shape_failed,
    )

    if not linears or not all(isinstance(lin, TeNvfp4Linear) for lin in linears):
        return None
    if any(lin.bias is not None for lin in linears):
        return None
    if any(bool(lin.weight.requires_grad) for lin in linears):
        return None
    in_f = int(linears[0].in_features)
    if any(int(lin.in_features) != in_f for lin in linears):
        return None
    packs = [getattr(lin, "_te_pack", None) for lin in linears]
    if any(p is None for p in packs):
        return None
    out_f = sum(int(lin.out_features) for lin in linears)
    k = int(x.shape[-1])
    if k % 16 != 0 or out_f % 16 != 0:
        return None
    lead = x.shape[:-1]
    owner = linears[0]
    key = tuple(id(lin) for lin in linears)
    cache = getattr(owner, "_te_fused_cat", None)
    layer = recipe = te = None
    if isinstance(cache, list) and cache and cache[0] == key:
        _, layer, recipe, te = cache
    else:
        te = packs[0][2]
        recipe = packs[0][1]
        try:
            fused = te.Linear(in_f, out_f, bias=False, params_dtype=linears[0].weight.dtype)
        except TypeError:
            fused = te.Linear(in_f, out_f, bias=False)
        try:
            fused = fused.to(device=x.device, dtype=linears[0].weight.dtype)
        except Exception:
            fused = fused.to(device=x.device)
        with torch.no_grad():
            fused.weight.copy_(torch.cat([lin.weight.detach() for lin in linears], dim=0))
        layer = fused
        owner._te_fused_cat = [key, layer, recipe, te]
    x2, n = pad_leading_to_block(x)
    n_pad = int(x2.size(0))
    if shape_failed(n_pad, k, out_f):
        return None
    if not nvfp4_leading_ok(n_pad, k, out_f) and n_pad > 0:
        return None
    fprop_only = not bool(x.requires_grad)
    inner = torch.no_grad() if fprop_only else nullcontext()
    try:
        with inner:
            with te.autocast(enabled=True, recipe=recipe):
                y = layer(x2)
        if n_pad != n:
            y = y[:n]
        return y.reshape(*lead, out_f)
    except Exception:
        shape_fail_note(n_pad, k, out_f)
        return None


def _probe_te_nvfp4_once() -> bool:
    """sm_120-only share probe. SM100 uses ``TeNvfp4Linear`` copy-once wrap.

    Throwaway 16×128 Linear. Never touches 12B masters. Share must keep the
    dummy Parameter identity. ``copy_`` into TE is not a fallback on sm_120
    (float4 ``copy_`` is NotImplemented).
    """
    from cat_yoko.nvfp4_hw import compute_family, te_nvfp4_recipe as family_recipe

    state = getattr(_try_te_nvfp4_linear, "_state", None)
    if state is not None:
        return bool(state)
    fam = compute_family()
    if fam in {"sm100", "sm103"}:
        # Datacenter Blackwell goes through TeNvfp4Linear, not Parameter share.
        _try_te_nvfp4_linear._state = False
        return False
    if not torch.cuda.is_available():
        _try_te_nvfp4_linear._state = False
        return False
    try:
        import transformer_engine.pytorch as te
    except Exception:
        _try_te_nvfp4_linear._state = False
        return False
    recipe = family_recipe(fam)
    if recipe is None:
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
    """sm_120 frozen share path. SM100 returns None (``TeNvfp4Linear`` owns GEMM).

    If TE refuses to share a Parameter or the sm_120 kernel faults, disable
    this share path for the rest of the process. Never ``copy_`` every call.
    """
    from cat_yoko.nvfp4_hw import compute_family, nvfp4_leading_ok

    if compute_family() in {"sm100", "sm103"}:
        return None
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
    if not nvfp4_leading_ok(n, k, n_out):
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


class TeNvfp4Linear(Nvfp4Linear):
    """Copy-once ``te.Linear`` under the family NVFP4 recipe.

    ``self.weight`` is the TE Parameter (state_dict / Adam unchanged). The TE
    module sits in a plain list so it is not a registered submodule. One
    illegal shape falls back to STE emulation for that call; SM100 is not
    process-wide disabled.
    """

    _te_pack: list | None = None

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "TeNvfp4Linear":
        from cat_yoko.nvfp4_hw import note_te_wrap, te_module, te_nvfp4_recipe

        te = te_module()
        recipe = te_nvfp4_recipe()
        if recipe is None:
            raise RuntimeError("NVFP4BlockScaling missing")
        device = lin.weight.device
        dtype = lin.weight.dtype
        try:
            layer = te.Linear(
                lin.in_features,
                lin.out_features,
                bias=lin.bias is not None,
                params_dtype=dtype,
            )
        except TypeError:
            layer = te.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
        try:
            layer = layer.to(device=device, dtype=dtype)
        except Exception:
            layer = layer.to(device=device)
        with torch.no_grad():
            if tuple(layer.weight.shape) != tuple(lin.weight.shape):
                raise RuntimeError(
                    f"te.Linear weight {tuple(layer.weight.shape)} "
                    f"!= nn.Linear {tuple(lin.weight.shape)}"
                )
            layer.weight.copy_(lin.weight.detach())
            layer.weight.requires_grad_(bool(lin.weight.requires_grad))
            if lin.bias is not None:
                bias = getattr(layer, "bias", None)
                if bias is None:
                    raise RuntimeError("te.Linear dropped bias")
                bias.copy_(lin.bias.detach())
                bias.requires_grad_(bool(lin.bias.requires_grad))
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.in_features = lin.in_features
        obj.out_features = lin.out_features
        obj.weight = layer.weight
        obj.bias = lin.bias if lin.bias is None else layer.bias
        obj._wq_cache = None
        obj._wq_ver = None
        obj._te_pack = [layer, recipe, te]
        note_te_wrap()
        return obj

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, te_nvfp4=True"

    def quantized_weight(self) -> torch.Tensor:
        """Master TE Parameter. Do not stack this into bf16 ``grouped_mm``."""
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from cat_yoko.nvfp4_hw import (
            nvfp4_leading_ok,
            pad_leading_to_block,
            shape_fail_note,
            shape_failed,
        )

        pack = self._te_pack
        if pack is None:
            return nvfp4_linear(x, self.weight, self.bias)
        k = int(x.shape[-1])
        n_out = int(self.weight.shape[0])
        if k % 16 != 0 or n_out % 16 != 0:
            return nvfp4_linear(x, self.weight, self.bias)
        lead = x.shape[:-1]
        x2, n = pad_leading_to_block(x)
        n_pad = int(x2.size(0))
        if n == 0:
            return F.linear(x, self.weight, self.bias)
        if shape_failed(n_pad, k, n_out) or not nvfp4_leading_ok(n_pad, k, n_out):
            return nvfp4_linear(x, self.weight, self.bias)
        layer, recipe, te = pack
        # Frozen + no dX: keep grad disabled so TE does not look up WGRAD.
        fprop_only = not bool(self.weight.requires_grad) and not bool(x.requires_grad)
        inner = torch.no_grad() if fprop_only else nullcontext()
        try:
            with inner:
                with te.autocast(enabled=True, recipe=recipe):
                    y = layer(x2)
            if n_pad != n:
                y = y[:n]
            return y.reshape(*lead, n_out)
        except Exception:
            shape_fail_note(n_pad, k, n_out)
            return nvfp4_linear(x, self.weight, self.bias)


def should_wrap_linear(name: str, lin: nn.Linear, phase: str) -> bool:
    """Router stays high-prec. B0 wraps frozen encoder GEMMs only.

    B1/B2 wrap every allowed GEMM leaf (attn QKV/O, MoE experts, cross Q/O,
    cache KV, lm_head, encoder linears). Embed / RMSNorm / QK-Norm are not
    ``nn.Linear`` and are never swapped.
    """
    if isinstance(lin, Nvfp4Linear):
        return False
    parts = name.split(".")
    if "indexer" in parts or "cross_indexer" in parts:
        return False
    if "kda" in parts:
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "router":
        return False
    key = policy_key(phase)
    pol = POLICY.get(key)
    if pol is None:
        return False
    if key in {"L0", "C"}:
        return False
    if key in {"B0", "C-index"}:
        # Published: frozen encoder forward NVFP4. C-index student is the
        # bf16 indexer; backbone GEMMs may still be NVFP4 FPROP.
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
    """Replace allowed ``nn.Linear`` with NVFP4 Linear. Idempotent.

    SM100/103: ``TeNvfp4Linear`` (copy-once TE). Elsewhere: ``Nvfp4Linear``
    emulation. A single TE wrap failure falls back that module; it does not
    disable the rest of the graph.
    """
    from cat_yoko.nvfp4_hw import prefer_te_linear

    swapped = 0
    use_te = prefer_te_linear()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or isinstance(mod, Nvfp4Linear):
            continue
        if not name or not should_wrap_linear(name, mod, phase):
            continue
        parent, leaf = _parent_and_leaf(model, name)
        wrapped: nn.Linear
        if use_te:
            try:
                wrapped = TeNvfp4Linear.from_linear(mod)
            except Exception:
                wrapped = Nvfp4Linear.from_linear(mod)
        else:
            wrapped = Nvfp4Linear.from_linear(mod)
        setattr(parent, leaf, wrapped)
        swapped += 1
    return swapped


def nvfp4_module_names(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, Nvfp4Linear)]


def apply_nvfp4(model: nn.Module, phase: str, *, enabled: bool) -> int:
    if not enabled:
        return 0
    if policy_key(phase) not in POLICY:
        return 0
    return wrap_nvfp4_linears(model, phase)
