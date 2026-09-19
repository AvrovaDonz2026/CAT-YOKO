"""B200 / SM100 NVFP4 hardware backend (recipe, probe, GroupedLinear).

Published TE training kernels: SM 10.0 / 10.3. Default recipe is
``NVFP4BlockScaling()`` (2D weights, RHT on WGRAD, SR on gradients).
sm_120 keeps the reduced recipe and ``Nvfp4Linear`` emulation.

``TeNvfp4Linear`` lives in ``nvfp4_linear.py`` so wrap tests keep
``isinstance(..., Nvfp4Linear)``.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ComputeFamily = str  # "sm100" | "sm103" | "sm120" | "other" | "cpu"

_TE_MODULE: Any = None
_FORCE_TE: bool | None = None
_PROBE_LINEAR: bool | None = None
_PROBE_GROUPED: bool | None = None
_WRAP_COUNT = 0
_SHAPE_FAILS: set[tuple[int, int, int]] = set()


def reset_te_probe() -> None:
    """Drop cached probes. Tests call this between cases."""
    global _PROBE_LINEAR, _PROBE_GROUPED
    _PROBE_LINEAR = None
    _PROBE_GROUPED = None
    _SHAPE_FAILS.clear()


def install_te_backend_for_tests(te_module: Any | None, *, force: bool | None = True) -> None:
    """Unit-test hook. ``te_module=None, force=False`` restores production."""
    global _TE_MODULE, _FORCE_TE, _WRAP_COUNT
    _TE_MODULE = te_module
    _FORCE_TE = force
    _WRAP_COUNT = 0
    reset_te_probe()


def note_te_wrap() -> None:
    global _WRAP_COUNT
    _WRAP_COUNT += 1


def te_nvfp4_wrap_count() -> int:
    return int(_WRAP_COUNT)


def shape_fail_note(n: int, k: int, n_out: int) -> None:
    """Record a per-shape TE miss. Does **not** disable SM100 process-wide."""
    _SHAPE_FAILS.add((n, k, n_out))


def shape_failed(n: int, k: int, n_out: int) -> bool:
    return (n, k, n_out) in _SHAPE_FAILS


def compute_capability(device: int | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    idx = 0 if device is None else int(device)
    return tuple(torch.cuda.get_device_capability(idx))


def compute_family(cap: tuple[int, int] | None = None) -> ComputeFamily:
    """SM family for the NVFP4 recipe. ``10.0`` / ``10.3`` train; ``12.0`` does not."""
    if cap is None:
        cap = compute_capability()
    if cap is None:
        return "cpu"
    major, minor = int(cap[0]), int(cap[1])
    if major == 10 and minor == 3:
        return "sm103"
    if major == 10:
        return "sm100"
    if major == 12:
        return "sm120"
    return "other"


def nvfp4_training_capable(family: str | None = None) -> bool:
    """True on SM 10.0 / 10.3, the TE NVFP4 **training** set."""
    fam = compute_family() if family is None else family
    return fam in {"sm100", "sm103"}


def te_nvfp4_recipe_kwargs(family: str | None = None) -> dict[str, bool]:
    """Kwargs for ``NVFP4BlockScaling``. Empty on SM100 (full RHT + 2D + SR)."""
    fam = compute_family() if family is None else family
    if fam in {"sm100", "sm103"}:
        return {}
    if fam == "sm120":
        return {
            "disable_rht": True,
            "disable_stochastic_rounding": True,
            "disable_2d_quantization": True,
        }
    return {"disable_rht": True, "disable_2d_quantization": True}


def te_nvfp4_recipe(family: str | None = None):
    """``NVFP4BlockScaling`` for this GPU, or None if TE is missing."""
    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except Exception:
        if _TE_MODULE is not None:
            recipe_cls = getattr(_TE_MODULE, "NVFP4BlockScaling", None)
            if recipe_cls is None:
                return object()
            NVFP4BlockScaling = recipe_cls
        else:
            return None
    kwargs = te_nvfp4_recipe_kwargs(family)
    try:
        return NVFP4BlockScaling(**kwargs)
    except TypeError:
        relaxed = {k: v for k, v in kwargs.items() if k != "disable_stochastic_rounding"}
        try:
            return NVFP4BlockScaling(**relaxed)
        except TypeError:
            try:
                return NVFP4BlockScaling()
            except Exception:
                return None


def te_module():
    if _TE_MODULE is not None:
        return _TE_MODULE
    import transformer_engine.pytorch as te

    return te


def te_pytorch_available() -> bool:
    if _TE_MODULE is not None:
        return True
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: F401
    except Exception:
        return False
    return True


def te_grouped_linear_cls():
    return getattr(te_module(), "GroupedLinear", None)


def nvfp4_leading_ok(n: int, k: int, n_out: int) -> bool:
    """TE NVFP4 wants each GEMM dim divisible by the 16-element block."""
    return n % 16 == 0 and k % 16 == 0 and n_out % 16 == 0


def nvfp4_grouped_token_align(family: str | None = None) -> int:
    """Token-count multiple for ``te.GroupedLinear`` splits.

    NVFP4 microblock is 16. Default SM100/103 ``NVFP4BlockScaling()`` keeps
    RHT on WGRAD, and TE requires ``split_sections[i] % 64 == 0``. Pad frozen
    FPROP to 64 as well so the same GroupedLinear stays on the kernel path
    instead of falling back to serial ``te.Linear``.
    """
    fam = compute_family() if family is None else family
    if fam in {"sm100", "sm103"}:
        kwargs = te_nvfp4_recipe_kwargs(fam)
        if not kwargs.get("disable_rht", False):
            return 64
    return 16


def nvfp4_pad_tokens(n: int, block: int = 16) -> int:
    """Smallest ``>= n`` multiple of ``block``. ``0`` stays ``0``."""
    if n <= 0:
        return 0
    r = n % block
    return n if r == 0 else n + (block - r)


def pad_leading_to_block(x: torch.Tensor, block: int = 16) -> tuple[torch.Tensor, int]:
    """Flatten ``[..., K]`` to ``[N, K]``, pad ``N`` to a multiple of ``block``.

    Returns ``(padded_2d, original_n)``. Zero rows do not change real-token
    GEMM rows; caller slices ``[:original_n]`` and reshapes.
    """
    k = int(x.shape[-1])
    x2 = x.reshape(-1, k)
    n = int(x2.size(0))
    pad = nvfp4_pad_tokens(n, block) - n
    if pad:
        x2 = F.pad(x2, (0, 0, 0, pad))
    return x2, n


def pad_packed_counts(x_sorted: torch.Tensor, counts: torch.Tensor, block: int = 16) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Insert zero rows so each expert's token count is 0 or a multiple of ``block``.

    ``x_sorted`` is packed by expert id. Returns padded packed tensor, new
    counts (length ``E``), and ``[(padded_start, n_real), ...]`` for experts
    with ``n_real > 0`` so the caller can drop pad rows.
    """
    counts_list = [int(v) for v in counts.tolist()]
    if not counts_list:
        return x_sorted, counts, []
    if all(c == 0 or c % block == 0 for c in counts_list):
        keeps = []
        off = 0
        for c in counts_list:
            if c:
                keeps.append((off, c))
            off += c
        return x_sorted, counts.to(dtype=torch.int64), keeps
    parts: list[torch.Tensor] = []
    new_counts: list[int] = []
    keeps: list[tuple[int, int]] = []
    offset = 0
    padded_off = 0
    k = int(x_sorted.shape[-1])
    for n in counts_list:
        if n <= 0:
            new_counts.append(0)
            continue
        sl = x_sorted.narrow(0, offset, n)
        pad = nvfp4_pad_tokens(n, block) - n
        if pad:
            sl = F.pad(sl, (0, 0, 0, pad))
        parts.append(sl)
        new_counts.append(n + pad)
        keeps.append((padded_off, n))
        padded_off += n + pad
        offset += n
    if not parts:
        return x_sorted, x_sorted.new_zeros(len(counts_list), dtype=torch.int64), []
    return torch.cat(parts, dim=0), x_sorted.new_tensor(new_counts, dtype=torch.int64), keeps


def unpad_packed(y_padded: torch.Tensor, keeps: list[tuple[int, int]], n_tok: int) -> torch.Tensor:
    """Gather real rows after ``pad_packed_counts``."""
    if not keeps:
        return y_padded.new_zeros(n_tok, y_padded.size(-1))
    if int(y_padded.size(0)) == n_tok:
        return y_padded
    parts = [y_padded.narrow(0, start, n) for start, n in keeps]
    return torch.cat(parts, dim=0)


def _env_te_flag() -> str | None:
    raw = os.environ.get("CAT_YOKO_TE_NVFP4", "").strip().lower()
    return raw or None


def prefer_te_linear() -> bool:
    """True when wrap should use ``TeNvfp4Linear`` (copy-once TE Parameter)."""
    env = _env_te_flag()
    if env in {"0", "off", "false", "emu", "no"}:
        return False
    if _FORCE_TE is False:
        return False
    if _FORCE_TE is True:
        return True
    if env in {"1", "on", "true", "te", "yes"}:
        return bool(te_pytorch_available() and te_linear_ready())
    if not nvfp4_training_capable():
        return False
    return bool(te_pytorch_available() and te_linear_ready())


def te_linear_ready() -> bool:
    """Probe a legal 16×128 ``te.Linear`` with the **family** recipe. Cached."""
    global _PROBE_LINEAR
    if _PROBE_LINEAR is not None:
        return _PROBE_LINEAR
    if _TE_MODULE is not None:
        _PROBE_LINEAR = True
        return True
    if not torch.cuda.is_available() or not te_pytorch_available():
        _PROBE_LINEAR = False
        return False
    try:
        te = te_module()
        recipe = te_nvfp4_recipe()
        if recipe is None:
            _PROBE_LINEAR = False
            return False
        device = torch.device("cuda")
        layer = te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16)
        layer = layer.to(device=device, dtype=torch.bfloat16)
        src = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            layer.weight.copy_(src)
        x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(x)
        ok = tuple(y.shape) == (16, 128) and bool(torch.isfinite(y.float()).all())
        _PROBE_LINEAR = ok
        return ok
    except Exception:
        _PROBE_LINEAR = False
        return False


def te_grouped_ready() -> bool:
    """Best-effort GroupedLinear probe. Failure does not disable ``te.Linear``."""
    global _PROBE_GROUPED
    if _PROBE_GROUPED is not None:
        return _PROBE_GROUPED
    if _TE_MODULE is not None and getattr(_TE_MODULE, "GroupedLinear", None) is not None:
        _PROBE_GROUPED = True
        return True
    if not prefer_te_linear():
        _PROBE_GROUPED = False
        return False
    cls = None
    try:
        cls = te_grouped_linear_cls()
    except Exception:
        _PROBE_GROUPED = False
        return False
    if cls is None:
        _PROBE_GROUPED = False
        return False
    if not torch.cuda.is_available():
        _PROBE_GROUPED = True
        return True
    try:
        te = te_module()
        recipe = te_nvfp4_recipe()
        device = torch.device("cuda")
        align = nvfp4_grouped_token_align()
        n = align * 2
        layer = cls(2, 128, 128, bias=False, params_dtype=torch.bfloat16, device=device)
        x = torch.randn(n, 128, device=device, dtype=torch.bfloat16)
        splits = torch.tensor([align, align], dtype=torch.int64)
        with te.autocast(enabled=True, recipe=recipe):
            y = layer(x, splits)
        _PROBE_GROUPED = tuple(y.shape) == (n, 128) and bool(torch.isfinite(y.float()).all())
        return _PROBE_GROUPED
    except Exception:
        _PROBE_GROUPED = False
        return False


def te_nvfp4_wrap_enabled() -> bool:
    return _WRAP_COUNT > 0 or (
        _PROBE_LINEAR is True and (nvfp4_training_capable() or _FORCE_TE is True)
    )


def te_grouped_swiglu(
    experts: nn.ModuleList,
    x_sorted: torch.Tensor,
    counts: torch.Tensor,
    owner: nn.Module | None,
) -> torch.Tensor | None:
    """NVFP4 GroupedLinear SwiGLU. None = caller should serial-TE.

    ``x_sorted`` is packed by expert id (same layout as the bf16 grouped path).
    Frozen: copy into ``weight{i}`` once. Trainable: alias expert Parameters
    so Adam still sees ``experts.i.gate_proj.weight``.
    """
    if not experts or not prefer_te_linear() or not te_grouped_ready():
        return None
    cls = te_grouped_linear_cls()
    if cls is None:
        return None
    n_tok = x_sorted.size(0)
    if n_tok == 0:
        return x_sorted
    te = te_module()
    recipe = te_nvfp4_recipe()
    if recipe is None:
        return None
    n_exp = len(experts)
    in_f = int(experts[0].gate_proj.in_features)
    mid = int(experts[0].gate_proj.out_features)
    dtype = experts[0].gate_proj.weight.dtype
    device = x_sorted.device
    trainable = any(bool(e.gate_proj.weight.requires_grad) for e in experts)
    align = nvfp4_grouped_token_align()
    x_use, counts_use, keeps = pad_packed_counts(
        x_sorted, counts.to(dtype=torch.int64), block=align
    )
    pack = None if owner is None else getattr(owner, "_te_grouped", None)
    if pack is None:

        def _make(in_features: int, out_features: int):
            try:
                return cls(
                    n_exp,
                    in_features,
                    out_features,
                    bias=False,
                    params_dtype=dtype,
                    device=device,
                )
            except TypeError:
                layer = cls(n_exp, in_features, out_features, bias=False, params_dtype=dtype)
                return layer.to(device=device, dtype=dtype)

        if trainable:
            gate_g = _make(in_f, mid)
            up_g = _make(in_f, mid)
            down_g = _make(mid, in_f)
            with torch.no_grad():
                for i, exp in enumerate(experts):
                    getattr(gate_g, f"weight{i}").copy_(exp.gate_proj.weight.detach())
                    getattr(up_g, f"weight{i}").copy_(exp.up_proj.weight.detach())
                    getattr(down_g, f"weight{i}").copy_(exp.down_proj.weight.detach())
            for i, exp in enumerate(experts):
                for proj, grouped in (
                    (exp.gate_proj, gate_g),
                    (exp.up_proj, up_g),
                    (exp.down_proj, down_g),
                ):
                    w = getattr(grouped, f"weight{i}")
                    w.requires_grad_(True)
                    proj.weight = w
                    pack_te = getattr(proj, "_te_pack", None)
                    if pack_te is not None:
                        try:
                            pack_te[0].weight = w
                        except Exception:
                            pass
            pack = ("split", gate_g, up_g, down_g)
        else:
            gu_g = _make(in_f, 2 * mid)
            down_g = _make(mid, in_f)
            with torch.no_grad():
                for i, exp in enumerate(experts):
                    getattr(gu_g, f"weight{i}").copy_(
                        torch.cat(
                            [exp.gate_proj.weight.detach(), exp.up_proj.weight.detach()],
                            dim=0,
                        )
                    )
                    getattr(down_g, f"weight{i}").copy_(exp.down_proj.weight.detach())
            for i in range(n_exp):
                getattr(gu_g, f"weight{i}").requires_grad_(False)
                getattr(down_g, f"weight{i}").requires_grad_(False)
            pack = ("fused_gu", gu_g, down_g)
        if owner is not None:
            owner._te_grouped = pack
    splits = counts_use.to(dtype=torch.int64)
    fprop_only = (not trainable) and (not bool(x_sorted.requires_grad))
    inner = torch.no_grad() if fprop_only else nullcontext()
    try:
        with inner:
            with te.autocast(enabled=True, recipe=recipe):
                if pack[0] == "fused_gu":
                    _, gu_g, down_g = pack
                    gu = gu_g(x_use, splits)
                    g, u = gu.chunk(2, dim=-1)
                    hidden = F.silu(g) * u
                    y = down_g(hidden, splits)
                else:
                    _, gate_g, up_g, down_g = pack
                    g = gate_g(x_use, splits)
                    u = up_g(x_use, splits)
                    hidden = F.silu(g) * u
                    y = down_g(hidden, splits)
        return unpad_packed(y, keeps, n_tok)
    except Exception:
        return None
