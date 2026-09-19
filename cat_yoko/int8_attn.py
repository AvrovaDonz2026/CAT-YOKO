"""SageBwd-style INT8 QK attention for dense causal SDPA.

Forward ``QK^T`` is INT8 (Triton flash kernel from luyanaa/flash-attn-triton,
BSD-3-Clause, Copyright 2025 Alyssa Vance, or ``torch._int_mm`` on Ampere).
Softmax stays fp32; PV and the entire backward stay FP16/BF16 (SageBwd).
Not a CSA CUDA kernel. Explicit window / document / ALiBi masks stay on SDPA.

The upstream launch is Turing sm_75. Ampere (sm_80 / sm_86) has INT8 tensor
cores; we try the same kernel, numerically probe it, and fall back.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

_BACKEND = "none"
_TRITON_STATE: str | None = None  # None=unprobed, "ok", "no"


def last_backend() -> str:
    return _BACKEND


def int8_shape_ok(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    causal: bool = True,
) -> bool:
    """Kernel constraints: equal lengths, hd=2^n>=32, seq%64==0, no extra mask."""
    if bias is not None or not causal:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    hd = int(q.shape[-1])
    if hd < 32 or hd & (hd - 1):
        return False
    sq, sk, sv = int(q.shape[-2]), int(k.shape[-2]), int(v.shape[-2])
    if sq != sk or sk != sv or sq % 64:
        return False
    if k.shape[-3] != q.shape[-3] and q.shape[-3] % k.shape[-3] != 0:
        return False
    return True


def _capability(device: torch.device | None = None) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(device))  # type: ignore[return-value]
    except Exception:
        return None


def int8_tc_ok(cap: tuple[int, int] | None) -> bool:
    """INT8 tensor cores: Turing sm_75 and later. Volta sm_70 has none."""
    if cap is None:
        return False
    major, minor = cap
    if major > 7:
        return True
    return major == 7 and minor >= 5


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, hd).reshape(b, n_kv * n_rep, s, hd)


def _int8_quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (last-dim) INT8 with scale, matching the Triton prepass."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1)
    scale = torch.clamp(amax, min=1.0) / 127.0
    hat = torch.round(xf / scale.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return hat, scale


def _int8_qk_scores(q: torch.Tensor, k: torch.Tensor, sm_scale: float) -> torch.Tensor:
    """INT8 ``QK^T`` → fp32 scores. Uses ``torch._int_mm`` on CUDA when it works."""
    global _BACKEND
    q8, qs = _int8_quantize_rows(q)
    k8, ks = _int8_quantize_rows(k)
    b, h, s, d = q8.shape
    scores: torch.Tensor | None = None
    if q8.is_cuda and hasattr(torch, "_int_mm"):
        try:
            q2 = q8.reshape(b * h, s, d).contiguous()
            k2 = k8.reshape(b * h, s, d).contiguous()
            out = torch.empty(b * h, s, s, device=q8.device, dtype=torch.int32)
            for i in range(b * h):
                out[i] = torch._int_mm(q2[i], k2[i].transpose(0, 1).contiguous())
            scores = out.view(b, h, s, s)
            _BACKEND = "int_mm"
        except Exception:
            scores = None
    if scores is None:
        scores = torch.matmul(q8.to(torch.int32), k8.to(torch.int32).transpose(-1, -2))
        _BACKEND = "int32_mm"
    return scores.float() * (qs.unsqueeze(-1) * ks.unsqueeze(-2)) * float(sm_scale)


def int8_qk_pv_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """INT8 QK + fp32 softmax + high-prec PV. CPU-safe (no Triton)."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    scores = _int8_qk_scores(q, k, float(sm_scale))
    if causal:
        s = scores.size(-1)
        mask = torch.ones(s, s, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    prob = torch.softmax(scores, dim=-1)
    return torch.matmul(prob.to(v.dtype), v)


def _sdpa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    *,
    causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    qn = q.detach().requires_grad_(True)
    kn = k.detach().requires_grad_(True)
    vn = v.detach().requires_grad_(True)
    extra: dict[str, Any] = {}
    with torch.enable_grad():
        try:
            extra["scale"] = sm_scale
            out = F.scaled_dot_product_attention(qn, kn, vn, is_causal=causal, **extra)
        except TypeError:
            out = F.scaled_dot_product_attention(qn, kn, vn, is_causal=causal)
        out.backward(do)
    return qn.grad, kn.grad, vn.grad


class _Int8AttnFn(torch.autograd.Function):
    """INT8 forward, FP16/BF16 SDPA backward (SageBwd)."""

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, backend):
        ctx.causal = bool(causal)
        ctx.sm_scale = float(sm_scale)
        ctx.backend = str(backend)
        ctx.save_for_backward(q, k, v)
        if backend == "triton":
            from cat_yoko.kernels import attention_kernel_int8 as kern

            q16 = q.contiguous() if q.dtype == torch.float16 else q.contiguous().to(torch.float16)
            k16 = k.contiguous() if k.dtype == torch.float16 else k.contiguous().to(torch.float16)
            v16 = v.contiguous() if v.dtype == torch.float16 else v.contiguous().to(torch.float16)
            out = kern.attention_int8(q16, k16, v16, causal=bool(causal), sm_scale=float(sm_scale))
            return out.to(dtype=q.dtype)
        return int8_qk_pv_forward(q, k, v, causal=bool(causal), sm_scale=float(sm_scale))

    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = _sdpa_backward(
            q, k, v, do, causal=ctx.causal, sm_scale=ctx.sm_scale
        )
        return dq, dk, dv, None, None, None


def _probe_triton(device: torch.device) -> bool:
    """Compile + cosine-check vs SDPA. INT8 packing on Ampere can be garbage."""
    global _TRITON_STATE
    if _TRITON_STATE is not None:
        return _TRITON_STATE == "ok"
    try:
        from cat_yoko.kernels import attention_kernel_int8 as kern  # noqa: F401
    except Exception:
        _TRITON_STATE = "no"
        return False
    cap = _capability(device)
    if not int8_tc_ok(cap):
        _TRITON_STATE = "no"
        return False
    try:
        from cat_yoko.kernels import attention_kernel_int8 as kern

        torch.manual_seed(0)
        q = torch.randn(1, 2, 64, 32, device=device, dtype=torch.float16)
        k = torch.randn(1, 2, 64, 32, device=device, dtype=torch.float16)
        v = torch.randn(1, 2, 64, 32, device=device, dtype=torch.float16)
        sm = 32 ** -0.5
        with torch.no_grad():
            hyp = kern.attention_int8(q, k, v, causal=True, sm_scale=sm)
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = hyp.float().reshape(1, -1)
        b = ref.float().reshape(1, -1)
        cos = float(F.cosine_similarity(a, b).item())
        ok = math.isfinite(cos) and cos >= 0.97
        _TRITON_STATE = "ok" if ok else "no"
        return ok
    except Exception:
        _TRITON_STATE = "no"
        return False


def try_int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor | None:
    """Dense causal INT8 attn, or ``None`` to keep Flash / math SDPA."""
    global _BACKEND, _TRITON_STATE
    _BACKEND = "none"
    if not causal or not q.is_cuda:
        return None
    if q.dtype not in (torch.float16, torch.bfloat16):
        return None
    if not int8_shape_ok(q, k, v, None, causal=causal):
        return None
    if k.size(-3) != q.size(-3):
        n_rep = q.size(-3) // k.size(-3)
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    sm = q.shape[-1] ** -0.5
    if _probe_triton(q.device):
        try:
            _BACKEND = "triton"
            return _Int8AttnFn.apply(q, k, v, True, sm, "triton")
        except Exception:
            _TRITON_STATE = "no"
    try:
        _BACKEND = "int_mm"
        return _Int8AttnFn.apply(q, k, v, True, sm, "int_mm")
    except Exception:
        _BACKEND = "none"
        return None


__all__ = [
    "int8_qk_pv_forward",
    "int8_shape_ok",
    "int8_tc_ok",
    "last_backend",
    "try_int8_attention",
]
