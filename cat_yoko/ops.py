"""Ampere BF16 operator snapshot for the dedicated mini-verify.

Dense causal (YOCO cross, or window when ``n_win`` covers seq): Flash → cuDNN
→ mem-efficient. Static sliding window with ``n_win < seq`` and
``seq >= 512`` uses fat tiles (width ``max(n_win, 256)``), not crumbs of
``n_win`` itself. Masked CSA/HCA ``attn_mask``: isolate
Efficient (seq<320) or cuDNN (longer) in bf16, then fp32 math. Flash rejects
``attn_mask``. Not a CSA CUDA kernel. Softmax accumulation stays fp32 inside
the fused kernel. Ampere MoE is padded bmm; ``grouped_mm`` is SM90+.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cat_yoko.attention import last_sdpa, sdpa_counts
from cat_yoko.config import CATYokoConfig
from cat_yoko.moe import grouped_mm_available
from cat_yoko.nvfp4_linear import fused_cat_linear
from cat_yoko.trainer import configure_cuda

# What each A→E phase should light on Ampere BF16. Probe seq keeps n_win < seq,
# so encoder window / CSA / HCA stay on the masked path; YOCO cross is dense.
PHASE_OPS: dict[str, tuple[str, ...]] = {
    "A": ("dense_cross", "masked_window", "fused_qkv", "tf32"),
    "B0": ("dense_cross", "masked_window", "fused_qkv", "tf32"),
    "B1": ("dense_cross", "masked_window", "fused_qkv", "tf32"),
    "B2": ("dense_cross", "masked_window", "fused_qkv", "tf32", "moe_bmm"),
    "C-index": ("indexer_fp32", "masked_window", "fused_qkv"),
    "C-topk": ("masked_csa_union", "fused_qkv", "tf32"),
    "C-hca": ("masked_hca_concat", "masked_csa_union", "fused_qkv", "tf32"),
    "C-win": ("masked_hca_concat", "masked_csa_union", "fused_qkv", "tf32"),
    "D-8k": ("masked_hca_concat", "dummy_needle", "fused_qkv", "tf32"),
    "E": ("masked_hca_concat", "wsd", "fused_qkv", "tf32"),
}


def _try_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backends: list, **extra) -> bool:
    try:
        from torch.nn.attention import sdpa_kernel
    except ImportError:
        return False
    try:
        with sdpa_kernel(backends):
            y = F.scaled_dot_product_attention(q, k, v, **extra)
        return bool(torch.isfinite(y).all().item())
    except Exception:
        return False


def _backend(name: str):
    try:
        from torch.nn.attention import SDPBackend
    except ImportError:
        return None
    return getattr(SDPBackend, name, None)


def probe_sdpa_backends(
    device: str,
    dtype: torch.dtype,
    *,
    seq: int = 128,
    n_heads: int = 4,
    n_kv: int = 2,
    head_dim: int = 32,
) -> dict[str, Any]:
    """Isolate Flash / cuDNN / Efficient on dense GQA vs masked equal-head."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "dense_gqa": "cpu",
            "dense_equal": "cpu",
            "masked_equal": "cpu",
            "hca_concat": "cpu",
            "head_dim": int(head_dim),
            "seq": int(seq),
        }
    configure_cuda()
    from cat_yoko.attention import _masked_backend_order

    dense_names = ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION")
    masked_names = _masked_backend_order(seq)
    q_g = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    k_g = torch.randn(1, n_kv, seq, head_dim, device=device, dtype=dtype)
    v_g = torch.randn(1, n_kv, seq, head_dim, device=device, dtype=dtype)
    q_e = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    k_e = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    v_e = torch.randn(1, n_heads, seq, head_dim, device=device, dtype=dtype)
    extra_gqa: dict[str, Any] = {}
    try:
        F.scaled_dot_product_attention(q_g, k_g, v_g, is_causal=True, enable_gqa=True)
        extra_gqa["enable_gqa"] = True
    except TypeError:
        pass

    def _first(q, k, v, names, **extra) -> str:
        for name in names:
            backend = _backend(name)
            if backend is None:
                continue
            if _try_sdpa(q, k, v, [backend], **extra):
                return name.lower().replace("_attention", "")
        return "math_fp32"

    bias = torch.zeros(seq, seq, device=device, dtype=dtype)
    future = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
    bias = bias.masked_fill(future, torch.finfo(dtype).min)
    n_slots = max(seq // 8, 1)
    k_h = torch.randn(1, n_heads, seq + n_slots, head_dim, device=device, dtype=dtype)
    v_h = torch.randn(1, n_heads, seq + n_slots, head_dim, device=device, dtype=dtype)
    bias_h = torch.zeros(seq, seq + n_slots, device=device, dtype=dtype)
    return {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "dense_gqa": _first(q_g, k_g, v_g, dense_names, is_causal=True, **extra_gqa),
        "dense_equal": _first(q_e, k_e, v_e, dense_names, is_causal=True),
        "masked_equal": _first(q_e, k_e, v_e, masked_names, attn_mask=bias),
        "hca_concat": _first(q_e, k_h, v_h, masked_names, attn_mask=bias_h),
        "head_dim": int(head_dim),
        "seq": int(seq),
        "flash_rejects_mask": True,
    }


def snapshot_ops(cfg: CATYokoConfig, device: str, *, phase: str | None = None) -> dict[str, Any]:
    """TF32 / fused QKV / grouped_mm / last SDPA. Cheap; no extra train step."""
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    tf32 = bool(cuda and torch.backends.cuda.matmul.allow_tf32)
    try:
        matmul_prec = torch.get_float32_matmul_precision()
    except Exception:
        matmul_prec = ""
    row: dict[str, Any] = {
        "phase": phase,
        "tf32": tf32,
        "matmul_precision": matmul_prec,
        "grouped_mm": grouped_mm_available(),
        "fused_qkv": True,
        "sdpa": last_sdpa(),
        "sdpa_counts": sdpa_counts(),
        "head_dim": int(cfg.head_dim),
        "seq_len": int(cfg.seq_len),
        "n_win": int(cfg.n_win),
        "use_fp8": bool(cfg.use_fp8),
        "use_nvfp4": bool(cfg.use_nvfp4),
        "intended": list(PHASE_OPS.get(phase or "", ())),
    }
    return row


def fused_qkv_smoke(device: str, dtype: torch.dtype, *, hidden: int = 128, kv: int = 64) -> bool:
    q = torch.nn.Linear(hidden, hidden, bias=False).to(device=device, dtype=dtype)
    k = torch.nn.Linear(hidden, kv, bias=False).to(device=device, dtype=dtype)
    v = torch.nn.Linear(hidden, kv, bias=False).to(device=device, dtype=dtype)
    x = torch.randn(2, 8, hidden, device=device, dtype=dtype)
    y = fused_cat_linear([q, k, v], x)
    return y.shape[-1] == hidden + kv + kv and bool(torch.isfinite(y).all().item())
