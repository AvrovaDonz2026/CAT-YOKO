"""Sliding-window self-attn and gated cross-attn (Phase B backend).

MiniCPM5-2B is Llama GQA (16 Q / 2 KV). Encoder/decoder self-attn and the
YOCO cache therefore use ``kv_dim = n_kv * head_dim``, not full MHA.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.config import CATYokoConfig
from cat_yoko.rope import RMSNorm, RotaryEmbedding, apply_rope


def _fused_qkv(
    q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One GEMM for Q/K/V. Same math as three Linears."""
    from cat_yoko.nvfp4_linear import fused_cat_linear

    qkv = fused_cat_linear([q_proj, k_proj, v_proj], x)
    d_q, d_k, d_v = q_proj.out_features, k_proj.out_features, v_proj.out_features
    return qkv.split((d_q, d_k, d_v), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, n_kv, S, hd]`` → ``[B, n_heads, S, hd]``."""
    if n_rep == 1:
        return x
    b, n_kv, s, hd = x.shape
    return x[:, :, None, :, :].expand(b, n_kv, n_rep, s, hd).reshape(b, n_kv * n_rep, s, hd)


def split_heads(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """``[B, S, n_heads * head_dim]`` → ``[B, S, n_heads, head_dim]``."""
    b, s, packed = x.shape
    if packed != n_heads * head_dim:
        raise ValueError(f"packed dim {packed} != {n_heads}*{head_dim}")
    return x.view(b, s, n_heads, head_dim)


def to_sdpa_layout(x: torch.Tensor) -> torch.Tensor:
    """``[B, S, H, D]`` → ``[B, H, S, D]`` without packing.

    Memory stays BSHD (stride ``(S·H·D, D, H·D, 1)``). Ampere Flash / FA2
    read that order. ``.contiguous()`` here would pack BHSD and add a copy
    on every layer (12B qk_norm used to do that after the transpose).
    """
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``[B, H, S, D]`` → ``[B, S, H*D]`` for ``o_proj``. View if BSHD memory."""
    b, h, s, d = x.shape
    return x.transpose(1, 2).reshape(b, s, h * d)


def rope_after_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    rope: RotaryEmbedding,
    q_norm: RMSNorm | None,
    k_norm: RMSNorm | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """qk_norm + RoPE on contiguous ``[B, S, H, D]`` (last dim = head_dim)."""
    if q_norm is not None:
        q = q_norm(q)
        k = k_norm(k)
    cos, sin = rope(int(q.size(1)), q.device, q.dtype)
    return apply_rope(q, k, cos, sin, seq_dim=1)


_SDPA_KERNEL = None  # dense cache: None=uninit, False=unavailable, else factory
_SDPA_MASKED_KERNEL: dict = {}  # "small"|"large" -> factory|False
_LAST_SDPA = {"kind": "uninit", "dtype": ""}
_SDPA_COUNTS = {"dense": 0, "masked_bf16": 0, "math_fp32": 0}
# Ampere 3090: Efficient SDPA wins below this seq; cuDNN wins at 384+.
MASKED_SDPA_SWITCH_SEQ = 320
# Below this width, tiles are launch-bound; keep the S×S mask.
BANDED_WINDOW_MIN = 16
# Ampere fused SDPA wants fat tiles. Tiling by ``n_win`` itself (e.g. 32)
# launches dozens of 32×64 kernels and loses to one S×S mask.
BANDED_TILE = 256
# S×S mask is cheaper than many tiles until the sequence is long enough.
# 3090 2026-09-20: seq=512 → 0.14× S×S; seq=1024 → 0.35×; seq=2048 → 1.17×;
# seq=4096 n_win=32 → 2.93×. Gate at the first non-regression.
BANDED_SEQ_MIN = 2048
_WINDOW_BIAS_CACHE: dict[tuple, torch.Tensor] = {}
_WINDOW_BIAS_BYTES = 0
_WINDOW_BIAS_BUDGET = 256 << 20


def last_sdpa() -> dict:
    """Last ``_sdpa`` path: dense Flash/cuDNN, masked bf16, or fp32 math."""
    return dict(_LAST_SDPA)


def sdpa_counts() -> dict:
    return dict(_SDPA_COUNTS)


def reset_sdpa_counts() -> None:
    for key in _SDPA_COUNTS:
        _SDPA_COUNTS[key] = 0
    _LAST_SDPA["kind"] = "uninit"
    _LAST_SDPA["dtype"] = ""


def reset_window_bias_cache() -> None:
    global _WINDOW_BIAS_BYTES
    _WINDOW_BIAS_CACHE.clear()
    _WINDOW_BIAS_BYTES = 0


def _note_sdpa(kind: str, dtype: torch.dtype) -> None:
    _LAST_SDPA["kind"] = kind
    _LAST_SDPA["dtype"] = str(dtype).replace("torch.", "")
    if kind in _SDPA_COUNTS:
        _SDPA_COUNTS[kind] += 1


def _sdpa_backend_ctx(names: tuple[str, ...]):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [
        backend
        for name in names
        if (backend := getattr(SDPBackend, name, None)) is not None
    ]
    if not backends:
        return None

    def _ctx():
        return sdpa_kernel(backends)

    return _ctx


def _cuda_sdpa_kernel():
    """Dense causal: Flash → cuDNN → mem-efficient. Pass a list, not a tuple."""
    global _SDPA_KERNEL
    if _SDPA_KERNEL is False:
        return nullcontext()
    if _SDPA_KERNEL is not None:
        return _SDPA_KERNEL()
    try:
        factory = _sdpa_backend_ctx(
            ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION")
        )
        if factory is None:
            _SDPA_KERNEL = False
            return nullcontext()
        _SDPA_KERNEL = factory
        return factory()
    except Exception:
        _SDPA_KERNEL = False
        return nullcontext()


def _cuda_masked_sdpa_kernel(seq: int = 512):
    """Preferred masked backend for this seq length. One backend, not Flash.

    Isolate a single kernel: with both cuDNN and Efficient enabled, Ampere
    dispatch at seq=128 still picks the slower cuDNN. seq<320 → Efficient;
    longer → cuDNN. ``_sdpa`` falls back to the other on RuntimeError.
    """
    kind = "small" if int(seq) < MASKED_SDPA_SWITCH_SEQ else "large"
    global _SDPA_MASKED_KERNEL
    if not isinstance(_SDPA_MASKED_KERNEL, dict):
        _SDPA_MASKED_KERNEL = {}
    cached = _SDPA_MASKED_KERNEL.get(kind)
    if cached is False:
        return nullcontext()
    if cached is not None:
        return cached()
    names = ("EFFICIENT_ATTENTION",) if kind == "small" else ("CUDNN_ATTENTION",)
    try:
        factory = _sdpa_backend_ctx(names)
        if factory is None:
            _SDPA_MASKED_KERNEL[kind] = False
            return nullcontext()
        _SDPA_MASKED_KERNEL[kind] = factory
        return factory()
    except Exception:
        _SDPA_MASKED_KERNEL[kind] = False
        return nullcontext()


def _masked_backend_order(seq: int) -> tuple[str, ...]:
    if int(seq) < MASKED_SDPA_SWITCH_SEQ:
        return ("EFFICIENT_ATTENTION", "CUDNN_ATTENTION")
    return ("CUDNN_ATTENTION", "EFFICIENT_ATTENTION")


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Softmax stays high-prec; CUDA QKV stay in ``q.dtype`` when a fused kernel runs.

    Dense causal (YOCO cross, or window when ``n_win`` covers seq): Flash / cuDNN
    / mem-efficient, with ``enable_gqa`` so 2 KV heads are not repeated.
    Masked CSA/HCA ``attn_mask``: isolate Efficient (seq<320) or cuDNN (longer)
    in bf16. Flash rejects a mask. fp32 math is the CPU / last fallback.
    Softmax accumulation stays fp32 inside the kernel.
    """
    def _call(qq, kk, vv, **extra):
        gqa_now = qq.size(-3) != kk.size(-3)
        if gqa_now:
            # Fused masked kernels on Ampere need equal heads. enable_gqa+attn_mask
            # does not raise; it silently drops to math. Repeat KV when masked.
            if extra.get("attn_mask") is None:
                try:
                    return F.scaled_dot_product_attention(
                        qq, kk, vv, enable_gqa=True, **extra
                    )
                except TypeError:
                    pass
            n_rep = qq.size(-3) // kk.size(-3)
            kk = _repeat_kv(kk, n_rep)
            vv = _repeat_kv(vv, n_rep)
        return F.scaled_dot_product_attention(qq, kk, vv, **extra)

    cuda_low = q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
    if bias is None and cuda_low:
        with _cuda_sdpa_kernel():
            out = _call(q, k, v, is_causal=causal)
        _note_sdpa("dense", q.dtype)
        return out
    if bias is not None and cuda_low:
        mask = bias if bias.dtype == q.dtype else bias.to(dtype=q.dtype)
        seq = int(q.size(-2))
        try:
            with _cuda_masked_sdpa_kernel(seq):
                out = _call(q, k, v, attn_mask=mask)
            _note_sdpa("masked_bf16", q.dtype)
            return out
        except RuntimeError:
            pass
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            SDPBackend = None  # type: ignore[assignment]
            sdpa_kernel = None  # type: ignore[assignment]
        if SDPBackend is not None and sdpa_kernel is not None:
            for name in _masked_backend_order(seq):
                backend = getattr(SDPBackend, name, None)
                if backend is None:
                    continue
                try:
                    with sdpa_kernel([backend]):
                        out = _call(q, k, v, attn_mask=mask)
                    _note_sdpa("masked_bf16", q.dtype)
                    return out
                except RuntimeError:
                    continue
    qf, kf, vf = q.float(), k.float(), v.float()
    if bias is None:
        out = _call(qf, kf, vf, is_causal=causal)
    else:
        out = _call(qf, kf, vf, attn_mask=bias.float())
    _note_sdpa("math_fp32", torch.float32)
    return out.to(q.dtype)


def collapse_doc_ids(doc_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Keep packed document ids only when a row actually crosses a boundary.

    DummyStream / single-doc rows are constant along the sequence. Passing
    those tensors into every attention layer used to ``.item()`` 60+ times
    per step and stall the CUDA pipeline. One check here, then ``None``.
    """
    if doc_ids is None or doc_ids.size(-1) <= 1:
        return None
    if not bool((doc_ids[..., 1:] != doc_ids[..., :-1]).any().item()):
        return None
    return doc_ids


def _mean_head_probs(
    q: torch.Tensor,
    k: torch.Tensor,
    window: int,
    doc_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-over-heads softmax weights ``[B, S, S]``. Used only for indexer KL."""
    h = q.size(-3)
    n_kv = k.size(-3)
    kk = k if n_kv == h else _repeat_kv(k, h // n_kv)
    scale = q.size(-1) ** -0.5
    logits = torch.matmul(q.float(), kk.float().transpose(-1, -2)) * scale
    s = q.size(-2)
    bias = _window_causal_bias(s, kk.size(-2), window, q.device, torch.float32, doc_ids)
    logits = logits + bias
    return torch.softmax(logits, dim=-1).mean(dim=1)


def _needs_explicit_mask(q_len: int, window: int, doc_ids: torch.Tensor | None) -> bool:
    """Dense causal SDPA is enough when the window covers the row and docs do not mix.

    Callers that went through ``collapse_doc_ids`` pass ``None`` for single-doc
    rows, so this is a Python branch with no GPU sync.
    """
    if window < q_len:
        return True
    return doc_ids is not None


def _use_banded_window(q_len: int, k_len: int, window: int) -> bool:
    """True when a static sliding window should be tiled instead of S×S.

    CSA/HCA union masks are not a band — those stay on ``_sdpa(..., bias)``.
    Published B0 ``n_win=8192 >= seq=4096`` never takes this path (dense Flash).
    Short seq stays on one S×S fused kernel (tile-by-window was a launch trap).
    """
    s = int(q_len)
    w = int(window)
    return k_len == s and w >= BANDED_WINDOW_MIN and w < s and s >= BANDED_SEQ_MIN


def _window_causal_bias(
    q_len: int,
    k_len: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive mask: 0 keep, -inf drop. Position t may see [t-window+1, t] ∩ [0, t].

    If ``doc_ids`` is ``[B, S]``, tokens from different packed documents cannot attend
    (document mask). Returned shape is ``[S, S]`` or ``[B, 1, S, S]``.

    DummyStream / single-doc rows rebuild the same ``S×S`` every layer. Cache
    that static tensor so masked SDPA is not preceded by a fresh mask GEMM.
    """
    global _WINDOW_BIAS_BYTES
    if doc_ids is None:
        dev = torch.device(device)
        key = (int(q_len), int(k_len), int(window), dev.type, dev.index, str(dtype))
        hit = _WINDOW_BIAS_CACHE.get(key)
        if hit is not None:
            return hit
    q = torch.arange(q_len, device=device)
    k = torch.arange(k_len, device=device)
    causal = k[None, :] <= q[:, None]
    win = (q[:, None] - k[None, :]) < window
    keep = causal & win
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    if doc_ids is None:
        nbytes = int(bias.numel() * bias.element_size())
        if _WINDOW_BIAS_BYTES + nbytes > _WINDOW_BIAS_BUDGET:
            reset_window_bias_cache()
        _WINDOW_BIAS_CACHE[key] = bias
        _WINDOW_BIAS_BYTES += nbytes
        return bias
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    bias = bias.view(1, q_len, k_len).expand(doc_ids.size(0), -1, -1)
    bias = bias.masked_fill(~same, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


def _tile_keep(
    n: int,
    tile: int,
    lookback: int,
    window: int,
    seq: int,
    device: torch.device,
) -> torch.Tensor:
    """``[n, tile, lookback+tile]`` bool: padded-left keys for each query tile."""
    k_width = lookback + tile
    i = torch.arange(tile, device=device)[None, :, None]
    j = torch.arange(k_width, device=device)[None, None, :]
    q0 = (torch.arange(n, device=device) * tile)[:, None, None]
    gq = q0 + i
    gk = q0 - lookback + j
    return (gk >= 0) & (gk <= gq) & ((gq - gk) < int(window)) & (gq < int(seq))


def _banded_window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Causal sliding window as batched fat tiles, not ``n_win×2 n_win`` crumbs.

    Query tiles of ``BANDED_TILE`` (or ``n_win`` if larger) attend to
    ``lookback + tile`` keys with a static keep-set. Same keep-set as
    ``_window_causal_bias``. Left-padded dummy keys on tile 0 are masked out.
    """
    w = int(window)
    tile = min(int(q.size(-2)), max(w, BANDED_TILE))
    if tile >= int(q.size(-2)):
        bias = _window_causal_bias(int(q.size(-2)), int(q.size(-2)), w, q.device, q.dtype, None)
        return _sdpa(q, k, v, bias)
    # Fat tiles need packed BHSD so unfold/reshape along S is a view.
    # Dense Flash keeps the BSHD-memory transpose from ``to_sdpa_layout``.
    if not q.is_contiguous():
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    b, hq, s, d = q.shape
    hk = k.size(-3)
    lookback = w - 1
    k_width = lookback + tile
    pad = (tile - (s % tile)) % tile
    q_pad = F.pad(q, (0, 0, 0, pad))
    k_pad = F.pad(k, (0, 0, lookback, pad))
    v_pad = F.pad(v, (0, 0, lookback, pad))
    s_pad = s + pad
    n = s_pad // tile
    q_b = q_pad.reshape(b, hq, n, tile, d)
    k_unf = k_pad.unfold(2, k_width, tile).permute(0, 1, 2, 4, 3).contiguous()
    v_unf = v_pad.unfold(2, k_width, tile).permute(0, 1, 2, 4, 3).contiguous()
    q_flat = q_b.permute(0, 2, 1, 3, 4).reshape(b * n, hq, tile, d)
    k_flat = k_unf.permute(0, 2, 1, 3, 4).reshape(b * n, hk, k_width, d)
    v_flat = v_unf.permute(0, 2, 1, 3, 4).reshape(b * n, hk, k_width, d)
    keep = _tile_keep(n, tile, lookback, w, s, q.device)
    bias = torch.zeros(n, tile, k_width, device=q.device, dtype=q.dtype)
    bias = bias.masked_fill(~keep, torch.finfo(q.dtype).min)
    bias = bias.unsqueeze(1).expand(n, 1, tile, k_width)
    # One row of tiles per original batch item; repeat bias across batch.
    bias = bias.repeat(b, 1, 1, 1)
    y = _sdpa(q_flat, k_flat, v_flat, bias)
    y = y.reshape(b, n, hq, tile, d).permute(0, 2, 1, 3, 4).reshape(b, hq, n * tile, d)
    return y[:, :, :s]


def _window_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
    doc_ids: torch.Tensor | None = None,
    extra_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sliding-window GQA: covering Flash, else banded tiles, else S×S mask."""
    s = int(q.size(-2))
    k_len = int(k.size(-2))
    if extra_bias is not None or doc_ids is not None or not _use_banded_window(s, k_len, window):
        if extra_bias is None and doc_ids is None and int(window) >= s and k_len == s:
            return _sdpa(q, k, v, causal=True)
        bias = _window_causal_bias(s, k_len, window, q.device, q.dtype, doc_ids)
        if extra_bias is not None:
            bias = bias + extra_bias.to(device=bias.device, dtype=bias.dtype)
        return _sdpa(q, k, v, bias)
    return _banded_window_sdpa(q, k, v, window)


class WindowAttention(nn.Module):
    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_win = cfg.n_win
        if self.n_heads % self.n_kv != 0:
            raise ValueError(f"num_heads={self.n_heads} must divide by num_kv_heads={self.n_kv}")
        d = cfg.hidden_size
        kv = cfg.kv_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv, bias=False)
        self.v_proj = nn.Linear(d, kv, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        *,
        extra_bias: torch.Tensor | None = None,
        return_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        h, hd, n_kv = self.n_heads, self.head_dim, self.n_kv
        q, k, v = _fused_qkv(self.q_proj, self.k_proj, self.v_proj, x)
        q = split_heads(q, h, hd)
        k = split_heads(k, n_kv, hd)
        v = split_heads(v, n_kv, hd)
        q, k = rope_after_qk_norm(
            q, k, rope=self.rope, q_norm=self.q_norm, k_norm=self.k_norm
        )
        qh, kh, vh = to_sdpa_layout(q), to_sdpa_layout(k), to_sdpa_layout(v)
        out = _window_sdpa(qh, kh, vh, self.n_win, doc_ids, extra_bias)
        y = self.o_proj(merge_heads(out))
        if return_probs:
            return y, _mean_head_probs(qh, kh, self.n_win, doc_ids)
        return y


class CrossAttention(nn.Module):
    """Decoder queries attend causally to the YOCO global cache (GQA K/V)."""

    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps) if cfg.qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_ids: torch.Tensor | None = None,
        *,
        extra_bias: torch.Tensor | None = None,
        return_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        h, hd, n_kv = self.n_heads, self.head_dim, self.n_kv
        q = split_heads(self.q_proj(x), h, hd)
        k = split_heads(k, n_kv, hd)
        v = split_heads(v, n_kv, hd)
        q, k = rope_after_qk_norm(
            q, k, rope=self.rope, q_norm=self.q_norm, k_norm=self.k_norm
        )
        qh, kh, vh = to_sdpa_layout(q), to_sdpa_layout(k), to_sdpa_layout(v)
        need_mask = extra_bias is not None or _needs_explicit_mask(s, s, doc_ids)
        if need_mask:
            bias = _window_causal_bias(s, s, s, x.device, q.dtype, doc_ids)
            if extra_bias is not None:
                bias = bias + extra_bias.to(device=bias.device, dtype=bias.dtype)
            out = _sdpa(qh, kh, vh, bias)
        else:
            out = _sdpa(qh, kh, vh, causal=True)
        y = self.o_proj(merge_heads(out))
        if return_probs:
            return y, _mean_head_probs(qh, kh, s, doc_ids)
        return y
