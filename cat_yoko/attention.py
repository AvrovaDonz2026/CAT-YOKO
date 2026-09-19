"""Sliding-window self-attn and gated cross-attn (Phase B backend).

MiniCPM5-2B is Llama GQA (16 Q / 2 KV). Encoder/decoder self-attn and the
YOCO cache therefore use ``kv_dim = n_kv * head_dim``, not full MHA.
"""

from __future__ import annotations

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


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Softmax stays high-prec; QKV stay in ``q.dtype`` on the CUDA fast path.

    Flash / cuDNN SDPA accumulate the softmax in fp32. Materializing fp32 Q/K/V
    disables those kernels and is only the CPU / explicit-mask fallback.
    ``enable_gqa`` keeps 2 KV heads instead of repeating to 16 Q heads.
    """
    gqa = k.size(-3) != q.size(-3)

    def _call(qq, kk, vv, **extra):
        if gqa:
            try:
                return F.scaled_dot_product_attention(
                    qq, kk, vv, enable_gqa=True, **extra
                )
            except TypeError:
                n_rep = qq.size(-3) // kk.size(-3)
                kk = _repeat_kv(kk, n_rep)
                vv = _repeat_kv(vv, n_rep)
        return F.scaled_dot_product_attention(qq, kk, vv, **extra)

    cuda_fast = (
        bias is None
        and q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
    )
    if cuda_fast:
        return _call(q, k, v, is_causal=causal)
    qf, kf, vf = q.float(), k.float(), v.float()
    if bias is None:
        out = _call(qf, kf, vf, is_causal=causal)
    else:
        out = _call(qf, kf, vf, attn_mask=bias.float())
    return out.to(q.dtype)


def _needs_explicit_mask(q_len: int, window: int, doc_ids: torch.Tensor | None) -> bool:
    """Dense causal SDPA is enough when the window covers the row and docs do not mix."""
    if window < q_len:
        return True
    if doc_ids is None or q_len <= 1:
        return False
    return bool((doc_ids[:, 1:] != doc_ids[:, :-1]).any().item())


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
    """
    q = torch.arange(q_len, device=device)
    k = torch.arange(k_len, device=device)
    causal = k[None, :] <= q[:, None]
    win = (q[:, None] - k[None, :]) < window
    keep = causal & win
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    if doc_ids is None:
        return bias
    same = doc_ids[:, :, None] == doc_ids[:, None, :]
    bias = bias.view(1, q_len, k_len).expand(doc_ids.size(0), -1, -1)
    bias = bias.masked_fill(~same, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


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

    def forward(self, x: torch.Tensor, doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, s, d = x.shape
        h, hd, n_kv = self.n_heads, self.head_dim, self.n_kv
        q, k, v = _fused_qkv(self.q_proj, self.k_proj, self.v_proj, x)
        q = q.view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, n_kv, hd).transpose(1, 2)
        v = v.view(b, s, n_kv, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(s, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if _needs_explicit_mask(s, self.n_win, doc_ids):
            k = _repeat_kv(k, h // n_kv)
            v = _repeat_kv(v, h // n_kv)
            bias = _window_causal_bias(s, s, self.n_win, x.device, q.dtype, doc_ids)
            out = _sdpa(q, k, v, bias)
        else:
            # GQA: 16 Q / 2 KV. Do not repeat KV; SDPA enable_gqa on CUDA.
            out = _sdpa(q, k, v, causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, d))


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
    ) -> torch.Tensor:
        b, s, d = x.shape
        h, hd, n_kv = self.n_heads, self.head_dim, self.n_kv
        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = k.view(b, s, n_kv, hd).transpose(1, 2)
        v = v.view(b, s, n_kv, hd).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self.rope(s, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if _needs_explicit_mask(s, s, doc_ids):
            k = _repeat_kv(k, h // n_kv)
            v = _repeat_kv(v, h // n_kv)
            bias = _window_causal_bias(s, s, s, x.device, q.dtype, doc_ids)
            out = _sdpa(q, k, v, bias)
        else:
            out = _sdpa(q, k, v, causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(b, s, d))
