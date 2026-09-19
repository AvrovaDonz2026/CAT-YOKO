"""Kimi Delta Attention (gated-delta linear attn) for KV savings.

Not a DPLR CUDA kernel. Sequential / chunk-boundary PyTorch. Published
Phase B stays window GQA (``use_kda=False``). When enabled, encoder
non-bootstrap layers follow 3:1 KDA:(CSA/HCA) and decoder self-attn
follows 3:1 KDA:sliding so those layers keep a fixed ``[H, d, d]``
state instead of an ``S``-length KV cache.

KDA does **not** shrink the YOCO global cache (that is M2). It does
not provide decoder query-aware retrieval (that is M3). Mid-context
exact recall still needs CSA / window anchors.

Pipeline: **implement then light**. ``use_kda=True`` on Phase B builds
the 3:1 graph (KDAGates exist) but compute stays window GQA. Phase C
lights ``C-kda → C-index → C-topk → C-hca → C-win``. Do not add KDA
modules at C onto a ``use_kda=False`` B overlay. Without ``use_kda``
the published chain is still indexer→topk→hca→win.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.attention import _fused_qkv, _repeat_kv
from cat_yoko.config import CATYokoConfig


class KDAGates(nn.Module):
    """Fine-grained decay (per channel) + head-wise write / output gates."""

    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        d, h, hd = cfg.hidden_size, cfg.num_heads, cfg.head_dim
        self.n_heads = h
        self.head_dim = hd
        self.dt_proj = nn.Linear(d, h, bias=True)
        self.beta_proj = nn.Linear(d, h, bias=True)
        self.out_gate = nn.Linear(d, d, bias=False)
        self.A_log = nn.Parameter(torch.empty(h, hd))
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.out_gate.weight)


def gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    doc_ids: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Causal gated delta rule. ``q,k,v,alpha`` are ``[B, H, T, D]``.

    ``alpha`` is per-channel decay in ``(0, 1]``. ``beta`` is ``[B, H, T]``
    write strength. ``state`` is ``[B, H, D, D]`` (value × key) in fp32.
    Document boundaries zero the recurrent state (packed-doc analogue of
    the window document mask).
    """
    b, h, t, d = q.shape
    qf = q.reshape(b * h, t, d).float()
    kf = k.reshape(b * h, t, d).float()
    vf = v.reshape(b * h, t, d).float()
    af = alpha.reshape(b * h, t, d).float().clamp(1e-6, 1.0)
    bf = beta.reshape(b * h, t).float().clamp(0.0, 1.0)
    if state is None:
        s = qf.new_zeros(b * h, d, d)
    else:
        s = state.reshape(b * h, d, d).float()
    docs = None
    if doc_ids is not None:
        docs = doc_ids.to(device=q.device).unsqueeze(1).expand(b, h, t).reshape(b * h, t)
    outs: list[torch.Tensor] = []
    for i in range(t):
        if docs is not None and i > 0:
            reset = docs[:, i] != docs[:, i - 1]
            if bool(reset.any()):
                s = s.masked_fill(reset.view(-1, 1, 1), 0)
        # Column-wise decay, then delta-rule write.
        s = s * af[:, i].unsqueeze(-2)
        k_i = kf[:, i]
        v_i = vf[:, i]
        k_beta = k_i * bf[:, i].unsqueeze(-1)
        sk = torch.bmm(s, k_i.unsqueeze(-1))
        s = s - torch.bmm(sk, k_beta.unsqueeze(-2))
        s = s + torch.bmm(v_i.unsqueeze(-1), k_beta.unsqueeze(-2))
        o_i = torch.bmm(s, qf[:, i].unsqueeze(-1)).squeeze(-1)
        outs.append(o_i)
    y = torch.stack(outs, dim=1).view(b, h, t, d).to(dtype=q.dtype)
    if return_state:
        return y, s.view(b, h, d, d)
    return y


def kda_attend(
    attn: nn.Module,
    gates: KDAGates,
    x: torch.Tensor,
    doc_ids: torch.Tensor | None = None,
    *,
    state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Linear self-attn using ``WindowAttention`` QKV/O (NoPE; decay carries recency)."""
    b, s, _d = x.shape
    h, hd, n_kv = attn.n_heads, attn.head_dim, attn.n_kv
    q, k, v = _fused_qkv(attn.q_proj, attn.k_proj, attn.v_proj, x)
    q = q.view(b, s, h, hd).transpose(1, 2)
    k = k.view(b, s, n_kv, hd).transpose(1, 2)
    v = v.view(b, s, n_kv, hd).transpose(1, 2)
    if attn.q_norm is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    k = _repeat_kv(k, h // n_kv)
    v = _repeat_kv(v, h // n_kv)
    q = q * (hd ** -0.5)
    dt = F.softplus(gates.dt_proj(x))
    a = -torch.exp(gates.A_log.float())
    log_alpha = dt.unsqueeze(-1) * a
    alpha = torch.exp(log_alpha).permute(0, 2, 1, 3).to(dtype=q.dtype)
    beta = torch.sigmoid(gates.beta_proj(x)).permute(0, 2, 1)
    scanned = gated_delta_scan(
        q, k, v, alpha, beta, doc_ids=doc_ids, state=state, return_state=return_state
    )
    if return_state:
        y, new_state = scanned
    else:
        y = scanned
        new_state = None
    y = y.transpose(1, 2).contiguous().view(b, s, h * hd)
    y = y * torch.sigmoid(gates.out_gate(x))
    out = attn.o_proj(y)
    if return_state:
        return out, new_state
    return out


def kda_state_bytes(cfg: CATYokoConfig, n_layers: int, *, dtype_bytes: float = 2.0) -> float:
    """One recurrent ``[H, d_h, d_h]`` map per KDA layer (independent of ``S``)."""
    return float(n_layers) * cfg.num_heads * cfg.head_dim * cfg.head_dim * dtype_bytes


def window_kv_bytes(
    cfg: CATYokoConfig,
    seq_len: int,
    n_layers: int,
    *,
    dtype_bytes: float = 2.0,
    cap_window: bool = True,
) -> float:
    """GQA K+V cache. Decode typically caps at ``n_win``."""
    s = min(int(seq_len), cfg.n_win) if cap_window else int(seq_len)
    return float(n_layers) * 2.0 * cfg.kv_dim * s * dtype_bytes


def yoco_cache_bytes(cfg: CATYokoConfig, seq_len: int, *, dtype_bytes: float = 2.0) -> float:
    """Single global ``K̂,V̂`` (M2 off: slot count = ``S``). KDA does not change this."""
    return 2.0 * cfg.kv_dim * int(seq_len) * dtype_bytes


@dataclass(frozen=True)
class KvLedger:
    seq_len: int
    yoco_cache_bytes: float
    encoder_window_layers: int
    encoder_kda_layers: int
    decoder_window_layers: int
    decoder_kda_layers: int
    encoder_self_bytes: float
    decoder_self_bytes: float
    decode_bytes: float
    decode_all_window_bytes: float
    decode_fullseq_decoder_bytes: float
    saved_vs_all_window: float
    saved_vs_fullseq_decoder: float
    note: str


def kv_ledger(
    cfg: CATYokoConfig,
    seq_len: int,
    *,
    use_kda: bool | None = None,
    dtype_bytes: float = 2.0,
    cap_window: bool = True,
) -> KvLedger:
    """Decode-time KV: YOCO global + decoder self. Encoder self is prefill-only."""
    from cat_yoko.config import decoder_layer_kind, encoder_layer_kind

    on = bool(cfg.use_kda if use_kda is None else use_kda)
    enc = [
        encoder_layer_kind(i, cfg.encoder_layers, use_kda=on, kda_group=cfg.kda_group)
        for i in range(cfg.encoder_layers)
    ]
    dec = [
        decoder_layer_kind(
            i,
            cfg.decoder_layers,
            use_kda=on and cfg.kda_decoder,
            kda_group=cfg.kda_group,
        )
        for i in range(cfg.decoder_layers)
    ]
    enc_kda = sum(1 for k in enc if k == "kda")
    enc_win = len(enc) - enc_kda
    dec_kda = sum(1 for k in dec if k == "kda")
    dec_win = len(dec) - dec_kda
    yoco = yoco_cache_bytes(cfg, seq_len, dtype_bytes=dtype_bytes)
    enc_self = window_kv_bytes(
        cfg, seq_len, enc_win, dtype_bytes=dtype_bytes, cap_window=cap_window
    ) + kda_state_bytes(cfg, enc_kda, dtype_bytes=dtype_bytes)
    dec_self = window_kv_bytes(
        cfg, seq_len, dec_win, dtype_bytes=dtype_bytes, cap_window=cap_window
    ) + kda_state_bytes(cfg, dec_kda, dtype_bytes=dtype_bytes)
    decode = yoco + dec_self
    all_win_dec = window_kv_bytes(
        cfg, seq_len, cfg.decoder_layers, dtype_bytes=dtype_bytes, cap_window=cap_window
    )
    decode_all_win = yoco + all_win_dec
    full_dec = window_kv_bytes(
        cfg, seq_len, cfg.decoder_layers, dtype_bytes=dtype_bytes, cap_window=False
    )
    decode_full = yoco + full_dec
    return KvLedger(
        seq_len=int(seq_len),
        yoco_cache_bytes=yoco,
        encoder_window_layers=enc_win,
        encoder_kda_layers=enc_kda,
        decoder_window_layers=dec_win,
        decoder_kda_layers=dec_kda,
        encoder_self_bytes=enc_self,
        decoder_self_bytes=dec_self,
        decode_bytes=decode,
        decode_all_window_bytes=decode_all_win,
        decode_fullseq_decoder_bytes=decode_full,
        saved_vs_all_window=decode_all_win - decode,
        saved_vs_fullseq_decoder=decode_full - decode,
        note=(
            "KDA replaces per-layer S-length self KV with [H,d,d] state. "
            "YOCO global cache is unchanged (M2). Not mid-context exact recall."
        ),
    )


LATE_KDA_IMPLEMENT = (
    "implement-then-light: pass --use-kda on B0/B1/B2 so the 3:1 graph exists "
    "(Phase B is still window GQA). Phase C only lights C-kda. Refusing to add "
    "KDA modules onto a use_kda=False overlay."
)


def extra_use_kda(extra: dict | None) -> bool | None:
    """KDA implement flag stored on a checkpoint. ``None`` = no overlay info."""
    if not extra:
        return None
    if "use_kda" in extra:
        return bool(extra["use_kda"])
    cfg = extra.get("cfg")
    if isinstance(cfg, dict) and "use_kda" in cfg:
        return bool(cfg["use_kda"])
    if isinstance(cfg, dict) and cfg:
        return False
    return None


def resolve_implemented_kda(*, cli: bool, extra: dict | None) -> bool:
    """Inherit ``use_kda`` from B overlay. Error if C tries to implement late."""
    ckpt = extra_use_kda(extra)
    if ckpt is False and cli:
        raise RuntimeError(LATE_KDA_IMPLEMENT)
    if ckpt is True:
        return True
    return bool(cli)


def model_has_kda(model: nn.Module) -> bool:
    for blk in list(getattr(model, "encoder", [])) + list(getattr(model, "decoder", [])):
        if getattr(blk, "kind", None) == "kda":
            return True
        if getattr(blk, "kda", None) is not None:
            return True
    return False
