"""Lightning Indexer: layer-internal dense-attn KL (Phase C). Not a CSA kernel.

Indexer GEMMs stay bf16/fp32 (KEEP_HIGH_PREC). Attached only when a C–G
stage needs them; Phase B graphs stay indexer-free.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.attention import _window_causal_bias
from cat_yoko.config import CATYokoConfig


class LightningIndexer(nn.Module):
    """Low-rank ReLU indexer. ``I_{t,s} = ReLU(q_t) · k_s / √d``.

    ``index_head_dim`` matches the param-budget stand-in (64 on 12B, 16 on tiny).
    """

    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.d_idx = int(cfg.indexer_dim)
        self.q_proj = nn.Linear(d, self.d_idx, bias=False)
        self.k_proj = nn.Linear(d, self.d_idx, bias=False)
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D] → [B, S, S]`` fp32 scores."""
        xf = x.float()
        q = F.relu(self.q_proj(xf))
        k = self.k_proj(xf)
        scale = self.d_idx ** -0.5
        return torch.matmul(q, k.transpose(-1, -2)) * scale


def _bias_b_s_s(
    window: int,
    seq: int,
    device: torch.device,
    doc_ids: torch.Tensor | None,
    batch: int,
) -> torch.Tensor:
    bias = _window_causal_bias(seq, seq, window, device, torch.float32, doc_ids)
    if bias.dim() == 2:
        return bias.unsqueeze(0).expand(batch, -1, -1)
    if bias.dim() == 4:
        return bias[:, 0]
    return bias


def indexer_align_kl(
    indexer: LightningIndexer,
    hidden: torch.Tensor,
    dense_probs: torch.Tensor,
    window: int,
    doc_ids: torch.Tensor | None,
) -> torch.Tensor:
    """KL(dense ‖ indexer). ``dense_probs`` must be detached (layer-internal)."""
    b, s, _ = hidden.shape
    bias = _bias_b_s_s(window, s, hidden.device, doc_ids, b)
    log_q = F.log_softmax(indexer.scores(hidden) + bias, dim=-1).clamp(min=-50.0)
    p = dense_probs.float().clamp_min(1e-8)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (p * (p.log() - log_q)).sum(dim=-1).mean()


def indexer_topk_bias(
    indexer: LightningIndexer,
    hidden: torch.Tensor,
    window: int,
    doc_ids: torch.Tensor | None,
    topk: int,
) -> torch.Tensor:
    """Additive mask: keep indexer top-k inside the causal window. Detach scores."""
    b, s, _ = hidden.shape
    with torch.no_grad():
        bias = _bias_b_s_s(window, s, hidden.device, doc_ids, b)
        scores = indexer.scores(hidden) + bias
        k = max(min(int(topk), s), 1)
        thresh = scores.topk(k, dim=-1).values[..., -1:]
        keep = scores >= thresh
        extra = torch.zeros_like(scores)
        extra = extra.masked_fill(~keep, torch.finfo(scores.dtype).min)
    return extra.unsqueeze(1)


def ensure_indexers(model: nn.Module, cfg: CATYokoConfig) -> int:
    """Attach indexer modules to CSA encoder layers and decoder cross-attn."""
    n = 0
    for blk in getattr(model, "encoder", []):
        if getattr(blk, "kind", "sliding") != "csa":
            continue
        if getattr(blk, "indexer", None) is None:
            blk.indexer = LightningIndexer(cfg).to(
                device=next(blk.parameters()).device,
                dtype=next(blk.parameters()).dtype,
            )
            n += 1
    for blk in getattr(model, "decoder", []):
        if getattr(blk, "cross_indexer", None) is None:
            blk.cross_indexer = LightningIndexer(cfg).to(
                device=next(blk.parameters()).device,
                dtype=next(blk.parameters()).dtype,
            )
            n += 1
    return n


def set_align_indexer(model: nn.Module, flag: bool) -> None:
    for blk in getattr(model, "encoder", []):
        blk.align_indexer = bool(flag and getattr(blk, "indexer", None) is not None)
    for blk in getattr(model, "decoder", []):
        blk.align_indexer = bool(flag and getattr(blk, "cross_indexer", None) is not None)


def set_sparse_mode(model: nn.Module, mode: str) -> None:
    """``window`` | ``topk`` | ``hca``. ``hca`` keeps top-k on CSA layers."""
    for blk in getattr(model, "encoder", []):
        kind = getattr(blk, "kind", "sliding")
        if mode == "hca" and kind == "hca":
            blk.sparse_mode = "hca"
        elif mode in {"topk", "hca"} and kind == "csa":
            blk.sparse_mode = "topk"
        else:
            blk.sparse_mode = "window"
    for blk in getattr(model, "decoder", []):
        blk.sparse_mode = "window"


def phase_needs_indexer(phase: str) -> bool:
    from cat_yoko.phases import PHASES

    ph = PHASES.get(phase)
    if ph is None:
        return False
    return bool(ph.align_indexer or ph.sparse in {"topk", "hca"} or ph.freeze == "indexer")


def indexer_param_names(model: nn.Module) -> list[str]:
    names = []
    for n, _p in model.named_parameters():
        parts = n.split(".")
        if "indexer" in parts or "cross_indexer" in parts:
            names.append(n)
    return names
