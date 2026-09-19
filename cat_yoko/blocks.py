"""Encoder / decoder blocks."""

from __future__ import annotations

import torch
from torch import nn

from cat_yoko.attention import CrossAttention, WindowAttention
from cat_yoko.config import CATYokoConfig
from cat_yoko.moe import MoE, SwiGLU
from cat_yoko.rope import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(self, cfg: CATYokoConfig, *, kind: str = "sliding", dense: bool = False) -> None:
        super().__init__()
        self.kind = kind  # sliding | csa | hca; Phase B compute is still window GQA
        ns, nr, tk = cfg.expert_count("encoder")
        del ns
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.attn = WindowAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        if dense:
            self.mlp = SwiGLU(cfg.hidden_size, cfg.dense_intermediate_size)
        else:
            self.mlp = MoE(cfg, nr, tk, hash_route=False)
        self.res = cfg.residual_scale
        self.index_topk = cfg.index_topk
        self.compress_m = cfg.compress_m
        self.compress_m_hca = cfg.compress_m_hca
        self.sparse_mode = "window"
        self.align_indexer = False

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        mode = getattr(self, "sparse_mode", "window")
        indexer = getattr(self, "indexer", None)
        self.last_indexer_kl = None
        self.last_indexer_recall = None
        if mode == "hca":
            from cat_yoko.sparse import hca_attend

            attn_out = hca_attend(self.attn, h, doc_ids, self.compress_m_hca)
        elif mode == "topk" and indexer is not None:
            from cat_yoko.indexer import indexer_compressed_keep
            from cat_yoko.sparse import csa_attend

            keep = indexer_compressed_keep(
                indexer, h, self.compress_m, self.index_topk, doc_ids
            )
            attn_out = csa_attend(self.attn, h, doc_ids, keep)
        elif bool(getattr(self, "align_indexer", False)) and indexer is not None:
            attn_out, probs = self.attn(h, doc_ids, return_probs=True)
            from cat_yoko.indexer import (
                indexer_align_kl,
                indexer_compressed_keep,
                indexer_recall_at_k,
            )
            from cat_yoko.sparse import compressed_keep_matrix

            self.last_indexer_kl = indexer_align_kl(
                indexer, h.detach(), probs.detach(), self.attn.n_win, doc_ids
            )
            selected = indexer_compressed_keep(
                indexer, h.detach(), self.compress_m, self.index_topk, doc_ids
            )
            comp = compressed_keep_matrix(h.size(1), self.compress_m, h.device)
            self.last_indexer_recall = indexer_recall_at_k(probs.detach(), comp, selected)
        else:
            attn_out = self.attn(h, doc_ids)
        x = x + self.res * attn_out
        kwargs = {"token_ids": token_ids} if isinstance(self.mlp, MoE) else {}
        x = x + self.res * self.mlp(self.ln2(x), **kwargs)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg: CATYokoConfig, *, hash_route: bool, dense: bool = False) -> None:
        super().__init__()
        ns, nr, tk = cfg.expert_count("decoder")
        del ns
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.self_attn = WindowAttention(cfg)
        self.ln_cross = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.cross_attn = CrossAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        if dense:
            self.mlp = SwiGLU(cfg.hidden_size, cfg.dense_intermediate_size)
        else:
            self.mlp = MoE(cfg, nr, tk, hash_route=hash_route)
        self.res = cfg.residual_scale
        self.register_buffer("gate", torch.zeros(()))
        self.sparse_mode = "window"
        self.align_indexer = False

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.res * self.self_attn(self.ln1(x), doc_ids)
        h = self.ln_cross(x)
        self.last_indexer_kl = None
        c = self.cross_attn(h, k, v, doc_ids)
        x = x + self.res * (self.gate * c)
        kwargs = {"token_ids": token_ids} if isinstance(self.mlp, MoE) else {}
        x = x + self.res * self.mlp(self.ln2(x), **kwargs)
        return x
