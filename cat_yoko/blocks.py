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

    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.res * self.attn(self.ln1(x), doc_ids)
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

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_ids: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.res * self.self_attn(self.ln1(x), doc_ids)
        x = x + self.res * (self.gate * self.cross_attn(self.ln_cross(x), k, v, doc_ids))
        kwargs = {"token_ids": token_ids} if isinstance(self.mlp, MoE) else {}
        x = x + self.res * self.mlp(self.ln2(x), **kwargs)
        return x
