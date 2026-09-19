"""YOCO causal encoder-decoder LM (Phase B window attention)."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.attention import collapse_doc_ids
from cat_yoko.blocks import DecoderBlock, EncoderBlock
from cat_yoko.config import CATYokoConfig, encoder_layer_kind
from cat_yoko.offload import move_module, offload_checkpoint_block
from cat_yoko.rope import RMSNorm


class CATYokoForCausalLM(nn.Module):
    def __init__(self, cfg: CATYokoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, v = cfg.hidden_size, cfg.vocab_size
        self.embed = nn.Embedding(v, d)
        self.encoder = nn.ModuleList(
            EncoderBlock(
                cfg,
                kind=encoder_layer_kind(i),
                dense=cfg.first_dense and i == 0,
            )
            for i in range(cfg.encoder_layers)
        )
        self.cache_k = nn.Linear(d, cfg.kv_dim, bias=False)
        self.cache_v = nn.Linear(d, cfg.kv_dim, bias=False)
        self.decoder = nn.ModuleList(
            DecoderBlock(
                cfg,
                hash_route=(i < cfg.hash_moe_decoder_layers),
                dense=cfg.first_dense and i == 0,
            )
            for i in range(cfg.decoder_layers)
        )
        self.norm = RMSNorm(d, cfg.rms_eps)
        self.lm_head = nn.Linear(d, v, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.scale_emb = cfg.embed_scale
        self.logit_scale = cfg.logit_scale
        self.detach_cache = True
        self.grad_checkpoint = False
        self.offload_encoder = False
        self.offload_blocks = False
        # Trainer turns this off unless KD needs student logits.
        self.return_logits = True
        p0 = next(self.parameters(), None)
        if p0 is not None and p0.device.type != "meta":
            from cat_yoko.upcycle import init_new_modules

            init_new_modules(self)

    def set_detach(self, flag: bool) -> None:
        self.detach_cache = flag

    def _run_block(self, blk: nn.Module, *tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.offload_blocks:
            return offload_checkpoint_block(blk, *tensors)
        ckpt = self.grad_checkpoint and self.training
        if ckpt and any(t.requires_grad for t in tensors):
            y = torch.utils.checkpoint.checkpoint(blk, *tensors, use_reentrant=False)
        else:
            y = blk(*tensors)
        aux = getattr(getattr(blk, "mlp", None), "last_aux", None)
        if aux is None:
            aux = y.new_zeros(())
        elif not any(p.requires_grad for p in blk.mlp.parameters()):
            aux = y.new_zeros(())
        return y, aux

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.offload_encoder and not self.offload_blocks:
            move_module(self.encoder, input_ids.device)
        x = self.embed(input_ids) * self.scale_emb
        # DummyStream / single-doc batches must not become zero tensors: that
        # forced a host sync in every attention layer. Packed multi-doc rows
        # still pass the real ids.
        doc_ids = collapse_doc_ids(doc_ids)
        aux = x.new_zeros(())
        # B0/B1: cache is detached, so encoder FPROP does not need an autograd
        # graph (TE otherwise saves activations / looks up NVFP4 WGRAD).
        enc_ctx = torch.no_grad() if self.detach_cache else nullcontext()
        with enc_ctx:
            for blk in self.encoder:
                x, a = self._run_block(blk, x, input_ids, doc_ids)
                if not self.detach_cache:
                    aux = aux + a
        if self.offload_encoder and self.detach_cache and not self.offload_blocks:
            move_module(self.encoder, "cpu")
        hidden = x.detach() if self.detach_cache else x
        from cat_yoko.nvfp4_linear import fused_cat_linear

        kv = fused_cat_linear([self.cache_k, self.cache_v], hidden)
        k, v = kv.split((self.cfg.kv_dim, self.cfg.kv_dim), dim=-1)
        y = hidden
        for blk in self.decoder:
            y, a = self._run_block(blk, y, k, v, input_ids, doc_ids)
            aux = aux + a
        logits = None
        if labels is None or self.return_logits:
            logits = self.lm_head(self.norm(y)) / self.logit_scale
        out: dict[str, torch.Tensor] = {}
        if logits is not None:
            out["logits"] = logits
        if labels is not None:
            if logits is None:
                from cat_yoko.loss import linear_cross_entropy

                nll, n_valid = linear_cross_entropy(
                    self.norm(y)[:, :-1],
                    labels[:, 1:],
                    self.lm_head,
                    logit_scale=self.logit_scale,
                )
            else:
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                nll = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
                n_valid = (shift_labels != -100).sum()
            out["aux"] = aux
            out["loss"] = nll + aux
            out["nll"] = nll
            out["n_valid"] = n_valid
        return out

    def step_router_bias(self) -> None:
        for blk in list(self.encoder) + list(self.decoder):
            mlp = getattr(blk, "mlp", None)
            fn = getattr(mlp, "step_router_bias", None)
            if not callable(fn):
                continue
            if not any(p.requires_grad for p in mlp.parameters()):
                mlp.last_load = None
                mlp._load_n = 0
                continue
            fn()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        buckets = {"embed": 0, "encoder": 0, "cache": 0, "decoder": 0, "norm": 0, "other": 0}
        for name, p in self.named_parameters():
            n = p.numel()
            if name.startswith("embed") or name.startswith("lm_head"):
                buckets["embed"] += n
            elif name.startswith("encoder"):
                buckets["encoder"] += n
            elif name.startswith("cache_"):
                buckets["cache"] += n
            elif name.startswith("decoder"):
                buckets["decoder"] += n
            elif name.startswith("norm"):
                buckets["norm"] += n
            else:
                buckets["other"] += n
        buckets["total"] = sum(v for k, v in buckets.items() if k != "total")
        return buckets
