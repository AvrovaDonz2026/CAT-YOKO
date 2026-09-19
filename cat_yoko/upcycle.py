"""MiniCPM5-2B dense → CAT-YOKO MoE (copy-and-scale, not exact identity)."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from cat_yoko.config import CATYokoConfig
from cat_yoko.model import CATYokoForCausalLM

NEW_MODULE_INIT_STD = 0.02


def _small_linear(lin: nn.Linear, std: float) -> None:
    if lin.weight.device.type == "meta":
        return
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    if lin.bias is not None and lin.bias.device.type != "meta":
        nn.init.zeros_(lin.bias)


def init_new_modules(model: CATYokoForCausalLM, std: float = NEW_MODULE_INIT_STD) -> None:
    """Small-scale random init for modules MiniCPM5 does not provide.

    Frozen spec: cross-attn, cache W_K/W_V, router. Skip on the meta device.
    """
    p0 = next(model.parameters(), None)
    if p0 is not None and p0.device.type == "meta":
        return
    _small_linear(model.cache_k, std)
    _small_linear(model.cache_v, std)
    for n, mod in model.named_modules():
        parts = n.split(".")
        if "indexer" in parts or "cross_indexer" in parts or "kda" in parts:
            for lin in mod.modules():
                if isinstance(lin, nn.Linear):
                    _small_linear(lin, std)
    for blk in model.decoder:
        cross = getattr(blk, "cross_attn", None)
        if cross is None:
            continue
        for name in ("q_proj", "o_proj"):
            lin = getattr(cross, name, None)
            if isinstance(lin, nn.Linear):
                _small_linear(lin, std)
    for blk in list(model.encoder) + list(model.decoder):
        router = getattr(getattr(blk, "mlp", None), "router", None)
        if isinstance(router, nn.Linear):
            _small_linear(router, std)


def _scale(n_experts: int, n_groups: int = 1, top_k: int = 1) -> float:
    return (n_experts * n_groups**2 / max(top_k, 1)) ** (1.0 / 3.0)


def _copy_exact(dst: nn.Linear, src: torch.Tensor, name: str) -> None:
    if tuple(dst.weight.shape) != tuple(src.shape):
        raise ValueError(
            f"{name}: expected MiniCPM5-2B GQA shape {tuple(dst.weight.shape)}, "
            f"got {tuple(src.shape)}. MiniCPM-2B MHA / other graphs are rejected."
        )
    dst.weight.data.copy_(src)


def _copy_linear(dst: nn.Linear, src: torch.Tensor) -> None:
    rows = min(dst.weight.shape[0], src.shape[0])
    cols = min(dst.weight.shape[1], src.shape[1])
    dst.weight.data.zero_()
    dst.weight.data[:rows, :cols] = src[:rows, :cols]


def upcycle_from_minicpm(
    model: CATYokoForCausalLM,
    src: Mapping[str, Any],
    cfg: CATYokoConfig | None = None,
) -> CATYokoForCausalLM:
    """Map MiniCPM5 Llama keys `layers.{i}.*` / `embed_tokens` into CAT-YOKO.

    MiniCPM5 dense SwiGLU 6144 / moe 2048 = 3 (exact groups). Each expert still
    takes the leading ``moe_intermediate_size`` rows, then scaled. Phase B
    recovers the rest. GQA K/V are ``(kv_dim, d)``. Attention, embedding, and
    final ``model.norm.weight`` copies are exact-shape; MiniCPM-2B (d=2304, MHA)
    cannot silently truncate.
    """
    cfg = cfg or model.cfg

    def _take(*keys: str) -> torch.Tensor | None:
        for key in keys:
            if key in src:
                return src[key]
        return None

    emb = _take("embed_tokens.weight", "model.embed_tokens.weight")
    if emb is not None:
        if tuple(emb.shape) != tuple(model.embed.weight.shape):
            raise ValueError(
                f"embed_tokens: expected MiniCPM5-2B {tuple(model.embed.weight.shape)}, "
                f"got {tuple(emb.shape)}"
            )
        model.embed.weight.data.copy_(emb)
    if not cfg.tie_embeddings:
        head = _take("lm_head.weight", "model.lm_head.weight")
        if head is not None:
            if tuple(head.shape) != tuple(model.lm_head.weight.shape):
                raise ValueError(
                    f"lm_head: expected MiniCPM5-2B {tuple(model.lm_head.weight.shape)}, "
                    f"got {tuple(head.shape)}"
                )
            model.lm_head.weight.data.copy_(head)
    norm = _take("model.norm.weight", "norm.weight")
    if norm is not None:
        if tuple(norm.shape) != tuple(model.norm.weight.shape):
            raise ValueError(
                f"norm: expected MiniCPM5-2B {tuple(model.norm.weight.shape)}, "
                f"got {tuple(norm.shape)}"
            )
        model.norm.weight.data.copy_(norm)

    def layer_prefix(i: int) -> str:
        for p in (f"model.layers.{i}.", f"layers.{i}."):
            if any(k.startswith(p) for k in src):
                return p
        return f"model.layers.{i}."

    def copy_attn(dst_attn: nn.Module, prefix: str) -> None:
        mapping = {
            "q_proj": "self_attn.q_proj.weight",
            "k_proj": "self_attn.k_proj.weight",
            "v_proj": "self_attn.v_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
        }
        for dst_name, src_name in mapping.items():
            key = prefix + src_name
            if key in src:
                _copy_exact(getattr(dst_attn, dst_name), src[key], key)

    def copy_ffn_to_moe(moe: nn.Module, prefix: str, top_k: int) -> None:
        gkey, ukey, dkey = (
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
            prefix + "mlp.down_proj.weight",
        )
        if gkey not in src:
            return
        want_up = (cfg.dense_intermediate_size, cfg.hidden_size)
        want_down = (cfg.hidden_size, cfg.dense_intermediate_size)
        if tuple(src[gkey].shape) != want_up or tuple(src[ukey].shape) != want_up:
            raise ValueError(
                f"{gkey}: expected MiniCPM5 dense SwiGLU {want_up}, "
                f"got gate={tuple(src[gkey].shape)} up={tuple(src[ukey].shape)}"
            )
        if tuple(src[dkey].shape) != want_down:
            raise ValueError(
                f"{dkey}: expected MiniCPM5 dense SwiGLU {want_down}, got {tuple(src[dkey].shape)}"
            )
        scale = _scale(moe.n_routed + moe.n_shared, top_k=top_k)
        for expert in list(moe.shared) + list(moe.experts):
            _copy_linear(expert.gate_proj, src[gkey] / scale)
            _copy_linear(expert.up_proj, src[ukey] / scale)
            # down is [hidden, intermediate]
            rows = min(expert.down_proj.weight.shape[0], src[dkey].shape[0])
            cols = min(expert.down_proj.weight.shape[1], src[dkey].shape[1])
            expert.down_proj.weight.data.zero_()
            expert.down_proj.weight.data[:rows, :cols] = src[dkey][:rows, :cols] / scale

    n_enc = cfg.encoder_layers
    for i, blk in enumerate(model.encoder):
        p = layer_prefix(i)
        copy_attn(blk.attn, p)
        if p + "input_layernorm.weight" in src:
            blk.ln1.weight.data.copy_(src[p + "input_layernorm.weight"])
        if p + "post_attention_layernorm.weight" in src:
            blk.ln2.weight.data.copy_(src[p + "post_attention_layernorm.weight"])
        copy_ffn_to_moe(blk.mlp, p, cfg.top_k_enc)

    for j, blk in enumerate(model.decoder):
        p = layer_prefix(n_enc + j)
        copy_attn(blk.self_attn, p)
        if p + "input_layernorm.weight" in src:
            blk.ln1.weight.data.copy_(src[p + "input_layernorm.weight"])
        if p + "post_attention_layernorm.weight" in src:
            blk.ln2.weight.data.copy_(src[p + "post_attention_layernorm.weight"])
        copy_ffn_to_moe(blk.mlp, p, cfg.top_k_dec)
    return model


def dummy_minicpm_state(cfg: CATYokoConfig) -> dict[str, torch.Tensor]:
    """Tiny dense teacher weights for tests (same hidden/vocab as cfg)."""
    d, v = cfg.hidden_size, cfg.vocab_size
    kv = cfg.kv_dim
    mid = cfg.dense_intermediate_size
    n = cfg.encoder_layers + cfg.decoder_layers
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(v, d) * 0.02,
        "lm_head.weight": torch.randn(v, d) * 0.02,
        "model.norm.weight": torch.randn(d),
    }
    for i in range(n):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(d, d) * 0.02
        sd[p + "self_attn.k_proj.weight"] = torch.randn(kv, d) * 0.02
        sd[p + "self_attn.v_proj.weight"] = torch.randn(kv, d) * 0.02
        sd[p + "self_attn.o_proj.weight"] = torch.randn(d, d) * 0.02
        sd[p + "mlp.gate_proj.weight"] = torch.randn(mid, d) * 0.02
        sd[p + "mlp.up_proj.weight"] = torch.randn(mid, d) * 0.02
        sd[p + "mlp.down_proj.weight"] = torch.randn(d, mid) * 0.02
        sd[p + "input_layernorm.weight"] = torch.ones(d)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(d)
    return sd


dummy_minicpm5_state = dummy_minicpm_state
upcycle_from_minicpm5 = upcycle_from_minicpm
