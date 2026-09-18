"""Map CATYokoConfig → Megatron-Core TransformerConfig dicts (no megatron import).

Field names follow NVIDIA Megatron-LM ``TransformerConfig`` (Core ≥0.12;
CSA / ``sqrtsoftplus`` fields exist on current main, ~0.19). Unknown keys
must be filtered when constructing a live config (see provider.py).
"""

from __future__ import annotations

from cat_yoko.config import (
    CATYokoConfig,
    C1_SPLIT,
    KEEP_HIGH_PREC,
    encoder_layer_kind,
)
from cat_yoko.nvfp4 import policy_for
from cat_yoko.parallel import ParallelPlan, validate_parallel

MEGATRON_LM = "https://github.com/NVIDIA/Megatron-LM"

# Custom surface a Megatron GPTModel / T5Model cannot provide.
MEGATRON_CUSTOM_SURFACE = (
    "two TransformerConfigs (encoder top-k=7 vs decoder top-k=10)",
    "causal encoder; do not use T5 bidirectional encoder",
    "GQA 16 Q / 2 KV; cache W_K/W_V is d→kv_dim (not per-layer 4d² K/V)",
    "cache.detach() in B0/B1 (Theorem D)",
    "scheduled gated cross-attn (buffer, not AdamW)",
    "no MiniCPM μP (scale_emb=1, residual_scale=1, logit_scale=1)",
    "untied embeddings: freeze embed B0/B1; freeze lm_head in B0; train lm_head in B1",
    "Hash-MoE on decoder first 2 layers",
    "EP in {1,2,4,5,10,20} because n_routed=20",
    "do not instantiate megatron.core.models.gpt.GPTModel for this graph",
)


def moe_layer_freq(n_layers: int, first_dense: bool) -> list[int]:
    if n_layers <= 0:
        return []
    if first_dense:
        return [0] + [1] * (n_layers - 1)
    return [1] * n_layers


def encoder_csa_compress_ratios(cfg: CATYokoConfig) -> list[int]:
    """Megatron CSA ratios must be 0 (sliding), 4 (CSA), or 128 (HCA). Matches m / m'."""
    out: list[int] = []
    for i in range(cfg.encoder_layers):
        kind = encoder_layer_kind(i)
        if kind == "sliding":
            out.append(0)
        elif kind == "csa":
            out.append(cfg.compress_m)  # 4
        else:
            out.append(cfg.compress_m_hca)  # 128
    return out


def _prec_fields(phase: str, cfg: CATYokoConfig, *, stack: str) -> dict:
    """Published NVFP4 + Hopper FP8 fallback. B0 student stays bf16.

    Frozen-encoder forward is still an NVFP4 GEMM slot even when the
    decoder student is bf16 (B0 / Phase C).
    """
    pol = policy_for(phase)
    nvfp4_on = bool(cfg.use_nvfp4)
    fp8_on = bool(cfg.use_fp8)
    encoder_low = pol.frozen_encoder_gemm in {"nvfp4", "fp8"}
    student_low = pol.student in {"nvfp4", "nvfp4_moe", "fp8", "fp8_moe"}
    if stack == "encoder":
        low = student_low if pol.frozen_encoder_gemm == "n/a" else encoder_low
    else:
        low = student_low
    if not low or not (nvfp4_on or fp8_on):
        return {"fp8": None, "nvfp4": False, "bf16": True}
    out = {"bf16": True, "nvfp4": bool(nvfp4_on and low)}
    if nvfp4_on and low:
        out["nvfp4_recipe"] = "te_nvfp4"
    if fp8_on and low:
        out["fp8"] = "hybrid"
        out["fp8_recipe"] = "delayed"
    else:
        out["fp8"] = None
    return out


def transformer_config_dict(
    cfg: CATYokoConfig,
    stack: str,
    plan: ParallelPlan,
    *,
    phase: str,
) -> dict:
    """One Megatron TransformerConfig-shaped dict for encoder or decoder."""
    if stack == "encoder":
        n_layers = cfg.encoder_layers
        top_k = cfg.top_k_enc
        n_routed = cfg.n_routed_enc
        window = [cfg.n_win, 0]
    elif stack == "decoder":
        n_layers = cfg.decoder_layers
        top_k = cfg.top_k_dec
        n_routed = cfg.n_routed_dec
        window = [cfg.n_win, 0]
    else:
        raise ValueError(stack)

    prec = _prec_fields(phase, cfg, stack=stack)
    return {
        "num_layers": n_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_heads,
        "num_query_groups": cfg.num_kv_heads,
        "kv_channels": cfg.head_dim,
        "ffn_hidden_size": cfg.dense_intermediate_size,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "normalization": "RMSNorm",
        "layernorm_epsilon": cfg.rms_eps,
        "qk_layernorm": cfg.qk_norm,
        "gated_linear_unit": True,
        "add_bias_linear": False,
        "add_qkv_bias": False,
        "position_embedding_type": "rope",
        "rotary_base": cfg.rope_theta,
        "rotary_interleaved": False,
        "window_size": window,
        "mtp_num_layers": None,
        "num_moe_experts": n_routed,
        "moe_ffn_hidden_size": cfg.moe_intermediate_size,
        "moe_shared_expert_intermediate_size": cfg.n_shared * cfg.moe_intermediate_size,
        "moe_shared_expert_gate": False,
        "moe_router_topk": top_k,
        "moe_router_pre_softmax": True,
        "moe_router_score_function": "sqrtsoftplus",
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 1e-3,
        "moe_aux_loss_coeff": cfg.seq_balance_loss,
        "moe_z_loss_coeff": cfg.router_z_loss,
        "moe_router_dtype": "fp32",
        "moe_layer_freq": moe_layer_freq(n_layers, cfg.first_dense),
        "tensor_model_parallel_size": plan.tensor_parallel,
        "pipeline_model_parallel_size": plan.pipeline_parallel,
        "expert_model_parallel_size": plan.expert_parallel,
        "context_parallel_size": plan.context_parallel,
        "sequence_parallel": plan.sequence_parallel,
        "variable_seq_lengths": True,
        **prec,
    }


def yoco_extras(cfg: CATYokoConfig, phase: str) -> dict:
    return {
        "architecture": "yoco_causal_encoder_decoder",
        "detach_cache": phase != "B2",
        "gate_schedule": {"B0": [0.0, 0.3], "B1": [0.3, 1.0], "B2": [1.0, 1.0]}[phase],
        "scale_emb": cfg.embed_scale,
        "residual_scale": cfg.residual_scale,
        "logit_scale": cfg.logit_scale,
        "tied_embeddings": cfg.tie_embeddings,
        "hash_moe_decoder_layers": cfg.hash_moe_decoder_layers,
        "hash_moe_encoder": False,
        "first_dense": cfg.first_dense,
        "attention_backend": cfg.attention_backend,
        "fp8_keep_high_prec": list(KEEP_HIGH_PREC),
        "nvfp4_keep_high_prec": list(KEEP_HIGH_PREC),
        "c1_split_tokens": dict(C1_SPLIT),
        "phase_c_csa": {
            "experimental_attention_variant": "dsv4_hybrid",
            "csa_window_size": cfg.n_win,
            "csa_compress_ratios": encoder_csa_compress_ratios(cfg),
            "csa_dense_mode": False,
            "dsa_indexer_topk": cfg.index_topk,
            "note": "Phase C only. Phase B keeps csa_dense_mode conceptually on (window GQA).",
        },
        "custom_surface": list(MEGATRON_CUSTOM_SURFACE),
    }


def megatron_training_args(cfg: CATYokoConfig, plan: ParallelPlan, phase: str) -> dict:
    seq = cfg.seq_len
    gbs = cfg.global_batch_tokens // seq
    warmup_iters = max(int(cfg.warmup_tokens // cfg.global_batch_tokens), 1)
    return {
        "seq_length": seq,
        "max_position_embeddings": seq,
        "global_batch_size": int(gbs),
        "lr": cfg.lr_b2 if phase == "B2" else cfg.lr,
        "min_lr": 0.0,
        "lr_warmup_tokens": int(cfg.warmup_tokens),
        "lr_warmup_iters": warmup_iters,
        "clip_grad": cfg.grad_clip,
        "weight_decay": cfg.weight_decay,
        "adam_beta1": cfg.adam_beta1,
        "adam_beta2": cfg.adam_beta2,
        "bf16": True,
        "use_distributed_optimizer": True,
        "tokenizer_type": "NullTokenizer",
        "vocab_size": cfg.vocab_size,
        "untie_embeddings_and_output_weights": not cfg.tie_embeddings,
        "tensor_model_parallel_size": plan.tensor_parallel,
        "pipeline_model_parallel_size": plan.pipeline_parallel,
        "expert_model_parallel_size": plan.expert_parallel,
        "context_parallel_size": plan.context_parallel,
        "sequence_parallel": plan.sequence_parallel,
        "pipeline_split_rank": plan.pipeline_split_rank
        if plan.pipeline_split_rank is not None
        else (cfg.encoder_layers if plan.pipeline_parallel > 1 else None),
        "extra_args_provider": "cat_yoko.megatron.provider.add_cat_yoko_args",
        "model_type": "encoder_and_decoder",
        "optimizer": "adam",
        "use_muon": cfg.use_muon,
    }


def megatron_blueprint(
    cfg: CATYokoConfig,
    plan: ParallelPlan | None = None,
    *,
    phase: str = "B0",
) -> dict:
    """JSON-serialisable mapping a later Megatron pretrain script consumes."""
    plan = plan or ParallelPlan()
    validate_parallel(cfg, plan)
    return {
        "upstream": MEGATRON_LM,
        "phase": phase,
        "encoder": transformer_config_dict(cfg, "encoder", plan, phase=phase),
        "decoder": transformer_config_dict(cfg, "decoder", plan, phase=phase),
        "yoco": yoco_extras(cfg, phase),
        "parallel": plan.as_dict(),
        "training": megatron_training_args(cfg, plan, phase),
    }
