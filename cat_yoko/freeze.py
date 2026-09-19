"""C1 freeze boundaries (Theorems D/E).

Duck-typed for the torch graph and a future MegatronModule: ``encoder``,
``decoder``, ``embed``, ``norm``, ``cache_k`` / ``cache_v``, ``set_detach``,
and ``decoder[i].{self_attn, mlp, ln1, ln2, cross_attn, ln_cross, gate}``.
"""

from __future__ import annotations

from cat_yoko.config import C1_SPLIT
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import unwrap


def set_gate(model: CATYokoForCausalLM, value: float) -> None:
    """Scheduled mix, not an AdamW parameter (Theorem A)."""
    model = unwrap(model)
    for blk in model.decoder:
        unwrap(blk).gate.fill_(float(value))


def gate_schedule(phase: str, progress: float) -> float:
    """progress in [0, 1] within the phase."""
    p = min(max(progress, 0.0), 1.0)
    if phase == "B0":
        return 0.3 * p
    if phase == "B1":
        return 0.3 + 0.7 * p
    return 1.0


def _b0_new_module_params(model: CATYokoForCausalLM):
    """YOCO new modules: cache projections and decoder cross-attn / ln_cross."""
    yield from model.cache_k.parameters()
    yield from model.cache_v.parameters()
    for blk in model.decoder:
        blk = unwrap(blk)
        yield from blk.cross_attn.parameters()
        yield from blk.ln_cross.parameters()


def _clear_trainable_cache(model: CATYokoForCausalLM) -> None:
    for m in model.modules():
        if hasattr(m, "_has_trainable_params"):
            delattr(m, "_has_trainable_params")


def apply_freeze(model: CATYokoForCausalLM, phase: str) -> None:
    if phase not in {"B0", "B1", "B2"}:
        raise ValueError(phase)
    model = unwrap(model)
    for p in model.parameters():
        p.requires_grad = True
    model.set_detach(phase != "B2")
    if phase == "B2":
        _clear_trainable_cache(model)
        return
    # Freeze encoder + input embedding (Theorem E). Untied lm_head is frozen
    # in B0 (new-modules only) and trained in B1 (does not drift X^0).
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.embed.weight.requires_grad = False
    if model.cfg.tie_embeddings or phase == "B0":
        model.lm_head.weight.requires_grad = False
    if phase == "B0":
        for blk in model.decoder:
            blk = unwrap(blk)
            for p in blk.self_attn.parameters():
                p.requires_grad = False
            for p in blk.mlp.parameters():
                p.requires_grad = False
            blk.ln1.weight.requires_grad = False
            blk.ln2.weight.requires_grad = False
            # cross_attn, ln_cross stay trainable; cache_k/v stay trainable.
            # gate is a scheduled buffer, not a Parameter.
        # MiniCPM5 final RMSNorm is backbone leftover, not a new module.
        # B1 trains it with the decoder stack; B2 already unfreezes everything.
        model.norm.weight.requires_grad = False
        keep = {id(p) for p in _b0_new_module_params(model)}
        for p in model.parameters():
            if id(p) not in keep:
                p.requires_grad = False
    # B1: decoder stack (self-attn, mlp, ln1/ln2, final norm) stays trainable.
    _clear_trainable_cache(model)


def trainable_names(model: CATYokoForCausalLM) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def tokens_for_phase(phase: str) -> float:
    return C1_SPLIT[phase]
