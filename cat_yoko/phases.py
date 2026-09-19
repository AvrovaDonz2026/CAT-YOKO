"""Published C1 Phase B launch specs. B0 / B1 / B2 are separate envelopes.

Wall-clock target is C1+NVFP4 (571 H100-h for 8+27+15B tokens; RTX PRO
6000 / 6000D). A 32GB card can run the same graph and freeze curriculum;
it cannot finish the token envelopes and cannot run NVFP4. ``--try``
writes a real (short) trainable overlay for HuggingFace Hub.
"""

from __future__ import annotations

from dataclasses import dataclass

from cat_yoko.config import C1_SPLIT


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    tokens: float
    student: str
    detach: bool
    gate_start: float
    gate_end: float
    offload_encoder: bool
    offload_blocks: bool
    optim_cpu: bool
    notes: str


PHASES: dict[str, PhaseSpec] = {
    "B0": PhaseSpec(
        name="B0",
        tokens=C1_SPLIT["B0"],
        student="bf16",
        detach=True,
        gate_start=0.0,
        gate_end=0.3,
        offload_encoder=True,
        offload_blocks=False,
        optim_cpu=False,
        notes="new modules only: cross-attn, cache_k/v, ln_cross; freeze encoder+embed+lm_head+final RMSNorm",
    ),
    "B1": PhaseSpec(
        name="B1",
        tokens=C1_SPLIT["B1"],
        student="nvfp4",
        detach=True,
        gate_start=0.3,
        gate_end=1.0,
        offload_encoder=True,
        offload_blocks=False,
        optim_cpu=True,
        notes="train decoder stack + untied lm_head + final RMSNorm; encoder+embed frozen",
    ),
    "B2": PhaseSpec(
        name="B2",
        tokens=C1_SPLIT["B2"],
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="full model; per-block CPU offload; accum=1",
    ),
}


# Hub shard cap. 4GiB keeps autodl-tmp / copy tools comfortable.
HUB_SHARD_MAX_BYTES = 4 * (1 << 30)
TIGHT_GPU_SEQ = 64
TRY_STEPS = 32
# Published envelope on a rental GPU: write trainable.pt often. 0 would only
# save at the end of 8/27/15B tokens, which this box will not reach.
PUBLISHED_SAVE_EVERY = 20
PUBLISHED_SEQ = 4096  # CATYokoConfig.middle_12b().seq_len


def spec(phase: str) -> PhaseSpec:
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase}; choose B0/B1/B2")
    return PHASES[phase]
