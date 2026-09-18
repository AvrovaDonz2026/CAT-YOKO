"""C1+FP8 module policy. Compute dtype only; master weights stay bf16/fp32.

True FP8 GEMM needs Transformer Engine on CUDA. This repo records the published
policy and falls back to bf16 autocast (or plain bf16 on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass

from cat_yoko.config import FP8_KEEP_HIGH_PREC


@dataclass(frozen=True)
class Fp8Policy:
    phase: str
    student: str
    frozen_encoder_gemm: str


POLICY = {
    "L0": Fp8Policy("L0", "bf16", "bf16"),
    "B0": Fp8Policy("B0", "bf16", "fp8"),
    "B1": Fp8Policy("B1", "fp8_moe", "fp8"),
    "B2": Fp8Policy("B2", "fp8_moe", "n/a"),
    "C": Fp8Policy("C", "bf16", "fp8"),
}


def policy_for(phase: str) -> Fp8Policy:
    return POLICY[phase]


def should_autocast(phase: str, *, cuda: bool, enabled: bool) -> bool:
    if not enabled or not cuda:
        return False
    return POLICY[phase].student in {"fp8", "fp8_moe"}


KEEP_HIGH_PREC = FP8_KEEP_HIGH_PREC
