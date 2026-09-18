"""Hopper/Ada FP8 fallback. Published compute dtype is NVFP4 (cat_yoko.nvfp4).

True FP8 GEMM needs Transformer Engine on CUDA. This module keeps the
fallback policy object; autocast still goes through nvfp4.should_autocast.
"""

from __future__ import annotations

from dataclasses import dataclass

from cat_yoko.config import FP8_KEEP_HIGH_PREC
from cat_yoko.nvfp4 import LOW_PREC_STUDENTS
from cat_yoko.nvfp4 import should_autocast as should_autocast

KEEP_HIGH_PREC = FP8_KEEP_HIGH_PREC


@dataclass(frozen=True)
class Fp8Policy:
    phase: str
    student: str
    frozen_encoder_gemm: str


# Fallback only. Published B1/B2 student is nvfp4 (all allowed GEMMs).
POLICY = {
    "L0": Fp8Policy("L0", "bf16", "bf16"),
    "B0": Fp8Policy("B0", "bf16", "fp8"),
    "B1": Fp8Policy("B1", "fp8", "fp8"),
    "B2": Fp8Policy("B2", "fp8", "n/a"),
    "C": Fp8Policy("C", "bf16", "fp8"),
}


def policy_for(phase: str) -> Fp8Policy:
    return POLICY[phase]


def should_fp8_autocast(phase: str, *, cuda: bool, enabled: bool) -> bool:
    if not enabled or not cuda:
        return False
    return POLICY[phase].student in LOW_PREC_STUDENTS
