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
    "C-index": Fp8Policy("C-index", "bf16", "fp8"),
    "C-kda": Fp8Policy("C-kda", "fp8", "n/a"),
    "C-topk": Fp8Policy("C-topk", "fp8", "n/a"),
    "C-hca": Fp8Policy("C-hca", "fp8", "n/a"),
    "C-win": Fp8Policy("C-win", "fp8", "n/a"),
    "D-8k": Fp8Policy("D-8k", "fp8", "n/a"),
    "D-32k": Fp8Policy("D-32k", "fp8", "n/a"),
    "D-128k": Fp8Policy("D-128k", "fp8", "n/a"),
    "E": Fp8Policy("E", "fp8", "n/a"),
    "F": Fp8Policy("F", "fp8", "n/a"),
    "G": Fp8Policy("G", "fp8", "n/a"),
    "G-dpo": Fp8Policy("G-dpo", "fp8", "n/a"),
}


def policy_for(phase: str) -> Fp8Policy:
    from cat_yoko.nvfp4 import policy_key

    return POLICY[policy_key(phase) if phase not in POLICY else phase]


def should_fp8_autocast(phase: str, *, cuda: bool, enabled: bool) -> bool:
    if not enabled or not cuda:
        return False
    return POLICY[phase].student in LOW_PREC_STUDENTS
