"""Published C1+NVFP4 compute policy. Master weights stay bf16.

Anything that is not a must-bf16/fp32 op is an NVFP4 linear GEMM on
Blackwell. B200 / SM 10.0 / 10.3 wrap ``TeNvfp4Linear`` (Transformer Engine
``NVFP4BlockScaling`` default recipe). sm_120 / CPU use ``Nvfp4Linear``
E2M1/16 emulation. Outer ``torch.autocast(bf16)`` still covers unwrapped
ops. Attn softmax / SDPA stay fp32 in ``attention._sdpa``.

Must-high-prec (not NVFP4): embed lookup, RMSNorm / QK-Norm, router,
scalar gate, indexer, attn softmax / SDPA scores. B0 student stays
bf16 (Theorem A). lm_head and attn QKV/O projections are NVFP4 GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass

from cat_yoko.config import KEEP_HIGH_PREC, NVFP4_GEMM_SLOTS

LOW_PREC_STUDENTS = frozenset({"nvfp4", "nvfp4_moe", "fp8", "fp8_moe"})


@dataclass(frozen=True)
class Nvfp4Policy:
    phase: str
    student: str
    frozen_encoder_gemm: str


POLICY = {
    "L0": Nvfp4Policy("L0", "bf16", "bf16"),
    "B0": Nvfp4Policy("B0", "bf16", "nvfp4"),
    "B1": Nvfp4Policy("B1", "nvfp4", "nvfp4"),
    "B2": Nvfp4Policy("B2", "nvfp4", "n/a"),
    "C": Nvfp4Policy("C", "bf16", "nvfp4"),
}


def policy_for(phase: str) -> Nvfp4Policy:
    return POLICY[phase]


def should_autocast(phase: str, *, cuda: bool, enabled: bool) -> bool:
    """Outer bf16 autocast for B1/B2. Wrapped GEMMs quantize inside forward."""
    if not enabled or not cuda:
        return False
    return POLICY[phase].student in LOW_PREC_STUDENTS


def low_prec_enabled(cfg) -> bool:
    return bool(getattr(cfg, "use_nvfp4", False) or getattr(cfg, "use_fp8", False))


def is_nvfp4_gemm(slot: str) -> bool:
    return slot in NVFP4_GEMM_SLOTS


GEMM_SLOTS = NVFP4_GEMM_SLOTS

__all__ = [
    "GEMM_SLOTS",
    "KEEP_HIGH_PREC",
    "LOW_PREC_STUDENTS",
    "Nvfp4Policy",
    "POLICY",
    "is_nvfp4_gemm",
    "low_prec_enabled",
    "policy_for",
    "should_autocast",
]
