"""CE is in the model; KD is optional when a MiniCPM teacher is present."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    t = temperature
    log_p = F.log_softmax(student_logits.float() / t, dim=-1)
    q = F.softmax(teacher_logits.float() / t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (t * t)


def kd_weight(step: int, steps: int, start: float) -> float:
    if steps <= 1:
        return 0.0
    return start * max(1.0 - (step + 1) / steps, 0.0)


def safe_ppl(nll: float) -> float | None:
    """Token PPL from CE nats. None when NLL is not a useful finite value."""
    if not math.isfinite(nll) or nll < 0 or nll > 20:
        return None
    return math.exp(nll)
