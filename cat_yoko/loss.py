"""CE is in the model; KD is optional when a MiniCPM5 teacher is present."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def kd_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    ignore: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-mean KL. ``ignore`` is shifted labels; ``-100`` drops packed boundaries.

    Flatten to ``[N, V]`` before ``kl_div``. 3D ``batchmean`` would divide by B
    only and scale the KD term by sequence length.
    """
    t = temperature
    log_p = F.log_softmax(student_logits.float() / t, dim=-1)
    q = F.softmax(teacher_logits.float() / t, dim=-1)
    token_kl = F.kl_div(
        log_p.reshape(-1, log_p.size(-1)),
        q.reshape(-1, q.size(-1)),
        reduction="none",
    ).sum(dim=-1) * (t * t)
    if ignore is not None:
        mask = ignore.reshape(-1) != -100
        if not bool(mask.any()):
            return student_logits.reshape(-1)[:1].sum() * 0
        return token_kl[mask].mean()
    return token_kl.mean()


def kd_weight(step: int, steps: int, start: float) -> float:
    if steps <= 1:
        return 0.0
    return start * max(1.0 - (step + 1) / steps, 0.0)


def safe_ppl(nll: float) -> float | None:
    """Token PPL from CE nats. None when NLL is not a useful finite value."""
    if not math.isfinite(nll) or nll < 0 or nll > 20:
        return None
    return math.exp(nll)
