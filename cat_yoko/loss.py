"""CE is in the model; KD is optional when a MiniCPM5 teacher is present."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_CE_CHUNK_CACHE: dict[tuple[str, int, int], int] = {}


def _ce_chunk_tokens(
    n_tok: int,
    vocab: int,
    requested: int | None,
    device: torch.device,
) -> int:
    """Chunk size for lm_head×CE. ``None`` = fit free HBM; small GPUs stay at 512."""
    if requested is not None:
        return max(int(requested), 1)
    if n_tok <= 512:
        return max(n_tok, 1)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 512
    key = (str(device), int(vocab), int(n_tok))
    hit = _CE_CHUNK_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        _CE_CHUNK_CACHE[key] = 512
        return 512
    # fp32 logits are ``chunk * V * 4`` bytes; 8× slack leaves room for dlogits.
    max_chunk = int(free // (32 * max(int(vocab), 1)))
    step = min(n_tok, max(512, max_chunk))
    _CE_CHUNK_CACHE[key] = step
    return step


def linear_cross_entropy(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    logit_scale: float = 1.0,
    ignore_index: int = -100,
    chunk_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-mean CE without materializing full ``[B, S, V]`` logits.

    Same reduction as ``F.cross_entropy(..., ignore_index)``: mean over
    non-ignored tokens. Chunks along the token axis so V=130560 at seq=4096
    never allocates a 2GiB logit tensor on a 32GB card. B200 has enough
    free HBM to do the whole 4095-token GEMM in one shot.
    """
    h = hidden.reshape(-1, hidden.size(-1))
    y = labels.reshape(-1)
    valid = y != ignore_index
    n_valid = valid.sum()
    total = h.new_zeros(())
    vocab = int(getattr(lm_head, "out_features", h.size(-1)))
    step = _ce_chunk_tokens(h.size(0), vocab, chunk_tokens, h.device)
    for i in range(0, h.size(0), step):
        logits = lm_head(h[i : i + step])
        if logit_scale != 1.0:
            logits = logits / logit_scale
        # CUDA CE softmax-accum is fp32 on fp16/bf16 logits. Skip a V-wide
        # ``.float()`` copy (chunk×130560×4). CPU non-fp32 still upcasts.
        if logits.dtype != torch.float32 and not logits.is_cuda:
            logits = logits.float()
        total = total + F.cross_entropy(
            logits,
            y[i : i + step],
            ignore_index=ignore_index,
            reduction="sum",
        )
    denom = n_valid.clamp_min(1).to(dtype=total.dtype)
    nll = total / denom
    return nll, n_valid


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


def kd_weight(
    step: int,
    steps: int | None,
    start: float,
    *,
    tokens_in_phase: float = 0.0,
    phase_budget: float | None = None,
) -> float:
    """Linear decay from ``start`` → 0 over the phase.

    Prefer ``(step+1)/steps``. When ``steps`` is None (``--tokens`` without
    ``--steps``), use ``tokens_in_phase/phase_budget`` like ``gate_schedule``.
    """
    if steps is not None:
        if steps <= 1:
            return 0.0
        progress = (step + 1) / steps
    elif phase_budget is not None and phase_budget > 1:
        progress = min((tokens_in_phase + 1) / phase_budget, 1.0)
    else:
        return 0.0
    return start * max(1.0 - progress, 0.0)


def safe_ppl(nll: float) -> float | None:
    """Token PPL from CE nats. None when NLL is not a useful finite value."""
    if not math.isfinite(nll) or nll < 0 or nll > 20:
        return None
    return math.exp(nll)


def response_only_labels(ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Ignore CE on the prompt prefix. ``prompt_len`` is along the last dim."""
    labels = ids.clone()
    cut = max(int(prompt_len), 0)
    if cut <= 0:
        return labels
    labels[..., :cut] = -100
    return labels


def token_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per-token log p(id_t | ctx_<t) from ``logits[:, :-1]`` vs ``ids[:, 1:]``."""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:].unsqueeze(-1)
    return logp.gather(-1, tgt).squeeze(-1)


def masked_seq_logprob(token_lp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum logprobs where ``mask`` is True. ``token_lp`` is ``[B, S-1]``."""
    m = mask.to(dtype=token_lp.dtype)
    return (token_lp * m).sum(dim=-1)


def grpo_loss(seq_logprob: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Token-sum GRPO: ``-E[A * log π]``. ``advantages`` are stop-grad."""
    return -(seq_logprob * advantages.detach()).mean()


def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logits = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean()
