"""Phase G GRPO / DPO helpers. Dummy RLVR for ``--try``; no PDSA/PS-PPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cat_yoko.loss import dpo_loss, grpo_loss, masked_seq_logprob, token_logprobs


@torch.no_grad()
def dummy_rlvr_reward(prompt: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    """Verifiable-shaped dummy: needle-in-completion + token-parity.

    Last prompt token and the mid-prompt token both count (long-ctx shaped).
    """
    needle = prompt[:, -1]
    hit = (completion == needle.unsqueeze(-1)).any(dim=-1).float()
    mid = prompt[:, prompt.size(1) // 2]
    hit_mid = (completion == mid.unsqueeze(-1)).any(dim=-1).float()
    parity = (completion[:, 0] % 2).float()
    return hit + 0.5 * hit_mid + 0.25 * parity


def group_advantages(rewards: torch.Tensor, group: int) -> torch.Tensor:
    """``rewards`` is ``[B*G]``. Group-relative z-score, std clamped."""
    g = max(int(group), 1)
    n = rewards.numel()
    if n % g:
        raise ValueError(f"reward count {n} not divisible by group {g}")
    r = rewards.view(-1, g)
    mean = r.mean(dim=-1, keepdim=True)
    std = r.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((r - mean) / std).reshape(-1)


@torch.no_grad()
def sample_completions(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new: int,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Greedy / multinomial continuation. Tiny ``--try`` path; not a fast generate kernel."""
    ids = prompt
    was_train = model.training
    model.eval()
    try:
        for _ in range(max(int(max_new), 0)):
            out = model(input_ids=ids)
            logits = out["logits"][:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(dim=-1)
            else:
                nxt = torch.multinomial(
                    F.softmax(logits / max(temperature, 1e-5), dim=-1), 1
                ).squeeze(-1)
            ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
    finally:
        if was_train:
            model.train()
    return ids


def completion_logprob(
    model: nn.Module,
    ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum log π of completion tokens (ids after ``prompt_len``)."""
    out = model(input_ids=ids)
    if "logits" not in out:
        raise RuntimeError("GRPO/DPO need logits; set model.return_logits=True")
    lp = token_logprobs(out["logits"], ids)
    # token_lp[t] predicts ids[t+1]; completion starts at index prompt_len.
    mask = torch.zeros_like(lp, dtype=torch.bool)
    start = max(int(prompt_len) - 1, 0)
    mask[:, start:] = True
    return masked_seq_logprob(lp, mask)


def grpo_step_loss(
    model: nn.Module,
    prompt: torch.Tensor,
    completions: torch.Tensor,
    *,
    group: int,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, mean_reward)``. Completions are ``[B*G, Sc]``."""
    ids = torch.cat([prompt, completions], dim=1)
    logp = completion_logprob(model, ids, prompt_len)
    with torch.no_grad():
        reward = dummy_rlvr_reward(prompt, completions)
        adv = group_advantages(reward, group)
    return grpo_loss(logp, adv), reward.mean()


def dpo_step_loss(
    model: nn.Module,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    *,
    prompt_len: int,
    beta: float,
) -> torch.Tensor:
    pi_c = completion_logprob(model, chosen, prompt_len)
    pi_r = completion_logprob(model, rejected, prompt_len)
    return dpo_loss(pi_c, pi_r, pi_c.detach(), pi_r.detach(), beta=beta)
