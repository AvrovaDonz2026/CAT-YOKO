"""Cluster parallelism. Architecture stays in CATYokoConfig; this is runtime-only.

Legal defaults for CAT-YOKO-12B (d=2048, 16 Q / 2 KV GQA, 20 routed experts):
- tensor parallel must divide 16 heads and 2048 → {1,2,4,8,16}
- expert parallel must divide 20 → {1,2,4,5,10,20}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cat_yoko.config import CATYokoConfig


@dataclass(frozen=True)
class ParallelPlan:
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: bool = False
    # PP rank after which the decoder stack starts. None → encoder_layers if PP>1.
    pipeline_split_rank: int | None = None

    def as_dict(self) -> dict[str, int | bool | None]:
        return asdict(self)


def legal_tensor_parallel(num_heads: int, hidden: int) -> tuple[int, ...]:
    return tuple(i for i in range(1, num_heads + 1) if num_heads % i == 0 and hidden % i == 0)


def legal_expert_parallel(n_routed: int) -> tuple[int, ...]:
    return tuple(i for i in range(1, n_routed + 1) if n_routed % i == 0)


def validate_parallel(cfg: CATYokoConfig, plan: ParallelPlan) -> None:
    tp_ok = legal_tensor_parallel(cfg.num_heads, cfg.hidden_size)
    if plan.tensor_parallel not in tp_ok:
        raise ValueError(f"TP={plan.tensor_parallel} illegal; need a divisor of heads={cfg.num_heads}: {tp_ok}")
    for n_routed, name in ((cfg.n_routed_enc, "encoder"), (cfg.n_routed_dec, "decoder")):
        ep_ok = legal_expert_parallel(n_routed)
        if plan.expert_parallel not in ep_ok:
            raise ValueError(
                f"EP={plan.expert_parallel} does not divide {name} routed experts {n_routed}; legal={ep_ok}"
            )
    if plan.pipeline_parallel < 1 or plan.context_parallel < 1:
        raise ValueError("PP and CP must be >= 1")
    if plan.sequence_parallel and plan.tensor_parallel == 1:
        raise ValueError("sequence_parallel requires TP>1")
    if plan.pipeline_split_rank is not None:
        if not (0 < plan.pipeline_split_rank < cfg.encoder_layers + cfg.decoder_layers):
            raise ValueError("pipeline_split_rank must sit inside the 16+26 stack")
