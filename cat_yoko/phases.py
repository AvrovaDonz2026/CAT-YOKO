"""Published launch specs: C1 Phase B (B0/B1/B2) plus C–G training envelopes.

Wall-clock target for B is C1+NVFP4 (571 H100-h). C–G reuse the same trainer:
window GQA (no CSA CUDA kernel), overlay checkpoints, ``--try`` 32 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    seq_len: int = 4096
    freeze: str = "b2"  # b0 | b1 | b2 | indexer | none
    loss: str = "ce"  # ce | indexer_kl | sft | grpo | dpo
    lr_mode: str = "stable"  # stable | b2 | decay
    sparse: str = "window"  # window | kda | topk | hca
    align_indexer: bool = False
    tokens_offset: float = 0.0
    default_steps: int | None = None


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
        freeze="b0",
        lr_mode="stable",
        tokens_offset=0.0,
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
        freeze="b1",
        lr_mode="stable",
        tokens_offset=C1_SPLIT["B0"],
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
        freeze="b2",
        lr_mode="b2",
        tokens_offset=C1_SPLIT["B0"] + C1_SPLIT["B1"],
    ),
    # Phase C: 25e9 inside the published 20–50B band. No CSA CUDA kernel.
    # Default chain (use_kda=False): indexer → topk → hca → win (2+7+7).
    # Implement-then-light: --use-kda on B builds 3:1 + KDAGates (still window);
    # C then lights C-kda first, CSA/HCA later. Do not add KDA modules at C.
    "C-kda": PhaseSpec(
        name="C-kda",
        tokens=5e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes=(
            "light KDA-kind layers only; CSA/HCA-kind stay window. "
            "hybrid-linear: majority linear path before strong retrieval. "
            "Opt-in use_kda; not a DPLR CUDA kernel"
        ),
        freeze="none",
        loss="ce",
        lr_mode="b2",
        sparse="kda",
        tokens_offset=50e9,
    ),
    "C-index": PhaseSpec(
        name="C-index",
        tokens=10e9,
        student="bf16",
        detach=True,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=False,
        optim_cpu=False,
        notes="freeze backbone; bf16 Lightning Indexer layer-internal KL vs dense window attn",
        freeze="indexer",
        loss="indexer_kl",
        lr_mode="stable",
        sparse="window",
        align_indexer=True,
        tokens_offset=50e9,
    ),
    "C-topk": PhaseSpec(
        name="C-topk",
        tokens=5e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="adapt backbone to indexer top-k mask on CSA-labeled layers (PyTorch SDPA mask, not a CSA kernel)",
        freeze="none",
        loss="ce",
        lr_mode="b2",
        sparse="topk",
        tokens_offset=60e9,
    ),
    "C-hca": PhaseSpec(
        name="C-hca",
        tokens=5e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="HCA-labeled encoder layers mean-pool KV (m'=128); CSA layers keep top-k mask",
        freeze="none",
        loss="ce",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=65e9,
    ),
    "C-win": PhaseSpec(
        name="C-win",
        tokens=5e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="8K window + CSA top-k + HCA concat (theorem B union); seq=8192",
        seq_len=8192,
        freeze="none",
        loss="ce",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=70e9,
    ),
    "D-8k": PhaseSpec(
        name="D-8k",
        tokens=10e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="long-context 8K; keep C-win lighting (CSA top-k + HCA); MiniCPM5 rope_theta=5e6",
        seq_len=8192,
        freeze="none",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=75e9,
    ),
    "D-32k": PhaseSpec(
        name="D-32k",
        tokens=15e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="long-context 32K; sparse stays hca after C",
        seq_len=32768,
        freeze="none",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=85e9,
    ),
    "D-128k": PhaseSpec(
        name="D-128k",
        tokens=15e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="long-context 128K (MiniCPM5 native); sparse stays hca after C",
        seq_len=131072,
        freeze="none",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=100e9,
    ),
    "E": PhaseSpec(
        name="E",
        tokens=20e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="WSD decay to ~1/100 peak LR; HQ math/code/long/instruction mix",
        seq_len=8192,
        freeze="none",
        lr_mode="decay",
        sparse="hca",
        tokens_offset=115e9,
    ),
    "F": PhaseSpec(
        name="F",
        tokens=1e9,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="SFT; loss on response tokens only (prompt labels = -100)",
        seq_len=8192,
        freeze="none",
        loss="sft",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=135e9,
    ),
    "G": PhaseSpec(
        name="G",
        tokens=0.0,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="GRPO; dummy RLVR reward on --try; steps-based (not a token envelope)",
        freeze="none",
        loss="grpo",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=136e9,
        default_steps=10_000,
    ),
    "G-dpo": PhaseSpec(
        name="G-dpo",
        tokens=0.0,
        student="nvfp4",
        detach=False,
        gate_start=1.0,
        gate_end=1.0,
        offload_encoder=False,
        offload_blocks=True,
        optim_cpu=True,
        notes="DPO preference scaffold; stop-grad current policy as reference unless --teacher",
        freeze="none",
        loss="dpo",
        lr_mode="b2",
        sparse="hca",
        tokens_offset=136e9,
        default_steps=10_000,
    ),
}

C_STAGES = {
    "kda": "C-kda",
    "indexer": "C-index",
    "topk": "C-topk",
    "hca": "C-hca",
    "win": "C-win",
}
D_STAGES = {
    "8k": "D-8k",
    "32k": "D-32k",
    "128k": "D-128k",
}
G_ALGOS = {
    "grpo": "G",
    "dpo": "G-dpo",
}
C_CHAIN = ("C-index", "C-topk", "C-hca", "C-win")
C_CHAIN_KDA = ("C-kda", "C-index", "C-topk", "C-hca", "C-win")
D_CHAIN = ("D-8k", "D-32k", "D-128k")
_POST_C = frozenset({"D-8k", "D-32k", "D-128k", "E", "F", "G", "G-dpo"})

# Hub shard cap. 4GiB keeps autodl-tmp / copy tools comfortable.
HUB_SHARD_MAX_BYTES = 4 * (1 << 30)
TIGHT_GPU_SEQ = 64
TRY_STEPS = 32
# Published envelope on a rental GPU: write trainable.pt often. 0 would only
# save at the end of 8/27/15B tokens, which this box will not reach.
PUBLISHED_SAVE_EVERY = 20
PUBLISHED_SEQ = 4096  # CATYokoConfig.middle_12b().seq_len; Phase B default


def spec(phase: str) -> PhaseSpec:
    if phase not in PHASES:
        raise KeyError(f"unknown phase {phase}; choose from {sorted(PHASES)}")
    return PHASES[phase]


def c_chain(*, use_kda: bool = False) -> tuple[str, ...]:
    """Published C lighting. KDA-majority graphs light the linear path first."""
    return C_CHAIN_KDA if use_kda else C_CHAIN


def resolve_phase_spec(name: str, *, use_kda: bool = False) -> PhaseSpec | None:
    """PhaseSpec with KDA-aware sparse flags and the carved C-index envelope.

    Without ``use_kda`` this is ``PHASES[name]``. With it: C-index keeps KDA
    lit (CSA stays window for indexer KL) and carves 5e9 from the 10e9
    indexer budget so C still sums to 25e9. D–G already pin ``sparse=hca``
    (last C lighting). ``use_kda`` only keeps KDA-kind layers on gated-delta
    inside that hca mode — it must not reset D–G back to window.
    """
    ph = PHASES.get(name)
    if ph is None:
        return None
    if not use_kda:
        return ph
    if name == "C-index":
        return replace(
            ph,
            tokens=5e9,
            tokens_offset=55e9,
            sparse="kda",
            notes=(
                "keep KDA-kind layers on gated-delta; freeze backbone; "
                "bf16 Lightning Indexer KL vs dense window on remaining CSA "
                "anchors (2 on 12B). Not top-k yet"
            ),
        )
    if name in _POST_C and ph.sparse == "window":
        return replace(ph, sparse="hca")
    return ph


def family(phase: str) -> str:
    """B / C / D / E / F / G."""
    if phase in {"B0", "B1", "B2"}:
        return "B"
    if phase.startswith("C"):
        return "C"
    if phase.startswith("D"):
        return "D"
    if phase == "E":
        return "E"
    if phase == "F":
        return "F"
    if phase.startswith("G"):
        return "G"
    return phase
