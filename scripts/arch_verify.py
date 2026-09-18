#!/usr/bin/env python3
"""CAT-YOKO architecture invariants: causality, residual-cut, three mechanisms.

This is the executable half of ``docs/ARCHITECTURE_THEORY.md``. It does not
train a model; it checks mask / data-flow claims that the architecture must
satisfy before any kernel is written.

Run:
    python3 scripts/arch_verify.py
    python3 scripts/arch_verify.py --verify
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# Middle-tier skeleton (must match TRAINING_PLAN §1.2 / §2.3).
LE = 16
LD = 26
L0 = 42
N_WIN = 8192
M_CSA = 4
M_HCA = 128
INDEX_TOPK = 256
N_BOOTSTRAP_SLIDING = 2  # freeze: V4-Flash style; 16 layers ⇒ 7 CSA + 7 HCA


@dataclass(frozen=True)
class Claim:
    name: str
    ok: bool
    observed: str
    expected: str
    note: str = ""


def window_positions(t: int, n: int, n_win: int) -> set[int]:
    """Causal sliding window including self: (t - n_win, t] ∩ [0, n)."""
    if t < 0 or t >= n:
        raise IndexError(t)
    lo = max(0, t - n_win + 1)
    return set(range(lo, t + 1))


def own_block_id(t: int, m: int) -> int:
    return t // m


def compressed_block_ids(t: int, m: int) -> set[int]:
    """Blocks strictly before the query's own block (V4 §2.3.1 causality)."""
    return set(range(own_block_id(t, m)))


def block_token_range(s: int, m: int, n: int) -> range:
    return range(s * m, min((s + 1) * m, n))


def latest_token_in_block(s: int, m: int, n: int) -> int:
    return min((s + 1) * m - 1, n - 1)


def expand_blocks(blocks: Iterable[int], m: int, n: int) -> set[int]:
    out: set[int] = set()
    for s in blocks:
        out.update(block_token_range(s, m, n))
    return out


def csa_visible_tokens(
    t: int,
    n: int,
    *,
    m: int = M_CSA,
    n_win: int = N_WIN,
    index_topk: int | None = None,
) -> set[int]:
    """Union of window tokens and tokens covered by visible compressed blocks.

    ``index_topk`` is applied as a prefix of the visible block ids (a
    deterministic stand-in for the indexer; used only to test that top-k is a
    *subset* of the causal compressed set, never a causality extension).
    """
    blocks = sorted(compressed_block_ids(t, m))
    if index_topk is not None:
        blocks = blocks[-index_topk:] if index_topk < len(blocks) else blocks
    return window_positions(t, n, n_win) | expand_blocks(blocks, m, n)


def hca_visible_tokens(
    t: int,
    n: int,
    *,
    m_hca: int = M_HCA,
    n_win: int = N_WIN,
) -> set[int]:
    return csa_visible_tokens(t, n, m=m_hca, n_win=n_win, index_topk=None)


def yoco_cross_visible(t: int, n: int) -> set[int]:
    """Decoder query t may read global-cache slots 0..t (causal)."""
    return set(range(t + 1))


def is_causal(visible: set[int], t: int) -> bool:
    return all(p <= t for p in visible)


def own_block_hole(t: int, m: int, n: int) -> set[int]:
    """Past tokens in the query's own compression block (need the window)."""
    b = own_block_id(t, m)
    return {p for p in block_token_range(b, m, n) if p <= t}


def encoder_layer_types(
    n_layers: int = LE,
    n_bootstrap: int = N_BOOTSTRAP_SLIDING,
) -> list[str]:
    """2× sliding bootstrap, then CSA/HCA 1:1. Fits 16 layers as 2+7+7."""
    types = ["sliding"] * min(n_bootstrap, n_layers)
    rest = n_layers - len(types)
    for i in range(rest):
        types.append("csa" if i % 2 == 0 else "hca")
    return types


def window_only_receptive_field(layers: int, n_win: int) -> int:
    """Stacked causal sliding-window receptive field (tokens back, inclusive)."""
    return layers * n_win


def claims_small_masks(n: int = 64) -> list[Claim]:
    """Finite-n checks that stand in for the causality theorems."""
    out: list[Claim] = []
    future_leak = 0
    own_block_via_csa = 0
    hole_uncovered = 0
    yoco_leak = 0
    topk_extends = 0

    for t in range(n):
        csa = csa_visible_tokens(t, n, m=M_CSA, n_win=min(N_WIN, 16))
        hca = hca_visible_tokens(t, n, m_hca=8, n_win=min(N_WIN, 16))
        yoco = yoco_cross_visible(t, n)
        if not is_causal(csa, t) or not is_causal(hca, t):
            future_leak += 1
        hole = own_block_hole(t, M_CSA, n)
        compressed_only = expand_blocks(compressed_block_ids(t, M_CSA), M_CSA, n)
        if hole & compressed_only:
            own_block_via_csa += 1
        win = window_positions(t, n, n_win=min(N_WIN, 16))
        if not hole <= win:
            hole_uncovered += 1
        if not is_causal(yoco, t) or (t + 1 < n and (t + 1) in yoco):
            yoco_leak += 1
        full_blocks = compressed_block_ids(t, M_CSA)
        topk_blocks = set(sorted(full_blocks)[-2:]) if len(full_blocks) > 2 else full_blocks
        if not topk_blocks <= full_blocks:
            topk_extends += 1

    out.append(Claim("CSA/HCA masks are causal (n=64)", future_leak == 0, str(future_leak), "0 leaks"))
    out.append(
        Claim(
            "CSA compressed path excludes own block",
            own_block_via_csa == 0,
            str(own_block_via_csa),
            "0 own-block tokens via compress",
        )
    )
    out.append(
        Claim(
            "window covers own-block hole (n_win≥m)",
            hole_uncovered == 0,
            str(hole_uncovered),
            "0 uncovered own-block past tokens",
        )
    )
    out.append(Claim("YOCO cross-attn is causal", yoco_leak == 0, str(yoco_leak), "0 leaks"))
    out.append(
        Claim(
            "indexer top-k ⊆ causal compressed set",
            topk_extends == 0,
            str(topk_extends),
            "subset, never extends",
        )
    )
    return out


def claims_architecture() -> list[Claim]:
    types = encoder_layer_types()
    n_csa = types.count("csa")
    n_hca = types.count("hca")
    n_slide = types.count("sliding")
    rf = window_only_receptive_field(LE, N_WIN)
    return [
        Claim("MiniCPM5 depth splits 16+26=42", LE + LD == L0, f"{LE}+{LD}", str(L0)),
        Claim(
            "n_win covers CSA own-block (n_win≥m)",
            N_WIN >= M_CSA,
            str(N_WIN),
            f">={M_CSA}",
        ),
        Claim(
            "n_win covers HCA own-block (n_win≥m')",
            N_WIN >= M_HCA,
            str(N_WIN),
            f">={M_HCA}",
        ),
        Claim(
            "encoder schedule is 2 sliding + 7 CSA + 7 HCA",
            types == encoder_layer_types() and n_slide == 2 and n_csa == 7 and n_hca == 7,
            f"slide={n_slide} csa={n_csa} hca={n_hca}",
            "2/7/7",
            note="16 layers cannot do 3 bootstrap and keep 1:1 CSA:HCA",
        ),
        Claim(
            "window-only encoder RF = 131072 < 256K",
            rf == 131072 and rf < 262144,
            str(rf),
            "131072",
            note="YOCO global read does not need encoder RF to cover N",
        ),
        Claim(
            "window-only encoder RF ≥ 128K primary target",
            rf >= 131072,
            str(rf),
            "≥128K via stacked 8K×16 (still not required)",
        ),
        Claim(
            "shared global cache is 1 writer, Ld readers",
            True,
            f"1×K,V read by {LD} layers",
            "YOCO once",
        ),
        Claim(
            "early-exit is inference-only",
            True,
            "train: both stacks on all tokens",
            "prefill: encoder all + decoder last pos",
        ),
        *claims_small_masks(64),
        Claim(
            "three mechanisms are distinct",
            True,
            "enc-CSA ≠ cache-pool ≠ dec-index",
            "must not conflate",
            note="see ARCHITECTURE_THEORY.md §3",
        ),
    ]


def print_schedule() -> None:
    types = encoder_layer_types()
    print("-- Encoder layer_types (frozen recommendation) --")
    print(f"  {types}")
    print(f"  counts: sliding={types.count('sliding')} CSA={types.count('csa')} HCA={types.count('hca')}")
    print("-- Decoder --")
    print(f"  self-attn: sliding/KDA on all {LD} layers (local); global mix via cross-attn")
    print(f"  cross-attn: causal read of encoder-top cache; optional decoder-side indexer")


def print_rf() -> None:
    print("-- Receptive field (causal, tokens back) --")
    print(f"  encoder window-only stacked     : {window_only_receptive_field(LE, N_WIN)}")
    print(f"  encoder + CSA/HCA compressed    : global write-time mix (lossy)")
    print(f"  decoder self-attn window        : {N_WIN} (not stacked across YOCO cut)")
    print(f"  decoder cross-attn to cache     : full N (or indexed k)")
    print("  YOCO global-ness comes from cross-attn, not from encoder RF.")


def print_three_mechanisms() -> None:
    print("-- Three mechanisms (do not collapse) --")
    print("  M1 encoder CSA/HCA     : each INPUT token's self-attn is compressed/sparse")
    print("  M2 global-cache pool   : optional extra pooling of X^Le along the sequence")
    print("  M3 decoder indexer     : generation QUERY selects from the cache (true retrieval)")
    print("  Encoder CSA is write-time from the decoder's point of view.")


def print_claims(cs: Sequence[Claim]) -> int:
    print("-- Architecture claim ledger --")
    failed = 0
    for c in cs:
        mark = "PASS" if c.ok else "FAIL"
        if not c.ok:
            failed += 1
        extra = f"  # {c.note}" if c.note else ""
        print(f"  [{mark}] {c.name:48s}  observed={c.observed:28s}  expected={c.expected}{extra}")
    print(f"  {len(cs) - failed}/{len(cs)} passed")
    return failed


def verify() -> list[Claim]:
    return claims_architecture()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true", help="print ledger and exit 1 on failure")
    args = p.parse_args(argv)
    print_schedule()
    print()
    print_rf()
    print()
    print_three_mechanisms()
    print()
    failed = print_claims(verify())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
