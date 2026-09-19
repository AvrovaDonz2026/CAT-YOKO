"""Isolated BF16 / INT8 mini full-pipeline verify.

Default graph is tiny (bf16) or ``int8_probe`` (hd=32, seq=64). ``--12b`` is
dummy-upcycle ``--try`` scale (seq=64). Does not download Ultra-FineWeb /
UltraChat. DummyStream only.

Recipes:

- ``bf16``: ``use_int8=False``, ``use_fp8=False``, ``use_nvfp4=False``
- ``int8``: SageBwd INT8 ``QK^T`` on dense causal SDPA (softmax fp32; PV / bwd
  high-prec). Ampere tries the Turing Triton kernel then ``torch._int_mm``.

Write overlays under ``--out/{bf16,int8}/<phase>/``. Not GitHub LFS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cat_yoko.config import CATYokoConfig
from cat_yoko.int8_attn import last_backend
from cat_yoko.nvfp4 import low_prec_enabled, should_autocast
from cat_yoko.phases import TIGHT_GPU_SEQ
from cat_yoko.trainer import run_phase_chain
from cat_yoko.upcycle import dummy_minicpm_state

MINI_PHASES = (
    "B0",
    "B1",
    "B2",
    "C-index",
    "C-topk",
    "C-hca",
    "C-win",
    "D-8k",
    "E",
    "F",
)

RECIPES = ("bf16", "int8")


def _finite_pos(x: float) -> bool:
    return bool(math.isfinite(x)) and x > 0


def recipe_cfg(name: str, *, twelve: bool) -> CATYokoConfig:
    if name not in RECIPES:
        raise KeyError(f"unknown recipe {name}; choose {list(RECIPES)}")
    if name == "bf16":
        base = CATYokoConfig.middle_12b() if twelve else CATYokoConfig.tiny()
        return replace(base, use_fp8=False, use_nvfp4=False, use_int8=False)
    if twelve:
        return replace(
            CATYokoConfig.middle_12b(),
            use_fp8=False,
            use_nvfp4=False,
            use_int8=True,
        )
    return CATYokoConfig.int8_probe()


def run_recipe(
    name: str,
    *,
    device: str,
    out: Path,
    twelve: bool = False,
    steps: int = 2,
    seq_len: int | None = None,
) -> dict:
    cfg = recipe_cfg(name, twelve=twelve)
    sl = seq_len
    if sl is None:
        sl = TIGHT_GPU_SEQ if twelve else cfg.seq_len
    save = Path(out) / name
    save.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "steps": steps,
        "dtype": "bf16" if str(device).startswith("cuda") else "fp32",
        "seq_len": sl,
        "micro_batch": 1 if twelve else 2,
        "accum": 1,
        "save_dir": save,
        "save_every": max(steps, 1),
        "save_full": False,
        "save_trainable": True,
        "save_optim": False,
        "log_path": save / "metrics.jsonl",
    }
    if twelve:
        kwargs["upcycle_src"] = dummy_minicpm_state(cfg)
    t0 = time.perf_counter()
    results = run_phase_chain(cfg, device, MINI_PHASES, **kwargs)
    elapsed = time.perf_counter() - t0
    phases = {}
    ok = True
    for phase, row in results.items():
        finite = _finite_pos(float(row.nll))
        ok = ok and finite
        phases[phase] = {
            "nll": float(row.nll),
            "step": int(row.step),
            "tokens_seen": float(row.tokens_seen),
            "peak_mib": float(row.peak_mib),
            "ok": finite,
        }
    summary = {
        "recipe": name,
        "twelve": twelve,
        "device": device,
        "seq_len": sl,
        "steps": steps,
        "use_fp8": bool(cfg.use_fp8),
        "use_nvfp4": bool(cfg.use_nvfp4),
        "use_int8": bool(cfg.use_int8),
        "int8_backend": last_backend(),
        "head_dim": int(cfg.head_dim),
        "n_win": int(cfg.n_win),
        "low_prec": low_prec_enabled(cfg),
        "b0_autocast": should_autocast(
            "B0", cuda=str(device).startswith("cuda"), enabled=low_prec_enabled(cfg)
        ),
        "b1_autocast": should_autocast(
            "B1", cuda=str(device).startswith("cuda"), enabled=low_prec_enabled(cfg)
        ),
        "phases": phases,
        "elapsed_s": round(elapsed, 3),
        "ok": ok,
        "out": str(save),
    }
    (save / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAT-YOKO mini BF16/INT8 full-pipeline verify")
    p.add_argument("--out", type=Path, required=True, help="dedicated run directory (not B0-full)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--recipe", choices=["bf16", "int8", "both"], default="both")
    p.add_argument("--12b", action="store_true", dest="twelve", help="12B dummy-upcycle, seq=64")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    names = list(RECIPES) if args.recipe == "both" else [args.recipe]
    reports = []
    overall = True
    for name in names:
        row = run_recipe(
            name,
            device=args.device,
            out=args.out,
            twelve=bool(args.twelve),
            steps=max(int(args.steps), 1),
            seq_len=args.seq_len,
        )
        reports.append(row)
        overall = overall and bool(row["ok"])
        if not args.json:
            print(
                f"{name} ok={row['ok']} seq={row['seq_len']} steps={row['steps']} "
                f"int8={row['use_int8']} backend={row['int8_backend']} "
                f"nvfp4={row['use_nvfp4']} {row['elapsed_s']}s",
                flush=True,
            )
            for phase, st in row["phases"].items():
                print(
                    f"  {phase} nll={st['nll']:.4f} ok={st['ok']} peak_mib={st['peak_mib']:.0f}",
                    flush=True,
                )
    blob = {"ok": overall, "runs": reports}
    (Path(args.out) / "summary.json").write_text(json.dumps(blob, indent=2) + "\n")
    if args.json:
        print(json.dumps(blob, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
