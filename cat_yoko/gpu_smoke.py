"""CUDA smoke for the C1 trainer. Never allocates the 12B graph on GPU."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.prepare import prepare
from cat_yoko.tokenizer import HashTokenizer
from cat_yoko.trainer import Trainer, train_loop


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(idx),
        "index": idx,
        "capability": f"{props.major}.{props.minor}",
        "total_gib": round(props.total_memory / 1024**3, 2),
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _finite(x: float) -> bool:
    return bool(x == x) and abs(x) != float("inf")


def run_tiny_cuda(*, steps: int = 2, micro_batch: int = 2) -> dict:
    """B0/B1/B2 + bf16 + packed bin + resume. Tiny graph only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gpu_smoke")
    device = "cuda"
    cfg = CATYokoConfig.tiny()
    torch.cuda.reset_peak_memory_stats()
    out: dict = {"info": cuda_info(), "phases": {}}
    for phase in ("B0", "B1", "B2"):
        nll = train_loop(cfg, phase, steps=steps, device=device, accum=1, micro_batch=micro_batch)
        out["phases"][phase] = {"nll": float(nll), "ok": _finite(nll) and nll > 0}
    if torch.cuda.is_bf16_supported():
        bf = Trainer(
            cfg, "B0", device, steps=1, accum=1, micro_batch=micro_batch, dtype="bf16"
        ).run()
        out["bf16"] = {"nll": float(bf.nll), "ok": _finite(bf.nll) and bf.nll > 0}
    fp8_cfg = replace(CATYokoConfig.tiny(), use_fp8=True)
    out["fp8_policy"] = {
        "b1_autocast": should_autocast("B1", cuda=True, enabled=True),
        "b0_autocast": should_autocast("B0", cuda=True, enabled=True),
    }
    nll_fp8 = train_loop(
        fp8_cfg, "B1", steps=1, device=device, accum=1, micro_batch=micro_batch
    )
    out["b1_fp8_autocast"] = {"nll": float(nll_fp8), "ok": _finite(nll_fp8) and nll_fp8 > 0}
    tok = HashTokenizer(cfg.vocab_size)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "docs.jsonl"
        src.write_text(
            '{"text": "' + ("the quick brown fox jumps over the lazy dog. " * 12) + '"}\n' * 4
        )
        packed = td / "t.bin"
        prepare(
            mix="local",
            out=packed,
            max_tokens=64,
            seq_len=cfg.seq_len,
            tokenizer=tok,
            local_jsonl=src,
        )
        save = td / "ckpt"
        first = Trainer(
            cfg,
            "B0",
            device,
            steps=1,
            accum=1,
            micro_batch=micro_batch,
            data=packed,
            save_dir=save,
            save_every=1,
        ).run()
        second = Trainer(
            cfg, "B0", device, steps=2, accum=1, micro_batch=micro_batch, data=packed, resume=save / "latest.pt"
        ).run()
        out["packed_resume"] = {
            "first_nll": float(first.nll),
            "second_step": int(second.step),
            "ok": second.step == 2 and _finite(second.nll),
        }
    model = CATYokoForCausalLM(cfg).to(device)
    apply_freeze(model, "B0")
    ids = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.seq_len), device=device)
    model(input_ids=ids, labels=ids)["loss"].backward()
    enc_ok = all((not p.requires_grad) and p.grad is None for p in model.encoder.parameters())
    out["b0_encoder_frozen"] = {"ok": enc_ok}
    out["peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    out["ok"] = all(
        [
            all(v["ok"] for v in out["phases"].values()),
            out.get("bf16", {"ok": True})["ok"],
            out["b1_fp8_autocast"]["ok"],
            out["packed_resume"]["ok"],
            out["b0_encoder_frozen"]["ok"],
            out["fp8_policy"]["b1_autocast"] is True,
            out["fp8_policy"]["b0_autocast"] is False,
        ]
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CAT-YOKO tiny CUDA smoke (never builds 12B on GPU)")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    info = cuda_info()
    if not info.get("cuda"):
        print("SKIP: no CUDA")
        return 0
    result = run_tiny_cuda(steps=args.steps)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"gpu {result['info']['device']} {result['info']['total_gib']} GiB "
            f"cap={result['info']['capability']} torch={result['info']['torch']}"
        )
        for phase, row in result["phases"].items():
            print(f"  {phase} nll={row['nll']:.4f} ok={row['ok']}")
        if "bf16" in result:
            print(f"  bf16 nll={result['bf16']['nll']:.4f} ok={result['bf16']['ok']}")
        print(f"  packed_resume ok={result['packed_resume']['ok']} peak_mib={result['peak_mib']}")
        print(f"  overall ok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
