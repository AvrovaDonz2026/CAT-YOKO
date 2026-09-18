"""CUDA smoke for the C1 trainer. Tiny always; 12B with --middle / --c1 on ≥28GiB."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from dataclasses import replace
from pathlib import Path

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.ddp_smoke import run_fsdp_one, run_gloo_c1, run_gloo_ddp
from cat_yoko.fp8 import should_autocast
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import trim_host_allocator
from cat_yoko.prepare import prepare
from cat_yoko.teacher import DummyTeacher
from cat_yoko.tokenizer import HashTokenizer
from cat_yoko.trainer import Trainer, build_model, run_c1_chain, train_loop
from cat_yoko.upcycle import dummy_minicpm_state


def cuda_info() -> dict:
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if not torch.cuda.is_available():
        return {"cuda": False, "alloc_conf": alloc_conf}
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
        "alloc_conf": alloc_conf,
    }


def _finite(x: float) -> bool:
    return bool(x == x) and abs(x) != float("inf")


def run_tiny_cuda(*, steps: int = 2, micro_batch: int = 2) -> dict:
    """B0/B1/B2 + bf16 + packed bin + resume + offload path. Tiny graph only."""
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
    nll_ckpt = train_loop(
        cfg, "B2", steps=1, device=device, accum=1, micro_batch=micro_batch, grad_ckpt=True
    )
    out["grad_ckpt"] = {"nll": float(nll_ckpt), "ok": _finite(nll_ckpt) and nll_ckpt > 0}
    nll_off = train_loop(
        cfg,
        "B1",
        steps=1,
        device=device,
        accum=1,
        micro_batch=micro_batch,
        offload_encoder=True,
        optim_cpu=True,
        dtype="bf16",
    )
    out["b1_offload"] = {"nll": float(nll_off), "ok": _finite(nll_off) and nll_off > 0}
    nll_blk = train_loop(
        cfg,
        "B2",
        steps=1,
        device=device,
        accum=1,
        micro_batch=micro_batch,
        offload_blocks=True,
        optim_cpu=True,
        dtype="bf16",
    )
    out["b2_block_offload"] = {"nll": float(nll_blk), "ok": _finite(nll_blk) and nll_blk > 0}
    chain = None
    with tempfile.TemporaryDirectory() as c1td:
        c1_root = Path(c1td)
        chain = run_c1_chain(
            cfg,
            device,
            steps=1,
            accum=1,
            micro_batch=micro_batch,
            dtype="bf16",
            grad_ckpt=True,
            save_dir=c1_root,
            save_optim=False,
        )
        out["c1_ckpts"] = {
            "ok": all((c1_root / p / "latest.pt").is_file() for p in ("B0", "B1", "B2"))
        }
    out["c1_chain"] = {
        phase: {"nll": float(r.nll), "step": int(r.step), "ok": _finite(r.nll) and r.step == 1}
        for phase, r in chain.items()
    }
    out["c1_chain"]["ok"] = all(out["c1_chain"][p]["ok"] for p in ("B0", "B1", "B2"))
    teacher = DummyTeacher(cfg.vocab_size, cfg.hidden_size)
    nll_kd = train_loop(
        cfg,
        "B0",
        steps=2,
        device=device,
        accum=1,
        micro_batch=micro_batch,
        teacher=teacher,
        dtype="bf16",
    )
    out["kd"] = {"nll": float(nll_kd), "ok": _finite(nll_kd) and nll_kd > 0}
    del teacher
    src = dummy_minicpm_state(cfg)
    nll_up = train_loop(
        cfg,
        "B0",
        steps=1,
        device=device,
        accum=1,
        micro_batch=micro_batch,
        upcycle_src=src,
        dtype="bf16",
    )
    out["upcycle"] = {"nll": float(nll_up), "ok": _finite(nll_up) and nll_up > 0}
    ev = Trainer(
        cfg,
        "B0",
        device,
        steps=1,
        accum=1,
        micro_batch=micro_batch,
        eval_every=1,
        dtype="bf16",
    ).run()
    out["eval"] = {"nll": float(ev.nll), "ok": _finite(ev.nll) and ev.nll > 0}
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
        b0 = Trainer(
            cfg,
            "B0",
            device,
            steps=1,
            accum=1,
            micro_batch=micro_batch,
            save_dir=td / "b0",
            save_every=1,
            dtype="bf16",
        ).run()
        b1 = Trainer(
            cfg,
            "B1",
            device,
            steps=1,
            accum=1,
            micro_batch=micro_batch,
            resume=td / "b0" / "latest.pt",
            dtype="bf16",
        ).run()
        out["b0_to_b1"] = {
            "b0_nll": float(b0.nll),
            "b1_step": int(b1.step),
            "b1_phase": b1.phase,
            "ok": b1.step == 1 and b1.phase == "B1" and _finite(b1.nll),
        }
    model = CATYokoForCausalLM(cfg).to(device)
    apply_freeze(model, "B0")
    ids = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.seq_len), device=device)
    model(input_ids=ids, labels=ids)["loss"].backward()
    enc_ok = all((not p.requires_grad) and p.grad is None for p in model.encoder.parameters())
    out["b0_encoder_frozen"] = {"ok": enc_ok}
    ddp = run_gloo_ddp(device="cpu", world=2, steps=1)
    out["ddp_gloo"] = ddp
    out["ddp_cuda"] = run_gloo_ddp(device="cuda", world=2, steps=1)
    out["ddp_cuda_c1"] = run_gloo_c1(device="cuda", world=2, steps=1)
    out["fsdp"] = run_fsdp_one(device="cuda", steps=1)
    out["peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    out["ok"] = all(
        [
            all(v["ok"] for v in out["phases"].values()),
            out.get("bf16", {"ok": True})["ok"],
            out["b1_fp8_autocast"]["ok"],
            out["grad_ckpt"]["ok"],
            out["b1_offload"]["ok"],
            out["b2_block_offload"]["ok"],
            out["c1_chain"]["ok"],
            out["c1_ckpts"]["ok"],
            out["packed_resume"]["ok"],
            out["b0_to_b1"]["ok"],
            out["kd"]["ok"],
            out["upcycle"]["ok"],
            out["eval"]["ok"],
            out["b0_encoder_frozen"]["ok"],
            out["ddp_gloo"]["ok"],
            out["ddp_cuda"]["ok"],
            out["ddp_cuda_c1"]["ok"],
            out["fsdp"]["ok"],
            out["fp8_policy"]["b1_autocast"] is True,
            out["fp8_policy"]["b0_autocast"] is False,
        ]
    )
    return out


TWELVE_B_MIN_GIB = 28.0


def enough_vram_for_12b(min_gib: float = TWELVE_B_MIN_GIB) -> bool:
    info = cuda_info()
    return bool(info.get("cuda") and info.get("total_gib", 0) >= min_gib)


def _drop_cuda_refs(obj: object) -> None:
    """Clear ``.grad`` on a module so GC can collect the 12B graph."""
    params = getattr(obj, "parameters", None)
    if not callable(params):
        return
    for p in params():
        p.grad = None


def _teardown_cuda(*holders: object) -> None:
    """Free CUDA graphs so a later 12B ``build_model`` can fit on 32GB.

    Drops ``model`` / ``opt`` / ``trainer`` dict keys or attributes on
    ``holders``, then GC + caching-allocator flush + sync + peak reset.
    Does not unbind names in the caller: pass the dict/object that holds
    them, and ``del`` locals too. Safe with no CUDA (CPU CI).
    """
    for obj in holders:
        if obj is None:
            continue
        if isinstance(obj, dict):
            for key in ("model", "opt", "optimizer", "trainer"):
                val = obj.pop(key, None)
                _drop_cuda_refs(val)
                del val
            continue
        for name in ("model", "opt", "optimizer", "trainer"):
            if hasattr(obj, name):
                val = getattr(obj, name)
                _drop_cuda_refs(val)
                setattr(obj, name, None)
                del val
        _drop_cuda_refs(obj)
    del holders
    gc.collect()
    trim_host_allocator()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc):
            ipc()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


def run_middle_12b_phase(
    phase: str,
    *,
    seq_len: int = 64,
    steps: int = 1,
    micro_batch: int = 1,
    reuse_model: CATYokoForCausalLM | None = None,
    save_dir: Path | None = None,
    resume: Path | None = None,
) -> dict:
    """One C1 step of the real 12B graph. Auto encoder-offload / CPU Adam / block offload.

    Isolated ``--middle --phase B0/B1/B2`` each build once. Pass ``reuse_model``
    so a C1 chain never allocates a second 12.25B graph in the same process.
    On OOM, a reused graph is kept; only a graph built here is dropped before
    retrying a shorter sequence.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 12B GPU smoke")
    info = cuda_info()
    if info["total_gib"] < TWELVE_B_MIN_GIB:
        raise RuntimeError(
            f"12B GPU smoke needs ≥{TWELVE_B_MIN_GIB}GiB, got {info['total_gib']}"
        )
    cfg = CATYokoConfig.middle_12b()
    last_err: BaseException | None = None
    tried: list[int] = []
    for sl in (seq_len, 32, 16):
        if sl in tried:
            continue
        tried.append(sl)
        if reuse_model is None:
            _teardown_cuda()
        else:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model = reuse_model
        trainer = None
        opt = None
        built_here = False
        try:
            if model is None:
                built_here = True
                model = build_model(cfg, "cuda", dtype="bf16")
            trainer = Trainer(
                cfg,
                phase,
                "cuda",
                steps=steps,
                micro_batch=micro_batch,
                accum=1,
                dtype="bf16",
                grad_ckpt=True,
                seq_len=sl,
                seed=0,
                reuse_model=model,
                save_dir=save_dir,
                save_every=1 if save_dir is not None else 0,
                resume=resume,
                save_optim=False,
            )
            out = trainer.run()
            peak = out.peak_mib / 1024.0 if out.peak_mib else torch.cuda.max_memory_allocated() / 1024**3
            ok = _finite(out.nll) and out.nll > 0 and out.step == steps
            result = {
                "info": info,
                "phase": phase,
                "nll": float(out.nll),
                "step": int(out.step),
                "seq_len": sl,
                "peak_gib": round(peak, 2),
                "ok": ok,
                "model": model,
            }
            if save_dir is not None:
                latest = Path(save_dir) / "latest.pt"
                step_p = Path(save_dir) / f"step_{out.step}.pt"
                result["latest"] = latest.is_file()
                result["step_ckpt"] = step_p.is_file()
                if latest.is_file() and step_p.is_file():
                    result["latest_hardlink"] = latest.stat().st_ino == step_p.stat().st_ino
                    result["latest_nlink"] = latest.stat().st_nlink
            trainer.reuse_model = None
            del trainer, opt
            return result
        except torch.cuda.OutOfMemoryError as exc:
            last_err = exc
            if trainer is not None:
                trainer.reuse_model = None
            del trainer, opt
            if built_here:
                _drop_cuda_refs(model)
                del model
                _teardown_cuda()
                reuse_model = None
            else:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    assert last_err is not None
    raise last_err


def run_middle_12b_b0(*, seq_len: int = 64, steps: int = 1, micro_batch: int = 1, save_dir: Path | None = None, resume: Path | None = None) -> dict:
    """One C1 B0 step of the real 12B graph. Needs ~24GiB weights + a little activation."""
    result = run_middle_12b_phase(
        "B0", seq_len=seq_len, steps=steps, micro_batch=micro_batch, save_dir=save_dir, resume=resume
    )
    model = result.pop("model", None)
    del model
    _teardown_cuda(result)
    return result


def run_middle_12b_c1(*, seq_len: int = 64, steps: int = 1, micro_batch: int = 1) -> dict:
    """B0 then B1 then B2 on one 12B graph. B2 may OOM on 32GB even with block offload.

    Builds the 12.25B graph once (first phase) and reuses it. Never calls
    ``build_model`` again in this process — a second graph is what OOMs 32GB.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 12B C1 smoke")
    info = cuda_info()
    if info["total_gib"] < TWELVE_B_MIN_GIB:
        raise RuntimeError(
            f"12B C1 smoke needs ≥{TWELVE_B_MIN_GIB}GiB, got {info['total_gib']}"
        )
    _teardown_cuda()
    torch.cuda.reset_peak_memory_stats()
    model = None
    out: dict = {"info": info, "phases": {}, "seq_len": seq_len}
    try:
        for phase in ("B0", "B1", "B2"):
            try:
                row = run_middle_12b_phase(
                    phase,
                    seq_len=seq_len,
                    steps=steps,
                    micro_batch=micro_batch,
                    reuse_model=model,
                )
                model = row.pop("model", None)
                out["phases"][phase] = row
            except torch.cuda.OutOfMemoryError as exc:
                out["phases"][phase] = {
                    "ok": False,
                    "oom": True,
                    "err": str(exc).split("\n")[0][:240],
                }
                if phase == "B2":
                    break
                raise
        b0 = out["phases"].get("B0", {})
        b1 = out["phases"].get("B1", {})
        b2 = out["phases"].get("B2", {})
        out["ok"] = bool(b0.get("ok") and b1.get("ok") and b2.get("ok"))
        out["b1_ok"] = bool(b1.get("ok"))
        out["b2_oom"] = bool(b2.get("oom"))
        return out
    finally:
        _drop_cuda_refs(model)
        del model
        _teardown_cuda(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="CAT-YOKO CUDA smoke (tiny, 12B B0 with --middle, C1 chain with --c1)"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="tiny default 2; 12B --middle/--c1 default 1",
    )
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--middle",
        action="store_true",
        help="one CAT-YOKO-12B step on CUDA bf16 (needs ≥28GiB; default phase B0)",
    )
    p.add_argument(
        "--c1",
        action="store_true",
        help="12B B0→B1→B2 one step each (encoder offload + CPU Adam; B2 block offload)",
    )
    p.add_argument("--phase", choices=["B0", "B1", "B2"], default="B0")
    p.add_argument("--seq-len", type=int, default=64, help="12B smoke sequence length")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="12B --middle: write step_N.pt and hardlink latest.pt (use a large volume, not /tmp)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="12B --middle: resume file or directory (latest.pt else newest step_*.pt)",
    )
    args = p.parse_args(argv)
    if args.save_dir is not None and not args.middle:
        p.error("--save-dir is for --middle 12B smoke")
    if args.resume is not None and not args.middle:
        p.error("--resume is for --middle 12B smoke")
    info = cuda_info()
    if not info.get("cuda"):
        print("SKIP: no CUDA")
        return 0
    if args.c1:
        steps = 1 if args.steps is None else args.steps
        result = run_middle_12b_c1(seq_len=args.seq_len, steps=steps)
        try:
            if args.json:
                print(json.dumps({k: v for k, v in result.items() if k != "model"}, indent=2, default=str))
            else:
                print(f"12b C1 {result['info']['device']} seq={result.get('seq_len')} ok={result['ok']}")
                for phase, row in result["phases"].items():
                    if row.get("oom"):
                        print(f"  {phase} OOM {row.get('err', '')}")
                    else:
                        print(
                            f"  {phase} nll={row['nll']:.4f} seq={row['seq_len']} "
                            f"peak_gib={row['peak_gib']} ok={row['ok']}"
                        )
            return 0 if result["ok"] or (result.get("b1_ok") and result.get("b2_oom")) else 1
        finally:
            _teardown_cuda(result)
    if args.middle:
        steps = 1 if args.steps is None else args.steps
        result = run_middle_12b_phase(
            args.phase,
            seq_len=args.seq_len,
            steps=steps,
            save_dir=args.save_dir,
            resume=args.resume,
        )
        model = result.pop("model", None)
        del model
        try:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(
                    f"12b {result['phase']} {result['info']['device']} nll={result['nll']:.4f} "
                    f"seq={result['seq_len']} peak_gib={result['peak_gib']} ok={result['ok']}"
                )
                if result.get("latest"):
                    print(
                        f"  ckpt latest={result.get('latest')} step={result.get('step_ckpt')} "
                        f"hardlink={result.get('latest_hardlink')} nlink={result.get('latest_nlink')}"
                    )
            return 0 if result["ok"] else 1
        finally:
            _teardown_cuda(result)
    result = run_tiny_cuda(steps=2 if args.steps is None else args.steps)
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
        print(f"  b0_to_b1 ok={result['b0_to_b1']['ok']}")
        print(f"  kd ok={result['kd']['ok']} upcycle ok={result['upcycle']['ok']} eval ok={result['eval']['ok']}")
        print(f"  grad_ckpt ok={result['grad_ckpt']['ok']}")
        print(f"  b1_offload ok={result['b1_offload']['ok']}")
        print(f"  b2_block_offload ok={result['b2_block_offload']['ok']}")
        print(f"  c1_chain ok={result['c1_chain']['ok']} ckpts ok={result['c1_ckpts']['ok']}")
        print(f"  ddp_gloo ok={result['ddp_gloo']['ok']}")
        print(f"  ddp_cuda ok={result['ddp_cuda']['ok']}")
        print(f"  ddp_cuda_c1 ok={result['ddp_cuda_c1']['ok']}")
        print(f"  fsdp ok={result['fsdp']['ok']}")
        print(f"  overall ok={result['ok']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
