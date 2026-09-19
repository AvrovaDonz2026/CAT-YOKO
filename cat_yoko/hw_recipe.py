"""Card-agnostic B0 launch recipe.

Unknown next GPU → probe SM family + VRAM (optional TE FPROP) and dispatch
the published PyTorch B0 envelope. This is **not** Megatron EP/TP, **not** a
CSA kernel, and **not** a 50B download.

VRAM bands (measured / refused):

- ``<40GiB``: refuse the 8e9 envelope (same as ``TIGHT_12B_GPU_GIB``). ``--try``
  seq=64 / 32 steps only.
- ``>=90GiB``: published seq=4096, encoder on GPU (6000D ~96GiB; B200 mb=1 ~96GiB).
- ``>=160GiB`` on SM100/103: micro-batch 2, no activation checkpoint (B200 ~138GiB).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass

from cat_yoko.nvfp4_hw import compute_capability, compute_family, nvfp4_training_capable
from cat_yoko.phases import PUBLISHED_SAVE_EVERY, TIGHT_GPU_SEQ, TRY_STEPS
from cat_yoko.train import TIGHT_12B_GPU_GIB

PUBLISHED_SEQ = 4096
COMFORTABLE_GIB = 90.0
WIDE_MB2_GIB = 160.0
HUB_OVERLAY = "b0-full"
PHASE = "B0"

_OFF_TE = frozenset({"0", "off", "false", "emu", "no"})
_ON_TE = frozenset({"1", "on", "true", "te", "yes"})


@dataclass(frozen=True)
class HwSnapshot:
    family: str
    arch: str
    cap: tuple[int, int] | None
    gpu_gib: float
    name: str | None = None
    te_nvfp4_fprop: bool | None = None
    te_pytorch: bool | None = None
    env_te: str | None = None


@dataclass(frozen=True)
class B0Recipe:
    profile: str
    phase: str
    published: bool
    try_run: bool
    launch: bool
    seq_len: int
    micro_batch: int
    offload_encoder: bool
    grad_ckpt: bool
    optim_cpu: bool
    nvfp4_path: str
    te_env: str | None
    megatron: bool
    save_full: bool
    resume_hub: str
    notes: str
    refuse_reason: str | None
    argv: tuple[str, ...]


def parse_cap(raw: str | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    if raw is None:
        return None
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    text = str(raw).strip()
    if not text:
        return None
    if "." in text:
        major_s, minor_s = text.split(".", 1)
        return int(major_s), int(minor_s)
    return int(text), 0


def arch_kind(cap: tuple[int, int] | None, family: str | None = None) -> str:
    """SKU class for the recipe table. TE still keys off ``compute_family``."""
    fam = family if family is not None else compute_family(cap)
    if fam in {"cpu", "sm100", "sm103", "sm120"}:
        return fam
    if cap is None:
        return "cpu"
    major, minor = int(cap[0]), int(cap[1])
    if major == 9:
        return "hopper"
    if major == 8 and minor >= 9:
        return "ada"
    if major == 8:
        return "ampere"
    return "other"


def _env_te(raw: str | None = None) -> str | None:
    if raw is not None:
        text = str(raw).strip().lower()
        return text or None
    text = os.environ.get("CAT_YOKO_TE_NVFP4", "").strip().lower()
    return text or None


def snapshot(
    *,
    cap: tuple[int, int] | None = None,
    gpu_gib: float | None = None,
    family: str | None = None,
    name: str | None = None,
    te_nvfp4_fprop: bool | None = None,
    te_pytorch: bool | None = None,
    env_te: str | None = None,
) -> HwSnapshot:
    fam = family if family is not None else compute_family(cap)
    gib = 0.0 if gpu_gib is None else float(gpu_gib)
    if fam == "cpu":
        gib = 0.0
    return HwSnapshot(
        family=fam,
        arch=arch_kind(cap, fam),
        cap=None if cap is None else (int(cap[0]), int(cap[1])),
        gpu_gib=gib,
        name=name,
        te_nvfp4_fprop=te_nvfp4_fprop,
        te_pytorch=te_pytorch,
        env_te=_env_te(env_te),
    )


def detect_snapshot(*, probe_te: bool = False) -> HwSnapshot:
    """Live CUDA snapshot. ``probe_te`` runs the 4096×2048 FPROP check."""
    cap = compute_capability()
    fam = compute_family(cap)
    name = None
    gib = 0.0
    te_pytorch = None
    te_fprop: bool | None = None
    if cap is not None:
        import torch

        props = torch.cuda.get_device_properties(0)
        gib = float(props.total_memory) / 1024**3
        name = str(getattr(props, "name", "") or "") or None
    if probe_te and fam != "cpu":
        from cat_yoko.nvfp4_hw import te_linear_ready, te_pytorch_available

        te_pytorch = bool(te_pytorch_available())
        te_fprop = bool(te_linear_ready()) if te_pytorch else False
    return snapshot(
        cap=cap,
        gpu_gib=gib,
        family=fam,
        name=name,
        te_nvfp4_fprop=te_fprop,
        te_pytorch=te_pytorch,
    )


def _nvfp4_path(snap: HwSnapshot) -> tuple[str, str | None]:
    env = snap.env_te
    if snap.family == "cpu":
        return "off", None
    if env in _OFF_TE:
        return "emu", "0"
    if env in _ON_TE:
        return "te", "1"
    if nvfp4_training_capable(snap.family):
        if snap.te_nvfp4_fprop is False:
            return "emu", "0"
        return "te", None
    return "emu", None


def _b0_argv(
    *,
    seq_len: int,
    micro_batch: int,
    offload_encoder: bool,
    grad_ckpt: bool,
    try_run: bool,
) -> tuple[str, ...]:
    out: list[str] = [
        "--seq-len",
        str(int(seq_len)),
        "--micro-batch",
        str(max(int(micro_batch), 1)),
    ]
    out.append("--offload-encoder" if offload_encoder else "--no-offload-encoder")
    out.append("--grad-ckpt" if grad_ckpt else "--no-grad-ckpt")
    if try_run:
        out.append("--try")
    return tuple(out)


def recipe_for(snap: HwSnapshot, *, force_try: bool = False) -> B0Recipe:
    """Pure SM + VRAM → B0 flags. ``force_try`` is the 32-step overlay path."""
    nvfp4, te_env = _nvfp4_path(snap)
    megatron = False
    save_full = False
    resume_hub = HUB_OVERLAY
    phase = PHASE

    if snap.family == "cpu" or snap.gpu_gib <= 0:
        notes = (
            "no CUDA; dump recipe JSON only. Do not build the 12B graph on CPU. "
            "Megatron EP/TP is not the unknown-card adapter."
        )
        return B0Recipe(
            profile="cpu",
            phase=phase,
            published=False,
            try_run=False,
            launch=False,
            seq_len=TIGHT_GPU_SEQ,
            micro_batch=1,
            offload_encoder=True,
            grad_ckpt=True,
            optim_cpu=False,
            nvfp4_path="off",
            te_env=None,
            megatron=megatron,
            save_full=save_full,
            resume_hub=resume_hub,
            notes=notes,
            refuse_reason="cpu",
            argv=_b0_argv(
                seq_len=TIGHT_GPU_SEQ,
                micro_batch=1,
                offload_encoder=True,
                grad_ckpt=True,
                try_run=True,
            ),
        )

    tight = snap.gpu_gib < TIGHT_12B_GPU_GIB
    try_run = bool(force_try or tight)
    comfortable = snap.gpu_gib >= COMFORTABLE_GIB
    wide = snap.gpu_gib >= WIDE_MB2_GIB and snap.family in {"sm100", "sm103"}

    if try_run:
        profile = "ada_tight" if snap.arch in {"ada", "other"} else f"{snap.arch}_try"
        if snap.arch in {"sm100", "sm103", "sm120"}:
            profile = f"{snap.arch}_try"
        elif snap.arch == "hopper":
            profile = "hopper_try"
        elif snap.arch == "ampere":
            profile = "ampere_try"
        seq_len = TIGHT_GPU_SEQ
        micro_batch = 1
        offload_encoder = True
        grad_ckpt = True
        published = False
        launch = True
        notes = (
            f"<{int(TIGHT_12B_GPU_GIB)}GiB (this card {snap.gpu_gib:.1f}GiB) cannot "
            f"finish the published 8e9 envelope. --try is {TRY_STEPS} steps, "
            f"seq={TIGHT_GPU_SEQ}. Do not resume Hub checkpoints/b0. "
            "Do not wire Megatron."
        )
        if force_try and not tight:
            notes = (
                f"forced --try on {snap.arch} {snap.gpu_gib:.1f}GiB: {TRY_STEPS} steps, "
                f"seq={TIGHT_GPU_SEQ}. Published 8e9 is off. Do not wire Megatron."
            )
            profile = f"{snap.arch}_try"
        refuse_reason = "tight_vram" if tight else "force_try"
    else:
        seq_len = PUBLISHED_SEQ
        published = True
        launch = True
        refuse_reason = None
        if wide:
            profile = "sm100_b200" if snap.family == "sm100" else "sm103_b200"
            micro_batch = 2
            offload_encoder = False
            grad_ckpt = False
            notes = (
                f"{snap.family} {snap.gpu_gib:.1f}GiB: published B0 seq={PUBLISHED_SEQ} "
                "micro-batch=2, encoder on GPU, no grad-ckpt, TE NVFP4 FPROP when ready. "
                "Resume Hub checkpoints/b0-full. Not Megatron."
            )
        elif snap.family in {"sm100", "sm103"} and comfortable:
            profile = snap.family
            micro_batch = 1
            offload_encoder = False
            grad_ckpt = False
            notes = (
                f"{snap.family} {snap.gpu_gib:.1f}GiB: published B0 seq={PUBLISHED_SEQ} "
                "micro-batch=1, encoder on GPU, no grad-ckpt (B200 mb=1 ~96GiB). "
                "Resume Hub b0-full. Not Megatron."
            )
        elif snap.family == "sm120" and comfortable:
            profile = "sm120_6000d"
            micro_batch = 1
            offload_encoder = False
            grad_ckpt = True
            notes = (
                f"sm_120 {snap.gpu_gib:.1f}GiB: published B0 seq={PUBLISHED_SEQ} "
                "micro-batch=1, encoder on GPU, grad-ckpt on, Nvfp4Linear emulation. "
                "Resume Hub b0-full. CAT_YOKO_TE_NVFP4=0 if a B200 launcher is nearby. "
                "Not Megatron."
            )
        elif snap.arch == "hopper":
            profile = "hopper_h100"
            micro_batch = 1
            offload_encoder = not comfortable
            grad_ckpt = True
            notes = (
                f"Hopper SM90 {snap.gpu_gib:.1f}GiB: published B0 seq={PUBLISHED_SEQ} "
                "micro-batch=1, Nvfp4Linear emulation (no FP4 tensor core). "
                "FP8 is the documented fallback (docs/FP8_THEORY.md), not a new wrap. "
                f"{'encoder CPU offload + ' if not comfortable else ''}"
                "grad-ckpt on. Resume Hub b0-full. Not Megatron."
            )
        else:
            profile = {
                "ada": "ada",
                "ampere": "ampere_a100",
            }.get(snap.arch, "other")
            micro_batch = 1
            offload_encoder = not comfortable
            grad_ckpt = True
            notes = (
                f"{snap.arch} {snap.gpu_gib:.1f}GiB: published B0 seq={PUBLISHED_SEQ} "
                "micro-batch=1, Nvfp4Linear emulation, "
                f"{'encoder offload, ' if offload_encoder else 'encoder on GPU, '}"
                "grad-ckpt on. Resume Hub b0-full. Not Megatron."
            )

    if nvfp4 == "emu" and snap.family in {"sm100", "sm103"} and not try_run:
        notes += " TE FPROP miss or CAT_YOKO_TE_NVFP4=0 → emulation this process."

    argv = _b0_argv(
        seq_len=seq_len,
        micro_batch=micro_batch,
        offload_encoder=offload_encoder,
        grad_ckpt=grad_ckpt,
        try_run=try_run,
    )
    return B0Recipe(
        profile=profile,
        phase=phase,
        published=published,
        try_run=try_run,
        launch=launch,
        seq_len=int(seq_len),
        micro_batch=int(micro_batch),
        offload_encoder=bool(offload_encoder),
        grad_ckpt=bool(grad_ckpt),
        optim_cpu=False,
        nvfp4_path=nvfp4,
        te_env=te_env,
        megatron=megatron,
        save_full=save_full,
        resume_hub=resume_hub,
        notes=notes,
        refuse_reason=refuse_reason,
        argv=argv,
    )


def recipe_payload(snap: HwSnapshot, rec: B0Recipe) -> dict:
    snap_d = asdict(snap)
    if snap_d["cap"] is not None:
        snap_d["cap"] = list(snap_d["cap"])
    rec_d = asdict(rec)
    rec_d["argv"] = list(rec.argv)
    rec_d["save_every"] = TRY_STEPS // 4 if rec.try_run else PUBLISHED_SAVE_EVERY
    rec_d["try_steps"] = TRY_STEPS if rec.try_run else None
    return {"snapshot": snap_d, "recipe": rec_d}


def shell_export(snap: HwSnapshot, rec: B0Recipe) -> str:
    payload = recipe_payload(snap, rec)
    argv = shlex.join(rec.argv)
    te = "" if rec.te_env is None else rec.te_env
    lines = [
        f"CAT_YOKO_HW_PROFILE={shlex.quote(rec.profile)}",
        f"CAT_YOKO_HW_ARCH={shlex.quote(snap.arch)}",
        f"CAT_YOKO_HW_FAMILY={shlex.quote(snap.family)}",
        f"CAT_YOKO_HW_LAUNCH={'1' if rec.launch else '0'}",
        f"CAT_YOKO_HW_TRY={'1' if rec.try_run else '0'}",
        f"CAT_YOKO_HW_PUBLISHED={'1' if rec.published else '0'}",
        f"CAT_YOKO_HW_NVFP4={shlex.quote(rec.nvfp4_path)}",
        f"CAT_YOKO_HW_TE_ENV={shlex.quote(te)}",
        f"CAT_YOKO_HW_RESUME_HUB={shlex.quote(rec.resume_hub)}",
        f"CAT_YOKO_HW_ARGV={shlex.quote(argv)}",
        f"CAT_YOKO_HW_JSON={shlex.quote(json.dumps(payload, default=str))}",
    ]
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Probe SM/VRAM and print the B0 launch recipe. "
            "Does not train. Does not download 50B tokens. Does not start Megatron."
        )
    )
    p.add_argument("--json", action="store_true", help="print snapshot+recipe JSON (default)")
    p.add_argument("--argv", action="store_true", dest="print_argv", help="print cat_yoko.b0 flags")
    p.add_argument("--shell", action="store_true", help="print eval-able CAT_YOKO_HW_* assignments")
    p.add_argument("--family", choices=["sm100", "sm103", "sm120", "other", "cpu"], default=None)
    p.add_argument("--cap", default=None, help="compute capability, e.g. 10.0 or 9.0")
    p.add_argument("--gib", type=float, default=None, help="device memory in GiB")
    p.add_argument("--name", default=None, help="optional GPU name override")
    p.add_argument("--te-fprop", choices=["0", "1"], default=None, help="override TE NVFP4 FPROP probe")
    p.add_argument("--try", action="store_true", dest="force_try", help="force 32-step --try recipe")
    p.add_argument(
        "--probe-te",
        action="store_true",
        help="live 4096x2048 te.Linear FPROP probe (needs CUDA + TE)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cap = parse_cap(args.cap)
    fake = any(v is not None for v in (args.family, args.cap, args.gib))
    te_fprop = None if args.te_fprop is None else args.te_fprop == "1"
    if fake:
        if cap is None and args.family == "sm100":
            cap = (10, 0)
        elif cap is None and args.family == "sm103":
            cap = (10, 3)
        elif cap is None and args.family == "sm120":
            cap = (12, 0)
        fam = args.family if args.family is not None else compute_family(cap)
        snap = snapshot(
            cap=cap,
            gpu_gib=args.gib,
            family=fam,
            name=args.name,
            te_nvfp4_fprop=te_fprop,
        )
    else:
        snap = detect_snapshot(probe_te=bool(args.probe_te))
        if args.name:
            snap = snapshot(
                cap=snap.cap,
                gpu_gib=snap.gpu_gib,
                family=snap.family,
                name=args.name,
                te_nvfp4_fprop=snap.te_nvfp4_fprop if te_fprop is None else te_fprop,
                te_pytorch=snap.te_pytorch,
                env_te=snap.env_te,
            )
        elif te_fprop is not None:
            snap = snapshot(
                cap=snap.cap,
                gpu_gib=snap.gpu_gib,
                family=snap.family,
                name=snap.name,
                te_nvfp4_fprop=te_fprop,
                te_pytorch=snap.te_pytorch,
                env_te=snap.env_te,
            )
    rec = recipe_for(snap, force_try=bool(args.force_try))
    if args.shell:
        sys.stdout.write(shell_export(snap, rec))
        return 0
    if args.print_argv:
        sys.stdout.write(shlex.join(rec.argv) + "\n")
        return 0
    json.dump(recipe_payload(snap, rec), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
