"""DDP / FSDP smokes so wrap, no_sync, and reduce_mean actually run."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from cat_yoko.dist_util import free_tcp_port


def _env(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)


def _destroy() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _worker(
    rank: int,
    world: int,
    port: int,
    device: str,
    out_path: str,
    steps: int,
    accum: int,
    mode: str,
) -> None:
    _env(rank, world, port)
    try:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.trainer import Trainer, run_c1_chain

        cfg = CATYokoConfig.tiny()
        dtype = "bf16" if str(device).startswith("cuda") else "fp32"
        if mode == "c1":
            chain = run_c1_chain(
                cfg,
                device,
                steps=steps,
                accum=accum,
                micro_batch=1,
                ddp=True,
                dtype=dtype,
            )
            if rank == 0:
                Path(out_path).write_text(
                    json.dumps(
                        {
                            "nll": float(chain["B2"].nll),
                            "step": int(chain["B2"].step),
                            "world": world,
                            "phase": "C1",
                            "b0": float(chain["B0"].nll),
                            "b1": float(chain["B1"].nll),
                            "b2": float(chain["B2"].nll),
                            "device": device,
                        }
                    )
                )
            return
        tr = Trainer(
            cfg,
            "B0",
            device,
            steps=steps,
            accum=accum,
            ddp=mode == "ddp",
            fsdp=mode == "fsdp",
            micro_batch=1,
            seed=0,
            dtype=dtype,
        )
        if mode == "ddp" and tr.world != world:
            raise RuntimeError(f"DDP world={tr.world} expected {world}")
        out = tr.run()
        if rank == 0:
            Path(out_path).write_text(
                json.dumps(
                    {
                        "nll": float(out.nll),
                        "step": int(out.step),
                        "world": int(tr.world),
                        "phase": out.phase,
                        "device": device,
                        "mode": mode,
                        "accum": accum,
                    }
                )
            )
    finally:
        _destroy()


def _spawn(mode: str, *, device: str, world: int, steps: int, accum: int) -> dict:
    from torch.multiprocessing import spawn

    port = free_tcp_port()
    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "dist.json")
        spawn(
            _worker,
            args=(world, port, device, out_path, steps, accum, mode),
            nprocs=world,
            join=True,
        )
        row = json.loads(Path(out_path).read_text())
    nll = float(row["nll"])
    row["ok"] = int(row["step"]) == steps and nll == nll and nll > 0
    return row


def run_gloo_ddp(*, device: str = "cpu", world: int = 2, steps: int = 1, accum: int = 1) -> dict:
    """Spawn ``world`` ranks, tiny B0, gloo (or NCCL when every rank has a GPU)."""
    return _spawn("ddp", device=device, world=world, steps=steps, accum=accum)


def run_gloo_c1(*, device: str = "cpu", world: int = 2, steps: int = 1) -> dict:
    """Two-rank C1 chain so B0→B1 rebuilds the DDP reducer after freeze."""
    return _spawn("c1", device=device, world=world, steps=steps, accum=1)


def run_fsdp_one(*, device: str = "cuda", steps: int = 1) -> dict:
    """1-rank FSDP (``init_distributed(force=True)``) on a tiny graph."""
    return _spawn("fsdp", device=device, world=1, steps=steps, accum=1)
