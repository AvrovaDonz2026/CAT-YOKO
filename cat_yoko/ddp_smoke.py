"""Two-rank gloo smoke so DDP wrap / reduce_mean actually run."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _worker(rank: int, world: int, port: int, device: str, out_path: str, steps: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.trainer import Trainer

        cfg = CATYokoConfig.tiny()
        tr = Trainer(
            cfg,
            "B0",
            device,
            steps=steps,
            accum=1,
            ddp=True,
            micro_batch=1,
            seed=0,
        )
        if tr.world != world:
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
                    }
                )
            )
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_gloo_ddp(*, device: str = "cpu", world: int = 2, steps: int = 1) -> dict:
    """Spawn ``world`` ranks, one tiny B0 step, gloo. Returns rank-0 metrics."""
    from torch.multiprocessing import spawn

    port = _free_port()
    with tempfile.TemporaryDirectory() as td:
        out_path = str(Path(td) / "ddp.json")
        spawn(
            _worker,
            args=(world, port, device, out_path, steps),
            nprocs=world,
            join=True,
        )
        row = json.loads(Path(out_path).read_text())
    nll = float(row["nll"])
    row["ok"] = int(row["step"]) == steps and nll == nll and nll > 0
    return row
