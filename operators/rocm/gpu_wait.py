"""Read-only ROCm GPU availability wait through KFD's registered process list."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

KFD_PROC_ROOT = Path("/sys/class/kfd/kfd/proc")


def other_gpu_pids(proc_root: Path = KFD_PROC_ROOT) -> list[int]:
    """Return other registered KFD PIDs; never infer idle from unreadable sysfs."""
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError(
            f"cannot read ROCm GPU process list {proc_root}: {exc}; refusing to assume GPU idle"
        ) from exc
    own_pid = os.getpid()
    return sorted(int(entry.name) for entry in entries if entry.name.isdigit()
                  and int(entry.name) > 0 and int(entry.name) != own_pid)


def wait_for_gpu_idle(
    *,
    max_wait_seconds: float | None = None,
    proc_root: Path = KFD_PROC_ROOT,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Observe two consecutive idle polls, two seconds apart, before returning.

    Other processes are only observed. This does not suspend, signal, or change
    any process, and it does not reserve the GPU after the last observation.
    The default wait has no time limit. Status reports repeat every 60 seconds.
    """
    if max_wait_seconds is not None and max_wait_seconds <= 0:
        raise ValueError("GPU idle max wait must be positive")
    if on_event is None:
        def on_event(event: dict) -> None:
            print(json.dumps(event), flush=True)
    begin = time.monotonic()
    last_report = float("-inf")
    idle_polls = 0
    while True:
        elapsed = time.monotonic() - begin
        pids = other_gpu_pids(proc_root)
        idle_polls = idle_polls + 1 if not pids else 0
        if elapsed - last_report >= 60:
            on_event({"event": "gpu_wait", "elapsed_s": elapsed,
                      "other_gpu_pids": pids, "idle_confirmations": idle_polls,
                      "proc_root": str(proc_root)})
            last_report = elapsed
        if idle_polls >= 2:
            result = {"event": "gpu_idle", "elapsed_s": elapsed,
                      "other_gpu_pids": [], "idle_confirmations": idle_polls,
                      "proc_root": str(proc_root)}
            on_event(result)
            return result
        if max_wait_seconds is not None and elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"GPU remained busy or unconfirmed after {elapsed:.1f}s; other KFD PIDs: {pids}"
            )
        sleep_seconds = 2.0
        if max_wait_seconds is not None:
            sleep_seconds = min(sleep_seconds, max_wait_seconds - elapsed)
        time.sleep(sleep_seconds)
