#!/usr/bin/env python3
"""Run isolated operator comparisons and resume this task's paused B0 process."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--only", nargs="+", choices=("projection", "frozen_moe", "frozen_moe_actual", "attention"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    commands = [
        ("projection", [sys.executable, str(root / "operators/rocm/projection_bench.py"),
         "--device", "cuda", "--tokens", "4096", "--warmup", "2", "--repeats", "5",
         "--base", str(args.base), "--out", str(args.out / "projection.json")], {}),
        ("frozen_moe", [sys.executable, str(root / "operators/rocm/frozen_moe.py"),
         "--device", "cuda", "--tokens", "64,256,1024", "--warmup", "2", "--iterations", "5",
         "--json", str(args.out / "frozen_moe.json")], {}),
        ("frozen_moe_actual", [sys.executable, str(root / "operators/rocm/frozen_moe.py"),
         "--device", "cuda", "--base", str(args.base), "--layer", "16",
         "--tokens", "4096", "--topk", "10", "--modes", "bf16",
         "--warmup", "2", "--iterations", "5",
         "--json", str(args.out / "frozen_moe_actual.json")], {}),
        ("attention", [sys.executable, str(root / "operators/rocm/attention_bench.py"),
         "--seq-lens", "128", "512", "4096", "--dtypes", "bf16",
         "--layouts", "bhsd", "bshd", "--warmup", "1", "--repeats", "2",
         "--reference-chunk-size", "512", "--chunk-size", "512",
         "--output", str(args.out / "attention.jsonl")],
         {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}),
    ]
    if args.only:
        commands = [item for item in commands if item[0] in args.only]
    paused = False
    results = []

    def interrupted(_signum, _frame):
        raise SystemExit("operator runner interrupted")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        if args.pause_pid is not None:
            commandline = Path(f"/proc/{args.pause_pid}/cmdline").read_bytes().decode().replace("\0", " ")
            if "scripts/run_b0_rocm.py" not in commandline:
                raise ValueError("pause-pid must identify this task's B0 ROCm launcher")
            os.kill(args.pause_pid, signal.SIGSTOP)
            paused = True
            print(json.dumps({"event": "training_paused", "pid": args.pause_pid}), flush=True)
            time.sleep(1)
        for name, command, extra_env in commands:
            print(json.dumps({"event": "benchmark_start", "name": name}), flush=True)
            start = time.time()
            env = {**os.environ, "OMP_NUM_THREADS": "4", **extra_env}
            with (args.out / f"{name}.log").open("w") as log:
                try:
                    result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=180)
                    status = result.returncode
                except subprocess.TimeoutExpired:
                    status = "timeout"
            row = {"name": name, "status": status, "elapsed_s": time.time() - start}
            results.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if paused:
            os.kill(args.pause_pid, signal.SIGCONT)
            print(json.dumps({"event": "training_resumed", "pid": args.pause_pid}), flush=True)
        ledger = args.out / "run.json"
        previous = json.loads(ledger.read_text()) if ledger.exists() else []
        ledger.write_text(json.dumps(previous + results, indent=2) + "\n")


if __name__ == "__main__":
    main()
