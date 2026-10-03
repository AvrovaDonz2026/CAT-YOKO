#!/usr/bin/env python3
"""Run isolated operator experiments after a verified long run releases GPU.

Only experiment children are launched. The queue reads the training supervisor
and KFD process list; it never signals any existing process. Every comparison
resumes one immutable copy of the final checkpoint, including Adam and cursor.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm.gpu_wait import wait_for_gpu_idle


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def wait_for_training(run: Path, update, *, max_seconds: float, sleep=time.sleep,
                      clock=time.monotonic) -> dict:
    begin = clock()
    while True:
        status = json.loads((run / "status.json").read_text())
        state = status.get("status")
        if state == "complete":
            checkpoint = run / "long/train/trainable.pt"
            report = json.loads((run / "long/parity.json").read_text())
            if not checkpoint.is_file() or report.get("status") != "training_complete":
                raise ValueError("completed source must have a final checkpoint and training report")
            if not report.get("source_optimizer_present") or not report.get("save_optimizer"):
                raise ValueError("source training must preserve and save Adam state")
            return status
        if state == "failed":
            raise RuntimeError(f"source training failed: {status.get('error', 'see source status')}")
        elapsed = clock() - begin
        if elapsed >= max_seconds:
            raise TimeoutError("source training did not finish within the queue wait limit")
        update({"status": "waiting_for_training", "source_status": state,
                "source_child_pid": status.get("child_pid"), "wait_elapsed_s": elapsed})
        sleep(min(60, max_seconds - elapsed))


def steady_metrics(path: Path, *, discard: int = 5) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "tok_s" in row][discard:]
    if not rows or any(not math.isfinite(row["tok_s"]) or row["tok_s"] <= 0 for row in rows):
        raise ValueError("missing or invalid steady update metrics")
    return {"measured_updates": len(rows), "discarded_warmup_updates": discard,
            "median_loop_tokens_per_second": statistics.median(row["tok_s"] for row in rows),
            "mean_loop_tokens_per_second": statistics.mean(row["tok_s"] for row in rows),
            "peak_allocated_mib": max(row["mem_mib"] for row in rows),
            "scope": "Trainer loop metrics; excludes initial/final evaluation and final checkpoint save"}


def benchmark_args(args, checkpoint: Path, out: Path, steps: int) -> list[str]:
    return ["--base", str(args.base), "--resume", str(checkpoint), "--out", str(out),
            "--data", str(args.data), "--eval-data", str(args.eval_data), "--eos-id", "1",
            "--moe-layout", "shared-storage", "--parity-seqs", "4096", "--seq-len", "4096",
            "--run-steps", str(steps), "--save-every", "0", "--keep-last", "1",
            "--eval-batches", "2", "--save-optim", "--deterministic-parity", "--reference-repeat",
            "--wait-gpu-idle", "--gpu-idle-max-wait", "3600"]


def supervise(args) -> dict:
    for key in ("training_run", "source_dir", "base", "data", "eval_data", "out"):
        setattr(args, key, getattr(args, key).resolve())
    if not math.isfinite(args.max_wait_hours) or args.max_wait_hours <= 0:
        raise ValueError("max-wait-hours must be finite and positive")
    if args.data == args.eval_data:
        raise ValueError("training and held-out data must differ")
    for protected in (args.training_run, args.source_dir, args.base, args.data.parent, args.eval_data.parent):
        if args.out == protected or args.out.is_relative_to(protected):
            raise ValueError("queue output must be outside training, source, model and data directories")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite or restart an existing experiment queue")
    for name in ("candidate_bench.py", "profile_training.py", "packed_attention_bench.py", "grad_norm_bench.py"):
        if not (args.source_dir / "operators/rocm" / name).is_file():
            raise FileNotFoundError(f"isolated source lacks {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "queue_pid": os.getpid(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_training_run": str(args.training_run), "source_dir": str(args.source_dir),
              "controls_other_gpu_jobs": False, "stages": []}

    def update(values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.out / "status.json", status)
        print(json.dumps(values), flush=True)

    def run_stage(name, script, flags):
        # Observe before every child, including after an earlier experiment.
        update({"status": "waiting_for_gpu", "stage": name})
        wait_for_gpu_idle(max_wait_seconds=3600, on_event=lambda event: update({"gpu_wait": event}))
        command = [args.python, "-u", str(args.source_dir / "operators/rocm" / script), *flags]
        begin = time.monotonic()
        with (args.out / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=args.source_dir, stdout=log,
                                     stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            update({"status": "running", "stage": name, "child_pid": child.pid, "child_command": command})
            try:
                code = child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                # This is only the experiment child created immediately above.
                child.kill()
                code = child.wait()
        status["stages"].append({"name": name, "exit_code": code,
                                 "elapsed_s": time.monotonic() - begin,
                                 "log": f"{name}.log"})
        update({"child_pid": None})
        return code == 0

    update({"status": "starting"})
    try:
        source = wait_for_training(args.training_run, update, max_seconds=args.max_wait_hours * 3600)
        update({"status": "waiting_for_gpu", "source_final_step": source.get("final_step")})
        wait_for_gpu_idle(max_wait_seconds=args.max_wait_hours * 3600,
                          on_event=lambda event: update({"gpu_wait": event}))
        checkpoint = args.out / "source_checkpoint/trainable.pt"
        checkpoint.parent.mkdir()
        shutil.copy2(args.training_run / "long/train/trainable.pt", checkpoint)
        update({"immutable_source_checkpoint": str(checkpoint), "source_bytes": checkpoint.stat().st_size})
        # Verify sufficient remaining real rows and lossless Adam before any
        # comparison. Never allow a shorter source to wrap silently or reset
        # its cursor merely to make an experiment run.
        preflight = subprocess.run([
            args.python, "-c",
            "import json,sys; from pathlib import Path; "
            "from operators.rocm.profile_training import check_resume; "
            "print(json.dumps(check_resume(Path(sys.argv[1]),Path(sys.argv[2]),4096,20)))",
            str(checkpoint), str(args.data)], cwd=args.source_dir, capture_output=True,
            text=True, timeout=120,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
        (args.out / "source_preflight.log").write_text(preflight.stdout + preflight.stderr)
        if preflight.returncode:
            raise ValueError("source checkpoint failed CPU Adam/cursor preflight; see source_preflight.log")
        update({"source_metadata": json.loads(preflight.stdout.splitlines()[-1])})
        packed_ok = run_stage("packed_operator", "packed_attention_bench.py", [
            "--data", str(args.data), "--eos-id", "1", "--seq-lens", "4096",
            "--windows", "8192", "32", "--layouts", "packed_qkv", "cross_cache",
            "--warmup", "3", "--repeats", "5", "--wait-gpu-idle",
            "--output", str(args.out / "packed_operator.jsonl")])
        norm_ok = run_stage("norm_operator", "grad_norm_bench.py", [
            "--resume", str(checkpoint), "--json", str(args.out / "norm_operator.json"), "--wait-gpu-idle"])
        baseline_ok = run_stage("baseline_steps", "candidate_bench.py",
                                benchmark_args(args, checkpoint, args.out / "baseline_steps", 20))
        run_stage("baseline_profile", "profile_training.py",
                  benchmark_args(args, checkpoint, args.out / "baseline_profile", 6))
        flags = (["--packed-attention"] if packed_ok else []) + (["--batched-grad-norm"] if norm_ok else [])
        status["candidate_flags"] = flags
        candidate_ok = False
        if flags and baseline_ok:
            candidate_ok = run_stage("candidate_steps", "candidate_bench.py",
                benchmark_args(args, checkpoint, args.out / "candidate_steps", 20) + flags)
            if candidate_ok:
                run_stage("candidate_profile", "profile_training.py",
                    benchmark_args(args, checkpoint, args.out / "candidate_profile", 6) + flags)
        if baseline_ok and candidate_ok:
            baseline = steady_metrics(args.out / "baseline_steps/train/metrics.jsonl")
            candidate = steady_metrics(args.out / "candidate_steps/train/metrics.jsonl")
            status["comparison"] = {"baseline": baseline, "candidate": candidate,
                "loop_speedup": candidate["median_loop_tokens_per_second"] / baseline["median_loop_tokens_per_second"],
                "same_source_checkpoint_and_adam": True, "experimental_updates_per_run": 20,
                "production_training_modified": False,
                "notes": "Idle checks do not reserve GPU; inspect PID observations and traces before attributing small gains."}
        update({"status": "complete" if all(stage["exit_code"] == 0 for stage in status["stages"])
                else "complete_with_rejections", "child_pid": None})
        write_json(args.out / "summary.json", status)
        return status
    except Exception as error:
        update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-wait-hours", type=float, default=36)
    args = parser.parse_args(argv)
    supervise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
