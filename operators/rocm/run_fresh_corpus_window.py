#!/usr/bin/env python3
"""Supervise exactly 4000 B0 updates on a verified fresh corpus from step 81864."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import fresh_corpus_handoff as handoff
from operators.rocm import run_operator_switch as checks
from operators.rocm import run_round3_switch as round3
from operators.rocm.run_corpus_pass import completed_metrics, protected_sources
from operators.rocm.gpu_wait import wait_for_gpu_idle


def fresh_plan(plan):
    handoff.validate_handoff(plan)
    return {"updates": plan["updates"], "target_step": plan["target_step"],
            "target_cursor": plan["target_cursor"], "target_adam_step": plan["target_cursor"],
            "target_tokens_in_phase": plan["target_phase_tokens"], "data_rows": plan["new_nseq"],
            "audit_max_cursor": plan["absolute_cursor_origin"] + plan["new_nseq"],
            "start_next_row": 0, "target_next_row": 4000, "target_logical_row": 4000,
            "scope": "4000 new rows, with explicit absolute-to-corpus coordinates and no wrap"}


def worker_command(args, plan):
    flags = round3.common(args, args.resume, args.out / "continuation", plan["updates"], production=True)
    flags += round3.ROUND3_FLAGS + ["--accepted-run", str(args.accepted_run),
             "--extra-heldout-start-row", "32", "--extra-heldout-batches", "32",
             "--data-handoff", str(args.handoff_manifest), "--fresh-eval-data", str(args.fresh_eval_data)]
    return [args.python, "-u", str(args.source_dir / "operators/rocm/fresh_corpus_continuation.py"), *flags]


def supervise(args):
    for name in ("source_dir", "resume", "old_data", "data", "eval_data", "fresh_eval_data",
                 "corpus_manifest", "out", "lock_file", "accepted_run"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    handoff.need(args.run_updates == 4000, "authorized fresh window is exactly 4000 updates")
    handoff.need(args.out.is_dir() and args.resume == args.out / "source_checkpoint/trainable.pt",
                 "use a separate run with its immutable complete source checkpoint")
    handoff.need(not (args.out / "continuation").exists() and not (args.out / "status.json").exists(),
                 "refusing to overwrite a previous fresh run")
    for path in (args.source_dir, args.base, args.data.parent, args.old_data.parent,
                 args.eval_data.parent, args.fresh_eval_data.parent):
        handoff.need(not (args.out == path or args.out.is_relative_to(path) or path.is_relative_to(args.out)),
                     "run output must be separate from source/base/corpora")
    handoff.need(args.eval_data != args.fresh_eval_data and args.data != args.eval_data,
                 "old and fresh held-out corpora must be independent")
    state = {"status": "preflight", "supervisor_pid": os.getpid(), "child_pid": None,
             "source_checkpoint": str(args.resume), "source_dir": str(args.source_dir),
             "phase": "B0", "reference_backend": "previous_packed_production",
             "cursor_reset": False, "optimizer_reset": False, "rng_reset": False,
             "corpus_local_position_reset": True, "new_unique_corpus_downloaded": True,
             "checkpoint_every_seconds": 300, "keep_last": 3,
             "controls_existing_processes": False, "extra_heldout_start_row": 32,
             "extra_heldout_batches": 32, "selected_operators": "split_attention_and_cached_cpu_adam"}

    def update(**values):
        state.update(values, updated_at=checks.utc_now().isoformat())
        checks.write_json(args.out / "status.json", state)
        print(json.dumps(values, allow_nan=False), flush=True)

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from operators.rocm.production_continuation import validate_accepted_run
        args.accepted_run_receipt = validate_accepted_run(args.accepted_run, args.resume, args.source_dir)
        source_args = SimpleNamespace(**vars(args))
        source_args.data = args.old_data
        old = checks.cpu_check(source_args, args.resume, 0, args.out / "old_source_preflight.log", max_cursor=58593)
        plan = handoff.build_handoff(old, args.resume, args.old_data, args.data, args.fresh_eval_data,
                                     args.corpus_manifest, updates=args.run_updates)
        args.handoff = plan
        args.handoff_manifest = args.out / "data_handoff.json"
        plan_sha = handoff.write_handoff(args.handoff_manifest, plan)
        metadata = handoff.cpu_check(args, args.resume, 0, args.out / "source_preflight.log")
        window = fresh_plan(plan)
        args.native_failure_receipt = None
        verify_old, source_sha = protected_sources(args)
        extra_paths = [args.old_data, args.fresh_eval_data, args.corpus_manifest, args.handoff_manifest]
        extra_stamps = {str(path): checks.fingerprint(path) for path in extra_paths}

        def verify():
            verify_old()
            handoff.need(all(checks.fingerprint(Path(path)) == stamp for path, stamp in extra_stamps.items()),
                         "protected fresh handoff, old corpus or fresh validation changed")
            handoff.need(handoff.digest(args.handoff_manifest) == plan_sha, "immutable handoff metadata changed")

        required_free = 5 * args.resume.stat().st_size + 1024**3
        handoff.need(shutil.disk_usage(args.out).free >= required_free,
                     f"fresh run needs at least {required_free} free bytes for keep3, verified recovery and atomic save")
        update(source_sha256=source_sha, source_metadata=metadata, accepted_run=args.accepted_run_receipt,
               data_handoff=plan, data_handoff_path=str(args.handoff_manifest), data_handoff_sha256=plan_sha,
               plan=window, target_step=85864, target_cursor=51062, audit_max_cursor=window["audit_max_cursor"],
               requested_run_updates=4000, required_free_bytes=required_free, status="waiting_for_gpu")
        wait_for_gpu_idle(max_wait_seconds=None, on_event=lambda row: update(gpu_wait=row))
        verify()
        command = worker_command(args, plan)
        checks.write_json(args.out / "continuation.command.json", {"command": command})
        with (args.out / "continuation.log").open("w") as log:
            child = None
            try:
                child = subprocess.Popen(command, cwd=args.source_dir, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONOPTIMIZE="0"))
                update(status="running_continuation", child_pid=child.pid)
                seen, last = set(), -1
                while child.poll() is None:
                    rows = completed_metrics(args.out / "continuation/train/metrics.jsonl", 81864)
                    handoff.need(len(rows) <= 4000, "fresh training exceeded the authorized 4000 updates")
                    if rows and rows[-1]["step"] != last:
                        last = rows[-1]["step"]
                        update(last_completed_step=last, completed_updates=len(rows),
                               completed_fresh_rows=len(rows), actual_new_operator_flags=True,
                               actual_deterministic_algorithms=False)
                    current = args.out / "continuation/train/trainable.pt"
                    if current.exists() and checks.fingerprint(current) not in seen:
                        directory = args.out / "verified_checkpoint"
                        directory.mkdir(exist_ok=True)
                        stable = directory / "pending.pt"
                        stable.unlink(missing_ok=True)
                        try:
                            os.link(current, stable)
                            stamp = checks.fingerprint(stable)
                            step = round3.numbered_step(current.parent, stable)
                            delta = step - 81864
                            handoff.need(0 < delta <= 4000, "fresh saved step lies outside the window")
                            checked = handoff.cpu_check(args, stable, delta, args.out / "latest_checkpoint_check.log")
                            stable.replace(directory / "latest_verified.pt")
                            seen.add(stamp)
                            update(first_checkpoint_verified=True, latest_verified_checkpoint={**checked,
                                   "file": str(directory / "latest_verified.pt")}, latest_verified_at=checks.utc_now().isoformat())
                        finally:
                            stable.unlink(missing_ok=True)
                    time.sleep(2)
                code = child.wait()
            finally:
                if child is not None:
                    round3.stop_owned_child(child)
        update(child_pid=None)
        handoff.need(code == 0, "fresh worker failed; immutable and verified recovery points retained")
        result = checks.validate_run(args.out / "continuation", source=args.resume, source_step=81864,
                                    steps=4000, max_updates=4000, variant="attention", source_dir=args.source_dir,
                                    expected_names=metadata["trainable_names"], args=args, eval_batches=32,
                                    reference_backend="previous_packed_production")
        round3.validate_round3_report(args, args.out / "continuation", "combined", 4000, timing=False,
                                     reference_backend="previous_packed_production")
        round3.validate_training_policy(args.out / "continuation", source_step=81864, updates=4000, deterministic_training=False)
        final = handoff.cpu_check(args, Path(result["checkpoint"]), 4000, args.out / "final_checkpoint_check.log")
        handoff.need(final["source_step"] == 85864 and final["source_stream_i"] == final["source_adam_step"] == 51062
                     and final["source_logical_row"] == 4000 and final["source_tokens_in_phase"] == 371394560,
                     "fresh final state differs from the exact endpoint")
        from operators.rocm.continuation_quality import validate_quality_report
        quality_path = args.out / "continuation/quality.json"
        quality = validate_quality_report(quality_path, source_dir=args.source_dir, start_row=32, batches=32)
        handoff.need(Path(quality["eval_data"]).resolve() == args.eval_data
                     and quality["initial_eval"]["step"] == 81864 and quality["final_eval"]["step"] == 85864,
                     "old paired evaluation differs from this fresh window")
        fresh_path = args.out / "continuation/fresh_quality.json"
        fresh_quality = handoff.validate_fresh_quality_report(fresh_path, source_dir=args.source_dir, handoff=plan)
        exercised = json.loads((args.out / "continuation/data_handoff.json").read_text())
        handoff.need(exercised.get("status") == "completed" and exercised.get("data_handoff") == plan
                     and exercised.get("data_handoff_sha256") == plan_sha and exercised.get("completed_updates") == 4000
                     and exercised.get("first_training_row") == 0 and exercised.get("last_training_row") == 3999,
                     "worker did not prove the explicit row mapping")
        verify()
        update(status="complete", final_checkpoint_verified=final, final_quality_done=True,
               final_quality={"path": str(quality_path), "sha256": handoff.digest(quality_path),
                              "initial_eval": quality["initial_eval"], "final_eval": quality["final_eval"]},
               fresh_quality_done=True, fresh_quality={"path": str(fresh_path), "sha256": handoff.digest(fresh_path),
                              "initial_eval": fresh_quality["initial_eval"], "final_eval": fresh_quality["final_eval"]})
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "resume", "base", "old-data", "data", "eval-data", "fresh-eval-data",
                 "corpus-manifest", "out", "lock-file", "accepted-run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-updates", type=int, default=4000)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("fresh supervisor signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        supervise(args)
    except BaseException as error:
        path = args.out / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status="failed", error=type(error).__name__ + ": " + str(error), child_pid=None,
                     updated_at=checks.utc_now().isoformat())
        checks.write_json(path, state)
        raise


if __name__ == "__main__":
    main()
