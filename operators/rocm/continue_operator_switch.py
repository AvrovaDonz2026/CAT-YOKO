#!/usr/bin/env python3
"""Resume a verified operator decision without repeating benchmark updates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import run_operator_switch as helpers
from operators.rocm.gpu_wait import wait_for_gpu_idle


def read_json(path):
    result = json.loads(path.read_text())
    helpers.require_finite(result)
    return result


def logged_metadata(path):
    for line in reversed(path.read_text().splitlines()):
        try:
            result = json.loads(line)
        except ValueError:
            continue
        if isinstance(result, dict) and result.get("checkpoint_verified") is True:
            return result
    raise ValueError("comparison source preflight log has no verified JSON metadata")


def safe_native_fallback(directory: Path, source_step: int):
    """Fail closed if any saved state or recorded optimizer update exists."""
    train = directory / "train"
    if train.exists() and any(train.glob("*.pt*")):
        return False
    try:
        report = read_json(directory / "parity.json") if (directory / "parity.json").exists() else {}
        if report.get("updates", 0) != 0:
            return False
        metrics = train / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
        return not any("tok_s" in row or row.get("step", source_step) > source_step for row in rows)
    except (OSError, ValueError, TypeError):
        return False


def prepare(args):
    args.comparison = args.comparison.resolve()
    decision = read_json(args.comparison / "decision.json")
    previous = read_json(args.comparison / "status.json")
    if previous.get("status") != "evaluated_only" or previous.get("child_pid") is not None:
        raise ValueError("comparison must have completed evaluate-only with no active child")
    args.resume = Path(decision["source_checkpoint"])
    args.benchmark_steps, args.preflight_dir, args.baseline_run = decision["benchmark_steps"], None, None
    helpers.validate_args(args)
    if args.out == args.comparison or args.out.is_relative_to(args.comparison) or args.comparison.is_relative_to(args.out):
        raise ValueError("continuation output must be separate from the comparison")
    if helpers.choose_variant(decision["baselines"], decision["candidates"])["selected"] != decision["selected"]:
        raise ValueError("recorded operator choice does not match the conservative selection")
    if decision.get("selected_flags") != helpers.FLAGS[decision["selected"]]:
        raise ValueError("recorded candidate flags disagree with the selection")
    if helpers.digest_file(args.resume) != decision["source_sha256"]:
        raise helpers.SourceIntegrityError("fixed source SHA differs from the comparison")
    return decision, previous


def supervise(args):
    decision, previous = prepare(args)
    args.out.mkdir(parents=True)
    state = {"status": "starting", "supervisor_pid": os.getpid(), "comparison": str(args.comparison),
             "source_checkpoint": str(args.resume), "source_sha256": decision["source_sha256"], "stages": [],
             "controls_existing_processes": False, "benchmark_updates_count_toward_continuation": False,
             "budget_scope": "Trainer.run; setup, parity and held-out evaluation are outside the core deadline timer"}

    def update(values):
        state.update(values, updated_at=helpers.utc_now().isoformat())
        helpers.write_json(args.out / "status.json", state)
        print(json.dumps(values, allow_nan=False), flush=True)

    try:
        metadata = helpers.cpu_check(args, args.resume, args.benchmark_steps, args.out / "source_check.log")
        preflight = logged_metadata(args.comparison / "source_preflight.log")
        if metadata != previous.get("source_metadata") or metadata != preflight:
            raise helpers.SourceIntegrityError("source metadata differs from the completed comparison")
        baselines, candidates = [], []
        for group, destination in (("baselines", baselines), ("candidates", candidates)):
            for index, recorded in enumerate(decision[group]):
                name = ("baseline_before" if index == 0 else "baseline_after") if group == "baselines" else recorded["variant"] + "_steps"
                receipt = read_json(args.comparison / (name + ".validation.json"))
                if receipt != recorded:
                    raise ValueError("validation receipt differs from the recorded decision")
                if not recorded.get("validated"):
                    destination.append(recorded)
                    continue
                directory = Path(recorded["checkpoint"]).parent.parent
                result = helpers.validate_run(directory, source=args.resume, source_step=metadata["source_step"],
                    steps=args.benchmark_steps, variant=recorded["variant"], source_dir=args.source_dir,
                    expected_names=metadata["trainable_names"], args=args)
                if any(result[key] != recorded.get(key) for key in result):
                    raise ValueError("actual benchmark differs from its saved validation")
                if group == "baselines" or recorded["variant"] == decision["selected"]:
                    result["verified_checkpoint_metadata"] = helpers.cpu_check(args, Path(result["checkpoint"]),
                        args.benchmark_steps, args.out / (name + ".checkpoint_check.log"))
                destination.append(result)
        if len(baselines) != 2 or not all(row.get("validated") for row in baselines):
            raise ValueError("both native baselines must validate before continuation")
        if helpers.choose_variant(baselines, candidates)["selected"] != decision["selected"]:
            raise ValueError("revalidated benchmark choice differs from the recorded decision")
        protected = [args.resume, args.data, args.eval_data, *(args.source_dir / "cat_yoko").glob("*.py"),
                     *(args.source_dir / "operators/rocm").glob("*.py")]
        snapshots = {str(path): helpers.fingerprint(path) for path in protected}

        def assert_unchanged():
            if any(helpers.fingerprint(Path(path)) != value for path, value in snapshots.items()) or helpers.digest_file(args.resume) != decision["source_sha256"]:
                raise helpers.SourceIntegrityError("source, code or corpus changed during continuation")

        remaining, variant = metadata["source_data_rows"] - metadata["source_stream_i"], decision["selected"]
        helpers.write_json(args.out / "decision.json", decision)
        update({"status": "verified", "selected": variant, "remaining_updates": remaining})
        for attempt in range(2):
            update({"status": "waiting_for_gpu", "selected": variant})
            wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait, on_event=lambda event: update({"gpu_wait": event}))
            assert_unchanged()
            hours = helpers.remaining_hours(args.deadline)
            if hours <= 0 or remaining <= 0:
                update({"status": "deadline_expired" if hours <= 0 else "corpus_complete"})
                break
            directory = args.out / ("continuation" if attempt == 0 else "native_fallback")
            flags = helpers.continuation_args(args, directory, remaining, hours, variant)
            command = [args.python, "-u", str(args.source_dir / "operators/rocm/candidate_bench.py"), *flags]
            receipt = {"variant": variant, "command": command, "max_hours": hours, "resume": str(args.resume)}
            helpers.write_json(args.out / (directory.name + ".command.json"), receipt)
            with (args.out / (directory.name + ".log")).open("w") as log:
                child = subprocess.Popen(command, cwd=args.source_dir, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED="1"))
                update({"status": "running", "child_pid": child.pid, "selected": variant})
                try:
                    code = child.wait(timeout=hours * 3600 + args.stage_timeout_seconds)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
                    receipt["timed_out"] = True
                except BaseException:
                    child.kill()
                    child.wait()
                    raise
            receipt.update(exit_code=code)
            state["stages"].append(receipt)
            helpers.write_json(args.out / (directory.name + ".receipt.json"), receipt)
            update({"child_pid": None})
            assert_unchanged()
            if code:
                if attempt == 0 and variant != "native" and safe_native_fallback(directory, metadata["source_step"]):
                    update({"native_fallback": "candidate failed before any recorded update or checkpoint"})
                    variant = "native"
                    continue
                raise RuntimeError("continuation failed; saved state and receipts are preserved for manual recovery")
            result = helpers.validate_run(directory, source=args.resume, source_step=metadata["source_step"], steps=None,
                variant=variant, max_updates=remaining, source_dir=args.source_dir,
                expected_names=metadata["trainable_names"], args=args, eval_batches=32)
            result["verified_checkpoint_metadata"] = helpers.cpu_check(args, Path(result["checkpoint"]),
                result["updates"], args.out / "final_checkpoint_check.log")
            assert_unchanged()
            update({"status": "complete", "continuation": result})
            break
    except BaseException as error:
        update({"status": "failed", "child_pid": None, "error": f"{type(error).__name__}: {error}"})
        helpers.write_json(args.out / "summary.json", state)
        raise
    helpers.write_json(args.out / "summary.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("comparison", "source-dir", "out", "base", "data", "eval-data"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--gpu-idle-max-wait", type=float, default=3600)
    parser.add_argument("--stage-timeout-seconds", type=float, default=3600)
    return supervise(parser.parse_args(argv))


if __name__ == "__main__":
    main()
