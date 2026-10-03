#!/usr/bin/env python3
"""Compare synchronized updates, then continue one fixed B0 checkpoint.

This standard-library supervisor only owns its children. Trial updates never
advance the real stream; checkpoints and reports are retained for review.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import run_operator_switch as helpers
from operators.rocm.gpu_wait import wait_for_gpu_idle

FLAGS = {"baseline": [], "gpu_adam": ["--gpu-fp32-adam"],
         "bucketed": ["--bucketed-attention"],
         "combined": ["--gpu-fp32-adam", "--bucketed-attention"]}
SPEEDS = ("median_completed_update_tokens_s", "aggregate_completed_update_tokens_s",
          "median_tokens_per_second")


def read_json(path):
    value = json.loads(path.read_text())
    helpers.require_finite(value)
    return value


def need(condition, message):
    if not condition:
        raise ValueError(message)


def numerical_adam_pass(report, source_sha, metadata, resume):
    need(report.get("status") == "complete" and report.get("device") == "cuda", "GPU Adam micro failed")
    need(report.get("source_sha256") == source_sha and report.get("source_step") == metadata["source_step"]
         and Path(report.get("source_checkpoint", "")).resolve() == resume, "GPU Adam micro used another source")
    need(report.get("checkpoint_moments_cpu_fp32") is True and report.get("persistent_fp32_master") is False,
         "GPU Adam did not export CPU FP32 moments")
    parity = report.get("parity", [])
    need(len(parity) >= 5 and [row.get("invocation_step") for row in parity] == list(range(1, len(parity) + 1)),
         "GPU Adam needs at least five consecutive numerical updates")
    for row in [*parity, report.get("terminal", {})]:
        need(row.get("pass") is True and row.get("parameters") == 132 and row.get("all_weights_bitwise") is True
             and row.get("gate", {}).get("fp32_moment_atol") == 1e-8
             and row.get("gate", {}).get("fp32_moment_rtol") == 1e-6, "GPU Adam numerical gate failed")
    need(report.get("installation", {}).get("state_exports", 0) >= 1, "GPU Adam export was not exercised")


def micro_gates(args, metadata, source_sha):
    status = read_json(args.micro_dir / "status.json")
    need(status.get("status") == "complete", "microbench controller must have completed")
    codes = {row["stage"]: row.get("exit_code") for row in status.get("results", [])}
    result = {}
    for variant in ("gpu_adam", "bucketed"):
        try:
            need(codes.get(variant) == 0, variant + " micro child did not exit successfully")
            names = (["gpu_adam.py", "gpu_adam_bench.py"] if variant == "gpu_adam" else
                     ["bucketed_attention.py", "bucketed_attention_bench.py", "packed_attention.py"])
            hashes = status.get("file_sha256", {})
            need(all(hashes.get("operators/rocm/" + name) == helpers.digest_file(args.source_dir / "operators/rocm" / name)
                     for name in names), variant + " micro source hashes differ from the isolated source")
            if variant == "gpu_adam":
                numerical_adam_pass(read_json(args.micro_dir / "gpu_adam.json"), source_sha, metadata, args.resume)
            else:
                need(status.get("source_sha256") == source_sha and status.get("source_step") == metadata["source_step"]
                     and Path(status.get("source_checkpoint", "")).resolve() == args.resume,
                     "bucketed micro lacks the fixed source identity")
                rows = [json.loads(line) for line in (args.micro_dir / "bucketed.jsonl").read_text().splitlines() if line.strip()]
                helpers.require_finite(rows)
                summaries = [row for row in rows if row.get("event") == "summary"]
                summary = summaries[-1] if summaries else {}
                cases = [row for row in rows if row.get("event") == "candidate"]
                need(summary.get("passed") == len(cases) > 0 and summary.get("failed") == summary.get("errors") == 0
                     and summary.get("bucketed_exercised", 0) > 0, "bucketed micro failed or was not exercised")
                need(all(row.get("status") == "pass" and row.get("output", {}).get("pass") is True
                         and set(row.get("gradients", {})) == {"dq", "dk", "dv"}
                         and all(error.get("pass") is True for error in row["gradients"].values()) for row in cases),
                     "bucketed output or input gradient failed")
                real = [row for row in cases if row.get("case") == "real"]
                need({(row.get("layout"), row.get("candidate")) for row in real} ==
                     {(layout, candidate) for layout in ("packed_qkv", "cross_cache") for candidate in ("packed", "bucketed")},
                     "bucketed micro lacks both real-data layouts")
                for row in real:
                    docs = row.get("documents", {})
                    need(Path(docs.get("path", "")).resolve() == args.data and docs.get("row") == metadata["source_stream_i"]
                         and docs.get("eos_id") == 1 and docs.get("source_pack_width") == 4096 and row.get("dtype") == "bf16",
                         "bucketed micro used a different packed row")
                environment = next(row for row in rows if row.get("event") == "environment")
                need(environment.get("window") == 8192 and environment.get("shape") == [1, 16, 4096, 128]
                     and environment.get("kv_heads") == 2 and environment.get("merge_heads_included") is True,
                     "bucketed micro changed the production attention shape")
            result[variant] = {"pass": True}
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
            result[variant] = {"pass": False, "reason": str(error)}
    return result


def validate_timing(path, updates):
    report = read_json(path)
    need(report.get("status") == "complete" and report.get("incomplete_update") is False
         and report.get("warmup_updates") == 5 and report.get("started_updates") == report.get("completed_updates") == updates
         and report.get("measured_updates") == updates - 5 and report.get("discarded_updates") == 5,
         "synchronized update timing is incomplete")
    samples = report.get("samples", [])
    need(len(samples) == updates and [row.get("invocation_update") for row in samples] == list(range(1, updates + 1)),
         "synchronized update samples are not consecutive")
    for row in samples:
        need(row.get("tokens") == 4096 and isinstance(row.get("elapsed_s"), (float, int)) and row["elapsed_s"] > 0
             and math.isclose(row.get("completed_update_tokens_s", 0), 4096 / row["elapsed_s"], rel_tol=1e-10),
             "invalid completed update timing sample")
    steady = samples[5:]
    recomputed = {"median_completed_update_tokens_s": statistics.median(row["completed_update_tokens_s"] for row in steady),
                  "aggregate_completed_update_tokens_s": sum(row["tokens"] for row in steady) / sum(row["elapsed_s"] for row in steady)}
    need(report.get("steady_tokens") == (updates - 5) * 4096 and math.isclose(report.get("sum_steady_elapsed_s", 0),
         sum(row["elapsed_s"] for row in steady), rel_tol=1e-10), "timing aggregate token/elapsed totals disagree")
    need(all(math.isclose(report.get(key, 0), value, rel_tol=1e-10) for key, value in recomputed.items()),
         "timing summary differs from raw synchronized samples")
    return recomputed


def validate_trial(args, directory, metadata, variant, *, steps, max_updates=None):
    result = helpers.validate_run(directory, source=args.resume, source_step=metadata["source_step"], steps=steps,
        variant="attention", source_dir=args.source_dir, expected_names=metadata["trainable_names"], args=args,
        max_updates=max_updates, eval_batches=2 if steps is not None else 32)
    report = read_json(directory / "next_operators.json")
    gpu, bucket = "--gpu-fp32-adam" in FLAGS[variant], "--bucketed-attention" in FLAGS[variant]
    need(report.get("status") == "completed" and report.get("patches_installed_after_native_reference") is True
         and report.get("gpu_fp32_adam") is gpu and report.get("bucketed_attention") is bucket
         and report.get("sync_update_timing") is (steps is not None), "next operator flags do not match this run")
    expected = {"operators/rocm/next_candidate_bench.py", "operators/rocm/candidate_bench.py"}
    expected.update("operators/rocm/" + name for enabled, name in
                    ((gpu, "gpu_adam.py"), (bucket, "bucketed_attention.py"), (steps is not None, "update_timing.py")) if enabled)
    hashes = report.get("file_sha256", {})
    if set(hashes) != expected or any(helpers.digest_file(args.source_dir / name) != value for name, value in hashes.items()):
        raise helpers.SourceIntegrityError("next operator source hashes differ from the isolated source")
    if gpu:
        calls = report.get("gpu_fp32_adam_calls", {})
        need(calls.get("state_loads", 0) >= 1 and calls.get("state_exports", 0) >= 1
             and calls.get("parameter_updates") == calls.get("optimized_calls") == 132 * result["updates"]
             and calls.get("persistent_fp32_master") is False, "GPU Adam update/load/export counts disagree")
    if bucket:
        need(all(report.get(key, {}).get("bucketed_calls", 0) > 0 for key in
                 ("bucketed_attention_parity_calls", "bucketed_attention_calls")), "bucketed attention was not exercised")
    result.update(variant=variant)
    if steps is not None:
        result.update(validate_timing(directory / "update_timing.json", steps))
    metrics = [json.loads(line) for line in (directory / "train/metrics.jsonl").read_text().splitlines() if line.strip()]
    need(all(row.get("actual_optimizer_device") == ("cuda" if gpu else "cpu") and row.get("bucketed_attention") is bucket
             for row in metrics if "tok_s" in row), "training metrics reported another optimizer/operator")
    result["verified_checkpoint_metadata"] = helpers.cpu_check(args, Path(result["checkpoint"]), result["updates"],
                                                                args.out / (directory.name + ".checkpoint_check.log"))
    return result


def choose_variant(baselines, candidates):
    need(len(baselines) == 2 and all(row.get("validated") is True for row in baselines), "both packed baselines must validate")
    limits = {key: max(row[key] for row in baselines) for key in SPEEDS}
    need(all(math.isfinite(value) and value > 0 for value in limits.values()), "invalid baseline throughput")
    eligible = [row for row in candidates if row.get("validated") is True and
                all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] >= 1.05 * limits[key] for key in SPEEDS)]
    winner = max(eligible, key=lambda row: row[SPEEDS[0]]) if eligible else {"variant": "baseline"}
    return {"selected": winner["variant"], "selected_flags": FLAGS[winner["variant"]], "minimum_gain": 0.05,
            "conservative_baseline": limits, "eligible": [row["variant"] for row in eligible],
            "reason": "all three throughput measures exceed both baselines by 5%" if eligible else "retain validated packed attention"}


def validate_profile(args, directory, metadata):
    result = helpers.validate_run(directory, source=args.resume, source_step=metadata["source_step"], steps=6,
        variant="attention", source_dir=args.source_dir, expected_names=metadata["trainable_names"], args=args)
    result["verified_checkpoint_metadata"] = helpers.cpu_check(args, Path(result["checkpoint"]), 6,
                                                               args.out / "profile_current.checkpoint_check.log")
    summary = read_json(directory / "profile/summary.json")
    need(summary.get("status") == "complete" and summary.get("recorded_updates") == summary.get("requested_active_updates") == 4
         and summary.get("completed_updates") == 6 and summary.get("warmup_updates") == 2
         and summary.get("phases", {}).get("backward", {}).get("device_work_us", 0) > 0
         and summary.get("top_steady_operators_by_device_work") and summary.get("top_steady_operators_by_self_cpu"),
         "profile did not record four updates with backward GPU attribution and steady tables")
    return {**result, "profile_summary": str(directory / "profile/summary.json"), "participates_in_selection": False}


def trial_args(args, directory, updates, variant, *, hours=None):
    flags = (helpers.benchmark_args(args, directory, updates) if hours is None else
             helpers.continuation_args(args, directory, updates, hours, "attention"))
    if hours is None:
        flags += ["--packed-attention", "--sync-update-timing"]
    flags += FLAGS[variant]
    if "--gpu-fp32-adam" in flags:
        flags += ["--gpu-adam-gate", str(args.micro_dir / "gpu_adam.json")]
    return flags


def prepare(args):
    args.preflight_dir = None
    helpers.validate_args(args)
    args.micro_dir = args.micro_dir.resolve()
    need(args.benchmark_steps == 20, "this round requires exactly 20 comparison updates")
    protected = (args.source_dir, args.resume.parent, args.base, args.data.parent, args.eval_data.parent, args.micro_dir)
    need(not any(args.out == path or args.out.is_relative_to(path) or path.is_relative_to(args.out) for path in protected),
         "output must be separate from all protected sources and micro reports")
    need(args.micro_dir.is_dir(), "micro-dir must contain completed independent microbench reports")
    if args.baseline_run is not None:
        need(args.baseline_run != args.out and not args.baseline_run.is_relative_to(args.out)
             and not args.out.is_relative_to(args.baseline_run), "baseline reuse and output must be separate")
    for name in ("next_candidate_bench.py", "gpu_adam.py", "bucketed_attention.py", "update_timing.py"):
        need((args.source_dir / "operators/rocm" / name).is_file(), "isolated source lacks " + name)


def supervise(args):
    prepare(args)
    args.out.mkdir(parents=True)
    state = {"status": "starting", "supervisor_pid": os.getpid(), "stages": [], "source_checkpoint": str(args.resume),
             "source_sha256": helpers.digest_file(args.resume), "controls_existing_processes": False,
             "benchmark_updates_count_toward_continuation": False, "checkpoints_removed": False,
             "deadline_utc": args.deadline.isoformat(),
             "budget_scope": "Trainer.run; setup, parity and held-out evaluation are outside the core deadline timer"}
    helpers.write_json(args.out / "status.json", state)
    waiting_since = time.monotonic()
    while read_json(args.micro_dir / "status.json").get("status") == "running":
        state.update(status="waiting_for_micro", updated_at=helpers.utc_now().isoformat())
        helpers.write_json(args.out / "status.json", state)
        need(time.monotonic() - waiting_since < args.gpu_idle_max_wait, "microbench completion wait timed out")
        time.sleep(2)
    protected = [args.resume, args.data, args.eval_data, *(args.source_dir / "cat_yoko").glob("*.py"),
                 *(args.source_dir / "operators/rocm").glob("*.py"), *args.micro_dir.glob("*.json*"),
                 *([args.base] if args.base.is_file() else args.base.rglob("*.safetensors"))]
    fingerprints = {str(path): helpers.fingerprint(path) for path in protected}
    hashes = {str(path): helpers.digest_file(path) for path in protected if path.suffix == ".py"}

    def update(values):
        state.update(values, updated_at=helpers.utc_now().isoformat())
        helpers.write_json(args.out / "status.json", state)
        print(json.dumps(values, allow_nan=False), flush=True)

    def assert_source():
        if (any(helpers.fingerprint(Path(path)) != value for path, value in fingerprints.items())
                or any(helpers.digest_file(Path(path)) != value for path, value in hashes.items())
                or helpers.digest_file(args.resume) != state["source_sha256"]):
            raise helpers.SourceIntegrityError("fixed source, code, base, micro report or corpus changed")

    def run_child(name, flags, timeout=None, script="next_candidate_bench.py"):
        assert_source()
        update({"status": "waiting_for_gpu", "stage": name})
        wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait, on_event=lambda event: update({"gpu_wait": event}))
        assert_source()
        if name == "continuation":
            hours = helpers.remaining_hours(args.deadline)
            need(hours > 0, "deadline expired while waiting for GPU")
            flags[flags.index("--max-hours") + 1] = str(hours)
            timeout = hours * 3600 + args.stage_timeout_seconds
        command = [args.python, "-u", str(args.source_dir / "operators/rocm" / script), *flags]
        record = {"name": name, "command": command, "started_at": helpers.utc_now().isoformat()}
        helpers.write_json(args.out / (name + ".command.json"), record)
        begin = time.monotonic()
        with (args.out / (name + ".log")).open("w") as log:
            child = subprocess.Popen(command, cwd=args.source_dir, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1"))
            update({"status": "running", "stage": name, "child_pid": child.pid})
            try:
                code = child.wait(timeout=timeout or args.stage_timeout_seconds)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
                record["timed_out"] = True
            except BaseException:
                child.kill()
                child.wait()
                raise
        record.update(exit_code=code, elapsed_s=time.monotonic() - begin)
        state["stages"].append(record)
        helpers.write_json(args.out / (name + ".receipt.json"), record)
        update({"child_pid": None})
        assert_source()
        return code == 0 and not record.get("timed_out")

    try:
        update({"status": "checking_source"})
        metadata = helpers.cpu_check(args, args.resume, 20, args.out / "source_preflight.log")
        update({"source_metadata": metadata})
        gates = micro_gates(args, metadata, state["source_sha256"])
        helpers.write_json(args.out / "micro_validation.json", gates)
        assert_source()
        baselines, candidates = [], []
        variants = ["baseline", *(key for key in ("gpu_adam", "bucketed") if gates[key]["pass"]),
                    *(["combined"] if all(row["pass"] for row in gates.values()) else []), "baseline"]
        for index, variant in enumerate(variants):
            name = ("baseline_before" if index == 0 else "baseline_after") if variant == "baseline" else variant + "_steps"
            try:
                existing = args.baseline_run if index == 0 else None
                if existing is not None:
                    update({"status": "waiting_for_gpu", "stage": "baseline_reuse"})
                    wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait, on_event=lambda event: update({"gpu_wait": event}))
                    assert_source()
                    try:
                        result = validate_trial(args, existing, metadata, variant, steps=20)
                        result["reused_from"] = str(existing)
                    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                        update({"rejected_reuse": str(existing), "rejection_reason": str(error)})
                        existing = None
                if existing is None:
                    need(run_child(name, trial_args(args, args.out / name, 20, variant)), "trial child failed or timed out")
                    result = validate_trial(args, args.out / name, metadata, variant, steps=20)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                result = {"variant": variant, "validated": False, "error": str(error)}
                if variant == "baseline":
                    helpers.write_json(args.out / (name + ".validation.json"), result)
                    raise RuntimeError("packed baseline failed; preserve outputs for manual recovery") from error
            helpers.write_json(args.out / (name + ".validation.json"), result)
            (baselines if variant == "baseline" else candidates).append(result)
        decision = {**choose_variant(baselines, candidates), "baselines": baselines, "candidates": candidates,
                    "source_checkpoint": str(args.resume), "source_sha256": state["source_sha256"], "benchmark_steps": 20}
        helpers.write_json(args.out / "decision.json", decision)
        update({"status": "comparison_complete", "decision": decision})
        assert_source()
        if args.profile_current:
            directory = args.out / "profile_current"
            try:
                flags = helpers.benchmark_args(args, directory, 6) + ["--packed-attention", "--profile-warmup", "2", "--profile-active", "4"]
                need(run_child("profile_current", flags, script="profile_training.py"), "profile child failed or timed out")
                profile = validate_profile(args, directory, metadata)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.TimeoutExpired) as error:
                profile = {"validated": False, "error": str(error), "participates_in_selection": False}
            helpers.write_json(args.out / "profile_current.validation.json", profile)
            update({"profile_current": profile})
            assert_source()
        remaining = metadata["source_data_rows"] - metadata["source_stream_i"]
        hours = helpers.remaining_hours(args.deadline)
        if args.evaluate_only or hours <= 0 or remaining <= 0:
            update({"status": "evaluated_only" if args.evaluate_only else "deadline_expired" if hours <= 0 else "corpus_complete"})
        else:
            variant = decision["selected"]
            need(run_child("continuation", trial_args(args, args.out / "continuation", remaining, variant, hours=hours),
                           timeout=hours * 3600 + args.stage_timeout_seconds), "continuation failed; manual recovery required")
            result = validate_trial(args, args.out / "continuation", metadata, variant, steps=None, max_updates=remaining)
            assert_source()
            update({"status": "complete", "continuation": result})
    except BaseException as error:
        update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        helpers.write_json(args.out / "summary.json", state)
        raise
    helpers.write_json(args.out / "summary.json", state)
    return state


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "resume", "base", "data", "eval-data", "micro-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--gpu-idle-max-wait", type=float, default=3600)
    parser.add_argument("--stage-timeout-seconds", type=float, default=3600)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--profile-current", action="store_true", help="record an auxiliary packed-attention 2+4-update profile")
    parser.add_argument("--baseline-run", type=Path, help="reuse one same-source synchronized packed baseline")
    return parser


def main(argv=None):
    supervise(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
