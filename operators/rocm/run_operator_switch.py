#!/usr/bin/env python3
"""Compare isolated operator candidates, then resume from one fixed checkpoint.

The supervisor uses only the standard library. Torch checkpoint checks run in
CPU-only children. It never signals an existing process or removes checkpoints.
Benchmark updates are independent experiments and never advance the real run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm.gpu_wait import wait_for_gpu_idle

FLAGS = {
    "native": [],
    "attention": ["--packed-attention"],
    "norm": ["--batched-grad-norm"],
    "combined": ["--packed-attention", "--batched-grad-norm"],
}
SCRIPTS = ("candidate_bench.py", "packed_attention_bench.py", "grad_norm_bench.py",
           "profile_training.py")


class SourceIntegrityError(RuntimeError):
    """A changed source or source implementation must stop the supervisor."""

CHECKPOINT_CHECK = r'''
import json, math, sys
from pathlib import Path
import torch

arguments = sys.argv[1:]
if len(arguments) not in (5, 6):
    raise ValueError("checkpoint check needs five arguments and an optional absolute cursor limit")
source_path, target_path, data_path, expected, window = arguments[:5]
expected, window = int(expected), int(window)
max_cursor = int(arguments[5]) if len(arguments) == 6 else None
if max_cursor is None and source_path == target_path:
    from operators.rocm.profile_training import check_resume
    check_resume(Path(source_path), Path(data_path), 4096, window)
source = torch.load(source_path, map_location="cpu", weights_only=False)
target = source if source_path == target_path else torch.load(target_path, map_location="cpu", weights_only=False)
def need(condition, message):
    if not condition:
        raise ValueError(message)
rows, remainder = divmod(Path(data_path).stat().st_size, 4096 * 4)
need(not remainder and rows > 0, "invalid packed data length")
need(max_cursor is None or (max_cursor > 0 and max_cursor >= rows and max_cursor % rows == 0),
     "max_cursor must be a positive whole-corpus multiple at least one corpus long")
absolute_limit = rows if max_cursor is None else max_cursor
def inspect(checkpoint):
    extra = checkpoint.get("extra") or {}
    need(checkpoint.get("kind") == "trainable", "expected trainable overlay")
    need(extra.get("phase") == "B0" and extra.get("name") == "CAT-YOKO-12B", "wrong model or phase")
    need(extra.get("seq_len") == 4096 and isinstance(extra.get("seed"), int), "invalid sequence or seed")
    cfg = extra.get("cfg") or {}
    need(all(cfg.get(key) == value for key, value in (("vocab_size",130560),("hidden_size",2048),
         ("encoder_layers",16),("decoder_layers",26),("n_routed_enc",20),("n_routed_dec",20),("top_k_dec",10))), "wrong architecture")
    need(not cfg.get("use_nvfp4") and not cfg.get("use_fp8"), "expected BF16 architecture")
    weights = checkpoint.get("trainable") or {}
    need(len(weights) == 132, "expected all 132 native weights")
    for name, value in weights.items():
        need(torch.is_tensor(value) and value.device.type == "cpu" and value.dtype == torch.bfloat16,
             "non-CPU/BF16 weight: " + name)
        need(bool(torch.isfinite(value).all()), "non-finite weight: " + name)
    step = extra.get("step")
    need(isinstance(step, int) and step >= 0, "invalid global step")
    for key in ("tokens_in_phase", "tokens_seen"):
        need(isinstance(extra.get(key), (int,float)) and math.isfinite(extra[key]) and extra[key] >= 0, "invalid token counter")
    stream = extra.get("stream") or {}
    need(stream.get("kind") == "packed" and stream.get("stride") == 1 and stream.get("nseq") == rows,
         "packed stream kind/stride/nseq mismatch")
    need(isinstance(stream.get("i"), int) and 0 <= stream["i"] <= absolute_limit, "invalid packed cursor")
    optimizer = checkpoint.get("optimizer") or {}
    groups, states = optimizer.get("param_groups") or [], optimizer.get("state") or {}
    ids = [parameter for group in groups for parameter in group.get("params", [])]
    need(len(ids) == len(set(ids)) == len(states) == 132 and set(ids) == set(states), "Adam ID mapping mismatch")
    def no_decay(name, value):
        parts = name.split(".")
        return value.ndim < 2 or parts[-1] == "bias" or any("norm" in part for part in parts) or "router" in parts
    names = [[name for name,value in weights.items() if not no_decay(name,value)],
             [name for name,value in weights.items() if no_decay(name,value)]]
    need(len(groups) == 2 and all(len(group["params"]) == len(values) for group,values in zip(groups,names)), "Adam decay groups mismatch")
    mapping = {parameter:name for group,values in zip(groups,names) for parameter,name in zip(group["params"],values)}
    counters = set()
    for parameter in ids:
        state = states[parameter]
        counter = state.get("step")
        counter = counter.item() if torch.is_tensor(counter) else counter
        need(isinstance(counter, (int,float)) and int(counter) == counter and counter > 0, "invalid Adam step")
        counters.add(int(counter))
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            need(torch.is_tensor(value) and value.device.type == "cpu" and value.dtype == torch.float32,
                 "Adam moments must stay CPU FP32")
            need(tuple(value.shape) == tuple(weights[mapping[parameter]].shape), "Adam moment shape mismatch")
            need(bool(torch.isfinite(value).all()), "non-finite Adam moment")
            if key == "exp_avg_sq":
                need(bool((value >= 0).all()), "negative squared Adam moment")
    need(len(counters) == 1, "Adam counters disagree")
    return extra, weights, groups, mapping, counters.pop()
sx, sw, sg, sm, sa = inspect(source)
tx, tw, tg, tm, ta = (sx,sw,sg,sm,sa) if target is source else inspect(target)
need(list(sw) == list(tw) and all(tuple(sw[name].shape) == tuple(tw[name].shape) for name in sw), "weight names/shapes changed")
need(sm == tm and [group["params"] for group in sg] == [group["params"] for group in tg], "Adam parameter groups/IDs changed")
for old,new in zip(sg,tg):
    need({key:value for key,value in old.items() if key != "lr"} == {key:value for key,value in new.items() if key != "lr"}, "Adam group options changed")
for key in ("phase","name","cfg","seed","seq_len","use_kda","sparse"):
    need(tx.get(key) == sx.get(key), "saved model/seed/config changed: " + key)
for key in ("step",):
    need(tx[key] == sx[key] + expected, "global step delta mismatch")
for key in ("tokens_in_phase", "tokens_seen"):
    need(tx[key] == sx[key] + expected * 4096, "token counter delta mismatch")
need(tx["stream"]["i"] == sx["stream"]["i"] + expected, "packed cursor delta mismatch")
need(ta == sa + expected, "Adam counter delta mismatch")
need(sx["stream"]["i"] + window <= absolute_limit,
     "requested window would wrap packed corpus" if max_cursor is None else "requested window would exceed packed cursor limit")
result = {"source_checkpoint": str(Path(target_path).resolve()), "source_step": tx["step"],
          "source_adam_step": ta, "source_stream_i": tx["stream"]["i"],
          "source_data_rows": rows, "optimizer_states": 132, "optimizer_moments": 264,
          "source_tokens_in_phase": tx["tokens_in_phase"], "source_tokens_seen": tx["tokens_seen"],
          "trainable_names": list(tw), "trainable_shapes": {name:list(value.shape) for name,value in tw.items()},
          "checkpoint_verified": True}
if max_cursor is not None:
    result.update(max_cursor=absolute_limit, source_completed_passes=tx["stream"]["i"] // rows,
                  source_modulo_row=tx["stream"]["i"] % rows)
print(json.dumps(result, allow_nan=False), flush=True)
'''


def utc_now():
    return datetime.now(timezone.utc)


def parse_deadline(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("deadline-utc must be an ISO-8601 timestamp with timezone") from exc
    if result.tzinfo is None:
        raise ValueError("deadline-utc must include its UTC timezone or offset")
    return result.astimezone(timezone.utc)


def remaining_hours(deadline: datetime, now: datetime | None = None) -> float:
    return max(0.0, (deadline - (now or utc_now())).total_seconds() / 3600)


def write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in experiment report")
    if isinstance(value, dict):
        for item in value.values():
            require_finite(item)
    elif isinstance(value, list):
        for item in value:
            require_finite(item)


def steady_metrics(path: Path, *, source_step: int, steps: int, discard=5):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "tok_s" in row]
    if len(rows) != steps or [row.get("step") for row in rows] != list(range(source_step + 1, source_step + steps + 1)):
        raise ValueError("metrics must contain exactly the requested consecutive updates")
    for row in rows:
        require_finite(row)
        if not isinstance(row["tok_s"], (int, float)) or row["tok_s"] <= 0:
            raise ValueError("invalid training throughput")
    measured = rows[discard:]
    if not measured:
        raise ValueError("no updates remain after warmup")
    return {"median_tokens_per_second": statistics.median(row["tok_s"] for row in measured),
            "measured_updates": len(measured), "discarded_updates": discard,
            "scope": "Trainer loop; initial/final evaluation and final save excluded"}


def validate_run(directory: Path, *, source: Path, source_step: int, steps: int | None,
                 variant: str, max_updates: int | None = None, source_dir: Path | None = None,
                 expected_names: list[str] | None = None, args=None, eval_batches=2,
                 reference_backend='native'):
    report = json.loads((directory / "parity.json").read_text())
    operators = json.loads((directory / "operators.json").read_text())
    require_finite(report)
    require_finite(operators)
    if reference_backend == 'previous_packed_production':
        from operators.rocm.production_continuation import validate_reference_evidence, WRAPPER_SOURCE
        validate_reference_evidence(report, source_dir=source_dir, source=source)
        validate_reference_evidence(operators, source_dir=source_dir, source=source)
        if operators.get('file_sha256', {}).get(WRAPPER_SOURCE) != operators['reference_file_sha256'][WRAPPER_SOURCE]:
            raise SourceIntegrityError('packed-production wrapper missing from operator source inventory')
    elif reference_backend != 'native' or report.get('reference_backend', 'native') != 'native' or operators.get('reference_backend', 'native') != 'native':
        raise ValueError('run used a different reference backend')
    if report.get("status") != "training_complete":
        raise ValueError("run did not finish training and held-out evaluation")
    for key in ("initial_eval", "final_eval"):
        evaluation = report.get(key, {})
        if evaluation.get("pass") is not True or not isinstance(evaluation.get("eval_nll"), (int, float)):
            raise ValueError("run is missing finite initial/final held-out evaluation")
    if args is not None:
        if (Path(report.get("data", "")).resolve() != args.data or Path(report.get("eval_data", "")).resolve() != args.eval_data
                or report.get("eos_id") != 1 or report.get("eval_eos_id") != 1 or report.get("dtype") != "bf16"
                or report.get("moe_layout") != "shared-storage" or report.get("parity_sequences") != [4096]
                or report.get("source_stream_kind") != "packed" or report.get("eval_batches") != eval_batches):
            raise ValueError("run used a different corpus, held-out set or execution configuration")
    if source_dir is not None:
        hashes = report.get("source_code_version", {}).get("file_sha256", {})
        required = {"operators/rocm/model_bench.py", "operators/rocm/shared_storage_moe.py",
                    *("cat_yoko/" + name for name in ("moe.py", "trainer.py", "checkpoint.py", "optim.py", "data.py", "attention.py", "loss.py"))}
        if set(hashes) != required:
            raise SourceIntegrityError("run did not record every native source file")
        for relative, expected in {**hashes, **operators.get("file_sha256", {})}.items():
            path = (source_dir / relative).resolve()
            if not path.is_relative_to(source_dir.resolve()) or digest_file(path) != expected:
                raise SourceIntegrityError("run source-code hashes differ from the isolated source")
    events = report.get("events", [])
    finish = [event for event in events if event.get("event") == "training_complete"]
    if not finish or finish[-1].get("optimizer_restored") is not True:
        raise ValueError("run did not restore the source Adam state")
    for key in ("source_optimizer_present", "save_optimizer", "deterministic_parity", "reference_repeat"):
        if report.get(key) is not True:
            raise ValueError(f"run is missing {key}")
    if any(finish[-1].get(key) is not True for key in ("source_optimizer_present", "save_optimizer")):
        raise ValueError("final training receipt did not preserve source Adam")
    if Path(report.get("source_checkpoint", "")).resolve() != source.resolve() or report.get("source_step") != source_step:
        raise ValueError("run resumed a different checkpoint")
    updates = report.get("updates")
    if not isinstance(updates, int) or updates < 0 or (steps is not None and updates != steps):
        raise ValueError("run completed the wrong number of updates")
    if max_updates is not None and updates > max_updates:
        raise ValueError("continuation exceeded the single-pass corpus")
    if not finish or finish[-1].get("step") != source_step + updates or finish[-1].get("updates") != updates:
        raise ValueError("final step does not match the fixed source and update count")
    thresholds = report.get("thresholds", {})
    if (thresholds.get("gradient_relative_l2_per_tensor_and_global") != 0.05
            or thresholds.get("selected_output_relative_l2") != 0.05
            or thresholds.get("loss_atol") != 0.02):
        raise ValueError("native parity thresholds changed")
    for name in ("reference_repeat", "parity"):
        comparisons = [event for event in events if event.get("event") == name and event.get("seq_len") == 4096]
        if not comparisons or not all(event.get("pass") is True for event in comparisons):
            raise ValueError(f"missing or failed 4096-token {name}")
        for event in comparisons:
            gradients = event.get("gradients", {})
            if len(gradients) != 132 or event.get("failed_gradients"):
                raise ValueError("parity did not validate all 132 native gradients")
            if expected_names is not None and set(gradients) != set(expected_names):
                raise ValueError("parity gradient names differ from the source checkpoint")
            if (event.get("global_gradient_relative_l2", math.inf) > 0.05
                    or event.get("loss_abs", math.inf) > 0.02):
                raise ValueError("parity exceeds the original numeric thresholds")
            if any(error.get("pass") is not True or error.get("finite") is not True
                   or error.get("relative_l2", math.inf) > 0.05 for error in gradients.values()):
                raise ValueError("a native gradient failed parity")
            for key in ("selected_logits", "final_hidden"):
                error = event.get(key, {})
                if error.get("finite") is not True or error.get("relative_l2", math.inf) > 0.05:
                    raise ValueError("parity output failed the original numeric threshold")
    attention = "--packed-attention" in FLAGS[variant]
    norm = "--batched-grad-norm" in FLAGS[variant]
    reference_installed = (operators.get("patches_installed_after_native_reference") is True
                           if reference_backend == 'native' else
                           operators.get("patches_installed_after_packed_production_reference") is True)
    if (operators.get("status") != "completed" or not reference_installed
            or operators.get("packed_attention") is not attention
            or operators.get("batched_grad_norm") is not norm):
        raise ValueError("operator installation does not match the requested candidate")
    if attention:
        for key in ("packed_attention_parity_calls", "packed_attention_calls"):
            if operators.get(key, {}).get("optimized_calls", 0) <= 0:
                raise ValueError("packed attention was not exercised")
    if not (directory / "train/trainable.pt").is_file():
        raise ValueError("successful run did not save its checkpoint")
    result = {"variant": variant, "updates": updates, "final_step": source_step + updates,
              "checkpoint": str(directory / "train/trainable.pt"), "validated": True}
    if steps is not None:
        result.update(steady_metrics(directory / "train/metrics.jsonl", source_step=source_step, steps=steps))
    return result


def microbench_passed(path: Path, kind: str):
    try:
        if kind == "attention":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require_finite(rows)
            summaries = [row for row in rows if row.get("event") == "summary"]
            return bool(summaries and summaries[-1].get("passed", 0) > 0
                        and summaries[-1].get("failed") == 0 and summaries[-1].get("errors") == 0)
        report = json.loads(path.read_text())
        require_finite(report)
        cases = report.get("cases", {})
        return report.get("status") == "complete" and set(cases) == {"clipped", "unclipped"} and all(
            case.get("parity", {}).get("pass") is True for case in cases.values())
    except (OSError, ValueError, TypeError):
        return False


def reusable_preflight(directory: Path, args, metadata, source_sha: str):
    report = json.loads((directory / "status.json").read_text())
    if report.get("status") != "complete":
        raise ValueError("reused preflight must be complete")
    codes = {row["stage"]: row.get("exit_code") for row in report.get("results", [])}
    if not {"packed", "norm"}.issubset(codes):
        raise ValueError("preflight must record packed and norm exit codes")
    packed, norm = directory / "packed.jsonl", directory / "norm.json"
    norm_ok = codes["norm"] == 0 and microbench_passed(norm, "norm")
    if norm.exists():
        data = json.loads(norm.read_text())
        if (data.get("source_sha256") != source_sha or data.get("source_step") != metadata["source_step"]
                or Path(data.get("source_checkpoint", "")).resolve() != args.resume):
            raise ValueError("reused norm preflight tested a different source")
    packed_ok = codes["packed"] == 0 and microbench_passed(packed, "attention")
    if packed_ok:
        rows = [json.loads(line) for line in packed.read_text().splitlines() if line.strip()]
        cases = [row for row in rows if row.get("event") == "candidate"]
        exercised = set()
        for row in cases:
            document = row.get("documents", {})
            if (Path(document.get("path", "")).resolve() != args.data
                    or document.get("row") != metadata["source_stream_i"] or document.get("eos_id") != 1
                    or document.get("source_pack_width") != 4096):
                raise ValueError("reused attention preflight used a different corpus row")
            if row.get("seq") == 4096 and row.get("window") == 8192 and row.get("dtype") == "bf16":
                if row.get("status") != "pass":
                    raise ValueError("reused production-shape attention case failed")
                if row.get("candidate") == "packed_attention" and row.get("correctness_path_counts", {}).get("optimized_calls", 0) <= 0:
                    raise ValueError("reused attention operator was not exercised")
                exercised.add((row.get("layout"), row.get("candidate")))
        expected = {(layout, candidate) for layout in ("packed_qkv", "cross_cache") for candidate in ("native", "packed_attention")}
        if not expected.issubset(exercised):
            raise ValueError("reused preflight lacks both 4096/8192 native and packed layouts")
    return packed_ok, norm_ok


def choose_variant(baselines: list[dict], candidates: list[dict], minimum_gain=0.05):
    decision = {"selected": "native", "selected_flags": [], "minimum_gain": minimum_gain,
                "reason": "no validated candidate improves both baseline runs by at least 5%"}
    if len(baselines) != 2 or not all(row.get("validated") is True for row in baselines):
        return {**decision, "reason": "both native baseline runs must validate before adopting a candidate"}
    speeds = [row["median_tokens_per_second"] for row in baselines]
    if any(not math.isfinite(value) or value <= 0 for value in speeds):
        raise ValueError("invalid baseline speed")
    conservative = max(speeds)  # Require improvement against the faster baseline.
    eligible = [row for row in candidates if row.get("validated") is True
                and math.isfinite(row["median_tokens_per_second"])
                and row["median_tokens_per_second"] >= conservative * (1 + minimum_gain)]
    decision.update(baseline_medians=speeds, conservative_baseline_median=conservative)
    if eligible:
        fastest = max(eligible, key=lambda row: row["median_tokens_per_second"])
        decision.update(selected=fastest["variant"], selected_flags=FLAGS[fastest["variant"]],
                        median_tokens_per_second=fastest["median_tokens_per_second"],
                        conservative_speedup=fastest["median_tokens_per_second"] / conservative,
                        reason="validated fastest candidate improves both baseline medians by at least 5%")
    return decision


def benchmark_args(args, out: Path, steps: int):
    return ["--base", str(args.base), "--resume", str(args.resume), "--out", str(out),
            "--data", str(args.data), "--eval-data", str(args.eval_data), "--eos-id", "1",
            "--moe-layout", "shared-storage", "--parity-seqs", "4096", "--seq-len", "4096",
            "--run-steps", str(steps), "--save-every", "0", "--save-every-seconds", "0",
            "--keep-last", "1", "--eval-every", "0", "--eval-batches", "2", "--save-optim",
            "--deterministic-parity", "--reference-repeat", "--loss-atol", "0.02",
            "--grad-relative-l2", "0.05", "--output-relative-l2", "0.05", "--wait-gpu-idle",
            "--gpu-idle-max-wait", str(args.gpu_idle_max_wait)]


def continuation_args(args, out: Path, updates: int, hours: float, variant: str):
    if updates <= 0 or not math.isfinite(hours) or hours <= 0:
        raise ValueError("continuation needs remaining rows and a positive deadline budget")
    flags = benchmark_args(args, out, updates)
    replacements = {"--save-every-seconds": "300", "--keep-last": "3",
                    "--eval-every": "250", "--eval-batches": "32"}
    for key, value in replacements.items():
        flags[flags.index(key) + 1] = value
    return flags + ["--max-hours", str(hours), *FLAGS[variant]]


def validate_args(args):
    for key in ("source_dir", "resume", "base", "data", "eval_data", "out"):
        setattr(args, key, getattr(args, key).resolve())
    args.deadline = parse_deadline(args.deadline_utc)
    for key in ("preflight_dir", "baseline_run"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if args.benchmark_steps <= 5:
        raise ValueError("benchmark-steps must exceed the five discarded warmup updates")
    for key in ("gpu_idle_max_wait", "stage_timeout_seconds"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key.replace('_', '-')} must be finite and positive")
    if not args.resume.is_file():
        raise ValueError("resume must be one fixed checkpoint file, not a live directory")
    if not args.base.exists() or not args.data.is_file() or not args.eval_data.is_file():
        raise FileNotFoundError("base, train and held-out data must exist")
    if args.data == args.eval_data:
        raise ValueError("train and held-out data must differ")
    for protected in (args.source_dir, args.resume.parent, args.base, args.data.parent, args.eval_data.parent):
        if args.out == protected or args.out.is_relative_to(protected):
            raise ValueError("output must be separate from source, checkpoint, base and data")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        raise FileExistsError("refusing to overwrite an existing operator switch")
    for script in SCRIPTS:
        if not (args.source_dir / "operators/rocm" / script).is_file():
            raise FileNotFoundError(f"isolated source lacks {script}")


def cpu_check(args, checkpoint: Path, updates: int, log: Path, *, max_cursor: int | None = None):
    """Audit native state; repeated corpus passes need an explicit cursor cap."""
    if max_cursor is not None:
        if type(max_cursor) is not int or max_cursor <= 0:
            raise ValueError("max_cursor must be a positive integer")
        rows, remainder = divmod(args.data.stat().st_size, 4096 * 4)
        if remainder or rows <= 0:
            raise ValueError("invalid packed data length")
        if max_cursor < rows or max_cursor % rows:
            raise ValueError("max_cursor must be a whole-corpus multiple at least one corpus long")
    same_source = checkpoint.resolve() == args.resume
    command = [args.python, "-c", CHECKPOINT_CHECK, str(args.resume), str(checkpoint),
               str(args.data), "0" if same_source else str(updates), str(updates) if same_source else "0"]
    if max_cursor is not None:
        command.append(str(max_cursor))
    result = subprocess.run(command,
                            cwd=args.source_dir, capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                                     ROCR_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"))
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError(f"CPU Adam/cursor verification failed; see {log.name}")
    metadata = json.loads(result.stdout.splitlines()[-1])
    if metadata.get("optimizer_states") != 132:
        raise ValueError("source must preserve all 132 native B0 Adam states")
    return metadata


def supervise(args):
    validate_args(args)
    args.out.mkdir(parents=True)
    status = {"status": "starting", "supervisor_pid": os.getpid(), "stages": [],
              "source_checkpoint": str(args.resume), "deadline_utc": args.deadline.isoformat(),
              "controls_existing_processes": False, "source_files_removed": False,
              "benchmark_updates_count_toward_continuation": False}

    def update(values):
        status.update(values, updated_at=utc_now().isoformat())
        write_json(args.out / "status.json", status)
        print(json.dumps(values, allow_nan=False), flush=True)

    source_stat = fingerprint(args.resume)
    code_paths = [*(args.source_dir / "cat_yoko" / name for name in
                    ("moe.py", "trainer.py", "checkpoint.py", "optim.py", "data.py", "attention.py", "loss.py")),
                  *(args.source_dir / "operators/rocm" / name for name in
                    ("model_bench.py", "shared_storage_moe.py", "candidate_bench.py", "profile_training.py",
                     "packed_attention.py", "grad_norm.py", "packed_attention_bench.py", "grad_norm_bench.py"))]
    code_hashes = {str(path): digest_file(path) for path in code_paths}
    data_stats = {str(path): fingerprint(path) for path in (args.data, args.eval_data)}

    def assert_source():
        if fingerprint(args.resume) != source_stat:
            raise SourceIntegrityError("fixed source checkpoint changed during comparison")
        if any(digest_file(Path(path)) != expected for path, expected in code_hashes.items()):
            raise SourceIntegrityError("isolated native source code changed during comparison")
        if any(fingerprint(Path(path)) != expected for path, expected in data_stats.items()):
            raise SourceIntegrityError("training or held-out corpus changed during comparison")

    def wait_idle(name):
        update({"status": "waiting_for_gpu", "stage": name})
        wait_for_gpu_idle(max_wait_seconds=args.gpu_idle_max_wait,
                          on_event=lambda event: update({"gpu_wait": event}))

    def run_child(name, script, flags, *, timeout=None, wait=True):
        assert_source()
        if wait:
            wait_idle(name)
        assert_source()
        command = [args.python, "-u", str(args.source_dir / "operators/rocm" / script), *flags]
        record = {"name": name, "command": command, "started_at": utc_now().isoformat()}
        write_json(args.out / f"{name}.command.json", record)
        begin = time.monotonic()
        with (args.out / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, cwd=args.source_dir, stdout=log,
                                     stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            update({"status": "running", "stage": name, "child_pid": child.pid})
            try:
                code = child.wait(timeout=timeout or args.stage_timeout_seconds)
            except subprocess.TimeoutExpired:
                # Only the child created in this scope can be signaled.
                child.kill()
                code = child.wait()
                record["timed_out"] = True
            except BaseException:
                child.kill()
                child.wait()
                raise
        record.update(exit_code=code, elapsed_s=time.monotonic() - begin, log=f"{name}.log")
        status["stages"].append(record)
        write_json(args.out / f"{name}.receipt.json", record)
        update({"child_pid": None})
        assert_source()
        return code == 0

    def benchmark(name, variant, metadata):
        directory = args.out / name
        if args.preflight_dir is not None and (args.preflight_dir / name).exists():
            existing = args.preflight_dir / name
            wait_idle(name + "_reuse")
            assert_source()
            try:
                result = validate_run(existing, source=args.resume, source_step=metadata["source_step"],
                                      steps=args.benchmark_steps, variant=variant, source_dir=args.source_dir,
                                      expected_names=metadata["trainable_names"], args=args)
                result["verified_checkpoint_metadata"] = cpu_check(args, Path(result["checkpoint"]), args.benchmark_steps,
                                                                    args.out / f"{name}.checkpoint_check.log")
                result["reused_from"] = str(existing)
                write_json(args.out / f"{name}.validation.json", result)
                update({"reused_experiment": name, "reused_directory": str(existing)})
                return result
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                update({"rejected_reuse": str(existing), "rejection_reason": str(error)})
        try:
            if not run_child(name, "candidate_bench.py", benchmark_args(args, directory, args.benchmark_steps) + FLAGS[variant]):
                raise ValueError("experiment child failed or timed out")
            result = validate_run(directory, source=args.resume, source_step=metadata["source_step"],
                                  steps=args.benchmark_steps, variant=variant, source_dir=args.source_dir,
                                  expected_names=metadata["trainable_names"], args=args)
            result["verified_checkpoint_metadata"] = cpu_check(args, Path(result["checkpoint"]), args.benchmark_steps,
                                                                args.out / f"{name}.checkpoint_check.log")
            write_json(args.out / f"{name}.validation.json", result)
            return result
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            result = {"variant": variant, "validated": False, "error": str(error)}
            write_json(args.out / f"{name}.validation.json", result)
            update({"rejected_experiment": name, "rejection_reason": str(error)})
            return result

    def run_microbench(metadata):
        packed_path, norm_path = args.out / "packed_operator.jsonl", args.out / "norm_operator.json"
        packed_ok = run_child("packed_operator", "packed_attention_bench.py", [
            "--data", str(args.data), "--data-row", str(metadata["source_stream_i"]), "--eos-id", "1",
            "--seq-lens", "4096", "--windows", "8192", "32", "--layouts", "packed_qkv", "cross_cache",
            "--warmup", "3", "--repeats", "5", "--wait-gpu-idle", "--gpu-wait-max-seconds",
            str(args.gpu_idle_max_wait), "--output", str(packed_path)]) and microbench_passed(packed_path, "attention")
        norm_ok = run_child("norm_operator", "grad_norm_bench.py", [
            "--resume", str(args.resume), "--json", str(norm_path), "--wait-gpu-idle",
            "--gpu-idle-max-wait", str(args.gpu_idle_max_wait)]) and microbench_passed(norm_path, "norm")
        return packed_ok, norm_ok

    update({"status": "starting"})
    try:
        metadata = cpu_check(args, args.resume, args.benchmark_steps, args.out / "source_preflight.log")
        status["source_metadata"] = metadata
        status["source_sha256"] = digest_file(args.resume)
        assert_source()
        update({"source_verified": True})
        reused_preflight = False
        if args.preflight_dir is not None:
            try:
                packed_ok, norm_ok = reusable_preflight(args.preflight_dir, args, metadata, status["source_sha256"])
                reused_preflight = True
                update({"reused_preflight": str(args.preflight_dir)})
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                update({"rejected_preflight_reuse": str(args.preflight_dir), "rejection_reason": str(error)})
        if not reused_preflight:
            packed_ok, norm_ok = run_microbench(metadata)
        update({"microbench_pass": {"attention": packed_ok, "norm": norm_ok}})
        first = None
        if args.baseline_run is not None:
            wait_idle("baseline_before_reuse")
            assert_source()
            try:
                first = validate_run(args.baseline_run, source=args.resume, source_step=metadata["source_step"],
                                     steps=args.benchmark_steps, variant="native", source_dir=args.source_dir,
                                     expected_names=metadata["trainable_names"], args=args)
                first["verified_checkpoint_metadata"] = cpu_check(args, Path(first["checkpoint"]), args.benchmark_steps,
                                                                   args.out / "baseline_before.checkpoint_check.log")
                first["reused_from"] = str(args.baseline_run)
                write_json(args.out / "baseline_before.validation.json", first)
                update({"reused_baseline": str(args.baseline_run)})
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                first = None
                update({"rejected_baseline_reuse": str(args.baseline_run), "rejection_reason": str(error)})
        if first is None:
            first = benchmark("baseline_before", "native", metadata)
        baselines = [first]
        candidates = []
        for variant, eligible in (("attention", packed_ok), ("norm", norm_ok), ("combined", packed_ok and norm_ok)):
            if eligible:
                candidates.append(benchmark(variant + "_steps", variant, metadata))
        baselines.append(benchmark("baseline_after", "native", metadata))
        assert_source()
        if digest_file(args.resume) != status["source_sha256"]:
            raise SourceIntegrityError("fixed source SHA changed during comparison")
        decision = {**choose_variant(baselines, candidates), "baselines": baselines, "candidates": candidates,
                    "source_checkpoint": str(args.resume), "source_sha256": status["source_sha256"],
                    "benchmark_steps": args.benchmark_steps, "continuation_source": "original fixed source, including Adam and cursor",
                    "deadline_utc": args.deadline.isoformat()}
        write_json(args.out / "decision.json", decision)
        update({"status": "evaluated", "decision": decision})
        remaining = metadata["source_data_rows"] - metadata["source_stream_i"]
        if args.evaluate_only:
            update({"status": "evaluated_only", "remaining_updates": remaining})
            write_json(args.out / "summary.json", status)
            return status
        wait_idle("continuation")
        hours = remaining_hours(args.deadline)
        if hours <= 0 or remaining <= 0:
            update({"status": "deadline_expired" if hours <= 0 else "corpus_complete", "remaining_updates": remaining})
            write_json(args.out / "summary.json", status)
            return status
        variant = decision["selected"]
        flags = continuation_args(args, args.out / "continuation", remaining, hours, variant)
        update({"status": "continuation_ready", "selected": variant, "remaining_updates": remaining,
                "max_hours": hours, "budget_scope": "Trainer.run; setup, parity and held-out evaluation are outside the core deadline timer"})
        write_json(args.out / "continuation_plan.json", {"variant": variant, "resume": str(args.resume),
                   "remaining_updates": remaining, "max_hours": hours, "flags": flags})
        if not run_child("continuation", "candidate_bench.py", flags, timeout=hours * 3600 + 3600, wait=False):
            raise RuntimeError("continuation failed; preserved its checkpoints and receipts for recovery")
        result = validate_run(args.out / "continuation", source=args.resume, source_step=metadata["source_step"],
                              steps=None, variant=variant, max_updates=remaining, source_dir=args.source_dir,
                              expected_names=metadata["trainable_names"], args=args, eval_batches=32)
        saved = cpu_check(args, Path(result["checkpoint"]), result["updates"], args.out / "continuation_checkpoint_check.log")
        for key in ("source_step", "source_adam_step", "source_stream_i"):
            if saved[key] != metadata[key] + result["updates"]:
                raise ValueError(f"saved continuation {key} does not match the source plus actual updates")
        result["verified_checkpoint_metadata"] = saved
        update({"status": "complete", "continuation": result})
        write_json(args.out / "summary.json", status)
        return status
    except BaseException as error:
        update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        write_json(args.out / "summary.json", status)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source-dir", "resume", "base", "data", "eval-data", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--gpu-idle-max-wait", type=float, default=3600)
    parser.add_argument("--stage-timeout-seconds", type=float, default=3600)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--preflight-dir", type=Path, help="reuse verified packed.jsonl/norm.json/status.json")
    parser.add_argument("--baseline-run", type=Path, help="reuse a completed native candidate_bench run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    supervise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
