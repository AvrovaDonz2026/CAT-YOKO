#!/usr/bin/env python3
"""Supervise verified data -> five-update smoke -> one-pass bounded B0 training.

This module uses only the standard library. Data preparation is an independent
job; it never downloads, prepares data, retries a failed child, or controls other
GPU jobs. The long-training time cap starts inside Trainer, after preparation,
smoke, model reconstruction, and parity. Checkpoint inspection runs in a separate
CPU-only Python process so the supervisor never imports torch or owns a GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_STEP = 34802
SMOKE_UPDATES = 5
SEQ_LEN = 4096

# Executed only by the supplied torch-enabled interpreter, with GPU visibility
# disabled and map_location=cpu. The supervisor itself has no torch dependency.
CHECKPOINT_CHECK = r'''
import json, math, sys
from pathlib import Path
import torch

path, output, mode, source_json, expected_updates = sys.argv[1:]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
extra = checkpoint.get("extra") or {}
assert checkpoint.get("kind") == "trainable", "checkpoint is not a trainable overlay"
assert extra.get("phase") == "B0" and extra.get("name") == "CAT-YOKO-12B", "wrong model/phase"
cfg = extra.get("cfg") or {}
assert int(cfg.get("vocab_size", 0)) == 130560, "wrong vocabulary"
assert all(int(cfg.get(key, 0)) == value for key, value in (
    ("hidden_size", 2048), ("encoder_layers", 16), ("decoder_layers", 26),
    ("n_routed_enc", 20), ("n_routed_dec", 20), ("top_k_dec", 10))), "wrong architecture"
assert not cfg.get("use_nvfp4") and not cfg.get("use_fp8"), "expected BF16 checkpoint graph"
assert int(extra.get("seq_len", 0)) == 4096, "wrong saved sequence length"
weights = checkpoint.get("trainable") or {}
assert len(weights) == 132, "expected 132 trainable tensors"
for name, value in weights.items():
    assert torch.is_tensor(value) and value.device.type == "cpu", "non-CPU weight: " + name
    assert value.dtype == torch.bfloat16, "expected BF16 trainable: " + name
    assert bool(torch.isfinite(value).all()), "nonfinite trainable: " + name
step = int(extra["step"])
tokens = float(extra["tokens_in_phase"])
seen = float(extra["tokens_seen"])
assert all(math.isfinite(value) and value >= 0 for value in (tokens, seen)), "invalid token counters"
stream = extra.get("stream") or {}
optimizer = checkpoint.get("optimizer")
result = {"checkpoint": str(Path(path).resolve()), "step": step,
          "tokens_in_phase": tokens, "tokens_seen": seen, "phase": "B0",
          "trainable_tensors": len(weights), "trainable_finite": True,
          "stream_kind": stream.get("kind"), "stream_i": stream.get("i"),
          "optimizer_present": bool(optimizer)}
result["trainable_shapes"] = {name: list(value.shape) for name, value in weights.items()}
if mode == "source":
    assert step == 34802, "source must be the verified step-34802 overlay"
    assert stream.get("kind") == "dummy" and float(stream.get("code_frac", 0)) == 0, "unexpected source stream"
    assert not optimizer, "source optimizer differs from the verified step-34802 overlay"
else:
    source = json.loads(Path(source_json).read_text())
    assert result["trainable_shapes"] == source["trainable_shapes"], "trainable names/shapes differ from source"
    updates = step - int(source["step"])
    maximum = int(expected_updates)
    assert updates == 5 if mode == "smoke" else 5 <= updates <= maximum, "invalid completed update count"
    assert tokens == float(source["tokens_in_phase"]) + updates * 4096, "phase token counter mismatch"
    assert seen == float(source["tokens_seen"]) + updates * 4096, "global token counter mismatch"
    assert stream.get("kind") == "packed" and int(stream.get("stride", 0)) == 1, "wrong training stream"
    assert int(stream.get("i", -1)) == updates, "checkpoint skipped or repeated a packed row"
    corpus = json.loads((Path(source_json).parent / "data_verification.json").read_text())
    assert int(stream.get("nseq", 0)) == corpus["splits"]["train"]["sequences"], "wrong packed corpus row count"
    assert optimizer and optimizer.get("state") and optimizer.get("param_groups"), "missing optimizer state"
    parameters = [value for group in optimizer["param_groups"] for value in group["params"]]
    states = optimizer["state"]
    assert len(parameters) == len(set(parameters)) == len(states) == 132 and set(parameters) == set(states), "optimizer parameter mapping mismatch"
    def no_decay(name, value):
        parts = name.split(".")
        return value.ndim < 2 or parts[-1] == "bias" or any("norm" in part for part in parts) or "router" in parts
    groups = optimizer["param_groups"]
    expected_groups = [[value for name, value in weights.items() if not no_decay(name, value)],
                       [value for name, value in weights.items() if no_decay(name, value)]]
    assert len(groups) == 2 and all(len(group["params"]) == len(values) for group, values in zip(groups, expected_groups)), "Adam decay-group layout mismatch"
    shapes = {parameter: tuple(value.shape) for group, values in zip(groups, expected_groups)
              for parameter, value in zip(group["params"], values)}
    moment_count = 0
    for parameter in parameters:
        state = states[parameter]
        state_step = state.get("step")
        state_step = int(state_step.item()) if torch.is_tensor(state_step) else int(state_step)
        assert state_step == updates, "Adam counter does not continue from smoke"
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            assert torch.is_tensor(value), "missing Adam moment"
            assert value.device.type == "cpu" and value.dtype == torch.float32, "Adam moments must be CPU FP32"
            assert tuple(value.shape) == shapes[parameter], "Adam moment does not match its parameter shape"
            assert bool(torch.isfinite(value).all()), "nonfinite Adam moment"
            if key == "exp_avg_sq":
                assert bool((value >= 0).all()), "negative Adam squared moment"
            moment_count += 1
    result.update(updates=updates, optimizer_states=len(states), optimizer_moments=moment_count,
                  optimizer_moments_cpu_fp32=True, optimizer_finite=True)
Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result, allow_nan=False), flush=True)
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"corpus file missing or outside data-dir: {name}")
    return path


def verified_hashes(path: Path, expected: dict) -> set[str]:
    if file_digest(path) != expected["sha256"]:
        raise ValueError(f"document-hash file SHA256 mismatch: {path}")
    values = path.read_text(encoding="ascii").splitlines()
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in values):
        raise ValueError(f"invalid document hashes: {path}")
    hashes = set(values)
    if not hashes or len(hashes) != len(values) or len(hashes) != int(expected["documents"]):
        raise ValueError(f"document-hash count mismatch or duplicates: {path}")
    return hashes


def validate_data_manifest(data_dir: Path) -> dict:
    """Verify complete bins and hash evidence, never accept random fallback data."""
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("hash_intersection") != 0:
        raise ValueError("data manifest is incomplete or declares overlapping train/eval documents")
    if (int(manifest.get("seq_len", 0)) != SEQ_LEN or int(manifest.get("eos_id", -1)) != 1
            or int(manifest.get("vocab_size", 0)) != 130560):
        raise ValueError("data must use 4096-token rows and MiniCPM vocabulary/EOS=1")
    verified = {"manifest": str(manifest_path.resolve()), "manifest_sha256": file_digest(manifest_path),
                "seq_len": SEQ_LEN, "eos_id": 1, "splits": {}}
    hash_sets = {}
    for split in ("train", "eval"):
        entry = manifest["outputs"][split]
        path = data_file(data_dir, entry["path"])
        sequences = int(entry["sequences"])
        tokens = sequences * SEQ_LEN
        if (sequences <= 0 or int(entry["tokens"]) != tokens or int(entry["bytes"]) != tokens * 4
                or path.stat().st_size != tokens * 4 or file_digest(path) != entry["sha256"]):
            raise ValueError(f"{split} bin size/count/SHA256 mismatch")
        if int(manifest["target_tokens"][split]) != tokens:
            raise ValueError(f"{split} packed rows do not meet the declared corpus target")
        sidecar = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
        if any(sidecar.get(key) != value for key, value in (
            ("seq_len", SEQ_LEN), ("eos_id", 1), ("vocab_size", 130560), ("split", split),
            ("tokens", tokens), ("sequences", sequences), ("sha256", entry["sha256"]),
        )):
            raise ValueError(f"{split} data sidecar differs from the verified manifest")
        hash_entry = manifest["hashes"][split]
        hash_sets[split] = verified_hashes(data_file(data_dir, hash_entry["path"]), hash_entry)
        verified["splits"][split] = {**entry, "path": str(path),
                                     "documents": len(hash_sets[split])}
    intersection = len(hash_sets["train"] & hash_sets["eval"])
    if intersection:
        raise ValueError(f"train/eval document hashes overlap: {intersection}")
    if verified["splits"]["train"]["sequences"] <= SMOKE_UPDATES:
        raise ValueError("corpus must contain more than five training rows")
    verified["hash_intersection"] = intersection
    return verified


def wait_for_data(data_dir: Path, on_status, *, max_wait_seconds: float = 7200,
                  poll_seconds: float = 10) -> dict:
    begin = time.monotonic()
    while True:
        status_path = data_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "missing"}
        if status.get("status") == "failed":
            raise RuntimeError(f"data preparation failed: {status.get('error', status)}")
        if status.get("status") == "complete":
            return validate_data_manifest(data_dir)
        if status.get("status") not in {"running", "planned", "missing"}:
            raise ValueError(f"unknown data preparation status: {status.get('status')}")
        elapsed = time.monotonic() - begin
        if elapsed >= max_wait_seconds:
            raise TimeoutError("data preparation did not complete within two hours")
        on_status({"status": "waiting_for_data", "data_preparation": status,
                   "data_wait_seconds": elapsed})
        time.sleep(min(poll_seconds, max_wait_seconds - elapsed))


def run_child(command: list[str], *, stage: str, log: Path, work_dir: Path,
              on_status, environment: dict | None = None) -> None:
    begin = time.monotonic()
    with log.open("w") as handle:
        child = subprocess.Popen(command, cwd=work_dir, stdout=handle, stderr=subprocess.STDOUT,
                                 env=environment)
        on_status({"status": stage, "child_pid": child.pid, "child_command": command,
                   "child_log": str(log), "child_started_at": utc_now()})
        code = child.wait()
    on_status({"status": stage + "_finished", "child_pid": None, "last_child_pid": child.pid,
               "child_exit_code": code, "child_elapsed_seconds": time.monotonic() - begin})
    if code:
        raise RuntimeError(f"{stage} child exited with status {code}; see {log}")


def verify_checkpoint(args, path: Path, *, mode: str, expected_updates: int, on_status) -> dict:
    result_path = args.out / f"{mode}_checkpoint.json"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                       ROCR_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", PYTHONOPTIMIZE="0")
    command = [args.python, "-c", CHECKPOINT_CHECK, str(path), str(result_path), mode,
               str(args.out / "source_checkpoint.json"), str(expected_updates)]
    run_child(command, stage=f"verify_{mode}", log=args.out / f"verify_{mode}.log",
              work_dir=args.work_dir, on_status=on_status, environment=environment)
    result = json.loads(result_path.read_text())
    if result.get("step") is None or not result.get("trainable_finite"):
        raise ValueError(f"invalid {mode} checkpoint verification result")
    return result


def training_command(args, corpus: dict, *, stage: str, resume: Path,
                     updates: int) -> list[str]:
    smoke = stage == "smoke"
    return [args.python, "-u", str(args.source_dir / "operators/rocm/model_bench.py"),
            "--base", str(args.base), "--resume", str(resume), "--out", str(args.out / stage),
            "--data", corpus["splits"]["train"]["path"],
            "--eval-data", corpus["splits"]["eval"]["path"], "--eos-id", "1",
            "--moe-layout", "shared-storage", "--deterministic-parity", "--reference-repeat",
            "--parity-seqs", "64,4096" if smoke else "4096", "--seq-len", "4096",
            "--run-steps", str(updates), "--max-hours", "0.25" if smoke else str(args.max_hours),
            "--save-every", "1" if smoke else ("0" if args.save_every_seconds else "100"),
            "--save-every-seconds", "0" if smoke else str(args.save_every_seconds),
            "--keep-last", "2" if smoke else str(args.keep_last),
            "--eval-every", "0" if smoke else "250", "--eval-batches", "4" if smoke else "32",
            "--save-optim", "--wait-gpu-idle"]


def verified_training_report(path: Path, *, expected_updates: int | None = None) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != "training_complete" or report.get("moe_layout") != "shared-storage":
        raise ValueError(f"training report is incomplete or uses the wrong layout: {path}")
    if not report.get("deterministic_parity") or not report.get("reference_repeat"):
        raise ValueError("training did not enforce deterministic parity and reference repeat")
    if not report.get("initial_eval", {}).get("pass") or not report.get("final_eval", {}).get("pass"):
        raise ValueError("training report lacks finite initial/final validation")
    parity = [event for event in report.get("events", []) if event.get("event") == "parity_complete"]
    repeat = [event for event in report.get("events", []) if event.get("event") == "reference_repeat"]
    if not parity or not all(event.get("pass") for event in parity + repeat) or not repeat:
        raise ValueError("training report does not contain passing parity and reference repeat")
    if expected_updates is not None and int(report.get("updates", -1)) != expected_updates:
        raise ValueError(f"smoke must finish exactly {expected_updates} updates")
    complete = [event for event in report["events"] if event.get("event") == "training_complete"]
    if not complete:
        raise ValueError("missing training completion event")
    return {"updates": int(report["updates"]), "initial_eval": report["initial_eval"],
            "final_eval": report["final_eval"], "completion": complete[-1]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True, help="remote run working directory")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable, help="ROCm torch-enabled Python for children")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True, help="exact verified step-34802 trainable.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=24,
                        help="long Trainer-loop cap; excludes preparation, parity, and smoke")
    parser.add_argument("--save-every-seconds", type=float, default=300,
                        help="long-run checkpoint interval; default 5 minutes after completed updates")
    parser.add_argument("--keep-last", type=int, default=3,
                        help="retain this many numbered checkpoints after successful publication")
    return parser


def supervise(args) -> dict:
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        raise ValueError("max-hours must be finite and positive")
    if not math.isfinite(args.save_every_seconds) or args.save_every_seconds < 0:
        raise ValueError("save-every-seconds must be finite and nonnegative")
    if args.keep_last < 1:
        raise ValueError("keep-last must be positive")
    for key in ("work_dir", "source_dir", "data_dir", "base", "resume", "out"):
        setattr(args, key, getattr(args, key).resolve())
    if not args.work_dir.is_dir() or not args.source_dir.is_dir():
        raise ValueError("work-dir and source-dir must exist")
    if not (args.base / "model.safetensors").is_file() or not args.resume.is_file():
        raise ValueError("verified base and exact resume file must exist")
    if not (args.source_dir / "operators/rocm/model_bench.py").is_file():
        raise ValueError("source-dir lacks the real-data model_bench launcher")
    if args.out == args.resume.parent or args.out == args.data_dir:
        raise ValueError("out must be separate from the source checkpoint and data directory")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to restart or overwrite an existing supervisor output")
    args.out.mkdir(parents=True, exist_ok=True)
    status = {"status": "starting", "supervisor_pid": os.getpid(), "started_at": utc_now(),
              "source_checkpoint": str(args.resume), "data_dir": str(args.data_dir),
              "max_long_train_hours": args.max_hours,
              "save_every_seconds": args.save_every_seconds, "keep_last": args.keep_last,
              "time_limit_scope": "long Trainer loop only; excludes preparation, smoke, reconstruction, parity",
              "retries": 0, "controls_other_gpu_jobs": False}

    def update(values: dict) -> None:
        status.update(values, updated_at=utc_now())
        write_json(args.out / "status.json", status)
        print(json.dumps({key: value for key, value in values.items() if key != "child_command"},
                         allow_nan=False), flush=True)

    update({"status": "starting"})
    try:
        corpus = wait_for_data(args.data_dir, update)
        write_json(args.out / "data_verification.json", corpus)
        update({"status": "data_verified", "train_sequences": corpus["splits"]["train"]["sequences"],
                "hash_intersection": 0})
        source = verify_checkpoint(args, args.resume, mode="source", expected_updates=0, on_status=update)
        smoke_command = training_command(args, corpus, stage="smoke", resume=args.resume, updates=SMOKE_UPDATES)
        run_child(smoke_command, stage="smoke", log=args.out / "smoke.log", work_dir=args.work_dir,
                  on_status=update, environment=dict(os.environ, PYTHONUNBUFFERED="1"))
        smoke_report = verified_training_report(args.out / "smoke/parity.json", expected_updates=SMOKE_UPDATES)
        smoke_path = args.out / "smoke/train/trainable.pt"
        smoke = verify_checkpoint(args, smoke_path, mode="smoke", expected_updates=SMOKE_UPDATES, on_status=update)
        updates = int(corpus["splits"]["train"]["sequences"]) - SMOKE_UPDATES
        update({"status": "smoke_verified", "smoke_checkpoint": smoke, "long_update_cap": updates})
        command = training_command(args, corpus, stage="long", resume=smoke_path, updates=updates)
        run_child(command, stage="long_training", log=args.out / "long.log", work_dir=args.work_dir,
                  on_status=update, environment=dict(os.environ, PYTHONUNBUFFERED="1"))
        report = verified_training_report(args.out / "long/parity.json")
        final = verify_checkpoint(args, args.out / "long/train/trainable.pt", mode="long",
                                  expected_updates=int(corpus["splits"]["train"]["sequences"]), on_status=update)
        if final["updates"] != SMOKE_UPDATES + report["updates"]:
            raise ValueError("long checkpoint does not match the completion report")
        summary = {"status": "complete", "completed_at": utc_now(), "source": source,
                   "data": corpus, "smoke": {"checkpoint": smoke, "validation": smoke_report},
                   "long": {"checkpoint": final, "validation": report},
                   "time_limit_scope": status["time_limit_scope"], "max_long_train_hours": args.max_hours,
                   "long_update_cap": updates, "total_updates": final["updates"],
                   "language_quality_claim": False}
        write_json(args.out / "summary.json", summary)
        update({"status": "complete", "summary": str(args.out / "summary.json"),
                "final_step": final["step"], "total_updates": final["updates"],
                "stop_reason": report["completion"].get("stop_reason")})
        return summary
    except Exception as error:
        update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        supervise(args)
    except Exception as error:
        print(f"supervisor failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
