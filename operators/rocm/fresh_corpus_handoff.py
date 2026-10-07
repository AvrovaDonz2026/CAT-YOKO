"""Explicit fresh-corpus coordinates and native CPU recovery audits.

The old checkpoint remains immutable. Its absolute input/Adam clock continues,
while new corpus coordinates start at zero. Imports here are CPU/stdlib only;
Torch is imported only by the isolated checkpoint audit command.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import os
from unittest.mock import patch

SEQ_LEN = 4096
MAPPING = "row=i-absolute_cursor_origin"
PROTOCOL = "fresh_validation_corpus_paired_heldout"
SOURCE_SHA = "20d3590756955195b8c9e2e8cc40e74767373a3cd1d1fa430ea04023f9b543dc"
HANDOFF_FILES = ("operators/rocm/fresh_corpus_handoff.py", "operators/rocm/handoff_stream.py",
                 "operators/rocm/fresh_corpus_continuation.py")


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def plan_digest(plan):
    return hashlib.sha256((json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False) + "\n").encode()).hexdigest()


def write_handoff(path, plan):
    validate_handoff(plan)
    path = Path(path)
    need(not path.exists(), "refusing to replace a data handoff")
    path.write_text(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return digest(path)


def _rows(path):
    rows, remainder = divmod(Path(path).stat().st_size, SEQ_LEN * 4)
    need(not remainder and rows > 0, "corpus is not complete 4096-token int32 rows")
    return rows


def _verify_manifest(plan):
    path = Path(plan["corpus_manifest_path"])
    manifest = json.loads(path.read_text())
    need(manifest.get("status") == "complete" and manifest.get("dataset_kind") == "fresh_exact_text_excluded"
         and manifest.get("seq_len") == SEQ_LEN and manifest.get("eos_id") == 1
         and manifest.get("vocab_size") == 130560, "fresh manifest uses a different packing/tokenizer recipe")
    for split, prefix in (("train", "new_train"), ("eval", "fresh_eval")):
        record = manifest["outputs"][split]
        actual = Path(plan[prefix + "_path"])
        declared = (path.parent / record["path"]).resolve()
        rows = plan["new_nseq"] if split == "train" else plan["fresh_eval_nseq"]
        need(actual.resolve() == declared and record["sha256"] == plan[prefix + "_sha256"]
             and record["sequences"] == rows and record["tokens"] == rows * SEQ_LEN
             and record["bytes"] == rows * SEQ_LEN * 4, "fresh output differs from its verified manifest")
    intersection = manifest.get("old_hash_intersections", {})
    expected = {"new_train_old_train", "new_train_old_eval", "new_eval_old_train", "new_eval_old_eval"}
    need(set(intersection) == expected and all(type(v) is int and v == 0 for v in intersection.values())
         and manifest.get("hash_intersection") == 0, "fresh corpus overlaps the old corpus or held-out data")
    exclusion = manifest.get("old_corpus_exclusion", {})
    old_manifest = Path(exclusion["manifest_path"])
    need(digest(old_manifest) == exclusion["manifest_sha256"], "old exclusion manifest changed")
    old = json.loads(old_manifest.read_text())
    need(old["outputs"]["train"]["sha256"] == plan["old_train_sha256"], "handoff old training corpus differs")
    need(manifest.get("tokenizer", {}).get("files_sha256") == old.get("tokenizer", {}).get("files_sha256")
         and bool(old.get("tokenizer", {}).get("files_sha256")), "fresh tokenizer differs from the original corpus")
    for split, record in exclusion.get("document_hashes", {}).items():
        need(split in ("train", "eval") and digest(record["path"]) == record["sha256"],
             "old exclusion document hashes changed")
    need(set(exclusion.get("document_hashes", {})) == {"train", "eval"}, "fresh exclusion lost an old split")
    for record in manifest.get("hashes", {}).values():
        need(digest(path.parent / record["path"]) == record["sha256"], "fresh document hash ledger changed")
    need(set(manifest.get("hashes", {})) == {"train", "eval"}, "fresh document hash ledger is incomplete")
    source_lock = manifest.get("source_lock", {})
    need(digest(source_lock["path"]) == source_lock["sha256"], "fresh source lock changed")
    return manifest


def build_handoff(source_metadata, source_checkpoint, old_data, new_data, fresh_eval, manifest, *, updates=4000):
    need(type(updates) is int and updates == 4000, "this authorization is exactly 4000 fresh updates")
    for key, value in (("source_step", 81864), ("source_stream_i", 47062), ("source_adam_step", 47062),
                       ("source_data_rows", 19531), ("source_tokens_in_phase", 355010560),
                       ("source_tokens_seen", 355010560)):
        need(source_metadata.get(key) == value, "wrong immutable 81864 source: " + key)
    paths = {key: str(Path(value).resolve()) for key, value in
             (("source_checkpoint", source_checkpoint), ("old_train", old_data), ("new_train", new_data),
              ("fresh_eval", fresh_eval), ("corpus_manifest", manifest))}
    plan = {"schema_version": 1, "mapping": MAPPING, "no_wrap": True,
            "seq_len": SEQ_LEN, "eos_id": 1, "old_nseq": 19531,
            "new_nseq": _rows(new_data), "fresh_eval_nseq": _rows(fresh_eval),
            "absolute_cursor_origin": 47062, "logical_row_origin": 0,
            "source_step": 81864, "source_phase_tokens": 355010560,
            "source_tokens_seen": 355010560, "source_real_tokens": 192765952,
            "source_unique_tokens": 79998976, "updates": updates,
            "target_step": 85864, "target_cursor": 51062, "target_logical_row": 4000,
            "target_phase_tokens": 371394560, "target_tokens_seen": 371394560,
            "target_real_tokens": 209149952}
    for key, value in paths.items():
        plan[key + "_path"] = value
        plan[key + "_sha256"] = digest(value)
    plan["new_unique_tokens"] = plan["new_nseq"] * SEQ_LEN
    plan["total_unique_tokens"] = plan["source_unique_tokens"] + plan["new_unique_tokens"]
    return validate_handoff(plan, verify_files=True)


def validate_handoff(plan, *, verify_files=False):
    need(isinstance(plan, dict), "data handoff must be an object")
    for key, value in (("schema_version", 1), ("mapping", MAPPING), ("no_wrap", True),
                       ("seq_len", SEQ_LEN), ("eos_id", 1), ("old_nseq", 19531),
                       ("absolute_cursor_origin", 47062), ("logical_row_origin", 0),
                       ("source_step", 81864), ("source_phase_tokens", 355010560),
                       ("source_tokens_seen", 355010560), ("source_real_tokens", 192765952),
                       ("source_unique_tokens", 79998976), ("updates", 4000),
                       ("target_step", 85864), ("target_cursor", 51062), ("target_logical_row", 4000),
                       ("target_phase_tokens", 371394560), ("target_tokens_seen", 371394560),
                       ("target_real_tokens", 209149952), ("source_checkpoint_sha256", SOURCE_SHA)):
        need(plan.get(key) == value and (type(plan.get(key)) is type(value)), "invalid explicit handoff field: " + key)
    for key, minimum in (("new_nseq", 4000), ("fresh_eval_nseq", 32)):
        need(type(plan.get(key)) is int and plan[key] >= minimum, "insufficient fresh corpus rows")
    need(plan.get("new_unique_tokens") == plan["new_nseq"] * SEQ_LEN
         and plan.get("total_unique_tokens") == plan["source_unique_tokens"] + plan["new_unique_tokens"],
         "unique corpus ledger changed")
    for prefix in ("source_checkpoint", "old_train", "new_train", "fresh_eval", "corpus_manifest"):
        path, sha = plan.get(prefix + "_path"), plan.get(prefix + "_sha256")
        need(isinstance(path, str) and Path(path).is_absolute()
             and isinstance(sha, str) and len(sha) == 64 and set(sha) <= set("0123456789abcdef"),
             "missing corpus/checkpoint identity: " + prefix)
        if verify_files:
            need(digest(path) == sha, "protected handoff file changed: " + prefix)
    need(len({plan[p + "_path"] for p in ("old_train", "new_train", "fresh_eval")}) == 3,
         "handoff needs different old/new training and new evaluation files")
    if verify_files:
        need(_rows(plan["old_train_path"]) == plan["old_nseq"]
             and _rows(plan["new_train_path"]) == plan["new_nseq"]
             and _rows(plan["fresh_eval_path"]) == plan["fresh_eval_nseq"], "corpus lengths changed")
        _verify_manifest(plan)
    return plan


def load_handoff(path, *, verify_files=True):
    path = Path(path)
    plan = json.loads(path.read_text())
    validate_handoff(plan, verify_files=verify_files)
    need(digest(path) == plan_digest(plan), "handoff JSON is not its canonical immutable identity")
    return plan


def stream_state(plan, absolute_cursor):
    validate_handoff(plan)
    origin = plan["absolute_cursor_origin"]
    need(type(absolute_cursor) is int and origin <= absolute_cursor <= origin + plan["new_nseq"],
         "absolute cursor would wrap or precede fresh corpus")
    return {"kind": "packed", "i": absolute_cursor, "stride": 1, "nseq": plan["new_nseq"],
            "corpus_sha256": plan["new_train_sha256"], "absolute_cursor_origin": origin,
            "logical_row_origin": 0, "corpus_row": absolute_cursor - origin, "no_wrap": True,
            "data_handoff_sha256": plan_digest(plan)}


def check_stream_state(state, plan, *, allow_source=False):
    need(isinstance(state, dict), "packed handoff state is absent")
    if allow_source and state == {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531}:
        return stream_state(plan, 47062)
    need(type(state.get("i")) is int and state == stream_state(plan, state["i"]),
         "fresh recovery corpus SHA/origin/row/cursor differs")
    return deepcopy(state)


def _rng_check(torch, extra):
    generator = random.Random()
    generator.setstate(extra["rng_py"])
    state = extra["rng_torch"]
    need(torch.is_tensor(state) and state.dtype == torch.uint8 and state.device.type == "cpu"
         and state.ndim == 1 and state.numel() > 0, "CPU RNG schema differs")
    torch.Generator(device="cpu").set_state(state)
    hip = extra.get("rng_cuda")
    need(isinstance(hip, list) and len(hip) == 1
         and all(torch.is_tensor(x) and x.dtype == torch.uint8 and x.ndim == 1
                 and x.device.type == "cpu" and x.numel() > 0 for x in hip), "saved HIP RNG schema differs")


def audit_checkpoint(source_path, target_path, plan, updates):
    """Audit real stream identity, then reuse every unchanged native-state check.

    The compatibility views replace only target stream.nseq for the legacy
    checker, after its real new-corpus coordinates have passed strict checks.
    No tensor, optimizer state, clock, RNG, source file or target file is edited.
    """
    import torch
    from operators.rocm import run_operator_switch as checks
    validate_handoff(plan, verify_files=True)
    source_path, target_path = Path(source_path).resolve(), Path(target_path).resolve()
    need(str(source_path) == plan["source_checkpoint_path"], "CPU audit source differs from handoff")
    need(type(updates) is int and 0 <= updates <= plan["updates"], "audit update bound differs")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = source if source_path == target_path else torch.load(target_path, map_location="cpu", weights_only=False)
    expected_source = {"kind": "packed", "i": 47062, "stride": 1, "nseq": 19531}
    need(source["extra"]["stream"] == expected_source, "immutable source stream differs")
    need(source["extra"]["step"] == 81864 and source["extra"]["tokens_in_phase"] == 355010560
         and source["extra"]["tokens_seen"] == 355010560, "immutable source clocks differ")
    _rng_check(torch, source["extra"])
    if target is source:
        need(updates == 0, "source-only audit cannot claim later updates")
        actual_state = stream_state(plan, 47062)
    else:
        actual_state = check_stream_state(target["extra"].get("stream"), plan)
        need(actual_state["i"] == 47062 + updates and actual_state["corpus_row"] == updates,
             "fresh checkpoint consumed skipped/repeated rows")
        _rng_check(torch, target["extra"])
    views = {}
    for path, checkpoint in ((source_path, source), (target_path, target)):
        view = dict(checkpoint)
        view["extra"] = dict(checkpoint["extra"])
        view["extra"]["stream"] = dict(checkpoint["extra"]["stream"], nseq=19531)
        views[path] = view
    limit = ((plan["target_cursor"] + 19530) // 19531) * 19531
    arguments = ["native-check", str(source_path), str(target_path), plan["old_train_path"],
                 str(updates), "0", str(limit)]
    output = io.StringIO()
    with patch.object(sys, "argv", arguments), patch.object(torch, "load", side_effect=lambda path, **kw: views[Path(path).resolve()]), redirect_stdout(output):
        exec(compile(checks.CHECKPOINT_CHECK, "native-complete-state-check", "exec"), {})
    metadata = json.loads(output.getvalue().splitlines()[-1])
    metadata.update(source_data_rows=plan["new_nseq"], saved_source_data_rows=19531,
                    source_logical_row=actual_state["corpus_row"], data_handoff_sha256=plan_digest(plan),
                    source_real_tokens=plan["source_real_tokens"] + updates * SEQ_LEN,
                    stream_mapping=MAPPING, no_wrap=True, rng_schema_verified=True,
                    immutable_source_unedited=True)
    return metadata


def cpu_check(args, checkpoint, updates, log):
    """Subprocess CPU audit with all GPU visibility disabled."""
    plan = getattr(args, "handoff", None)
    manifest = getattr(args, "handoff_manifest", None)
    if manifest is None:
        manifest = getattr(args, "data_handoff_path", None)
    need(manifest is not None, "CPU audit needs its immutable handoff JSON")
    loaded = load_handoff(manifest, verify_files=False)
    if plan is not None:
        need(plan == loaded, "CPU audit handoff object differs from file")
    need(Path(args.resume).resolve() == Path(loaded["source_checkpoint_path"])
         and Path(args.data).resolve() == Path(loaded["new_train_path"]), "CPU audit paths differ from handoff")
    command = [args.python, "-m", "operators.rocm.fresh_corpus_handoff", "--audit",
               "--data-handoff", str(manifest), "--source", str(args.resume),
               "--checkpoint", str(checkpoint), "--updates", str(updates)]
    result = subprocess.run(command, cwd=args.source_dir, capture_output=True, text=True, timeout=300,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                                     ROCR_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONOPTIMIZE="0"))
    Path(log).write_text(result.stdout + result.stderr)
    need(result.returncode == 0, "fresh native CPU audit failed; see " + Path(log).name)
    return json.loads(result.stdout.splitlines()[-1])


def validate_fresh_quality_report(path, *, source_dir, handoff):
    validate_handoff(handoff)
    report = json.loads(Path(path).read_text())
    need(report.get("status") == "completed" and report.get("protocol") == PROTOCOL
         and report.get("seq_len") == SEQ_LEN and report.get("eval_batches") == 32
         and report.get("row_start") == 0 and report.get("row_stop") == 32
         and report.get("eval_data") == handoff["fresh_eval_path"]
         and report.get("eval_data_sha256") == handoff["fresh_eval_sha256"]
         and digest(report["eval_data"]) == handoff["fresh_eval_sha256"]
         and report.get("data_handoff_sha256") == plan_digest(handoff)
         and report.get("source_checkpoint") == handoff["source_checkpoint_path"]
         and report.get("source_checkpoint_sha256") == handoff["source_checkpoint_sha256"],
         "fresh held-out report is incomplete or belongs to another data handoff")
    need(report.get("file_sha256") == {name: digest(Path(source_dir) / name) for name in HANDOFF_FILES},
         "fresh held-out source hashes differ")
    for event, step in (("initial_eval", handoff["source_step"]), ("final_eval", handoff["target_step"])):
        row = report.get(event, {})
        need(row.get("event") == "fresh_" + event and row.get("step") == step and row.get("pass") is True
             and row.get("eval_batches") == 32 and row.get("row_start") == 0 and row.get("row_stop") == 32,
             "fresh evaluation did not span the exact training window")
        for key in ("eval_nll", "eval_valid_tokens"):
            value = row.get(key)
            need(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                 and value >= 0, "fresh evaluation contains invalid numbers")
        need(0 < row["eval_valid_tokens"] <= 32 * SEQ_LEN, "fresh evaluation token count differs")
        for key, cursor in (("stream_before", 0), ("stream_after", 32)):
            need(row.get(key) == {"kind": "packed", "i": cursor, "stride": 1, "nseq": handoff["fresh_eval_nseq"]},
                 "fresh held-out cursor wrapped or used another corpus")
        need(all(row.get(key) is True for key in ("no_wrap", "rng_restored", "open_restored",
             "model_training_restored", "eval_batches_restored", "training_stream_untouched")),
             "fresh held-out observation changed training state")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", required=True)
    parser.add_argument("--data-handoff", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()
    plan = load_handoff(args.data_handoff, verify_files=False)
    print(json.dumps(audit_checkpoint(args.source, args.checkpoint, plan, args.updates), allow_nan=False))


if __name__ == "__main__":
    main()
