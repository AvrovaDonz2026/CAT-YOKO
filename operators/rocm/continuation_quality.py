"""Observe unused validation rows without changing the primary training score.

This is a second slice of the same validation corpus, not an external benchmark.
Imports stay standard-library-only; the loaded benchmark supplies its torch API.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

QUALITY_SOURCE = "operators/rocm/continuation_quality.py"
PROTOCOL = "same_validation_corpus_disjoint_rows_not_external_benchmark"


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _integer(value, name):
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_row(row, *, event, start_row, batches):
    if not isinstance(row, dict) or row.get("event") != event or row.get("pass") is not True:
        raise ValueError("independent held-out evaluation did not pass")
    _integer(row.get("step"), "evaluation step")
    if not _finite(row.get("eval_nll")) or row["eval_nll"] < 0:
        raise ValueError("independent held-out NLL is invalid")
    valid = row.get("eval_valid_tokens")
    if not _finite(valid) or not 0 < valid <= batches * 4096:
        raise ValueError("independent held-out token count is invalid")
    if (row.get("eval_batches") != batches or row.get("row_start") != start_row
            or row.get("row_stop") != start_row + batches):
        raise ValueError("independent held-out slice differs")
    before, after = row.get("stream_before", {}), row.get("stream_after", {})
    for state, cursor in ((before, start_row), (after, start_row + batches)):
        if (state.get("kind") != "packed" or state.get("i") != cursor
                or state.get("stride") != 1 or type(state.get("nseq")) is not int
                or state["nseq"] < start_row + batches):
            raise ValueError("independent held-out cursor wrapped or changed")
    if before["nseq"] != after["nseq"]:
        raise ValueError("independent held-out corpus size changed")
    for key in ("no_wrap", "rng_restored", "open_restored", "model_training_restored",
                "eval_batches_restored", "training_stream_untouched"):
        if row.get(key) is not True:
            raise ValueError(f"independent held-out restoration missing: {key}")


def validate_quality_report(path, *, source_dir, start_row=32, batches=32):
    """Require a completed, source-bound paired observation; return its receipt."""
    if _integer(start_row, "start_row") < 32 or _integer(batches, "batches") != 32:
        raise ValueError("independent observation needs at least row 32 and exactly 32 batches")
    report = json.loads(Path(path).read_text())
    expected = {QUALITY_SOURCE: _digest(Path(source_dir) / QUALITY_SOURCE)}
    if (report.get("status") != "completed" or report.get("protocol") != PROTOCOL
            or report.get("file_sha256") != expected or report.get("seq_len") != 4096
            or report.get("eval_row_start") != start_row
            or report.get("eval_row_stop") != start_row + batches
            or report.get("eval_batches") != batches
            or report.get("primary_eval_row_stop", start_row + 1) > start_row):
        raise ValueError("independent quality receipt is incomplete, overlapping, or from another source")
    initial, final = report.get("initial_eval", {}), report.get("final_eval", {})
    _validate_row(initial, event="independent_initial_eval", start_row=start_row, batches=batches)
    _validate_row(final, event="independent_final_eval", start_row=start_row, batches=batches)
    if not isinstance(report.get("eval_data"), str):
        raise ValueError("independent observation has no validation corpus identity")
    if (final["step"] <= initial["step"]
            or initial["stream_before"]["nseq"] != final["stream_before"]["nseq"]
            or report.get("eval_data_sha256") != _digest(report["eval_data"])):
        raise ValueError("independent observation did not span training or its corpus changed")
    return report


def _clone(value):
    return value.clone() if hasattr(value, "clone") else deepcopy(value)


@contextmanager
def quality_context(model_bench, out, eval_row_start=32, eval_batches=32):
    """Add only initial/final checks on disjoint rows, preserving primary rows."""
    start = _integer(eval_row_start, "eval_row_start")
    batches = _integer(eval_batches, "eval_batches")
    if start < 32 or batches != 32:
        raise ValueError("independent quality check needs row_start >= 32 and exactly 32 batches")
    path = Path(out) / "quality.json"
    if path.exists():
        raise FileExistsError(f"refusing to replace prior quality receipt: {path}")
    evaluate = model_bench.evaluate_heldout
    module_hash = _digest(__file__)
    report = {
        "status": "pending", "protocol": PROTOCOL,
        "scope": "Rows are disjoint from the repeatedly scored prefix; this is the same validation corpus.",
        "eval_row_start": start, "eval_row_stop": start + batches,
        "eval_batches": batches, "seq_len": 4096,
        "file_sha256": {QUALITY_SOURCE: module_hash},
    }
    paired_objects = []
    _write(path, report)

    def independent(trainer, model, *, event, step):
        if (trainer.seq_len != 4096 or trainer.micro_batch != 1
                or getattr(trainer, "world", 1) != 1 or trainer.eval_data is None
                or trainer.eval_batches > start):
            raise ValueError("independent observation requires 4096/mb1/world1 and disjoint validation rows")
        eval_path = Path(trainer.eval_data).resolve()
        if getattr(trainer, "data", None) is not None and eval_path == Path(trainer.data).resolve():
            raise ValueError("independent observation cannot use the training corpus")
        data_hash = _digest(eval_path)
        if "eval_data_sha256" in report and (
                report["eval_data"] != str(eval_path) or report["eval_data_sha256"] != data_hash):
            raise ValueError("independent validation corpus changed between observations")
        report.update(eval_data=str(eval_path), eval_data_sha256=data_hash,
                      primary_eval_row_stop=trainer.eval_batches,
                      source_checkpoint=str(trainer.resume) if getattr(trainer, "resume", None) else None)
        torch = model_bench.torch
        cpu_rng, py_rng = _clone(torch.get_rng_state()), random.getstate()
        cuda_rng = ([_clone(state) for state in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else None)
        was_train, original_batches, original_open = model.training, trainer.eval_batches, trainer._open
        opened, non_eval_opens = [], []

        def offset_open(stream_path, seed):
            stream = original_open(stream_path, seed)
            if stream_path is None or Path(stream_path).resolve() != eval_path:
                non_eval_opens.append(str(stream_path))
                return stream
            state = stream.state_dict()
            nseq = _integer(state.get("nseq"), "validation nseq")
            if (state.get("kind") != "packed" or state.get("stride") != 1
                    or nseq < start + batches or eval_path.stat().st_size != nseq * 4096 * 4):
                raise ValueError("independent validation stream cannot cover the requested rows without wrapping")
            stream.load_state_dict(dict(state, i=start))
            before = deepcopy(stream.state_dict())
            if before != dict(state, i=start):
                raise ValueError("independent validation stream rejected the offset")
            opened.append((stream, before))
            return stream

        try:
            trainer.eval_batches = batches
            with patch.object(trainer, "_open", offset_open):
                row = evaluate(trainer, model, event="independent_" + event, step=step)
        finally:
            trainer.eval_batches = original_batches
            model.train(was_train)
            model_bench.clear_moe_statistics(model)
            torch.set_rng_state(cpu_rng)
            random.setstate(py_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if len(opened) != 1 or non_eval_opens:
            raise ValueError("independent evaluation must open only one validation stream")
        stream, before = opened[0]
        row = dict(row, row_start=start, row_stop=start + batches,
                   stream_before=before, stream_after=deepcopy(stream.state_dict()),
                   no_wrap=True, open_restored=trainer._open == original_open,
                   model_training_restored=model.training == was_train,
                   eval_batches_restored=trainer.eval_batches == original_batches,
                   training_stream_untouched=not non_eval_opens,
                   rng_restored=(torch.equal(cpu_rng, torch.get_rng_state())
                                 and random.getstate() == py_rng
                                 and (cuda_rng is None or (
                                     len(cuda_rng) == len(torch.cuda.get_rng_state_all())
                                     and all(torch.equal(a, b) for a, b in
                                             zip(cuda_rng, torch.cuda.get_rng_state_all()))))))
        _validate_row(row, event="independent_" + event, start_row=start, batches=batches)
        if _digest(__file__) != module_hash or _digest(eval_path) != data_hash:
            raise ValueError("independent observation source changed during evaluation")
        return row

    def observed(trainer, model, *, event, step):
        try:
            if report["status"] == "failed":
                raise ValueError("independent observation already failed; continuation is refused")
            primary = evaluate(trainer, model, event=event, step=step)
            if event not in ("initial_eval", "final_eval"):
                return primary
            if event in report or (event == "final_eval" and "initial_eval" not in report):
                raise ValueError("independent observations must be exactly one initial/final pair")
            if paired_objects and (trainer is not paired_objects[0] or model is not paired_objects[1]):
                raise ValueError("independent observations must use the same trainer and model")
            if not paired_objects:
                paired_objects.extend((trainer, model))
            report[event] = independent(trainer, model, event=event, step=step)
            report["status"] = "initial_checked" if event == "initial_eval" else "final_checked"
            _write(path, report)
            return primary
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            _write(path, report)
            raise

    try:
        with patch.object(model_bench, "evaluate_heldout", observed):
            yield report
        if report["status"] == "failed":
            raise ValueError("independent observation failed inside the training context")
        if "initial_eval" not in report or "final_eval" not in report:
            raise ValueError("independent observation did not complete both evaluations")
        if report["final_eval"]["step"] <= report["initial_eval"]["step"]:
            raise ValueError("independent observation did not span any training updates")
        report["status"] = "completed"
        _write(path, report)
        validate_quality_report(path, source_dir=Path(__file__).resolve().parents[2],
                                start_row=start, batches=batches)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write(path, report)
        raise
