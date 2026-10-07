#!/usr/bin/env python3
"""Accepted B0 continuation with explicit fresh rows and a third held-out pair."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm.fresh_corpus_handoff import (
    HANDOFF_FILES, PROTOCOL, digest, load_handoff, need, plan_digest,
    validate_fresh_quality_report,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def fresh_quality_context(model_bench, out, handoff):
    """Evaluate new held-out rows, restoring all training and global RNG state."""
    path = Path(out) / "fresh_quality.json"
    need(not path.exists(), "refusing to replace a fresh held-out receipt")
    evaluate = model_bench.evaluate_heldout
    report = {"status": "pending", "protocol": PROTOCOL, "seq_len": 4096,
              "eval_batches": 32, "row_start": 0, "row_stop": 32,
              "eval_data": handoff["fresh_eval_path"], "eval_data_sha256": handoff["fresh_eval_sha256"],
              "source_checkpoint": handoff["source_checkpoint_path"],
              "source_checkpoint_sha256": handoff["source_checkpoint_sha256"],
              "data_handoff_sha256": plan_digest(handoff),
              "file_sha256": {name: digest(Path(__file__).resolve().parents[2] / name) for name in HANDOFF_FILES}}
    write_json(path, report)
    paired = []

    def fresh_evaluate(trainer, model, *, event, step):
        torch = model_bench.torch
        need(trainer.seq_len == 4096 and trainer.micro_batch == 1 and trainer.world == 1,
             "fresh evaluation requires 4096/mb1/world1")
        cpu_rng, py_rng = torch.get_rng_state().clone(), random.getstate()
        hip_rng = ([x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)
        was_training = model.training
        old_eval, old_batches, old_open = trainer.eval_data, trainer.eval_batches, trainer._open
        fresh_path = Path(handoff["fresh_eval_path"])
        opened = []

        def checked_open(stream_path, seed):
            need(stream_path is not None and Path(stream_path).resolve() == fresh_path,
                 "fresh evaluation tried to open a training stream")
            stream = old_open(stream_path, seed)
            before = deepcopy(stream.state_dict())
            need(before == {"kind": "packed", "i": 0, "stride": 1, "nseq": handoff["fresh_eval_nseq"]},
                 "fresh validation stream identity differs")
            opened.append((stream, before))
            return stream

        try:
            trainer.eval_data, trainer.eval_batches = fresh_path, 32
            with patch.object(trainer, "_open", checked_open):
                row = evaluate(trainer, model, event="fresh_" + event, step=step)
        finally:
            trainer.eval_data, trainer.eval_batches = old_eval, old_batches
            model.train(was_training)
            model_bench.clear_moe_statistics(model)
            torch.set_rng_state(cpu_rng)
            random.setstate(py_rng)
            if hip_rng is not None:
                torch.cuda.set_rng_state_all(hip_rng)
        need(len(opened) == 1, "fresh evaluation did not open exactly one validation stream")
        stream, before = opened[0]
        return dict(row, row_start=0, row_stop=32, stream_before=before,
                    stream_after=deepcopy(stream.state_dict()), no_wrap=True,
                    open_restored=trainer._open == old_open,
                    model_training_restored=model.training == was_training,
                    eval_batches_restored=(trainer.eval_batches == old_batches and trainer.eval_data == old_eval),
                    training_stream_untouched=True,
                    rng_restored=(torch.equal(cpu_rng, torch.get_rng_state()) and random.getstate() == py_rng
                                  and (hip_rng is None or (len(hip_rng) == len(torch.cuda.get_rng_state_all())
                                  and all(torch.equal(a, b) for a, b in zip(hip_rng, torch.cuda.get_rng_state_all()))))))

    def observed(trainer, model, *, event, step):
        primary = evaluate(trainer, model, event=event, step=step)
        if event not in ("initial_eval", "final_eval"):
            return primary
        need(event not in report and (event == "initial_eval" or "initial_eval" in report),
             "fresh paired observations repeated or reordered")
        if paired:
            need(trainer is paired[0] and model is paired[1], "fresh observations changed model/trainer")
        else:
            paired.extend((trainer, model))
        report[event] = fresh_evaluate(trainer, model, event=event, step=step)
        report["status"] = "initial_checked" if event == "initial_eval" else "final_checked"
        write_json(path, report)
        return primary

    try:
        with patch.object(model_bench, "evaluate_heldout", observed):
            yield report
        need("initial_eval" in report and "final_eval" in report, "fresh evaluation did not span training")
        report["status"] = "completed"
        write_json(path, report)
        validate_fresh_quality_report(path, source_dir=Path(__file__).resolve().parents[2], handoff=handoff)
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        write_json(path, report)
        raise


def main(argv=None):
    from operators.rocm import model_bench, production_continuation
    from operators.rocm.handoff_stream import handoff_context
    argv = list(sys.argv[1:] if argv is None else argv)
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-handoff", type=Path, required=True)
    parser.add_argument("--fresh-eval-data", type=Path, required=True)
    handoff_args, native_argv = parser.parse_known_args(argv)
    handoff = load_handoff(handoff_args.data_handoff)
    need(handoff_args.fresh_eval_data.resolve() == Path(handoff["fresh_eval_path"]), "fresh held-out CLI differs")
    from operators.rocm import round3_candidate_bench
    inherited_parser = round3_candidate_bench.build_parser()
    inherited, _ = inherited_parser.parse_known_args(native_argv)
    need(inherited.resume.resolve() == Path(handoff["source_checkpoint_path"])
         and inherited.data.resolve() == Path(handoff["new_train_path"])
         and inherited.run_steps == handoff["updates"] and inherited.eval_data is not None
         and inherited.eval_every == 250 and inherited.eval_batches == 32
         and inherited.save_optim and inherited.save_every_seconds == 300 and inherited.keep_last == 3
         and inherited.max_hours is None,
         "fresh worker source/corpus/update CLI differs from the explicit handoff")
    inherited.out.mkdir(parents=True, exist_ok=True)
    receipt_path = inherited.out / "data_handoff.json"
    need(not receipt_path.exists(), "refusing to replace an exercised handoff receipt")
    receipt = {"status": "pending", "data_handoff": handoff, "data_handoff_sha256": plan_digest(handoff),
               "file_sha256": {name: digest(Path(__file__).resolve().parents[2] / name) for name in HANDOFF_FILES}}
    write_json(receipt_path, receipt)
    try:
        with handoff_context(model_bench, handoff) as streams, fresh_quality_context(model_bench, inherited.out, handoff):
            code = production_continuation.main(native_argv)
        need(code == 0, "accepted production continuation failed")
        observations = [{"stream": stream.state_dict(), "batch_calls": stream.batch_calls,
                         "first_batch_row": stream.first_batch_row, "last_batch_row": stream.last_batch_row}
                        for stream in streams]
        trained = [row for row in observations if row["batch_calls"] == handoff["updates"]]
        need(len(trained) == 1 and trained[0]["first_batch_row"] == 0 and trained[0]["last_batch_row"] == 3999
             and trained[0]["stream"]["i"] == handoff["target_cursor"], "fresh training skipped or repeated rows")
        receipt.update(status="completed", opened_train_streams=observations, no_wrap=True,
                       first_training_row=0, last_training_row=3999, completed_updates=4000)
        write_json(receipt_path, receipt)
        return 0
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
