#!/usr/bin/env python3
"""Compare the next opt-in operators against the existing B0 reference.

Every run retains packed attention and shared-storage MoE. Extra patches enter
after native reference capture. Nothing changes another process or its files.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import candidate_bench, model_bench

EXTRA_FLAGS = {"--gpu-fp32-adam", "--bucketed-attention", "--sync-update-timing"}
NATIVE_ADAM_SOURCES = {"operators/rocm/gpu_adam.py", "operators/rocm/gpu_adam_bench.py", "cat_yoko/optim.py"}


def build_parser():
    parser = candidate_bench.build_parser()
    parser.description = __doc__
    parser.add_argument("--gpu-fp32-adam", action="store_true")
    parser.add_argument("--bucketed-attention", action="store_true")
    parser.add_argument("--sync-update-timing", action="store_true",
                        help="synchronize complete updates in isolated comparisons")
    parser.add_argument("--gpu-adam-gate", type=Path,
                        help="successful same-checkpoint GPU optimizer numerical report")
    return parser


def legacy_argv(argv):
    result = []
    iterator = iter(argv)
    for value in iterator:
        if value in EXTRA_FLAGS:
            continue
        if value == "--gpu-adam-gate":
            next(iterator)
            continue
        if value.startswith("--gpu-adam-gate="):
            continue
        result.append(value)
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_extra_args(parser, args):
    if not args.packed_attention or args.moe_layout != "shared-storage":
        parser.error("this round requires the validated packed attention and shared-storage layout")
    if args.bucketed_attention and not args.reference_repeat:
        parser.error("bucketed attention requires native reference repeat")
    if args.gpu_fp32_adam and not args.save_optim:
        parser.error("GPU Adam requires optimizer checkpoint export")
    if args.gpu_fp32_adam:
        if args.gpu_adam_gate is None:
            parser.error("GPU Adam requires --gpu-adam-gate from an independent optimizer check")
        gate = json.loads(args.gpu_adam_gate.read_text())
        source_root = Path(__file__).resolve().parents[2]
        hashes = gate.get("optimizer_source_sha256", {})
        if (gate.get("arithmetic_variant") != "native-order" or not isinstance(hashes, dict)
                or set(hashes) != NATIVE_ADAM_SOURCES
                or any(value != digest(source_root / name) for name, value in hashes.items())):
            parser.error("GPU Adam numerical report does not match the native-order implementation and source hashes")

        def numerical_row_passes(item):
            return (isinstance(item, dict) and item.get("pass") is True and item.get("parameters") == 132
                    and item.get("all_weights_bitwise") is True
                    and item.get("gate", {}).get("fp32_moment_atol") == 1e-8
                    and item.get("gate", {}).get("fp32_moment_rtol") == 1e-6)

        if (gate.get("status") != "complete" or gate.get("device") != "cuda"
                or gate.get("source_sha256") != digest(args.resume)
                or gate.get("checkpoint_moments_cpu_fp32") is not True
                or gate.get("persistent_fp32_master") is not False
                or len(gate.get("parity", [])) < 5
                or any(not numerical_row_passes(item) for item in gate.get("parity", []))
                or not numerical_row_passes(gate.get("terminal", {}))):
            parser.error("GPU Adam numerical report did not pass for all source parameters")
    if args.sync_update_timing and args.parity_only:
        parser.error("update timing requires optimizer updates")


class BucketParityStream:
    """Exercise a real bucketable row without advancing the training stream."""

    def __init__(self, stream, info, max_batches=32):
        self.stream, self.info, self.max_batches = stream, info, max_batches

    def batch(self, micro_batch, device):
        from operators.rocm.bucketed_attention import BucketConfig, build_bucket_plan
        from operators.rocm.packed_attention import _document_plan
        for scanned in range(1, self.max_batches + 1):
            batch = self.stream.batch(micro_batch, device)
            docs = batch.get("doc_ids")
            if docs is None:
                continue
            _, buckets, reason = build_bucket_plan(_document_plan(docs, None),
                                                   int(docs.size(-1)), BucketConfig())
            if reason is None:
                self.info["bucketed_parity_scanned_batches"] = scanned
                self.info["bucketed_parity_buckets"] = [len(bucket) for bucket in buckets]
                return batch
        raise RuntimeError("no bucketable real row in the bounded parity search; refusing updates")


@contextmanager
def next_context(args, report):
    original_context = candidate_bench.candidate_context
    installations = {}

    @contextmanager
    def extended_context(legacy_args):
        with original_context(legacy_args) as legacy_report, ExitStack() as stack:
            install = model_bench.install_moe_layout
            emit = model_bench.emit
            open_stream = model_bench.open_parity_stream

            def extra_stream(cfg, seq, seed, **kwargs):
                stream, info = open_stream(cfg, seq, seed, **kwargs)
                if args.bucketed_attention and seq == 4096:
                    if isinstance(stream, candidate_bench.PackedParityStream):
                        stream = stream.stream
                    stream = BucketParityStream(stream, info)
                return stream, info

            def install_extra(model, layout):
                result = install(model, layout)
                if args.bucketed_attention:
                    from operators.rocm.bucketed_attention import bucketed_attention_context
                    installations["bucketed_attention"] = stack.enter_context(bucketed_attention_context())
                if args.gpu_fp32_adam:
                    from operators.rocm.gpu_adam import use_gpu_fp32_adam
                    installations["gpu_fp32_adam"] = stack.enter_context(use_gpu_fp32_adam())
                if args.sync_update_timing:
                    from operators.rocm.update_timing import synchronized_update_timing
                    installations["update_timing"] = stack.enter_context(
                        synchronized_update_timing(args.out / "update_timing.json", warmup=5))
                report["patches_installed_after_native_reference"] = True
                return {**result, "next_operator_round": {
                    "gpu_fp32_adam": args.gpu_fp32_adam,
                    "bucketed_attention": args.bucketed_attention,
                    "sync_update_timing": args.sync_update_timing,
                }}

            def emitted(path, full_report, event):
                if event.get("event") == "parity_complete" and args.bucketed_attention:
                    calls = installations["bucketed_attention"].report()
                    report["bucketed_attention_parity_calls"] = calls
                    if event.get("pass") and calls.get("bucketed_calls", 0) <= 0:
                        full_report["status"] = "candidate_not_exercised"
                        emit(path, full_report, {"event": "bucketed_candidate_not_exercised", "pass": False})
                        raise RuntimeError("bucketed attention was not exercised before updates")
                return emit(path, full_report, event)

            # Keep the original historical flag, and add the actual device used
            # by the opt-in optimizer to each new experiment's metrics.
            from cat_yoko.trainer import Trainer
            log = Trainer._log

            def logged(owner, row):
                row = dict(row, actual_optimizer_device="cuda" if args.gpu_fp32_adam else "cpu",
                           bucketed_attention=args.bucketed_attention)
                return log(owner, row)

            try:
                with patch.object(model_bench, "install_moe_layout", install_extra), \
                        patch.object(model_bench, "emit", emitted), \
                        patch.object(model_bench, "open_parity_stream", extra_stream), \
                        patch.object(Trainer, "_log", logged):
                    yield legacy_report
            finally:
                for name, installation in installations.items():
                    report[name + "_calls"] = installation.report()

    with patch.object(candidate_bench, "candidate_context", extended_context):
        yield


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    model_bench.validate_args(parser, args)
    candidate_bench.validate_candidates(parser, args)
    validate_extra_args(parser, args)
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root / "operators/rocm/candidate_bench.py"]
    for enabled, name in ((args.gpu_fp32_adam, "gpu_adam.py"),
                          (args.bucketed_attention, "bucketed_attention.py"),
                          (args.sync_update_timing, "update_timing.py")):
        if enabled:
            paths.append(root / "operators/rocm" / name)
    report = {
        "gpu_fp32_adam": args.gpu_fp32_adam,
        "bucketed_attention": args.bucketed_attention,
        "sync_update_timing": args.sync_update_timing,
        "patches_installed_after_native_reference": False,
        "file_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths},
    }
    try:
        with next_context(args, report):
            code = candidate_bench.main(legacy_argv(argv))
        report["status"] = "completed" if code == 0 else "failed"
        return code
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if (args.out / "parity.json").is_file():
            (args.out / "next_operators.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
