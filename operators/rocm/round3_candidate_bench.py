#!/usr/bin/env python3
"""Isolated B0 validation/training with split attention and cached CPU Adam.

The original runner captures the native model and its repeated reference
before any new context is installed. The selected contexts then cover
candidate gradient parity, Trainer construction, optimizer restoration and
completed updates. Checkpoint, RNG and packed-cursor formats remain native.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import candidate_bench, model_bench


EXTRA_FLAGS = {"--split-attention", "--cached-cpu-adam", "--cpu-adam-cached",
               "--sync-update-timing", "--deterministic-training"}


def build_parser():
    parser = candidate_bench.build_parser()
    parser.description = __doc__
    parser.set_defaults(save_every_seconds=300.0, keep_last=3)
    parser.add_argument("--split-attention", action="store_true",
                        help="replace independent document slices with one FP32 split per Q/K/V")
    parser.add_argument("--cached-cpu-adam", "--cpu-adam-cached", dest="cached_cpu_adam",
                        action="store_true",
                        help="cache BF16 parameter readbacks; retain native CPU FP32 Adam arithmetic/state")
    parser.add_argument("--sync-update-timing", action="store_true",
                        help="record synchronized complete updates, discarding five warmup updates")
    parser.add_argument("--deterministic-training", action="store_true",
                        help="require deterministic algorithms through updates for controlled arithmetic checks")
    return parser


def legacy_argv(argv, args=None):
    """Remove new flags and forward this entry's resolved retention defaults."""
    result = [value for value in argv if value not in EXTRA_FLAGS]
    if args is not None:
        # The inherited runner parses argv again with its own older defaults.
        # Preserve explicit options and forward our defaults only if omitted.
        for flag, value in (("--save-every-seconds", args.save_every_seconds),
                            ("--keep-last", args.keep_last)):
            if not any(part == flag or part.startswith(flag + "=") for part in result):
                result.extend([flag, str(value)])
    return result


def validate_extra_args(parser, args):
    if not args.packed_attention or args.moe_layout != "shared-storage":
        parser.error("round-three runs require packed attention and shared-storage MoE")
    if not args.deterministic_parity or not args.reference_repeat:
        parser.error("round-three runs require deterministic parity and native reference repeat")
    if 4096 not in [int(value) for value in args.parity_seqs.split(",")]:
        parser.error("round-three runs require 4096-token full-model parity")
    if args.batched_grad_norm:
        parser.error("round-three comparisons retain the native gradient norm")
    if args.cached_cpu_adam and not args.parity_only and not args.save_optim:
        parser.error("cached CPU Adam training requires --save-optim")
    if args.sync_update_timing and args.parity_only:
        parser.error("update timing requires optimizer updates")
    for name, limit in (("loss_atol", 0.02), ("grad_relative_l2", 0.05),
                        ("output_relative_l2", 0.05)):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) > limit:
            parser.error("round-three runs cannot loosen the existing " + name + " gate")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


@contextmanager
def round3_context(args, report):
    """Extend the packed runner only after its native captures have completed."""
    original_context = candidate_bench.candidate_context
    installations = {}

    @contextmanager
    def extended_context(legacy_args):
        with original_context(legacy_args) as legacy_report, ExitStack() as stack:
            install = model_bench.install_moe_layout
            emit = model_bench.emit
            version = model_bench.source_code_version

            def install_extra(model, layout):
                # The parent first installs shared storage and packed attention.
                # model_bench invokes this only after native/reference-repeat.
                result = install(model, layout)
                if args.split_attention:
                    from operators.rocm.split_attention import split_attention_context
                    installations["split_attention"] = stack.enter_context(split_attention_context())
                if args.cached_cpu_adam:
                    from operators.rocm.cpu_adam_cached import use_cached_cpu_adam
                    installations["cached_cpu_adam"] = stack.enter_context(use_cached_cpu_adam())
                if args.sync_update_timing:
                    from operators.rocm.update_timing import synchronized_update_timing
                    installations["update_timing"] = stack.enter_context(
                        synchronized_update_timing(args.out / "update_timing.json", warmup=5))
                report["patches_installed_after_native_reference"] = True
                return {**result, "round3_operators": {
                    "split_attention": args.split_attention,
                    "cached_cpu_adam": args.cached_cpu_adam,
                    "sync_update_timing": args.sync_update_timing,
                    "deterministic_training": args.deterministic_training,
                }}

            def source_version(layout):
                return {**version(layout), "round3_operators": {
                    "split_attention": args.split_attention,
                    "cached_cpu_adam": args.cached_cpu_adam,
                    "sync_update_timing": args.sync_update_timing,
                    "deterministic_training": args.deterministic_training,
                    "file_sha256": dict(report.get("file_sha256", {})),
                }}

            def emitted(path, full_report, event):
                if event.get("event") == "training_complete":
                    event = dict(event, deterministic_training=args.deterministic_training,
                                 deterministic_algorithms=model_bench.torch.are_deterministic_algorithms_enabled())
                if event.get("event") == "parity_complete":
                    report["full_model_parity_passed"] = bool(event.get("pass"))
                    if args.split_attention:
                        calls = installations["split_attention"].report()
                        report["split_attention_parity_calls"] = calls
                        if event.get("pass") and calls.get("split_optimized_calls", 0) <= 0:
                            full_report["status"] = "round3_candidate_not_exercised"
                            emit(path, full_report, {"event": "split_candidate_not_exercised", "pass": False})
                            raise RuntimeError("split attention was not exercised before updates")
                if event.get("event") == "training_complete" and args.cached_cpu_adam:
                    calls = installations["cached_cpu_adam"].report()
                    report["cached_cpu_adam_training_calls"] = calls
                    updates = int(event.get("updates", 0))
                    expected = updates * 132
                    exercised = (updates > 0 and calls.get("gpu_parameter_updates") == expected
                                 and calls.get("parameter_updates") == expected
                                 and calls.get("fallback_parameter_updates") == 0)
                    restored = (not event.get("source_optimizer_present")
                                or event.get("optimizer_restored") is True)
                    report["cached_cpu_adam_training_exercised"] = exercised
                    report["source_optimizer_restored"] = bool(event.get("optimizer_restored"))
                    if not exercised or not restored:
                        full_report["status"] = "round3_optimizer_not_exercised_or_restored"
                        emit(path, full_report, {"event": "cached_cpu_adam_training_rejected", "pass": False,
                                                "expected_parameter_updates": expected,
                                                "optimizer_restored": restored})
                        raise RuntimeError("cached CPU Adam did not cover all 132 parameters or restore source moments")
                return emit(path, full_report, event)

            # These wrappers never run while native captures are collected.
            from cat_yoko.trainer import Trainer
            log = Trainer._log

            def logged(owner, row):
                return log(owner, dict(row, actual_optimizer_device="cpu",
                                       split_attention=args.split_attention,
                                       cached_cpu_adam=args.cached_cpu_adam,
                                       deterministic_training=args.deterministic_training,
                                       deterministic_algorithms=model_bench.torch.are_deterministic_algorithms_enabled()))

            try:
                with patch.object(model_bench, "install_moe_layout", install_extra), \
                        patch.object(model_bench, "source_code_version", source_version), \
                        patch.object(model_bench, "emit", emitted), \
                        patch.object(Trainer, "_log", logged):
                    yield legacy_report
            finally:
                # Capture caches/counters before context exit clears private state.
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
    names = ["operators/rocm/round3_candidate_bench.py", "operators/rocm/candidate_bench.py",
             "operators/rocm/model_bench.py", "operators/rocm/packed_attention.py",
             "operators/rocm/shared_storage_moe.py", "cat_yoko/optim.py", "cat_yoko/trainer.py",
             "cat_yoko/checkpoint.py"]
    if args.split_attention:
        names.append("operators/rocm/split_attention.py")
    if args.cached_cpu_adam:
        names.append("operators/rocm/cpu_adam_cached.py")
    if args.sync_update_timing:
        names.append("operators/rocm/update_timing.py")
    report = {
        "split_attention": args.split_attention,
        "cached_cpu_adam": args.cached_cpu_adam,
        "sync_update_timing": args.sync_update_timing,
        "deterministic_training": args.deterministic_training,
        "original_deterministic_algorithms": model_bench.torch.are_deterministic_algorithms_enabled(),
        "patches_installed_after_native_reference": False,
        "full_model_parity_passed": False,
        "cached_cpu_adam_training_exercised": False,
        "file_sha256": {name: digest(root / name) for name in names},
        "optimizer_device": "cpu", "checkpoint_format_changed": False,
        "persistent_fp32_master": False,
        "notes": ["Native/reference-repeat captures precede all new context installation.",
                  "The existing loss, sampled-output and all-132-gradient gates remain in force.",
                  "Gradient parity alone does not compare Adam arithmetic; independent optimizer checks are required.",
                  "Parity-only never verifies or claims cached optimizer updates.",
                  "No context changes another process or the source checkpoint."],
    }
    try:
        # Keep the opt-in policy around the complete inherited run. Its inner
        # parity context then restores this policy before Trainer's updates.
        # False leaves the caller's production policy intact; exit restores it.
        with model_bench.parity_determinism(args.deterministic_training), round3_context(args, report):
            report["deterministic_algorithms"] = model_bench.torch.are_deterministic_algorithms_enabled()
            code = candidate_bench.main(legacy_argv(argv, args))
        report["status"] = "completed" if code == 0 else "failed"
        return code
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        report["restored_deterministic_algorithms"] = model_bench.torch.are_deterministic_algorithms_enabled()
        # Keep the parent's source/output isolation guard: no preflight output.
        if (args.out / "parity.json").is_file():
            path = args.out / "round3_operators.json"
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
