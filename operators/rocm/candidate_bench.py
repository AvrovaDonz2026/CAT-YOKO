#!/usr/bin/env python3
"""Validate opt-in operators against native B0 before independent updates.

The native reference is captured before installing candidates. Patches stay
active for candidate parity and bounded training, and restore on every exit.
This entry never changes a separate training process or its source directory.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from operators.rocm import model_bench


def build_parser() -> argparse.ArgumentParser:
    parser = model_bench.build_parser()
    parser.description = __doc__
    parser.add_argument("--packed-attention", action="store_true",
                        help="FP32 per-document causal attention and window tiles after reference capture")
    parser.add_argument("--packed-attention-tile", type=int, default=256)
    parser.add_argument("--batched-grad-norm", action="store_true",
                        help="batch scalar norm readback, retaining native clipping arithmetic")
    return parser


def native_argv(argv: list[str]) -> list[str]:
    """Remove only this entry's options; retain all native CLI values verbatim."""
    result = []
    remaining = iter(argv)
    for value in remaining:
        if value in {"--packed-attention", "--batched-grad-norm"}:
            continue
        if value == "--packed-attention-tile":
            next(remaining)
            continue
        if value.startswith("--packed-attention-tile="):
            continue
        result.append(value)
    return result


def validate_candidates(parser, args) -> None:
    if args.packed_attention_tile <= 0:
        parser.error("packed-attention-tile must be positive")
    if not (args.packed_attention or args.batched_grad_norm):
        return
    if args.moe_layout != "shared-storage":
        parser.error("operator candidates require --moe-layout shared-storage")
    if not args.deterministic_parity:
        parser.error("operator candidates require --deterministic-parity")
    if args.packed_attention:
        if args.data is None:
            parser.error("packed attention parity requires real --data")
        if 4096 not in [int(value) for value in args.parity_seqs.split(",")]:
            parser.error("packed attention requires 4096-token full-model parity")
        if not args.reference_repeat:
            parser.error("packed attention requires --reference-repeat")
        if args.packed_attention_tile >= 4096:
            parser.error("packed-attention-tile must be smaller than 4096")


class PackedParityStream:
    """Find a real multi-document parity row without advancing training data.

    This wraps only the independent parity stream. Trainer later restores its
    original source cursor; rows scanned here never become optimizer updates.
    """

    def __init__(self, stream, info, max_batches=32):
        self.stream, self.info, self.max_batches = stream, info, max_batches

    def batch(self, micro_batch, device):
        for scanned in range(1, self.max_batches + 1):
            batch = self.stream.batch(micro_batch, device)
            docs = batch.get("doc_ids")
            if docs is not None and bool((docs[..., 1:] != docs[..., :-1]).any().item()):
                self.info["packed_parity_scanned_batches"] = scanned
                self.info["packed_parity_document_boundaries"] = int(
                    (docs[..., 1:] != docs[..., :-1]).sum().item())
                return batch
        raise RuntimeError("no multi-document parity row found; refusing unexercised operator validation")


@contextmanager
def candidate_context(args):
    """Enter patches once, after the native model's reference captures."""
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve()]
    if args.packed_attention:
        paths.append(root / "operators/rocm/packed_attention.py")
    if args.batched_grad_norm:
        paths.append(root / "operators/rocm/grad_norm.py")
    report = {
        "packed_attention": args.packed_attention,
        "packed_attention_tile": args.packed_attention_tile,
        "batched_grad_norm": args.batched_grad_norm,
        "patches_installed_after_native_reference": False,
        "file_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths},
    }
    install = model_bench.install_moe_layout
    version = model_bench.source_code_version
    emit = model_bench.emit
    open_stream = model_bench.open_parity_stream
    installation = None
    installed = False
    with ExitStack() as candidates:
        def install_candidates(model, layout):
            nonlocal installed, installation
            if installed:
                raise RuntimeError("candidate installer must run only once per experiment")
            layout_report = install(model, layout)
            if args.packed_attention:
                from operators.rocm.packed_attention import packed_attention_context

                installation = candidates.enter_context(
                    packed_attention_context(tile_size=args.packed_attention_tile))
            if args.batched_grad_norm:
                from operators.rocm.grad_norm import use_batched_grad_norm

                candidates.enter_context(use_batched_grad_norm())
            installed = True
            report["patches_installed_after_native_reference"] = True
            return {**layout_report, "experimental_operators": dict(report)}

        def candidate_version(layout):
            return {**version(layout), "experimental_operators": dict(report)}

        def candidate_stream(cfg, seq, seed, **kwargs):
            stream, info = open_stream(cfg, seq, seed, **kwargs)
            if args.packed_attention and seq == 4096:
                stream = PackedParityStream(stream, info)
            return stream, info

        def candidate_emit(path, full_report, event):
            if event.get("event") == "parity_complete" and installation is not None:
                stats = installation.report()
                report["packed_attention_parity_calls"] = stats
                if event.get("pass") and stats["optimized_calls"] == 0:
                    full_report["status"] = "candidate_not_exercised"
                    emit(path, full_report, {"event": "candidate_not_exercised", "pass": False,
                                            "reason": "parity batch did not exercise packed attention"})
                    raise RuntimeError("packed attention was not exercised; refusing candidate updates")
            return emit(path, full_report, event)

        try:
            with patch.object(model_bench, "install_moe_layout", install_candidates), \
                    patch.object(model_bench, "source_code_version", candidate_version), \
                    patch.object(model_bench, "emit", candidate_emit), \
                    patch.object(model_bench, "open_parity_stream", candidate_stream):
                yield report
        finally:
            if installation is not None:
                report["packed_attention_calls"] = installation.report()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    model_bench.validate_args(parser, args)
    validate_candidates(parser, args)
    report = None
    try:
        with candidate_context(args) as report:
            try:
                code = model_bench.main(native_argv(argv))
                report["status"] = "completed" if code == 0 else "failed"
            except BaseException as exc:
                report["status"] = "failed"
                report["error"] = f"{type(exc).__name__}: {exc}"
                raise
    finally:
        # Publish after every patch has restored and final call counts are known.
        # The native runner creates this file only after checking source/output
        # isolation; do not create an output on a rejected preflight path.
        if report is not None and (args.out / "parity.json").is_file():
            (args.out / "operators.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
