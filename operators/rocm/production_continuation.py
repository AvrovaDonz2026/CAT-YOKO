#!/usr/bin/env python3
"""Resume the accepted B0 backend using its previous packed FP32 reference.

Dense-native failures remain failures. This entry changes only the two
reference captures, before shared storage, split attention and cached Adam
are installed. All inherited full-model numeric and recovery gates remain.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REFERENCE_BACKEND = 'previous_packed_production'
WRAPPER_SOURCE = 'operators/rocm/production_continuation.py'
PACKED_SOURCE = 'operators/rocm/packed_attention.py'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_reference_evidence(report, *, source_dir, source=None):
    """Require an explicit, source-bound oracle, never infer one from PASS."""
    if source_dir is None:
        raise ValueError('packed-production reference requires its frozen source directory')
    if (report.get('reference_backend') != REFERENCE_BACKEND
            or report.get('patches_installed_after_native_reference') is not False
            or report.get('patches_installed_after_packed_production_reference') is not True
            or report.get('production_reference_context_restored') is not True):
        raise ValueError('missing truthful packed-production reference metadata')
    hashes = report.get('reference_file_sha256', {})
    if (set(hashes) != {WRAPPER_SOURCE, PACKED_SOURCE}
            or any(digest(Path(source_dir) / name) != sha for name, sha in hashes.items())):
        raise ValueError('packed-production reference source hashes differ')
    captures = report.get('production_reference_captures', [])
    if len(captures) != 2 or any(
        row.get('status') != 'completed' or row.get('seq_len') != 4096
        or row.get('offload_blocks') is not True or row.get('gradient_tensors') != 132
        or row.get('packed_attention_calls', {}).get('optimized_calls', 0) <= 0
        for row in captures
    ):
        raise ValueError('both full-model packed-production references must be exercised')
    if source is not None and report.get('reference_checkpoint_sha256') != digest(source):
        raise ValueError('packed-production reference used a different checkpoint')


def label_reference_metadata(value, metadata):
    """Relabel inherited nested receipts as well as their outer report."""
    if isinstance(value, dict):
        if 'patches_installed_after_native_reference' in value:
            value['patches_installed_after_native_reference'] = False
            value['patches_installed_after_packed_production_reference'] = metadata['patches_installed_after_packed_production_reference']
            value['reference_backend'] = REFERENCE_BACKEND
        for name, child in value.items():
            if name in ('experimental_operators', 'round3_operators') and isinstance(child, dict):
                child.update(reference_backend=REFERENCE_BACKEND, patches_installed_after_native_reference=False,
                             patches_installed_after_packed_production_reference=metadata['patches_installed_after_packed_production_reference'])
            label_reference_metadata(child, metadata)
    elif isinstance(value, list):
        for child in value:
            label_reference_metadata(child, metadata)


@contextmanager
def reference_context(model_bench, metadata, *, tile_size, packed_context):
    """Scope packed attention only around the inherited offload captures."""
    capture, emit = model_bench.capture, model_bench.emit

    def reference_capture(model, batch, *, device, offload):
        if not offload:
            return capture(model, batch, device=device, offload=offload)
        if len(metadata['production_reference_captures']) >= 2:
            raise RuntimeError('packed-production continuation requires exactly two references')
        metadata['production_reference_context_restored'] = False
        with packed_context(tile_size=tile_size) as installation:
            row, snapshot = capture(model, batch, device=device, offload=True)
            calls = installation.report()
        metadata['production_reference_context_restored'] = True
        metadata['production_reference_captures'].append({
            'status': 'completed', 'seq_len': row['seq_len'], 'offload_blocks': True,
            'gradient_tensors': row['gradient_tensors'], 'packed_attention_calls': calls,
        })
        if row['seq_len'] != 4096 or row['gradient_tensors'] != 132 or calls.get('optimized_calls', 0) <= 0:
            raise RuntimeError('packed-production reference did not cover the full exercised model')
        return row, snapshot

    def reference_emit(path, report, event):
        if event.get('event') == 'moe_layout_installed':
            metadata['patches_installed_after_packed_production_reference'] = True
        report.update(metadata)
        event = dict(event, reference_backend=REFERENCE_BACKEND)
        if event.get('event') == 'reference_repeat':
            event['moe_layout'] = 'previous-packed-production-reference-repeat'
        label_reference_metadata(report, metadata)
        label_reference_metadata(event, metadata)
        return emit(path, report, event)

    with patch.object(model_bench, 'capture', reference_capture), \
            patch.object(model_bench, 'emit', reference_emit):
        yield


def annotate_operator_reports(directory, metadata):
    """Correct oracle labels only; never turn a failed gate into PASS."""
    for name in ('operators.json', 'round3_operators.json'):
        path = directory / name
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        report.update(metadata)
        label_reference_metadata(report, metadata)
        report.setdefault('file_sha256', {})[WRAPPER_SOURCE] = metadata['reference_file_sha256'][WRAPPER_SOURCE]
        if 'notes' in report:
            report['notes'] = [note.replace('Native/reference-repeat captures', 'Previous packed-production/reference-repeat captures')
                               for note in report['notes']]
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def main(argv=None):
    # Heavyweight imports remain lazy so auditors and stdlib tests stay CPU-only.
    from operators.rocm import model_bench, round3_candidate_bench
    from operators.rocm.packed_attention import packed_attention_context

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = round3_candidate_bench.build_parser()
    parser.description = __doc__
    parser.add_argument('--native-failure-receipt', type=Path,
                        help='retain the earlier dense-native failure by recording its path and SHA')
    args = parser.parse_args(argv)
    sequences = model_bench.validate_args(parser, args)
    if (sequences != [4096] or not args.split_attention or not args.cached_cpu_adam
            or args.deterministic_training or args.sync_update_timing):
        parser.error('production continuation requires 4096 parity, split/cached Adam and the ordinary training policy')
    native_argv = []
    values = iter(argv)
    for value in values:
        if value == '--native-failure-receipt':
            next(values)
        elif not value.startswith('--native-failure-receipt='):
            native_argv.append(value)
    root = Path(__file__).resolve().parents[2]
    metadata = {
        'reference_backend': REFERENCE_BACKEND,
        'patches_installed_after_native_reference': False,
        'patches_installed_after_packed_production_reference': False,
        'production_reference_context_restored': False,
        'production_reference_captures': [],
        'reference_checkpoint_sha256': digest(args.resume),
        'reference_file_sha256': {name: digest(root / name) for name in (WRAPPER_SOURCE, PACKED_SOURCE)},
        'reference_scope': 'Previous accepted packed FP32 attention with native MoE offload; candidate retains shared storage/split/cached CPU Adam. Dense-native failures are not overridden.',
    }
    if args.native_failure_receipt is not None:
        receipt = args.native_failure_receipt.resolve()
        failed = json.loads(receipt.read_text())
        if failed.get('status') != 'parity_failure':
            parser.error('native-failure-receipt must preserve a failed dense-native parity report')
        metadata['native_failure_receipt'] = {'path': str(receipt), 'sha256': digest(receipt), 'status': 'parity_failure'}
    try:
        with reference_context(model_bench, metadata, tile_size=args.packed_attention_tile,
                               packed_context=packed_attention_context):
            return round3_candidate_bench.main(native_argv)
    finally:
        annotate_operator_reports(args.out, metadata)


if __name__ == '__main__':
    raise SystemExit(main())
