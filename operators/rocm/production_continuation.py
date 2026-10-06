#!/usr/bin/env python3
"""Resume the accepted B0 backend using its previous packed FP32 reference.

Dense-native failures remain failures. This entry changes only the two
reference captures, before shared storage, split attention and cached Adam
are installed. All inherited full-model numeric and recovery gates remain.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REFERENCE_BACKEND = 'previous_packed_production'
WRAPPER_SOURCE = 'operators/rocm/production_continuation.py'
PACKED_SOURCE = 'operators/rocm/packed_attention.py'
QUALITY_SOURCE = 'operators/rocm/continuation_quality.py'
MATH_OPERATOR_SOURCES = {
    'operators/rocm/model_bench.py', 'operators/rocm/candidate_bench.py',
    'operators/rocm/round3_candidate_bench.py', PACKED_SOURCE,
    'operators/rocm/split_attention.py', 'operators/rocm/cpu_adam_cached.py',
    'operators/rocm/shared_storage_moe.py',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_accepted_run(path, source_checkpoint, source_dir):
    """Revalidate a completed production run and bind its state and mathematics.

    This is deliberately stdlib-only. The supervisor separately loads the new
    fixed checkpoint on CPU to audit every weight, Adam moment and RNG/cursor.
    Historical receipts are read in place and are never rewritten.
    """
    from operators.rocm import run_operator_switch as checks, run_round3_switch as round3

    def need(condition, message):
        if not condition:
            raise ValueError(message)

    def read(filename):
        value = json.loads(filename.read_text())
        checks.require_finite(value)
        return value

    path, source_checkpoint, source_dir = (Path(value).resolve() for value in
                                          (path, source_checkpoint, source_dir))
    status_path = path / 'status.json' if path.is_dir() else path
    need(status_path.name == 'status.json', 'accepted-run must identify its original status.json or run directory')
    status = read(status_path)
    need(status.get('status') == 'complete' and status.get('child_pid') is None,
         'accepted production run did not finish successfully')
    need(status.get('reference_backend') == REFERENCE_BACKEND
         and status.get('phase') == 'B0'
         and all(status.get(key) is False for key in ('cursor_reset', 'optimizer_reset', 'rng_reset')),
         'accepted run used a different backend or reset recovery state')
    prior_source_dir = Path(status['source_dir']).resolve()
    prior_source = Path(status['source_checkpoint']).resolve()
    need(prior_source_dir.is_dir() and prior_source.is_file(), 'accepted run lost its frozen source or fixed checkpoint')
    need(status.get('source_sha256') == digest(prior_source), 'accepted run fixed source SHA differs')
    previous, final = status.get('source_metadata', {}), status.get('final_checkpoint_verified', {})
    for metadata in (previous, final):
        need(metadata.get('checkpoint_verified') is True
             and metadata.get('optimizer_states') == 132 and metadata.get('optimizer_moments') == 264,
             'accepted run lacks a complete 132-weight/264-moment CPU audit')
        names, shapes = metadata.get('trainable_names', []), metadata.get('trainable_shapes', {})
        need(len(names) == len(set(names)) == 132 and set(shapes) == set(names),
             'accepted run lacks the exact 132 parameter names and shapes')
        need(all(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape)
                 for shape in shapes.values()), 'accepted parameter shapes are invalid')
        need(all(type(metadata.get(key)) is int and metadata[key] >= 0 for key in
                 ('source_step', 'source_stream_i', 'source_adam_step', 'source_data_rows'))
             and metadata['source_data_rows'] > 0
             and metadata['source_stream_i'] == metadata['source_adam_step'],
             'accepted step, cursor and Adam counters disagree')
    need(Path(previous['source_checkpoint']).resolve() == prior_source,
         'accepted initial CPU audit belongs to a different checkpoint')
    need(set(previous['trainable_names']) == set(final['trainable_names'])
         and previous['trainable_shapes'] == final['trainable_shapes']
         and previous['source_data_rows'] == final['source_data_rows'],
         'accepted final CPU audit changed parameter identities or corpus size')
    updates = final['source_step'] - previous['source_step']
    need(0 < updates <= previous['source_data_rows']
         and all(final[key] == previous[key] + updates for key in ('source_stream_i', 'source_adam_step'))
         and all(final.get(key) == previous.get(key) + updates * 4096
                 for key in ('source_tokens_in_phase', 'source_tokens_seen')),
         'accepted final state does not contain every claimed update')
    plan = status.get('plan', {})
    need(plan.get('updates') == updates and plan.get('target_step') == final['source_step']
         and plan.get('target_cursor') == final['source_stream_i']
         and plan.get('target_adam_step') == final['source_adam_step']
         and plan.get('target_tokens_in_phase') == final['source_tokens_in_phase']
         and status.get('target_step') == final['source_step']
         and status.get('target_cursor') == final['source_stream_i'],
         'accepted run is missing its exact completed update plan')
    directory = status_path.parent / 'continuation'
    paths = {'status_path': status_path, 'parity_path': directory / 'parity.json',
             'operators_path': directory / 'operators.json',
             'round3_operators_path': directory / 'round3_operators.json',
             'metrics_path': directory / 'train/metrics.jsonl'}
    receipt_hashes = {str(filename): digest(filename) for filename in paths.values()}
    parity = read(paths['parity_path'])
    prior_args = SimpleNamespace(source_dir=prior_source_dir, resume=prior_source,
                                 data=Path(parity['data']).resolve(), eval_data=Path(parity['eval_data']).resolve())
    # These are the real full-model and optimizer installation validators. They
    # verify historical wrapper hashes against the historical frozen source.
    validated = checks.validate_run(directory, source=prior_source, source_step=previous['source_step'],
                                    steps=updates if updates > 5 else None, max_updates=updates,
                                    variant='attention', source_dir=prior_source_dir,
                                    expected_names=previous['trainable_names'], args=prior_args,
                                    eval_batches=32, reference_backend=REFERENCE_BACKEND)
    need(validated.get('updates') == updates, 'accepted full-model report has the wrong update count')
    round3.validate_round3_report(prior_args, directory, 'combined', updates, timing=False,
                                 reference_backend=REFERENCE_BACKEND)
    policy = round3.validate_training_policy(directory, source_step=previous['source_step'], updates=updates,
                                             deterministic_training=False)
    latest = Path(validated['checkpoint']).resolve()
    need(Path(final['source_checkpoint']).resolve() == latest,
         'accepted final CPU receipt describes a different saved checkpoint')
    numbered = latest.parent / ('trainable_step_' + str(final['source_step']) + '.pt')
    need(numbered.is_file() and numbered.samefile(latest), 'accepted final checkpoint has no atomic numbered publication')
    checkpoint_sha = digest(numbered)
    need(checkpoint_sha == digest(source_checkpoint), 'new fixed checkpoint differs from accepted final state SHA')
    prior_core = {str(filename.relative_to(prior_source_dir)) for filename in (prior_source_dir / 'cat_yoko').glob('*.py')}
    current_core = {str(filename.relative_to(source_dir)) for filename in (source_dir / 'cat_yoko').glob('*.py')}
    need(prior_core == current_core and {'cat_yoko/model.py', 'cat_yoko/trainer.py', 'cat_yoko/optim.py'} <= prior_core,
         'accepted and current core math inventories differ')
    math_hashes = {name: digest(prior_source_dir / name) for name in sorted(prior_core | MATH_OPERATOR_SOURCES)}
    need(all((source_dir / name).is_file() and digest(source_dir / name) == sha
             for name, sha in math_hashes.items()), 'new source changes previously accepted production mathematics')
    need(all(digest(Path(name)) == sha for name, sha in receipt_hashes.items()),
         'accepted receipts changed during verification')
    paths.update(checkpoint_path=numbered, final_latest_path=latest)
    return {**{name: str(filename) for name, filename in paths.items()},
            'file_sha256': {**receipt_hashes, str(numbered): checkpoint_sha},
            'checkpoint_sha256': checkpoint_sha, 'prior_source_checkpoint': str(prior_source),
            'prior_source_sha256': status['source_sha256'], 'prior_source_dir': str(prior_source_dir),
            'source_previous_status': 'complete', 'reference_backend': REFERENCE_BACKEND,
            'source_update_params': {'updates': updates, 'source_step': previous['source_step'],
                                     'final_step': final['source_step'], 'source_cursor': previous['source_stream_i'],
                                     'final_cursor': final['source_stream_i'], 'final_adam_step': final['source_adam_step'],
                                     'data_rows': final['source_data_rows'], 'final_tokens_in_phase': final['source_tokens_in_phase']},
            'math_file_sha256': math_hashes, 'previous_validation': validated, 'training_policy': policy,
            'checkpoint_tensor_audit': 'Prior complete CPU receipt; supervisor independently rechecks the new fixed checkpoint.'}


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
    accepted_event_emitted = False

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
        nonlocal accepted_event_emitted
        if event.get('event') == 'moe_layout_installed':
            metadata['patches_installed_after_packed_production_reference'] = True
        report.update(metadata)
        event = dict(event, reference_backend=REFERENCE_BACKEND)
        if event.get('event') == 'reference_repeat':
            event['moe_layout'] = 'previous-packed-production-reference-repeat'
        label_reference_metadata(report, metadata)
        label_reference_metadata(event, metadata)
        if metadata.get('accepted_run') is not None and not accepted_event_emitted:
            accepted_event_emitted = True
            emit(path, report, {'event': 'accepted_run_verified', 'reference_backend': REFERENCE_BACKEND,
                                'accepted_run': metadata['accepted_run']})
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
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument('--native-failure-receipt', type=Path,
                            help='retain the earlier dense-native failure by recording its path and SHA')
    provenance.add_argument('--accepted-run', type=Path,
                            help='revalidate a completed production run whose final state is this fixed checkpoint')
    parser.add_argument('--extra-heldout-start-row', type=int,
                        help='also evaluate this disjoint row range at initial/final evaluation')
    parser.add_argument('--extra-heldout-batches', type=int, default=32)
    args = parser.parse_args(argv)
    sequences = model_bench.validate_args(parser, args)
    if (sequences != [4096] or not args.split_attention or not args.cached_cpu_adam
            or args.deterministic_training or args.sync_update_timing):
        parser.error('production continuation requires 4096 parity, split/cached Adam and the ordinary training policy')
    if args.extra_heldout_start_row is not None and (
            args.extra_heldout_start_row < args.eval_batches or args.extra_heldout_batches <= 0):
        parser.error('extra held-out rows must be disjoint from the primary prefix and have positive batches')
    native_argv = []
    values = iter(argv)
    for value in values:
        if value in ('--native-failure-receipt', '--accepted-run', '--extra-heldout-start-row', '--extra-heldout-batches'):
            next(values)
        elif not any(value.startswith(name + '=') for name in
                     ('--native-failure-receipt', '--accepted-run', '--extra-heldout-start-row', '--extra-heldout-batches')):
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
    if args.accepted_run is not None:
        metadata['accepted_run'] = validate_accepted_run(args.accepted_run, args.resume, root)
        metadata['source_provenance_kind'] = 'accepted_completed_production'
    if args.extra_heldout_start_row is not None:
        metadata['quality_file_sha256'] = {QUALITY_SOURCE: digest(root / QUALITY_SOURCE)}
        metadata['quality_parameters'] = {'eval_row_start': args.extra_heldout_start_row,
                                          'eval_batches': args.extra_heldout_batches}
    try:
        with ExitStack() as contexts:
            if args.extra_heldout_start_row is not None:
                from operators.rocm.continuation_quality import quality_context
                contexts.enter_context(quality_context(model_bench, args.out,
                    eval_row_start=args.extra_heldout_start_row, eval_batches=args.extra_heldout_batches))
            contexts.enter_context(reference_context(model_bench, metadata, tile_size=args.packed_attention_tile,
                                                     packed_context=packed_attention_context))
            return round3_candidate_bench.main(native_argv)
    finally:
        annotate_operator_reports(args.out, metadata)


if __name__ == '__main__':
    raise SystemExit(main())
