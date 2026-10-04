#!/usr/bin/env python3
"""Continue one explicitly bounded packed-corpus pass with complete B0 state.

This supervisor does not reset the packed cursor or optimizer. The native
stream reads row ``i % nseq`` while retaining monotonically increasing ``i``.
Only the requested absolute pass boundary changes the first-pass audit policy.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import run_operator_switch as checks
from operators.rocm import run_round3_switch as round3
from operators.rocm.gpu_wait import wait_for_gpu_idle


def pass_plan(metadata, target_cursor):
    rows, cursor = metadata['source_data_rows'], metadata['source_stream_i']
    if isinstance(target_cursor, bool) or not isinstance(target_cursor, int):
        raise ValueError('target cursor must be an integer')
    if target_cursor != (cursor // rows + 1) * rows:
        raise ValueError('target must be the next exact corpus-pass boundary')
    updates = target_cursor - cursor
    if not 0 < updates <= rows:
        raise ValueError('continuation must finish at most one additional pass')
    if metadata['source_adam_step'] != cursor:
        raise ValueError('saved Adam counter and packed cursor differ')
    final_tokens = metadata['source_tokens_in_phase'] + updates * 4096
    if final_tokens >= 8e9:
        raise ValueError('bounded pass must remain inside the existing B0 token budget')
    return {'updates': updates, 'target_step': metadata['source_step'] + updates,
            'target_cursor': target_cursor, 'target_adam_step': cursor + updates,
            'target_tokens_in_phase': final_tokens, 'data_rows': rows,
            'start_completed_passes': cursor // rows, 'start_next_row': cursor % rows,
            'target_completed_passes': target_cursor // rows,
            'repeated_corpus': cursor >= rows,
            'scope': 'One explicitly requested pass over the same packed pilot; repetitions are not new unique data.'}


def protected_sources(args):
    paths = [args.resume, args.data, args.eval_data,
             *args.source_dir.glob('cat_yoko/*.py'),
             *args.source_dir.glob('operators/rocm/*.py'),
             *args.base.rglob('*.safetensors'), *args.base.glob('*.json')]
    if getattr(args, 'native_failure_receipt', None) is not None:
        paths.append(args.native_failure_receipt)
    stamps = {str(p): checks.fingerprint(p) for p in paths}
    hashes = {str(p): checks.digest_file(p) for p in paths if p.suffix == '.py'}
    source_sha = checks.digest_file(args.resume)

    def verify():
        if any(checks.fingerprint(Path(p)) != value for p, value in stamps.items()):
            raise ValueError('protected checkpoint, code, base or corpus changed')
        if any(checks.digest_file(Path(p)) != value for p, value in hashes.items()):
            raise ValueError('protected implementation changed')
        if checks.digest_file(args.resume) != source_sha:
            raise ValueError('fixed recovery checkpoint changed')

    return verify, source_sha


def completed_metrics(path, source_step):
    text = path.read_text() if path.exists() else ''
    # A concurrent writer can leave its last JSON line incomplete.
    lines = text.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith('\n') and line.strip()]
    checks.require_finite(rows)
    updates = [row for row in rows if 'tok_s' in row]
    if [row.get('step') for row in updates] != list(range(source_step + 1, source_step + len(updates) + 1)):
        raise ValueError('completed updates are not consecutive from the fixed source')
    for row in updates:
        if not (row.get('split_attention') is True and row.get('cached_cpu_adam') is True
                and row.get('actual_optimizer_device') == 'cpu'
                and row.get('deterministic_training') is False
                and row.get('deterministic_algorithms') is False):
            raise ValueError('actual training backend or original policy changed')
    return updates


def supervise(args):
    reference_backend = getattr(args, 'reference_backend', 'native')
    if reference_backend not in ('native', 'previous_packed_production'):
        raise ValueError('unsupported full-model reference backend')
    failure_receipt = getattr(args, 'native_failure_receipt', None)
    if reference_backend == 'previous_packed_production':
        if failure_receipt is None or not failure_receipt.is_file():
            raise ValueError('production-reference recovery must retain its failed native receipt')
        args.native_failure_receipt = failure_receipt.resolve()
    for name in ('source_dir', 'resume', 'base', 'data', 'eval_data', 'out', 'lock_file'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.out.is_dir() or args.resume != args.out / 'source_checkpoint/trainable.pt':
        raise ValueError('use a separate run containing its fixed complete source checkpoint')
    if any(args.out == p or args.out.is_relative_to(p) or p.is_relative_to(args.out)
           for p in (args.source_dir, args.base, args.data.parent, args.eval_data.parent)):
        raise ValueError('output must be separate from source, base and corpus')
    if (args.out / 'continuation').exists():
        raise ValueError('use a fresh continuation directory')
    state = {'status': 'preflight', 'supervisor_pid': os.getpid(),
             'source_checkpoint': str(args.resume), 'source_dir': str(args.source_dir),
             'explicit_target_cursor': args.target_cursor, 'controls_existing_processes': False,
             'checkpoint_every_seconds': 300, 'keep_last': 3,
             'selected_operators': 'split_attention_and_cached_cpu_adam',
             'reference_backend': reference_backend,
             'cursor_reset': False, 'optimizer_reset': False, 'rng_reset': False,
             'phase': 'B0', 'new_unique_corpus_downloaded': False}

    def update(**values):
        state.update(values, updated_at=checks.utc_now().isoformat())
        checks.write_json(args.out / 'status.json', state)
        print(json.dumps(values, allow_nan=False), flush=True)

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify, source_sha = protected_sources(args)
        update(source_sha256=source_sha)
        if reference_backend == 'previous_packed_production':
            failure = json.loads(args.native_failure_receipt.read_text())
            if not (failure.get('status') == 'parity_failure' and failure.get('updates') == 0
                    and failure.get('source_step') >= 0
                    and any(row.get('event') == 'parity' and row.get('pass') is False
                            for row in failure.get('events', []))
                    and checks.digest_file(Path(failure['source_checkpoint'])) == source_sha):
                raise ValueError('failed native receipt is not bound to this unchanged source')
            update(native_failure_receipt=str(args.native_failure_receipt),
                   native_failure_receipt_sha256=checks.digest_file(args.native_failure_receipt))
        metadata = checks.cpu_check(args, args.resume, 0, args.out / 'source_preflight.log',
                                    max_cursor=args.target_cursor)
        plan = pass_plan(metadata, args.target_cursor)
        checks.cpu_check(args, args.resume, plan['updates'], args.out / 'source_window_check.log',
                         max_cursor=args.target_cursor)
        update(source_metadata=metadata, plan=plan, target_step=plan['target_step'],
               target_cursor=plan['target_cursor'], status='waiting_for_gpu')
        wait_for_gpu_idle(max_wait_seconds=None, on_event=lambda row: update(gpu_wait=row))
        verify()
        flags = round3.common(args, args.resume, args.out / 'continuation', plan['updates'],
                              production=True) + round3.ROUND3_FLAGS
        if reference_backend == 'previous_packed_production':
            flags += ['--native-failure-receipt', str(args.native_failure_receipt)]
        entry = ('production_continuation.py' if reference_backend == 'previous_packed_production'
                 else 'round3_candidate_bench.py')
        command = [args.python, '-u', str(args.source_dir / 'operators/rocm' / entry), *flags]
        checks.write_json(args.out / 'continuation.command.json', {'command': command})
        with (args.out / 'continuation.log').open('w') as log:
            child = None
            try:
                child = subprocess.Popen(command, cwd=args.source_dir, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONOPTIMIZE='0'))
                update(status='running_continuation', child_pid=child.pid)
                seen = set()
                last_logged = -1
                while child.poll() is None:
                    rows = completed_metrics(args.out / 'continuation/train/metrics.jsonl', metadata['source_step'])
                    if rows and rows[-1]['step'] != last_logged:
                        if len(rows) > plan['updates']:
                            raise ValueError('training exceeded the explicit pass bound')
                        last_logged = rows[-1]['step']
                        update(last_completed_step=last_logged, completed_updates=len(rows),
                               actual_new_operator_flags=True, actual_deterministic_algorithms=False)
                    current = args.out / 'continuation/train/trainable.pt'
                    if current.exists() and checks.fingerprint(current) not in seen:
                        directory = args.out / 'verified_checkpoint'
                        directory.mkdir(exist_ok=True)
                        stable = directory / 'pending.pt'
                        stable.unlink(missing_ok=True)
                        try:
                            os.link(current, stable)
                            stamp = checks.fingerprint(stable)
                            step = round3.numbered_step(current.parent, stable)
                            delta = step - metadata['source_step']
                            if not 0 < delta <= plan['updates']:
                                raise ValueError('saved step lies outside the requested pass')
                            checked = checks.cpu_check(args, stable, delta, args.out / 'latest_checkpoint_check.log',
                                                       max_cursor=args.target_cursor)
                            stable.replace(directory / 'latest_verified.pt')
                            seen.add(stamp)
                            update(first_checkpoint_verified=True, latest_verified_checkpoint={**checked,
                                   'file': str(directory / 'latest_verified.pt')},
                                   latest_verified_at=checks.utc_now().isoformat())
                        finally:
                            stable.unlink(missing_ok=True)
                    time.sleep(2)
                code = child.wait()
            finally:
                if child is not None:
                    round3.stop_owned_child(child)
        update(child_pid=None)
        if code != 0:
            raise ValueError('continuation exited unsuccessfully; fixed and verified recovery points retained')
        result = checks.validate_run(args.out / 'continuation', source=args.resume,
                                     source_step=metadata['source_step'], steps=plan['updates'],
                                     variant='attention', source_dir=args.source_dir,
                                     expected_names=metadata['trainable_names'], args=args, eval_batches=32,
                                     reference_backend=reference_backend)
        round3.validate_round3_report(args, args.out / 'continuation', 'combined', plan['updates'], timing=False,
                                     reference_backend=reference_backend)
        round3.validate_training_policy(args.out / 'continuation', source_step=metadata['source_step'],
                                        updates=plan['updates'], deterministic_training=False)
        final = checks.cpu_check(args, Path(result['checkpoint']), plan['updates'],
                                 args.out / 'final_checkpoint_check.log', max_cursor=args.target_cursor)
        if not (final['source_step'] == plan['target_step']
                and final['source_stream_i'] == final['source_adam_step'] == args.target_cursor
                and final['source_tokens_in_phase'] == plan['target_tokens_in_phase']):
            raise ValueError('final checkpoint does not match the exact pass boundary')
        verify()
        update(status='complete', final_checkpoint_verified=final)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'resume', 'base', 'data', 'eval-data', 'out', 'lock-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--target-cursor', type=int, required=True)
    parser.add_argument('--reference-backend', choices=('native', 'previous_packed_production'), default='native')
    parser.add_argument('--native-failure-receipt', type=Path)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt('supervisor signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        supervise(args)
    except BaseException as error:
        path = args.out / 'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                     updated_at=checks.utc_now().isoformat())
        checks.write_json(path, state)
        raise


if __name__ == '__main__':
    main()
