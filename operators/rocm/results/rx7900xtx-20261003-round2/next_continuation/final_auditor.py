"""Audit completion without signalling or replacing the running training child."""
import datetime
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

root = pathlib.Path('/home/donz/cat-yoko-rocm-20261002')
out = root / 'runs/longtrain-45024-20261003T1520'
source_dir = root / 'source-operators-round2-20261003T0903'
source = out / 'source_checkpoint/trainable.pt'
supervisor_pid = 1566072
expected_sha = '655da42cf523ef7d512234247fcb229f70c8ab1c72a495fbb41cd82e778a4285'
sys.path.insert(0, str(source_dir))
from operators.rocm import run_operator_switch as helper

args = SimpleNamespace(
    python='/home/donz/revelation-rocm-venv/bin/python',
    resume=source.resolve(), source_dir=source_dir,
    data=root / 'data/phase-b-real-20261002/train.bin',
    eval_data=root / 'data/phase-b-real-20261002/eval.bin',
)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def need(condition, message):
    if not condition:
        raise RuntimeError(message)

def identity(pid):
    try:
        proc = pathlib.Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {
            'pid': pid, 'uid': proc.stat().st_uid, 'start_time': fields[19],
            'command': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1],
        }
    except FileNotFoundError:
        return None

audit = {
    'status': 'starting', 'auditor_pid': os.getpid(), 'started_at': now(),
    'supervisor_pid': supervisor_pid, 'source_sha256': expected_sha,
    'signals_sent': 0, 'training_processes_started': 0,
    'purpose': 'Independent final validation with explicit held-out corpus; does not control training',
}

def update(**values):
    audit.update(values, updated_at=now())
    helper.write_json(out / 'final_audit_status.json', audit)
    print(json.dumps(values), flush=True)

try:
    initial = read(out / 'status.json')
    need(initial['supervisor_pid'] == supervisor_pid and initial['status'] == 'running', 'unexpected supervisor state')
    need(initial['source_sha256'] == expected_sha and helper.digest_file(source) == expected_sha, 'source checkpoint changed')
    need(initial['remaining_updates'] == 9309 and initial['target_step'] == 54333, 'unexpected continuation bound')
    parent = identity(supervisor_pid)
    worker = identity(initial['child_pid'])
    need(parent is not None and parent['command'] == [args.python, '-u', str(root / 'run-corpus-45024-20261003T1520.py')], 'supervisor identity mismatch')
    command = read(out / 'continuation.command.json')['command']
    need(worker is not None and worker['command'] == command, 'training worker identity mismatch')
    need('--max-hours' not in command and command[command.index('--run-steps') + 1] == '9309', 'unexpected training time or update limit')
    need(command[command.index('--eval-data') + 1] == str(args.eval_data), 'unexpected held-out corpus')
    protected = [source, args.data, args.eval_data, *source_dir.joinpath('cat_yoko').glob('*.py'),
                 *source_dir.joinpath('operators/rocm').glob('*.py'),
                 *root.joinpath('hf/MiniCPM5-2B-Base').rglob('*.safetensors')]
    fingerprints = {str(p): helper.fingerprint(p) for p in protected}
    hashes = {str(p): helper.digest_file(p) for p in protected if p.suffix == '.py'}
    helper.write_json(out / 'final_audit_sources.json', {'fingerprints': fingerprints, 'python_sha256': hashes, 'source_sha256': expected_sha})

    def assert_source():
        need(all(helper.fingerprint(pathlib.Path(p)) == value for p, value in fingerprints.items()), 'protected source/base/corpus fingerprint changed')
        need(all(helper.digest_file(pathlib.Path(p)) == value for p, value in hashes.items()), 'protected Python source changed')
        need(helper.digest_file(source) == expected_sha, 'fixed source checkpoint changed')

    source_metadata = helper.cpu_check(args, source, 9309, out / 'final_audit_source_preflight.log')
    need(source_metadata == initial['source_metadata'], 'source metadata differs from launch preflight')
    assert_source()
    update(status='waiting_for_training_completion', supervisor_identity=parent, worker_identity=worker)
    while identity(supervisor_pid) == parent:
        time.sleep(20)
    update(status='validating_completed_run')
    state = read(out / 'status.json')
    helper.write_json(out / 'original_supervisor_status.json', state)
    if (out / 'summary.json').is_file():
        helper.write_json(out / 'original_supervisor_summary.json', read(out / 'summary.json'))
    need(identity(worker['pid']) != worker, 'supervisor ended while original training child still runs')
    receipt = read(out / 'continuation.receipt.json')
    need(receipt.get('exit_code') == 0, 'training child did not exit successfully')
    allowed_error = "AttributeError: 'types.SimpleNamespace' object has no attribute 'eval_data'"
    need(state.get('status') == 'complete' or (state.get('status') == 'failed' and state.get('error') == allowed_error), 'supervisor reported a substantive failure')
    assert_source()
    result = helper.validate_run(
        out / 'continuation', source=source, source_step=45024, steps=9309,
        variant='attention', source_dir=source_dir,
        expected_names=initial['source_metadata']['trainable_names'], args=args,
        max_updates=9309, eval_batches=32,
    )
    report = read(out / 'continuation/parity.json')
    next_operators = read(out / 'continuation/next_operators.json')
    need(next_operators.get('status') == 'completed'
         and next_operators.get('patches_installed_after_native_reference') is True
         and all(next_operators.get(key) is False for key in ('gpu_fp32_adam', 'bucketed_attention', 'sync_update_timing')),
         'unexpected next-round operators installed')
    expected_next_paths = {'operators/rocm/next_candidate_bench.py', 'operators/rocm/candidate_bench.py'}
    next_hashes = next_operators.get('file_sha256', {})
    need(set(next_hashes) == expected_next_paths, 'next-round entrypoint source ledger incomplete')
    need(all(hashes[str(source_dir / relative)] == digest for relative, digest in next_hashes.items()), 'next-round source differs from frozen launch source')
    finishes = [event for event in report['events'] if event.get('event') == 'training_complete']
    need(result['updates'] == 9309 and result['final_step'] == 54333, 'did not finish the complete remaining corpus')
    need(finishes[-1].get('stop_reason') == 'steps' and finishes[-1].get('max_train_seconds') is None
         and report.get('max_train_seconds') is None, 'unexpected completion reason or time limit')
    metadata = helper.cpu_check(args, pathlib.Path(result['checkpoint']), 9309, out / 'final_checkpoint_check.log')
    need(metadata['source_step'] == 54333 and metadata['source_adam_step'] == 19531 and metadata['source_stream_i'] == 19531, 'final global/Adam/cursor counters mismatch')
    for key in ('source_tokens_in_phase', 'source_tokens_seen'):
        need(metadata[key] == initial['source_metadata'][key] + 9309 * 4096, 'final token delta mismatch')
    result['verified_checkpoint_metadata'] = metadata
    result['independent_final_audit'] = str(out / 'final_audit_status.json')
    assert_source()
    helper.write_json(out / 'continuation.validation.json', result)
    original_status, original_error = state['status'], state.get('error')
    state.pop('error', None)
    state.update(status='complete', child_pid=None, continuation=result,
                 original_supervisor_terminal_status=original_status,
                 original_supervisor_terminal_error=original_error,
                 final_auditor_pid=os.getpid(), updated_at=now())
    helper.write_json(out / 'status.json', state)
    helper.write_json(out / 'summary.json', state)
    update(status='complete', continuation=result, original_supervisor_terminal_status=original_status,
           original_supervisor_terminal_error=original_error)
except BaseException as error:
    update(status='failed', error=type(error).__name__ + ': ' + str(error))
    raise
