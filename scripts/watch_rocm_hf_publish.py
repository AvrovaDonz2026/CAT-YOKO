#!/usr/bin/env python3
"""Wait locally for a verified ROCm window, then publish with local HF login.

The remote host only prepares public checkpoint files. No local Hugging Face
credential, environment variable, or authentication file is sent over SSH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


class RemoteUnavailable(RuntimeError):
    pass


class PreparationFailed(RuntimeError):
    pass


class PublicationFailed(RuntimeError):
    pass


REMOTE_STATUS = r'''
import json, sys
from pathlib import Path
run=Path(sys.argv[1]).resolve(); source=Path(sys.argv[2]).resolve()
p=run/'status.json'; d=json.loads(p.read_text())
m=d.get('source_metadata',{}); f=d.get('final_checkpoint_verified',{}); q=d.get('final_quality',{})
print(json.dumps({'status':d.get('status'),'run_path':str(run),
 'source_checkpoint':d.get('source_checkpoint'),'source_dir':d.get('source_dir'),
 'source_step':m.get('source_step'),'source_cursor':m.get('source_stream_i'),
 'source_adam_step':m.get('source_adam_step'),'source_tokens_in_phase':m.get('source_tokens_in_phase'),
 'target_step':d.get('target_step'),'target_cursor':d.get('target_cursor'),
 'updates':d.get('plan',{}).get('updates'),'child_pid':d.get('child_pid'),
 'final_verified':f.get('checkpoint_verified'),'final_step':f.get('source_step'),
 'final_checkpoint':f.get('source_checkpoint'),'final_cursor':f.get('source_stream_i'),
 'final_adam_step':f.get('source_adam_step'),'final_tokens_in_phase':f.get('source_tokens_in_phase'),
 'optimizer_states':f.get('optimizer_states'),'optimizer_moments':f.get('optimizer_moments'),
 'final_quality_done':d.get('final_quality_done'),
 'quality_path':q.get('path'),'quality_sha256':q.get('sha256'),
 'quality_initial':{k:q.get('initial_eval',{}).get(k) for k in ('step','pass','eval_batches','row_start','row_stop')},
 'quality_final':{k:q.get('final_eval',{}).get(k) for k in ('step','pass','eval_batches','row_start','row_stop')}},allow_nan=False))
'''

REMOTE_PACK = r'''
import hashlib, json, os, sys, tarfile
from pathlib import Path, PurePosixPath
stage=Path(sys.argv[1]).resolve(); step=int(sys.argv[2]); prefix=f'checkpoints/b0-rocm-realtext/step-{step}'
manifest=stage/'prepared.json'; d=json.loads(manifest.read_text())
if d.get('status')!='prepared' or d.get('step')!=step or d.get('checkpoint_prefix')!=prefix:
 raise ValueError('wrong prepared checkpoint')
files=d['files']
archive=stage/'payload.tar.gz'; temporary=stage/'payload.tar.gz.tmp'
with tarfile.open(temporary,'w:gz',dereference=True) as output:
 output.add(manifest,arcname='prepared.json',recursive=False)
 for name, entry in sorted(files.items()):
  relative=PurePosixPath(name)
  if relative.is_absolute() or '..' in relative.parts or '\\' in name or not name.startswith(prefix+'/'):
   raise ValueError('unsafe prepared path')
  path=stage.joinpath(*relative.parts)
  if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(stage):
   raise ValueError('prepared artifact must be a regular file')
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
  if h.hexdigest()!=entry['sha256'] or path.stat().st_size!=entry['bytes']:
   raise ValueError('prepared artifact changed')
  output.add(path,arcname=name,recursive=False)
os.replace(temporary,archive)
h=hashlib.sha256()
with archive.open('rb') as stream:
 for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
print(json.dumps({'status':'packed','step':step,'archive_path':str(archive),
 'archive_sha256':h.hexdigest(),'archive_bytes':archive.stat().st_size}))
'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_status(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def json_receipt(stdout):
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    raise ValueError('missing structured receipt')


def ssh_options(args):
    return ['-o', 'BatchMode=yes', '-o', 'PasswordAuthentication=no',
            '-o', 'KbdInteractiveAuthentication=no', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=3', '-o', 'ControlPath=' + args.control_path,
            '-o', 'UserKnownHostsFile=' + str(args.known_hosts)]


def remote(args, words, *, prepare=False, timeout=60):
    command = ['ssh', '-p', str(args.ssh_port), *ssh_options(args), args.ssh_host,
               shlex.join([str(word) for word in words])]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RemoteUnavailable(type(error).__name__) from None
    if result.returncode:
        error = PreparationFailed if prepare and result.returncode != 255 else RemoteUnavailable
        raise error('remote command did not complete')
    return json_receipt(result.stdout)


def completion_verified(row, args):
    """Check slim receipts before invoking the heavyweight remote CPU audit."""
    run = PurePosixPath(args.remote_run)
    if (row.get('run_path') != str(run) or row.get('source_dir') != args.remote_source
            or row.get('source_checkpoint') != str(run / 'source_checkpoint/trainable.pt')):
        raise ValueError('remote status belongs to another run or frozen source')
    if ((row.get('source_step') is not None and row['source_step'] != args.expected_source_step)
            or (row.get('target_step') is not None and row['target_step'] != args.expected_target_step)):
        raise ValueError('remote training has a different source or target step')
    if row.get('status') != 'complete':
        return False
    updates = args.expected_target_step - args.expected_source_step
    cursor = row.get('source_cursor')
    if (row.get('source_step') != args.expected_source_step
            or row.get('target_step') != args.expected_target_step or row.get('updates') != updates
            or row.get('child_pid') is not None or type(cursor) is not int or cursor != 43062
            or row.get('source_adam_step') != cursor
            or row.get('final_verified') is not True
            or row.get('final_step') != args.expected_target_step
            or row.get('target_cursor') != cursor + updates
            or row.get('final_cursor') != cursor + updates
            or row.get('final_adam_step') != cursor + updates
            or row.get('final_checkpoint') != str(run / 'continuation/train/trainable.pt')
            or row.get('optimizer_states') != 132 or row.get('optimizer_moments') != 264
            or not isinstance(row.get('source_tokens_in_phase'), (int, float))
            or row.get('final_tokens_in_phase') != row['source_tokens_in_phase'] + 4096 * updates
            or row.get('final_quality_done') is not True
            or row.get('quality_path') != str(run / 'continuation/quality.json')
            or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('quality_sha256', '')))):
        raise ValueError('completed run has incomplete state or quality receipts')
    for key, step in (('quality_initial', args.expected_source_step), ('quality_final', args.expected_target_step)):
        quality = row.get(key, {})
        if (quality.get('pass') is not True or quality.get('step') != step
                or quality.get('eval_batches') != 32 or quality.get('row_start') != 32
                or quality.get('row_stop') != 64):
            raise ValueError('paired quality evaluation did not cover the complete window')
    return True


def allowed_payload_path(name, step):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name:
        return False
    if name == 'prepared.json':
        return True
    prefix = f'checkpoints/b0-rocm-realtext/step-{step}/'
    if not name.startswith(prefix):
        return False
    relative = name[len(prefix):]
    fixed = {'trainable.pt', 'weights-only.pt', 'release.json', 'README.md',
             'evidence/status.json', 'evidence/cpu-check.json', 'evidence/cpu-export.json',
             'evidence/continuation/parity.json', 'evidence/continuation/operators.json',
             'evidence/continuation/round3_operators.json', 'evidence/continuation/quality.json',
             'evidence/continuation/train/metrics.jsonl'}
    if relative in fixed:
        return True
    if relative.startswith(('source/cat_yoko/', 'source/operators/rocm/')) and relative.endswith('.py'):
        return all(re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', part) for part in PurePosixPath(relative).parts)
    return False


def extract_payload(archive, local_stage, *, step, repo_id):
    """Extract only declared regular artifacts; never use tar's extract API."""
    local_stage = Path(local_stage)
    local_stage.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.incoming-', dir=local_stage) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive, 'r:gz') as package:
            members = package.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or 'prepared.json' not in names:
                raise ValueError('duplicate or missing archive manifest')
            if any(not member.isfile() or not allowed_payload_path(member.name, step) for member in members):
                raise ValueError('archive contains undeclared paths, links or nonregular files')
            manifest_member = package.getmember('prepared.json')
            if manifest_member.size > 4 << 20:
                raise ValueError('archive manifest is too large')
            with package.extractfile(manifest_member) as stream:
                manifest = json.load(stream)
            prefix = f'checkpoints/b0-rocm-realtext/step-{step}'
            files = manifest.get('files', {})
            if (manifest.get('status') != 'prepared' or manifest.get('step') != step
                    or manifest.get('repo_id') != repo_id or manifest.get('checkpoint_prefix') != prefix
                    or not isinstance(files, dict) or set(names) != {'prepared.json', *files}
                    or not {prefix + '/trainable.pt', prefix + '/weights-only.pt', prefix + '/release.json'} <= set(files)):
                raise ValueError('archive does not match the complete curated manifest')
            if any(not allowed_payload_path(name, step) or name == 'prepared.json' for name in files):
                raise ValueError('manifest artifact is outside the curated allowlist')
            for member in members:
                output = temporary.joinpath(*PurePosixPath(member.name).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(member) as source, output.open('xb') as destination:
                    shutil.copyfileobj(source, destination, length=8 << 20)
                if member.name != 'prepared.json':
                    expected = files[member.name]
                    if (type(expected.get('bytes')) is not int or expected['bytes'] < 0
                            or member.size != expected['bytes'] or digest(output) != expected.get('sha256')):
                        raise ValueError('downloaded artifact failed its SHA or byte count')
        ready = local_stage / 'prepared'
        if ready.exists():
            if ready.is_symlink():
                raise ValueError('local prepared stage must not be a symbolic link')
            previous = ready / 'prepared.json'
            if not previous.is_file() or json.loads(previous.read_text()) != manifest:
                raise ValueError('local prepared stage already contains another payload')
            for name, expected in files.items():
                path = ready.joinpath(*PurePosixPath(name).parts)
                if not path.is_file() or path.is_symlink() or digest(path) != expected['sha256']:
                    raise ValueError('existing local prepared stage changed')
        else:
            temporary.rename(ready)
        return ready


def local_publication(args, prepared):
    """Retry only explicit transient failures, preserving the same staged bytes."""
    command = [args.python, str(args.publisher), '--publish-prepared', '--stage-dir', str(prepared),
               '--repo-id', args.repo_id]
    for attempt in range(3):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=7200, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            result = None
            error_type = 'TimeoutError'
        except OSError as error:
            raise PublicationFailed(type(error).__name__) from None
        if result is not None and result.returncode == 0:
            receipt = json_receipt(result.stdout)
            if (receipt.get('status') != 'verified' or receipt.get('step') != args.expected_target_step
                    or receipt.get('repo_id') != args.repo_id
                    or not re.fullmatch(r'[0-9a-f]{40,64}', str(receipt.get('commit', '')))):
                raise PublicationFailed('local publisher did not return a verified commit')
            return {'status': 'published_verified', 'repo_id': args.repo_id, 'step': args.expected_target_step,
                    'commit': receipt['commit'], 'prepared_stage': str(prepared), 'publication_attempts': attempt + 1}
        if result is not None:
            try:
                failure = json_receipt(result.stdout)
                error_type = failure.get('error_type') if failure.get('status') == 'failed' else None
            except ValueError:
                # Compatibility with the publisher's previous type-only stderr.
                match = re.fullmatch(r'Completed-run publication failed: ([A-Za-z_][A-Za-z0-9_]*)\s*', result.stderr)
                error_type = match.group(1) if match else None
        if error_type not in ('RuntimeError', 'TimeoutError') or attempt == 2:
            raise PublicationFailed('local publisher did not complete')
        time.sleep(60)
    raise PublicationFailed('publication retry bound reached')


def prepared_resume(args):
    proof_path = args.local_stage / 'watch-prepared.json'
    if not proof_path.is_file():
        return None
    proof = json.loads(proof_path.read_text())
    prepared = args.local_stage / 'prepared'
    manifest_path = prepared / 'prepared.json'
    if (proof.get('remote_run') != args.remote_run or proof.get('remote_source') != args.remote_source
            or proof.get('source_step') != args.expected_source_step or proof.get('step') != args.expected_target_step
            or proof.get('repo_id') != args.repo_id or not manifest_path.is_file()
            or proof.get('prepared_manifest_sha256') != digest(manifest_path)):
        raise ValueError('local publication resume belongs to another verified payload')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('step') != args.expected_target_step or manifest.get('repo_id') != args.repo_id:
        raise ValueError('local publication resume has a different checkpoint')
    return prepared


def discard_verified_transport(args, receipt):
    archive = args.local_stage / 'incoming-payload.tar.gz'
    if not archive.exists():
        return
    proof = json.loads((args.local_stage / 'watch-prepared.json').read_text())
    if archive.is_symlink() or digest(archive) != proof.get('archive_sha256'):
        receipt['transport_retained'] = True
        return
    try:
        archive.unlink()
    except OSError as error:
        receipt['transport_cleanup_error_type'] = type(error).__name__


def publish_completed(args):
    remote(args, [args.remote_python, args.remote_helper, '--prepare-only', '--run-dir', args.remote_run,
                  '--source-dir', args.remote_source, '--python', args.remote_python,
                  '--stage-dir', args.remote_stage, '--template-release', args.remote_template,
                  '--base', args.remote_base, '--data', args.remote_data, '--eval-data', args.remote_eval_data,
                  '--code-commit', args.code_commit, '--repo-id', args.repo_id], prepare=True, timeout=1800)
    packed = remote(args, [args.remote_python, '-c', REMOTE_PACK, args.remote_stage,
                           args.expected_target_step], prepare=True, timeout=1800)
    expected_archive = str(PurePosixPath(args.remote_stage) / 'payload.tar.gz')
    if (packed.get('status') != 'packed' or packed.get('step') != args.expected_target_step
            or packed.get('archive_path') != expected_archive
            or not re.fullmatch(r'[0-9a-f]{64}', str(packed.get('archive_sha256', '')))
            or type(packed.get('archive_bytes')) is not int or packed['archive_bytes'] <= 0):
        raise ValueError('invalid remote payload receipt')
    args.local_stage.mkdir(parents=True, exist_ok=True)
    archive = args.local_stage / 'incoming-payload.tar.gz'
    command = ['scp', '-P', str(args.ssh_port), *ssh_options(args),
               args.ssh_host + ':' + shlex.quote(expected_archive), str(archive)]
    try:
        copied = subprocess.run(command, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RemoteUnavailable(type(error).__name__) from None
    if copied.returncode:
        raise RemoteUnavailable('payload transport did not complete')
    if archive.stat().st_size != packed['archive_bytes'] or digest(archive) != packed['archive_sha256']:
        raise ValueError('transported archive failed its SHA or byte count')
    prepared = extract_payload(archive, args.local_stage, step=args.expected_target_step, repo_id=args.repo_id)
    write_status(args.local_stage / 'watch-prepared.json', {
        'remote_run': args.remote_run, 'remote_source': args.remote_source,
        'source_step': args.expected_source_step, 'step': args.expected_target_step, 'repo_id': args.repo_id,
        'prepared_manifest_sha256': digest(prepared / 'prepared.json'), 'archive_sha256': packed['archive_sha256']})
    receipt = local_publication(args, prepared)
    # This is our verified transport file, outside the curated checkpoint stage.
    # Preserve prepared binaries/receipts for recovery and independent review.
    discard_verified_transport(args, receipt)
    return {**receipt, 'archive_sha256': packed['archive_sha256']}


def supervise(args):
    state = {'status': 'waiting_for_training', 'pid': os.getpid(), 'remote_run': args.remote_run,
             'expected_source_step': args.expected_source_step, 'expected_target_step': args.expected_target_step,
             'credentials_location': 'local_only', 'local_stage': str(args.local_stage)}

    def update(**values):
        state.update(values, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_status(args.status_file, state)
        print(json.dumps({key: state[key] for key in ('status', 'updated_at', 'commit', 'exception_type') if key in state}), flush=True)

    while True:
        try:
            existing = prepared_resume(args)
            if existing is not None:
                update(status='publishing_prepared_resume')
                receipt = local_publication(args, existing)
                discard_verified_transport(args, receipt)
                state.pop('exception_type', None)
                update(**receipt)
                return 0
            row = remote(args, [args.remote_python, '-c', REMOTE_STATUS, args.remote_run, args.remote_source])
            if row.get('status') == 'failed':
                # Check run ownership before accepting a terminal failure.
                completion_verified(row, args)
                update(status='training_failed')
                return 1
            if not completion_verified(row, args):
                update(status='waiting_for_training')
            else:
                update(status='preparing_publication')
                receipt = publish_completed(args)
                state.pop('exception_type', None)
                update(**receipt)
                return 0
        except RemoteUnavailable as error:
            update(status='waiting_for_remote', exception_type=type(error).__name__)
            if args.once:
                return 0
            time.sleep(60)
            continue
        except BaseException as error:
            update(status='publication_failed', exception_type=type(error).__name__)
            return 1
        if args.once:
            return 0
        time.sleep(30)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-host', required=True)
    parser.add_argument('--ssh-port', type=int, default=45535)
    parser.add_argument('--control-path', required=True)
    parser.add_argument('--known-hosts', type=Path, required=True)
    for name in ('remote-python', 'remote-helper', 'remote-run', 'remote-source', 'remote-stage',
                 'remote-template', 'remote-base', 'remote-data', 'remote-eval-data', 'code-commit'):
        parser.add_argument('--' + name, required=True)
    for name in ('local-stage', 'publisher', 'status-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--repo-id', default='AvrovaDonz/CAT-YOKO')
    parser.add_argument('--expected-source-step', type=int, default=77864)
    parser.add_argument('--expected-target-step', type=int, default=81864)
    parser.add_argument('--once', action='store_true', help='perform one polling round without waiting')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+', args.ssh_host)
            or not 1 <= args.ssh_port <= 65535):
        parser.error('invalid SSH host or port')
    for name in ('remote_python', 'remote_helper', 'remote_run', 'remote_source', 'remote_stage',
                 'remote_template', 'remote_base', 'remote_data', 'remote_eval_data'):
        path = PurePosixPath(getattr(args, name))
        if not path.is_absolute() or '..' in path.parts:
            parser.error('remote paths must be absolute without traversal')
        setattr(args, name, str(path))
    if ((args.expected_source_step, args.expected_target_step) != (77864, 81864)
            or not re.fullmatch(r'[0-9a-f]{40}', args.code_commit)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id)):
        parser.error('this watcher requires the approved 77864 to 81864 window and a fixed code commit')
    args.local_stage = args.local_stage.resolve()
    args.publisher = args.publisher.resolve()
    args.status_file = args.status_file.resolve()
    args.known_hosts = args.known_hosts.resolve()
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
