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
import stat
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
import hashlib, json, sys
from pathlib import Path
run=Path(sys.argv[1]).resolve(); source=Path(sys.argv[2]).resolve()
p=run/'status.json'; d=json.loads(p.read_text())
m=d.get('source_metadata',{}); f=d.get('final_checkpoint_verified',{}); q=d.get('final_quality',{})
h=d.get('data_handoff'); fresh={}; handoff_sha=None
if h is not None and d.get('status')=='complete':
 try:
  hp=Path(d['data_handoff_path'])
  if json.loads(hp.read_text())==h:
   handoff_sha=hashlib.sha256(hp.read_bytes()).hexdigest()
  fresh=json.loads((run/'continuation/fresh_quality.json').read_text())
 except (OSError,ValueError,KeyError,TypeError):
  handoff_sha=None; fresh={}
print(json.dumps({'status':d.get('status'),'run_path':str(run),
 'source_checkpoint':d.get('source_checkpoint'),'source_dir':d.get('source_dir'),
 'source_sha256':d.get('source_sha256'),'data_handoff':h,
 'data_handoff_path':d.get('data_handoff_path'),'data_handoff_sha256':d.get('data_handoff_sha256'),
 'actual_data_handoff_sha256':handoff_sha,'fresh_quality':fresh,
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
            or row.get('child_pid') is not None or type(cursor) is not int or cursor != args.expected_source_cursor
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
    if args.expected_source_sha256 is not None and row.get('source_sha256') != args.expected_source_sha256:
        raise ValueError('completed source checkpoint SHA differs from the requested window')
    if args.remote_handoff is not None:
        handoff = row.get('data_handoff', {})
        if (row.get('data_handoff_path') != args.remote_handoff
                or row.get('data_handoff_sha256') != args.expected_handoff_sha256
                or row.get('actual_data_handoff_sha256') != args.expected_handoff_sha256
                or handoff.get('source_checkpoint_path') != row['source_checkpoint']
                or handoff.get('source_checkpoint_sha256') != args.expected_source_sha256
                or handoff.get('source_step') != args.expected_source_step
                or handoff.get('target_step') != args.expected_target_step
                or handoff.get('absolute_cursor_origin') != cursor or handoff.get('logical_row_origin') != 0
                or handoff.get('updates') != updates or handoff.get('target_cursor') != cursor + updates
                or handoff.get('target_logical_row') != updates or handoff.get('no_wrap') is not True
                or handoff.get('mapping') != 'row=i-absolute_cursor_origin'
                or type(handoff.get('new_nseq')) is not int or handoff['new_nseq'] < updates
                or handoff.get('new_train_path') != args.remote_data
                or handoff.get('source_phase_tokens') != row['source_tokens_in_phase']
                or handoff.get('target_phase_tokens') != row['final_tokens_in_phase']):
            raise ValueError('completed fresh corpus mapping differs from the requested handoff')
        fresh = row.get('fresh_quality', {})
        if (fresh.get('status') != 'completed' or fresh.get('protocol') != 'fresh_validation_corpus_paired_heldout'
                or fresh.get('data_handoff_sha256') != args.expected_handoff_sha256
                or fresh.get('eval_data') != handoff.get('fresh_eval_path')
                or fresh.get('eval_data_sha256') != handoff.get('fresh_eval_sha256')
                or fresh.get('source_checkpoint_sha256') != args.expected_source_sha256
                or fresh.get('seq_len') != 4096 or fresh.get('eval_batches') != 32
                or fresh.get('row_start') != 0 or fresh.get('row_stop') != 32):
            raise ValueError('fresh held-out corpus did not produce a source-bound paired receipt')
        for key, step in (('initial_eval', args.expected_source_step), ('final_eval', args.expected_target_step)):
            observation = fresh.get(key, {})
            if (observation.get('pass') is not True or observation.get('step') != step
                    or observation.get('eval_batches') != 32 or observation.get('row_start') != 0
                    or observation.get('row_stop') != 32):
                raise ValueError('fresh paired evaluation did not cover the complete window')
    elif row.get('data_handoff') is not None:
        raise ValueError('fresh corpus requires an explicitly requested handoff identity')
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
             'evidence/data_handoff.json', 'evidence/corpus-manifest.json', 'evidence/continuation/fresh_quality.json',
             'evidence/continuation/data_handoff.json',
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
            verified_publication(args, prepared, expected_commit=receipt['commit'])
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
    legacy = (args.expected_source_step, args.expected_target_step, args.expected_source_cursor) == (77864, 81864, 43062)
    if not legacy or args.expected_source_sha256 is not None or args.remote_handoff is not None:
        if (proof.get('source_cursor') != args.expected_source_cursor
                or proof.get('original_source_checkpoint_sha256') != args.expected_source_sha256
                or proof.get('data_handoff_sha256') != args.expected_handoff_sha256):
            raise ValueError('local publication resume has a different source or handoff')
    if args.remote_handoff is not None:
        if (manifest.get('data_handoff_sha256') != args.expected_handoff_sha256
                or manifest.get('original_source_checkpoint_sha256') != args.expected_source_sha256):
            raise ValueError('prepared manifest does not bind the requested fresh corpus source')
        prefix = f'checkpoints/b0-rocm-realtext/step-{args.expected_target_step}'
        plan_path = prepared / prefix / 'evidence/data_handoff.json'
        release_path = prepared / prefix / 'release.json'
        regular_owned_file(plan_path, prepared)
        regular_owned_file(release_path, prepared)
        if digest(plan_path) != args.expected_handoff_sha256:
            raise ValueError('prepared handoff evidence SHA differs')
        plan = json.loads(plan_path.read_text())
        release = json.loads(release_path.read_text())
        if (release.get('data_handoff') != plan or release.get('data_handoff_sha256') != args.expected_handoff_sha256
                or plan.get('source_checkpoint_sha256') != args.expected_source_sha256
                or plan.get('source_step') != args.expected_source_step or plan.get('target_step') != args.expected_target_step
                or plan.get('absolute_cursor_origin') != args.expected_source_cursor
                or plan.get('target_cursor') != args.expected_source_cursor + 4000
                or plan.get('new_train_sha256') != manifest.get('train_corpus_sha256')):
            raise ValueError('prepared fresh corpus provenance differs from the requested window')
    elif manifest.get('data_handoff_sha256') is not None:
        raise ValueError('fresh payload cannot resume as a legacy publication')
    return prepared


def regular_owned_file(path, root, *, missing_ok=False):
    """Reject links in the file and its relative parents before reading/deleting."""
    path, root = Path(path), Path(root)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('local publication path escapes its stage') from None
    parent = root
    if root.is_symlink() or not root.is_dir():
        raise ValueError('local publication stage is not an owned directory')
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.is_symlink() or not parent.is_dir():
            raise ValueError('local publication parent is not a regular directory')
    try:
        stamp = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise ValueError('local publication evidence is missing') from None
    if not stat.S_ISREG(stamp.st_mode):
        raise ValueError('local publication file must be regular and not a link')
    return stamp


def artifact_record(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def file_identity(stamp):
    # Reads may change atime; only identity/content timestamps bind cleanup.
    return stamp.st_dev, stamp.st_ino, stamp.st_size, stamp.st_mtime_ns


def valid_artifact_record(value):
    return (isinstance(value, dict) and set(value) == {'sha256', 'bytes'}
            and type(value['bytes']) is int and value['bytes'] > 0
            and re.fullmatch(r'[0-9a-f]{64}', str(value['sha256'])) is not None)


def verified_publication(args, prepared, *, expected_commit=None):
    """Bind actual publisher evidence to the previously verified local transport.

    This does not require the weight binaries to remain after successful cleanup.
    A watcher status string or publisher stdout alone is never sufficient.
    """
    prepared = Path(prepared)
    if prepared != args.local_stage / 'prepared' or prepared_resume(args) != prepared:
        raise ValueError('verified publication belongs to another prepared stage')
    for path in (args.local_stage / 'watch-prepared.json', prepared / 'prepared.json', prepared / 'publish.json'):
        regular_owned_file(path, args.local_stage)
    manifest = json.loads((prepared / 'prepared.json').read_text())
    receipt = json.loads((prepared / 'publish.json').read_text())
    prefix = f'checkpoints/b0-rocm-realtext/step-{args.expected_target_step}'
    if (manifest.get('status') != 'prepared' or manifest.get('repo_id') != args.repo_id
            or manifest.get('step') != args.expected_target_step or manifest.get('checkpoint_prefix') != prefix
            or receipt.get('status') != 'verified' or receipt.get('repo_id') != args.repo_id
            or receipt.get('step') != args.expected_target_step or receipt.get('checkpoint_prefix') != prefix
            or not re.fullmatch(r'[0-9a-f]{40}', str(receipt.get('commit', '')))
            or (expected_commit is not None and receipt['commit'] != expected_commit)
            or not isinstance(manifest.get('files'), dict) or receipt.get('allowlist') != manifest['files']):
        raise ValueError('actual publisher receipt does not verify this immutable payload')
    records = {}
    for filename in ('trainable.pt', 'weights-only.pt'):
        name = prefix + '/' + filename
        expected = manifest['files'].get(name)
        if (not valid_artifact_record(expected)
                or receipt.get('checkpoint_lfs_verified', {}).get(filename) != expected):
            raise ValueError('published checkpoint LFS identities differ from the prepared manifest')
        records[name] = expected
    small = receipt.get('download_verified', {})
    for name in ('README.md', prefix + '/README.md', prefix + '/release.json'):
        path = prepared / name
        regular_owned_file(path, prepared)
        expected = small.get(name)
        if (not valid_artifact_record(expected) or artifact_record(path) != expected
                or (name != 'README.md' and manifest['files'].get(name) != expected)):
            raise ValueError('commit-pinned publication card or manifest evidence changed')
    release = json.loads((prepared / prefix / 'release.json').read_text())
    if release.get('step') != args.expected_target_step or release.get('repo_id') != args.repo_id:
        raise ValueError('publication manifest identifies another checkpoint')
    for name, expected in records.items():
        released = release.get('files', {}).get(PurePosixPath(name).name, {})
        if {key: released.get(key) for key in ('sha256', 'bytes')} != expected:
            raise ValueError('release artifact identities differ from the publisher LFS proof')
    return receipt


def cleanup_verified_payload(args, prepared):
    """Remove only this verified pair and transport; preserve evidence and login."""
    prepared = Path(prepared)
    receipt = verified_publication(args, prepared)
    manifest = json.loads((prepared / 'prepared.json').read_text())
    prefix = manifest['checkpoint_prefix']
    report = {'status': 'complete', 'repo_id': args.repo_id, 'step': args.expected_target_step,
              'commit': receipt['commit'], 'publish_receipt_sha256': digest(prepared / 'publish.json'),
              'prepared_manifest_sha256': digest(prepared / 'prepared.json'),
              'deleted_weights': [], 'missing_weights': [], 'retained_files': [],
              'scope': 'Only this verified trainable.pt, weights-only.pt and SHA-bound local transport; other files retained.'}
    cleanup_path = args.local_stage / 'local-cleanup.json'
    regular_owned_file(cleanup_path, args.local_stage, missing_ok=True)
    regular_owned_file(cleanup_path.with_name(cleanup_path.name + '.tmp'), args.local_stage, missing_ok=True)
    candidates = []
    # Validate the complete pair before deleting either member.
    for filename in ('trainable.pt', 'weights-only.pt'):
        name = prefix + '/' + filename
        path = prepared / name
        try:
            stamp = regular_owned_file(path, prepared, missing_ok=True)
            if stamp is None:
                report['missing_weights'].append(name)
            elif artifact_record(path) != manifest['files'][name]:
                raise ValueError('local checkpoint bytes changed after publication')
            else:
                candidates.append((name, path, stamp))
        except (ValueError, OSError) as error:
            report['retained_files'].append({'path': name, 'error_type': type(error).__name__})
    if report['retained_files']:
        report['status'] = 'retained'
        report['retained_files'].extend({'path': name, 'reason': 'paired_checkpoint_validation_failed'}
                                        for name, _, _ in candidates)
    else:
        # Also reject a replaced file between hashing the pair and deletion.
        changed = [name for name, path, stamp in candidates if file_identity(path.lstat()) != file_identity(stamp)]
        if changed:
            report['status'] = 'retained'
            report['retained_files'].extend({'path': name, 'reason': 'paired_checkpoint_changed'}
                                            for name, _, _ in candidates)
        else:
            for name, path, stamp in candidates:
                try:
                    path.unlink()
                    report['deleted_weights'].append(name)
                except OSError as error:
                    report['status'] = 'retained'
                    report['retained_files'].append({'path': name, 'error_type': type(error).__name__})
    archive = args.local_stage / 'incoming-payload.tar.gz'
    if report['status'] == 'complete':
        try:
            stamp = regular_owned_file(archive, args.local_stage, missing_ok=True)
            if stamp is None:
                report['transport_missing'] = True
            else:
                proof = json.loads((args.local_stage / 'watch-prepared.json').read_text())
                if (digest(archive) != proof.get('archive_sha256')
                        or ('archive_bytes' in proof and stamp.st_size != proof['archive_bytes'])):
                    raise ValueError('local transport changed after verification')
                if file_identity(archive.lstat()) != file_identity(stamp):
                    raise ValueError('local transport changed before cleanup')
                archive.unlink()
                report['transport_deleted'] = True
        except (ValueError, OSError) as error:
            report['status'] = 'retained'
            report['retained_files'].append({'path': archive.name, 'error_type': type(error).__name__})
    write_status(cleanup_path, report)
    return {'local_cleanup_status': report['status'], 'local_cleanup_receipt': str(cleanup_path),
            'local_weight_files_deleted': len(report['deleted_weights']),
            'local_weight_files_already_missing': len(report['missing_weights'])}


def publish_completed(args):
    prepare_command = [args.remote_python, args.remote_helper, '--prepare-only', '--run-dir', args.remote_run,
                  '--source-dir', args.remote_source, '--python', args.remote_python,
                  '--stage-dir', args.remote_stage, '--template-release', args.remote_template,
                  '--base', args.remote_base, '--data', args.remote_data, '--eval-data', args.remote_eval_data,
                  '--code-commit', args.code_commit, '--repo-id', args.repo_id]
    if args.remote_handoff is not None:
        prepare_command += ['--handoff-manifest', args.remote_handoff]
    remote(args, prepare_command, prepare=True, timeout=1800)
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
        'source_cursor': args.expected_source_cursor, 'original_source_checkpoint_sha256': args.expected_source_sha256,
        'data_handoff_sha256': args.expected_handoff_sha256,
        'prepared_manifest_sha256': digest(prepared / 'prepared.json'), 'archive_sha256': packed['archive_sha256'],
        'archive_bytes': packed['archive_bytes']})
    receipt = local_publication(args, prepared)
    receipt.update(cleanup_verified_payload(args, prepared))
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
                if (existing / 'publish.json').exists() or (existing / 'publish.json').is_symlink():
                    publication = json.loads((existing / 'publish.json').read_text())
                    if publication.get('status') == 'verified':
                        publication = verified_publication(args, existing)
                        cleanup = cleanup_verified_payload(args, existing)
                        update(status='published_verified', repo_id=args.repo_id, step=args.expected_target_step,
                               commit=publication['commit'], prepared_stage=str(existing),
                               publication_resumed_from_verified_receipt=True, **cleanup)
                        return 0
                update(status='publishing_prepared_resume')
                receipt = local_publication(args, existing)
                receipt.update(cleanup_verified_payload(args, existing))
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
    parser.add_argument('--expected-source-cursor', type=int, default=43062)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--remote-handoff')
    parser.add_argument('--expected-handoff-sha256')
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
    if args.remote_handoff is not None:
        path = PurePosixPath(args.remote_handoff)
        if not path.is_absolute() or '..' in path.parts:
            parser.error('remote handoff must be an absolute path without traversal')
        args.remote_handoff = str(path)
    if (args.expected_source_step <= 0 or args.expected_target_step - args.expected_source_step != 4000
            or args.expected_source_cursor < 0 or not re.fullmatch(r'[0-9a-f]{40}', args.code_commit)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo_id)):
        parser.error('this watcher requires an explicit 4000-update window, valid cursor and fixed code commit')
    if args.expected_source_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', args.expected_source_sha256):
        parser.error('expected source checkpoint SHA must be a full SHA256')
    if ((args.expected_source_step, args.expected_target_step, args.expected_source_cursor) != (77864, 81864, 43062)
            and args.expected_source_sha256 is None):
        parser.error('a new window requires its immutable source checkpoint SHA256')
    if ((args.remote_handoff is None) != (args.expected_handoff_sha256 is None)
            or args.remote_handoff is not None and (args.expected_source_sha256 is None
                or not re.fullmatch(r'[0-9a-f]{64}', args.expected_handoff_sha256))):
        parser.error('fresh corpus publication requires handoff path/SHA256 and immutable source SHA256')
    args.local_stage = args.local_stage.resolve()
    args.publisher = args.publisher.resolve()
    args.status_file = args.status_file.resolve()
    args.known_hosts = args.known_hosts.resolve()
    return supervise(args)


if __name__ == '__main__':
    raise SystemExit(main())
