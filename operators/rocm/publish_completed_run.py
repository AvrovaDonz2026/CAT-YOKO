#!/usr/bin/env python3
"""CPU-only validation, staging and atomic Hub publication of a completed B0 run.

No GPU is opened, no running trainer is signalled, and no folder upload is used.
--prepare-only produces an explicit file inventory without importing Hub tools.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FAMILY = 'checkpoints/b0-rocm-realtext'
MARKER_BEGIN = '<!-- CAT-YOKO latest completed release -->'
MARKER_END = '<!-- /CAT-YOKO latest completed release -->'
CPU_EXPORT = r'''
import json, random, sys
from pathlib import Path
import torch
from cat_yoko.optim import CPUOffloadAdamW
torch.set_num_threads(1)
full, light = map(Path, sys.argv[1:])
checkpoint = torch.load(full, map_location='cpu', weights_only=False)
weights, saved = checkpoint['trainable'], checkpoint['optimizer']
def need(condition, message):
    if not condition: raise ValueError(message)
def byte_equal(left, right):
    return (left.dtype == right.dtype and left.shape == right.shape and
            torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
need(len(weights) == 132, 'missing native trainable tensors')
def no_decay(name, value):
    parts = name.split('.')
    return value.ndim < 2 or parts[-1] == 'bias' or any('norm' in part for part in parts) or 'router' in parts
names = [[name for name, value in weights.items() if not no_decay(name, value)],
         [name for name, value in weights.items() if no_decay(name, value)]]
groups = [{**{key: value for key, value in group.items() if key != 'params'},
           'params': [torch.nn.Parameter(weights[name], requires_grad=True) for name in order]}
          for group, order in zip(saved['param_groups'], names)]
optimizer = CPUOffloadAdamW(groups, state_dtype=torch.float32)
optimizer.load_state_dict(saved)
exported = optimizer.state_dict()
need(exported['param_groups'] == saved['param_groups'], 'Adam export changed native groups')
need(set(exported['state']) == set(saved['state']), 'Adam export changed ID mapping')
moments = 0
for key, state in saved['state'].items():
    need(set(state) == set(exported['state'][key]), 'Adam state schema changed')
    for name, value in state.items():
        actual = exported['state'][key][name]
        if name == 'step':
            need((value.item() if torch.is_tensor(value) else value) == actual, 'Adam counter changed')
        elif torch.is_tensor(value):
            need(byte_equal(value, actual), 'Adam tensor bytes changed')
            if name in ('exp_avg', 'exp_avg_sq'): moments += 1
        else: need(value == actual, 'Adam option changed')
need(moments == 264, 'Adam export did not cover all moments')
extra = checkpoint['extra']
random.setstate(extra['rng_py'])
need(random.getstate() == extra['rng_py'], 'Python RNG restore changed')
torch.set_rng_state(extra['rng_torch'])
need(byte_equal(torch.get_rng_state(), extra['rng_torch']), 'Torch CPU RNG restore changed')
cuda_rng = extra.get('rng_cuda')
need(isinstance(cuda_rng, (list, tuple)) and len(cuda_rng) > 0 and all(
    torch.is_tensor(value) and value.device.type == 'cpu' and value.dtype == torch.uint8
    and value.ndim == 1 and value.numel() > 0 for value in cuda_rng), 'saved HIP RNG schema missing')
payload = dict(checkpoint, optimizer=None)
temporary = light.with_name(light.name + '.tmp')
torch.save(payload, temporary)
temporary.replace(light)
loaded = torch.load(light, map_location='cpu', weights_only=False)
need(loaded.get('kind') == checkpoint.get('kind') and loaded.get('optimizer') is None,
     'light overlay format changed')
need(set(loaded['trainable']) == set(weights) and all(
    byte_equal(value, loaded['trainable'][name]) for name, value in weights.items()), 'light weights differ')
need(loaded['n_tensors'] == checkpoint['n_tensors'] == 132 and loaded['nbytes'] == checkpoint['nbytes'],
     'light trainable metadata changed')
result = {'status': 'passed', 'trainable_tensors': 132, 'optimizer_states': 132,
          'optimizer_moments': moments, 'native_adam_export_byteexact': True,
          'weights_only_byteexact': True, 'python_rng_restored': True,
          'cpu_torch_rng_restored': True, 'saved_gpu_rng_schema_checked': True,
          'gpu_rng_restored': False, 'gpu_used': False,
          'extra': {name: extra[name] for name in ('phase','name','step','tokens_in_phase','tokens_seen','seq_len','seed','cfg','stream')}}
print(json.dumps(result, allow_nan=False))
'''


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text())
    def require_finite(value):
        import math
        if isinstance(value, float):
            need(math.isfinite(value), 'non-finite number in publication evidence')
        elif isinstance(value, dict):
            for item in value.values():
                require_finite(item)
        elif isinstance(value, list):
            for item in value:
                require_finite(item)
    require_finite(result)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def await_completion(run_dir, *, wait=False, timeout_seconds=7 * 86400, poll_seconds=30,
                     sleep=time.sleep, alive=process_alive, clock=time.monotonic):
    """Only a completed supervisor receipt permits publication, never worker exit."""
    need(timeout_seconds > 0 and 0 < poll_seconds <= 30, 'invalid completion wait bounds')
    deadline = clock() + timeout_seconds
    while True:
        status = read_json(Path(run_dir) / 'status.json')
        if status.get('status') == 'complete':
            need(status.get('child_pid') is None, 'completed run still owns a worker')
            return status
        need(status.get('status') not in ('failed', 'cancelled', 'interrupted'), 'run failed before publication')
        need(wait, 'run is not complete; use --wait-for-completion to wait')
        pids = [status.get(key) for key in ('supervisor_pid', 'child_pid') if status.get(key) is not None]
        need(pids and all(alive(pid) for pid in pids), 'run process ended without a complete receipt')
        need(clock() < deadline, 'completion wait timed out')
        sleep(min(poll_seconds, max(0, deadline - clock())))


def cpu_export(args, full, light):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', PYTHONOPTIMIZE='0')
    result = subprocess.run([args.python, '-c', CPU_EXPORT, str(full), str(light)],
                            cwd=args.source_dir, env=env, capture_output=True, text=True, timeout=900)
    # Keep errors local; SDK and subprocess exception text is never printed by CLI.
    (args.stage_dir / 'cpu-export.log').write_text(result.stdout + result.stderr)
    need(result.returncode == 0, 'CPU optimizer recovery or weights-only export failed; see cpu-export.log')
    audit = json.loads(result.stdout.splitlines()[-1])
    need(audit.get('status') == 'passed' and audit.get('optimizer_moments') == 264
         and audit.get('native_adam_export_byteexact') is True and audit.get('weights_only_byteexact') is True,
         'CPU export did not prove complete byte-identical state recovery')
    return audit


def file_record(path):
    return {'sha256': digest(path), 'bytes': Path(path).stat().st_size}


def deployed_sources(source_dir):
    files = {}
    for directory in ('cat_yoko', 'operators/rocm'):
        for path in sorted((source_dir / directory).rglob('*.py')):
            relative = path.relative_to(source_dir).as_posix()
            need(not path.is_symlink() and path.resolve().is_relative_to(source_dir), 'deployed source escapes its directory')
            files[relative] = path
    need({'cat_yoko/model.py', 'cat_yoko/optim.py', 'cat_yoko/trainer.py',
          'operators/rocm/production_continuation.py'} <= set(files), 'deployed source archive is incomplete')
    return files


def verified_assets(args, template):
    """Static mixture/tokenizer metadata is inherited only for the identical assets."""
    assets = {name: file_record(path) for name, path in (('train', args.data), ('eval', args.eval_data))}
    for name, record in assets.items():
        need(record['sha256'] == template['data'][name]['sha256']
             and record['bytes'] == template['data'][name]['bytes'], 'pilot corpus differs from the release template')
        record['seq_len'] = 4096
    base = deepcopy(template['base_model'])
    for name, expected in list(base.items()):
        if isinstance(expected, dict) and 'sha256' in expected:
            actual = file_record(args.base / name)
            need(actual == expected, 'base model differs from the release template')
            base[name] = actual
    auxiliaries = base.get('auxiliary_sha256', {})
    for name, expected in auxiliaries.items():
        need(digest(args.base / name) == expected, 'base auxiliary file differs from template')
    for name, expected in template.get('tokenizer', {}).get('files_sha256', {}).items():
        need(digest(args.base / name) == expected, 'tokenizer differs from template')
    return assets, base


def build_release(args, template, status, accepted, audit, cpu, quality, parity, operators, files, sources, assets, base):
    extra, cursor, rows = cpu['extra'], audit['source_stream_i'], audit['source_data_rows']
    primary = parity['final_eval']
    step = audit['source_step']
    # A whitelist avoids accidentally inheriting stale handoff counters or publication claims.
    result = {name: deepcopy(template[name]) for name in ('model_format', 'tokenizer', 'mixture', 'license') if name in template}
    result.update(schema_version=2, repo_id=args.repo_id, phase='B0', name='CAT-YOKO-12B',
                  snapshot_at_utc=datetime.fromtimestamp(Path(accepted['checkpoint_path']).stat().st_mtime, timezone.utc).isoformat(),
                  snapshot_time_scope='Atomic numbered checkpoint file modification time in UTC',
                  completed_at_utc=status.get('updated_at'),
                  step=step, adam_step=audit['source_adam_step'], tokens_in_phase=extra['tokens_in_phase'],
                  tokens_seen=extra['tokens_seen'],
                  token_counter_scope='Cumulative phase clock includes DummyStream history; packed inputs include ordered repetitions.',
                  real_text_tokens_processed=cursor * 4096, unique_training_tokens=rows * 4096,
                  completed_corpus_passes=cursor // rows, repeated_corpus=cursor >= rows,
                  repetition_scope='Same packed pilot in the original row order; repeated inputs are not new unique data.',
                  stream={'kind': 'packed', 'stride': 1, 'next_unread_row': cursor,
                          'next_row_mod_nseq': cursor % rows, 'nseq': rows, 'seq_len': 4096},
                  files=files, trainable_names=audit['trainable_names'], trainable_shapes=audit['trainable_shapes'],
                  base_model=base, data=assets,
                  latest_fixed_evaluation={'step': step, 'nll': primary['eval_nll'],
                                           'batches': primary['eval_batches'], 'valid_loss_tokens': primary['eval_valid_tokens'],
                                           'row_start': 0, 'row_stop': 32, 'scope': 'Fixed pilot held-out prefix; not a downstream benchmark.'},
                  additional_fixed_evaluation=quality['final_eval'],
                  additional_evaluation_scope=quality['protocol'],
                  runtime={'gpu': parity['gpu'], 'torch': parity['torch'], 'hip': parity.get('hip'),
                           'precision': 'bfloat16', 'attention': 'packed FP32 MATH with document split backward',
                           'moe_layout': 'shared-storage frozen experts', 'optimizer': 'cached BF16 CPU shadow with native CPU FP32 Adam moments',
                           'deterministic_training': False},
                  code_commit=args.code_commit,
                  code_commit_scope='Publication reference; archived deployed source SHA256 identities are authoritative.',
                  deployed_source_sha256={name: digest(path) for name, path in sources.items()},
                  deployed_source_archive='source/', full_12b_graph_uploaded=False,
                  checkpoint_verification={'passed': True, 'cpu_checkpoint_audit': audit, 'native_recovery_and_light_export': cpu},
                  operator_adoption={'split_attention': True, 'cached_cpu_adam': True,
                                     'verified_completed_updates': accepted['source_update_params']['updates'],
                                     'counts': {name: value for name, value in operators.items() if name.endswith('_calls')},
                                     'checkpoint_format_changed': False, 'moment_device': 'cpu', 'moment_dtype': 'float32',
                                     'reference_backend': accepted['reference_backend'],
                                     'selection': 'Previously accepted production backend; no new speed claim.'},
                  publication_source={'source_run': str(args.run_dir), 'source_dir': str(args.source_dir),
                                      'source_checkpoint': accepted['checkpoint_path'], 'source_sha256': accepted['checkpoint_sha256'],
                                      'snapshot_method': 'Hardlink from a completed atomic numbered checkpoint; no trainer signal.',
                                      'training_run_complete': True, 'updates': accepted['source_update_params']['updates'],
                                      'periodic_checkpoint_seconds': 300, 'keep_last': 3},
                  resume_requirements=['Rebuild frozen graph by upcycling the specified MiniCPM5 base.',
                                       'Use identical tokenizer and packed corpora for the saved absolute cursor.',
                                       'Restore native CPU FP32 Adam from trainable.pt; weights-only.pt omits optimizer history.'])
    return result


def snapshot_card(release):
    step, cursor = release['step'], release['adam_step']
    full, light = release['files']['trainable.pt'], release['files']['weights-only.pt']
    return (f'# Completed B0 step {step}\n\n'
            f'Global step **{step}**, Adam counter / absolute packed cursor **{cursor}**. '
            f'Next corpus row **{release["stream"]["next_row_mod_nseq"]}**; '
            f'{release["unique_training_tokens"]:,} unique training tokens, '
            f'{release["real_text_tokens_processed"]:,} packed inputs including repeated passes. B0 continues; this is not B1.\n\n'
            f'`trainable.pt` is the native trainable overlay plus optimizer, RNG and cursor: '
            f'{full["bytes"]:,} bytes, SHA256 `{full["sha256"]}`. '
            f'`weights-only.pt` has identical 132 BF16 weights and no Adam history: '
            f'{light["bytes"]:,} bytes, SHA256 `{light["sha256"]}`.\n\n'
            'Reconstruct the frozen 12B graph by MiniCPM5-2B-Base upcycling. These files are overlays. '
            'The complete recovery snapshot contains 264 CPU FP32 Adam moments; native load/export and light-weight byte comparisons pass. '
            'GPU RNG is serialized and schema-checked on CPU; this publisher does not restore GPU state.\n\n'
            f'Final primary held-out NLL: **{release["latest_fixed_evaluation"]["nll"]}** on rows 0–31; '
            f'additional NLL: **{release["additional_fixed_evaluation"]["eval_nll"]}** on rows '
            f'{release["additional_fixed_evaluation"]["row_start"]}–{release["additional_fixed_evaluation"]["row_stop"] - 1}. '
            'Both are slices of the same pilot validation corpus, not external benchmarks.\n\n'
            'See `release.json` for verified base/data/source identities and the immutable `source/` archive. '
            'Restore the packed absolute cursor without resetting Adam or RNG; streams read `i % nseq`. '
            'The deployed `source/operators/rocm/production_continuation.py` retains the accepted packed reference, '
            'split attention, shared frozen storage and cached CPU Adam. Its `--accepted-run` receipt references the archived complete-run evidence. '
            'Complete original run directories and corpora remain necessary for that supervisor provenance check; '
            'the standard checkpoint loader can restore the overlay from the recorded base, configuration and native Adam.\n')


def prepare(args, *, status=None):
    for name in ('run_dir', 'source_dir', 'base', 'data', 'eval_data', 'template_release', 'stage_dir'):
        setattr(args, name, Path(getattr(args, name)).resolve())
    # A standalone publisher is deployed outside the frozen training checkout.
    # All heavy CPU audit code is imported from that explicitly identified source.
    sys.path.insert(0, str(args.source_dir))
    from operators.rocm import run_operator_switch as checks
    from operators.rocm.production_continuation import validate_accepted_run
    from operators.rocm.continuation_quality import validate_quality_report
    need(re.fullmatch(r'[0-9a-fA-F]{40}', args.code_commit) is not None, 'code commit must be a complete Git SHA')
    status = status or await_completion(args.run_dir)
    need(status.get('status') == 'complete' and status.get('child_pid') is None, 'only a completed run can be staged')
    need(Path(status['source_dir']).resolve() == args.source_dir, 'source directory differs from completed receipt')
    final = status['final_checkpoint_verified']
    latest = Path(final['source_checkpoint']).resolve()
    numbered = latest.parent / f'trainable_step_{final["source_step"]}.pt'
    accepted = validate_accepted_run(args.run_dir, numbered, args.source_dir)
    parity = read_json(args.run_dir / 'continuation/parity.json')
    need(Path(parity['data']).resolve() == args.data and Path(parity['eval_data']).resolve() == args.eval_data,
         'publication corpus differs from completed training')
    quality = validate_quality_report(args.run_dir / 'continuation/quality.json', source_dir=args.source_dir,
                                      start_row=status.get('extra_heldout_start_row', 32), batches=32)
    need(quality['initial_eval']['step'] == accepted['source_update_params']['source_step']
         and quality['final_eval']['step'] == final['source_step']
         and Path(quality['eval_data']).resolve() == args.eval_data, 'quality observations belong to a different update window')
    for key in ('checkpoint_every_seconds', 'keep_last'):
        need(status.get(key) == {'checkpoint_every_seconds': 300, 'keep_last': 3}[key], 'completed run changed checkpoint retention policy')
    if (args.stage_dir / 'prepared.json').is_file():
        prepared = read_json(args.stage_dir / 'prepared.json')
        need(prepared.get('status') == 'prepared' and prepared.get('repo_id') == args.repo_id
             and prepared.get('step') == final['source_step']
             and prepared.get('source_checkpoint_sha256') == accepted['checkpoint_sha256']
             and prepared.get('run_dir') == str(args.run_dir) and prepared.get('source_dir') == str(args.source_dir),
             'existing staging belongs to a different completed run')
        prefix = prepared.get('checkpoint_prefix')
        need(prefix == f'{FAMILY}/step-{final["source_step"]}', 'existing staging checkpoint prefix differs')
        for name, expected in prepared['files'].items():
            path = args.stage_dir / name
            need(name.startswith(prefix + '/') and '..' not in PurePosixPath(name).parts
                 and not path.is_symlink() and path.resolve().is_relative_to(args.stage_dir),
                 'existing staging path escapes its immutable checkpoint')
            need(file_record(path) == expected, 'existing staged publication changed')
        release = read_json(args.stage_dir / prefix / 'release.json')
        need(release.get('code_commit') == args.code_commit
             and release['publication_source']['source_sha256'] == accepted['checkpoint_sha256']
             and release['deployed_source_sha256'] == {name: digest(path) for name, path in deployed_sources(args.source_dir).items()},
             'existing staging release no longer matches its source')
        verified_assets(args, read_json(args.template_release))
        return prepared
    need(not args.stage_dir.exists() or not any(args.stage_dir.iterdir()), 'use a fresh staging directory')
    need(not args.stage_dir.is_relative_to(args.run_dir) and not args.run_dir.is_relative_to(args.stage_dir)
         and not args.stage_dir.is_relative_to(args.source_dir), 'staging must be separate from training and frozen source')
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    audit_cap = status.get('audit_max_cursor', status['plan'].get('audit_max_cursor', status['plan']['target_cursor']))
    cpu_args = SimpleNamespace(python=args.python, source_dir=args.source_dir, data=args.data,
                               resume=Path(status['source_checkpoint']).resolve())
    audit = checks.cpu_check(cpu_args, numbered, accepted['source_update_params']['updates'],
                             args.stage_dir / 'cpu-check.log', max_cursor=audit_cap)
    need(audit['source_step'] == final['source_step'] and audit['source_stream_i'] == final['source_stream_i']
         and audit['source_adam_step'] == final['source_adam_step'], 'fresh CPU audit differs from completed final state')
    relative = f'{FAMILY}/step-{audit["source_step"]}'
    directory = args.stage_dir / relative
    directory.mkdir(parents=True)
    full = directory / 'trainable.pt'
    os.link(numbered, full)  # Fail closed across filesystems instead of copying a mutable latest pointer.
    cpu = cpu_export(args, full, directory / 'weights-only.pt')
    need(digest(full) == accepted['checkpoint_sha256'], 'numbered checkpoint changed during staging')
    template = read_json(args.template_release)
    assets, base = verified_assets(args, template)
    sources = deployed_sources(args.source_dir)
    files = {'trainable.pt': {**file_record(full), 'trainable_tensors': 132, 'optimizer_states': 132,
                             'optimizer_moments': 264, 'moment_device': 'cpu', 'moment_dtype': 'float32', 'rng_and_stream_saved': True},
             'weights-only.pt': {**file_record(directory / 'weights-only.pt'), 'trainable_tensors': 132,
                                 'optimizer_states': 0, 'weights_bitwise_equal_full_checkpoint': True, 'complete_optimizer_resume': False}}
    operators = read_json(args.run_dir / 'continuation/round3_operators.json')
    release = build_release(args, template, status, accepted, audit, cpu, quality, parity, operators,
                            files, sources, assets, base)
    write_json(directory / 'release.json', release)
    (directory / 'README.md').write_text(snapshot_card(release))
    for name, source in sources.items():
        destination = directory / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        need(digest(destination) == release['deployed_source_sha256'][name], 'source changed while archiving')
    for name in ('status.json', 'continuation/parity.json', 'continuation/operators.json',
                 'continuation/round3_operators.json', 'continuation/quality.json', 'continuation/train/metrics.jsonl'):
        source = args.run_dir / name
        destination = directory / 'evidence' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        need(digest(destination) == digest(source), 'run evidence changed while archiving')
    write_json(directory / 'evidence/cpu-check.json', audit)
    write_json(directory / 'evidence/cpu-export.json', cpu)
    # Revalidate all protected receipts and source after the relatively slow exports.
    again = validate_accepted_run(args.run_dir, numbered, args.source_dir)
    need(again == accepted and read_json(args.run_dir / 'status.json') == status, 'completed run changed during staging')
    allowlist = {path.relative_to(args.stage_dir).as_posix(): file_record(path)
                 for path in sorted(directory.rglob('*')) if path.is_file()}
    prepared = {'status': 'prepared', 'step': audit['source_step'], 'repo_id': args.repo_id,
                'checkpoint_prefix': relative, 'stage_dir': str(args.stage_dir), 'files': allowlist,
                'run_dir': str(args.run_dir), 'source_dir': str(args.source_dir),
                'source_checkpoint_sha256': accepted['checkpoint_sha256']}
    write_json(args.stage_dir / 'prepared.json', prepared)
    return prepared


def latest_card(current, release):
    """Replace only mutable release sections; preserve YAML/spec/history/license."""
    step, repo = release['step'], release['repo_id']
    seen = [int(value) for value in re.findall(r'\bstep[- ](\d+)\b', current, flags=re.I)]
    need(not seen or max(seen) <= step, 'refusing to downgrade a newer model card')
    prefix = f'https://huggingface.co/{repo}/blob/main/{FAMILY}/step-{step}'
    note = (f'{MARKER_BEGIN}\n## Current B0 snapshot\n\n'
            f'The latest completed B0 recovery snapshot is **step {step}**. '
            f'Adam / absolute cursor: **{release["adam_step"]}**; next corpus row: '
            f'**{release["stream"]["next_row_mod_nseq"]}**. B0 remains in its original budget.\n\n'
            f'Use [`trainable.pt`]({prefix}/trainable.pt) for optimizer/RNG/cursor recovery or '
            f'[`weights-only.pt`]({prefix}/weights-only.pt) for the identical overlay without Adam. '
            f'Both require MiniCPM5-2B-Base upcycling; no full 12B graph is uploaded. '
            f'[Snapshot instructions]({prefix}/README.md) and [verified manifest]({prefix}/release.json) '
            'record base, data, runtime and archived deployed code. Older downloader pins select historical snapshots.\n\n'
            f'Real packed inputs: **{release["real_text_tokens_processed"]:,}**, including repeated rows; '
            f'unique training tokens: **{release["unique_training_tokens"]:,}**.\n{MARKER_END}\n')
    replacement = note + '\n'
    if MARKER_BEGIN in current:
        current = re.sub(re.escape(MARKER_BEGIN) + r'.*?' + re.escape(MARKER_END) + r'\n?',
                         note, current, count=1, flags=re.S)
    elif '## Current B0 snapshot' in current:
        current = re.sub(r'(?ms)^## Current B0 snapshot\n.*?(?=^## Current weights\n|^## |\Z)', replacement, current, count=1)
    else:
        match = re.search(r'(?m)^# [^\n]+\n', current)
        need(match is not None, 'model card has no title for the release note')
        current = current[:match.end()] + '\n' + replacement + current[match.end():]
    current = current.replace('## Current weights\n', '## Historical weights\n')
    current = current.replace('Current recommended B0 recovery snapshot', 'Historical B0 recovery snapshot')
    primary, extra = release['latest_fixed_evaluation'], release['additional_fixed_evaluation']
    validation = (f'## Validation observation\n\nAt completed step **{step}**, fixed held-out rows 0–31 have '
                  f'NLL **{primary["nll"]}** ({primary["batches"]} batches, {primary["valid_loss_tokens"]:,} valid tokens). '
                  f'Rows {extra["row_start"]}–{extra["row_stop"] - 1} have NLL **{extra["eval_nll"]}** '
                  f'({extra["eval_batches"]} batches, {extra["eval_valid_tokens"]:,} valid tokens). '
                  'These are two slices of the same pilot validation corpus, not external benchmarks.\n\n')
    if '## Validation observation\n' in current:
        current = re.sub(r'(?ms)^## Validation observation\n.*?(?=^## |\Z)', validation, current, count=1)
    return current


@contextmanager
def network_timeout(seconds):
    """Bound each SDK operation in the CLI process, including upload attempts."""
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('bounded Hub operation timed out')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def request(call, *, attempts=3, timeout=900, sleep=time.sleep):
    for attempt in range(attempts):
        try:
            with network_timeout(timeout):
                return call()
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status in (400, 401, 403, 404, 409, 412):
                raise
            if attempt + 1 == attempts:
                raise RuntimeError('Hub operation failed after bounded retries') from None
            sleep(min(2 ** attempt, 8))


def sibling_map(info):
    return {item.rfilename: item for item in info.siblings}


def lfs_record(item):
    lfs = getattr(item, 'lfs', None)
    def get(name):
        return lfs.get(name) if isinstance(lfs, dict) else getattr(lfs, name, None)
    return {'sha256': get('sha256'), 'bytes': get('size')}


def assert_remote_file(item, expected):
    actual = lfs_record(item)
    need(actual == expected and getattr(item, 'size', expected['bytes']) == expected['bytes'],
         'immutable Hub checkpoint size/SHA differs from staged file')


def publish(args, prepared, *, api=None, download=None, addition=None):
    """CAS commit with immutable checkpoint files and a freshly derived root card."""
    if api is None or download is None or addition is None:
        os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
        os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
        os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
        api = api or HfApi()
        download = download or hf_hub_download
        addition = addition or CommitOperationAdd
    prefix, inventory = prepared['checkpoint_prefix'], prepared['files']
    need(prepared['repo_id'] == args.repo_id and prepared['status'] == 'prepared', 'prepared publication identity differs')
    need(type(prepared.get('step')) is int and prepared['step'] > 0
         and prefix == f'{FAMILY}/step-{prepared["step"]}', 'invalid prepared checkpoint prefix')
    need({prefix + '/' + name for name in ('trainable.pt', 'weights-only.pt', 'release.json', 'README.md')} <= set(inventory),
         'prepared publication is missing required checkpoint files')
    for relative, expected in inventory.items():
        path = PurePosixPath(relative)
        need(relative.startswith(prefix + '/') and '..' not in path.parts and not path.is_absolute(), 'file outside checkpoint allowlist')
        local = args.stage_dir / relative
        need(not local.is_symlink() and local.resolve().is_relative_to(args.stage_dir.resolve()), 'staged path escapes its directory')
        need(file_record(local) == expected, 'staged file changed before publication')
    release = read_json(args.stage_dir / prefix / 'release.json')
    need(release.get('step') == prepared['step'] and release.get('repo_id') == args.repo_id,
         'staged manifest differs from publication identity')
    for filename in ('trainable.pt', 'weights-only.pt'):
        need({key: release['files'][filename][key] for key in ('sha256', 'bytes')} == inventory[prefix + '/' + filename],
             'manifest checkpoint hashes differ from the staging inventory')
    for attempt in range(3):
        info = request(lambda: api.model_info(args.repo_id, files_metadata=True, timeout=60))
        parent, remote = info.sha, sibling_map(info)
        existing_steps = [int(match.group(1)) for name in remote
                          if (match := re.match(re.escape(FAMILY) + r'/step-(\d+)/', name))]
        need(not existing_steps or max(existing_steps) <= prepared['step'], 'Hub already contains a newer completed checkpoint')
        current_path = request(lambda: download(repo_id=args.repo_id, filename='README.md', revision=parent))
        card = latest_card(Path(current_path).read_text(), release)
        root_card = args.stage_dir / 'README.md'
        root_card.write_text(card)
        operations = []
        for name, expected in inventory.items():
            if name in remote:
                if name.endswith(('/trainable.pt', '/weights-only.pt')):
                    assert_remote_file(remote[name], expected)
                else:
                    downloaded = request(lambda name=name: download(repo_id=args.repo_id, filename=name, revision=parent))
                    need(file_record(downloaded) == expected, 'immutable Hub release file differs from staged file')
            else:
                operations.append(addition(path_in_repo=name, path_or_fileobj=str(args.stage_dir / name)))
        if Path(current_path).read_bytes() != root_card.read_bytes():
            operations.append(addition(path_in_repo='README.md', path_or_fileobj=str(root_card)))
        try:
            if operations:
                result = request(lambda: api.create_commit(repo_id=args.repo_id, repo_type='model',
                    operations=operations, commit_message=f'Publish completed B0 step {prepared["step"]}', parent_commit=parent))
                revision = result.oid
            else:
                revision = parent
        except Exception as error:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
            if status not in (409, 412) or attempt == 2:
                raise RuntimeError('Hub publication rejected; no verified receipt was issued') from None
            continue
        pinned = request(lambda: api.model_info(args.repo_id, revision=revision, files_metadata=True, timeout=60))
        need(pinned.sha == revision, 'Hub did not return the committed revision')
        published = sibling_map(pinned)
        for filename in ('trainable.pt', 'weights-only.pt'):
            name = prefix + '/' + filename
            need(name in published, 'Hub commit omitted a checkpoint')
            assert_remote_file(published[name], inventory[name])
        verified_small = {}
        for name in ('README.md', prefix + '/README.md', prefix + '/release.json'):
            path = request(lambda name=name: download(repo_id=args.repo_id, filename=name, revision=revision))
            expected = file_record(root_card) if name == 'README.md' else inventory[name]
            need(file_record(path) == expected, 'commit-pinned card or manifest differs after upload')
            verified_small[name] = expected
        receipt = {'status': 'verified', 'repo_id': args.repo_id, 'commit': revision, 'parent_commit': parent,
                   'step': prepared['step'], 'checkpoint_prefix': prefix,
                   'checkpoint_lfs_verified': {name: inventory[prefix + '/' + name] for name in ('trainable.pt', 'weights-only.pt')},
                   'download_verified': verified_small, 'allowlist': inventory,
                   'url': f'https://huggingface.co/{args.repo_id}/tree/{revision}/{prefix}'}
        write_json(args.stage_dir / 'publish.json', receipt)
        return receipt
    raise RuntimeError('Hub parent changed repeatedly; publication did not verify')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run-dir', 'source-dir', 'base', 'data', 'eval-data', 'template-release'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--code-commit')
    parser.add_argument('--repo-id', default='AvrovaDonz/CAT-YOKO')
    parser.add_argument('--wait-for-completion', action='store_true')
    parser.add_argument('--wait-timeout-seconds', type=float, default=7 * 86400)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--publish-prepared', action='store_true',
                        help='upload a portable prepared.json inventory from this staging directory; no Torch/source access')
    args = parser.parse_args(argv)
    args.stage_dir = args.stage_dir.resolve()
    if args.publish_prepared:
        if args.prepare_only or args.wait_for_completion:
            parser.error('publish-prepared cannot prepare or wait for a training run')
    elif any(getattr(args, name) is None for name in ('run_dir', 'source_dir', 'base', 'data', 'eval_data', 'template_release', 'code_commit')):
        parser.error('preparation requires run-dir, source-dir, base, data, eval-data, template-release and code-commit')
    try:
        if args.publish_prepared:
            result = publish(args, read_json(args.stage_dir / 'prepared.json'))
        else:
            status = await_completion(args.run_dir, wait=args.wait_for_completion, timeout_seconds=args.wait_timeout_seconds)
            prepared = prepare(args, status=status)
            result = prepared if args.prepare_only else publish(args, prepared)
        print(json.dumps({key: result[key] for key in ('status', 'step', 'repo_id', 'commit', 'url') if key in result}, allow_nan=False))
        return 0
    except Exception as error:
        if args.stage_dir.is_dir():
            write_json(args.stage_dir / 'publish.json', {'status': 'failed', 'error_type': type(error).__name__,
                                                       'message': 'Validation or publication failed; no verified receipt was issued.'})
        # Do not expose SDK exception bodies, credentials, or signed URLs.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}))
        print('Completed-run publication failed: ' + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
