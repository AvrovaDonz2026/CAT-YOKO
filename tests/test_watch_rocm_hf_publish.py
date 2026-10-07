"""Local publication watcher tests; no SSH, Hugging Face or torch required."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('watch_rocm_hf_publish',
    Path(__file__).resolve().parents[1] / 'scripts/watch_rocm_hf_publish.py')
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.argv = ['--ssh-host', 'donz@103.40.14.174', '--control-path', '/tmp/control-%C',
                     '--known-hosts', str(self.root / 'known-hosts'), '--remote-python', '/remote/venv/python',
                     '--remote-helper', '/remote/tools/publish.py', '--remote-run', '/remote/run',
                     '--remote-source', '/remote/source', '--remote-stage', '/remote/stage',
                     '--remote-template', '/remote/template/release.json', '--remote-base', '/remote/base',
                     '--remote-data', '/remote/data/train.bin', '--remote-eval-data', '/remote/data/eval.bin',
                     '--code-commit', 'a' * 40, '--local-stage', str(self.root / 'local'),
                     '--publisher', str(self.root / 'publisher.py'), '--python', '/local/venv/python',
                     '--status-file', str(self.root / 'status.json'), '--once']
        self.args = watch.build_parser().parse_args(self.argv)
        quality = {'pass': True, 'eval_batches': 32, 'row_start': 32, 'row_stop': 64}
        self.row = {'status': 'complete', 'run_path': '/remote/run', 'source_dir': '/remote/source',
                    'source_checkpoint': '/remote/run/source_checkpoint/trainable.pt', 'source_step': 77864,
                    'source_cursor': 43062, 'source_adam_step': 43062, 'source_tokens_in_phase': 338626560,
                    'target_step': 81864, 'target_cursor': 47062, 'updates': 4000, 'child_pid': None,
                    'final_verified': True, 'final_step': 81864, 'final_cursor': 47062,
                    'final_adam_step': 47062, 'final_tokens_in_phase': 355010560,
                    'final_checkpoint': '/remote/run/continuation/train/trainable.pt',
                    'optimizer_states': 132, 'optimizer_moments': 264, 'final_quality_done': True,
                    'quality_path': '/remote/run/continuation/quality.json', 'quality_sha256': 'b' * 64,
                    'quality_initial': {**quality, 'step': 77864}, 'quality_final': {**quality, 'step': 81864}}

    def read_status(self):
        return json.loads(self.args.status_file.read_text())

    def payload(self):
        step = self.args.expected_target_step
        prefix = f'checkpoints/b0-rocm-realtext/step-{step}/'
        files = {prefix + name: value for name, value in {
            'trainable.pt': b'complete weights + Adam', 'weights-only.pt': b'132 BF16 weights',
            'release.json': b'{"step":81864}', 'README.md': b'verified checkpoint',
            'source/cat_yoko/model.py': b'previous accepted math',
            'evidence/continuation/quality.json': b'{"status":"completed"}',
            'evidence/continuation/train/metrics.jsonl': b'{"step":81864}\n'}.items()}
        release = {'step': step, 'repo_id': self.args.repo_id,
            'files': {name: {'bytes': len(files[prefix + name]),
                'sha256': watch.hashlib.sha256(files[prefix + name]).hexdigest()}
                for name in ('trainable.pt', 'weights-only.pt')}}
        if self.args.remote_handoff is not None:
            release.update(data_handoff=self.fresh_plan, data_handoff_sha256=self.args.expected_handoff_sha256)
            files[prefix + 'evidence/data_handoff.json'] = self.fresh_plan_bytes
            files[prefix + 'evidence/corpus-manifest.json'] = b'{"fresh":"verified"}'
            files[prefix + 'evidence/continuation/fresh_quality.json'] = json.dumps(self.row['fresh_quality']).encode()
        files[prefix + 'release.json'] = json.dumps(release).encode()
        directory = self.root / 'payload'
        directory.mkdir(exist_ok=True)
        for name, value in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        manifest = {'status': 'prepared', 'step': step, 'repo_id': self.args.repo_id,
                    'checkpoint_prefix': prefix[:-1], 'stage_dir': '/remote/stage',
                    'files': {name: {'bytes': len(value), 'sha256': watch.digest(directory / name)}
                              for name, value in files.items()}}
        if self.args.remote_handoff is not None:
            manifest.update(data_handoff_sha256=self.args.expected_handoff_sha256,
                            original_source_checkpoint_sha256=self.args.expected_source_sha256,
                            train_corpus_sha256=self.fresh_plan['new_train_sha256'])
        (directory / 'prepared.json').write_text(json.dumps(manifest))
        archive = self.root / 'payload.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            for name in ['prepared.json', *files]:
                output.add(directory / name, arcname=name, recursive=False)
        return archive, manifest

    def stage_payload(self, *, transport=True):
        archive, manifest = self.payload()
        prepared = watch.extract_payload(archive, self.args.local_stage, step=self.args.expected_target_step, repo_id=self.args.repo_id)
        watch.write_status(self.args.local_stage / 'watch-prepared.json', {
            'remote_run': self.args.remote_run, 'remote_source': self.args.remote_source,
            'source_step': self.args.expected_source_step, 'step': self.args.expected_target_step, 'repo_id': self.args.repo_id,
            'source_cursor': self.args.expected_source_cursor, 'original_source_checkpoint_sha256': self.args.expected_source_sha256,
            'data_handoff_sha256': self.args.expected_handoff_sha256,
            'prepared_manifest_sha256': watch.digest(prepared / 'prepared.json'),
            'archive_sha256': watch.digest(archive), 'archive_bytes': archive.stat().st_size})
        if transport:
            shutil.copyfile(archive, self.args.local_stage / 'incoming-payload.tar.gz')
        return prepared, manifest

    def mark_published(self, prepared, *, commit='c' * 40):
        manifest = json.loads((prepared / 'prepared.json').read_text())
        prefix = manifest['checkpoint_prefix']
        (prepared / 'README.md').write_text('Verified latest model card\n')
        receipt = {'status': 'verified', 'repo_id': self.args.repo_id, 'step': self.args.expected_target_step,
                   'commit': commit, 'parent_commit': 'a' * 40, 'checkpoint_prefix': prefix,
                   'allowlist': manifest['files'],
                   'checkpoint_lfs_verified': {name: manifest['files'][prefix + '/' + name]
                        for name in ('trainable.pt', 'weights-only.pt')},
                   'download_verified': {name: watch.artifact_record(prepared / name)
                        for name in ('README.md', prefix + '/README.md', prefix + '/release.json')}}
        watch.write_status(prepared / 'publish.json', receipt)
        return receipt

    def fresh_window(self):
        self.args.expected_source_step = 81864; self.args.expected_target_step = 85864
        self.args.expected_source_cursor = 47062; self.args.expected_source_sha256 = 'd' * 64
        self.args.remote_handoff = '/remote/run/data_handoff.json'
        self.args.remote_data = '/remote/fresh/train.bin'
        self.fresh_plan = {'source_step': 81864, 'target_step': 85864,
            'source_checkpoint_path': self.row['source_checkpoint'], 'source_checkpoint_sha256': 'd' * 64,
            'absolute_cursor_origin': 47062, 'logical_row_origin': 0, 'updates': 4000,
            'target_cursor': 51062, 'target_logical_row': 4000, 'no_wrap': True,
            'mapping': 'row=i-absolute_cursor_origin', 'new_nseq': 19531,
            'new_train_path': self.args.remote_data, 'new_train_sha256': 'e' * 64,
            'fresh_eval_path': '/remote/fresh/eval.bin', 'fresh_eval_sha256': 'f' * 64,
            'source_phase_tokens': 355010560, 'target_phase_tokens': 371394560}
        self.fresh_plan_bytes = (json.dumps(self.fresh_plan, sort_keys=True, indent=2) + '\n').encode()
        self.args.expected_handoff_sha256 = watch.hashlib.sha256(self.fresh_plan_bytes).hexdigest()
        q = {'pass': True, 'eval_batches': 32, 'row_start': 0, 'row_stop': 32}
        fresh = {'status': 'completed', 'protocol': 'fresh_validation_corpus_paired_heldout',
                 'data_handoff_sha256': self.args.expected_handoff_sha256,
                 'eval_data': self.fresh_plan['fresh_eval_path'], 'eval_data_sha256': 'f' * 64,
                 'source_checkpoint_sha256': 'd' * 64, 'seq_len': 4096, 'eval_batches': 32,
                 'row_start': 0, 'row_stop': 32,
                 'initial_eval': {**q, 'step': 81864}, 'final_eval': {**q, 'step': 85864}}
        self.row.update(source_step=81864, target_step=85864, source_cursor=47062,
                        source_adam_step=47062, source_tokens_in_phase=355010560,
                        target_cursor=51062, final_step=85864, final_cursor=51062,
                        final_adam_step=51062, final_tokens_in_phase=371394560,
                        source_sha256='d' * 64, data_handoff=self.fresh_plan,
                        data_handoff_path=self.args.remote_handoff, data_handoff_sha256=self.args.expected_handoff_sha256,
                        actual_data_handoff_sha256=self.args.expected_handoff_sha256, fresh_quality=fresh)
        self.row['quality_initial']['step'] = 81864; self.row['quality_final']['step'] = 85864

    def test_fresh_completion_binds_source_global_and_local_rows_and_third_quality(self):
        self.fresh_window()
        self.assertTrue(watch.completion_verified(self.row, self.args))
        for change in ({'source_sha256': 'a' * 64}, {'actual_data_handoff_sha256': 'b' * 64},
                       {'data_handoff_path': '/other/plan.json'}, {'source_cursor': 43062},
                       {'data_handoff': dict(self.fresh_plan, target_logical_row=8000)},
                       {'data_handoff': dict(self.fresh_plan, new_train_path='/old/train.bin')},
                       {'fresh_quality': dict(self.row['fresh_quality'], status='failed')},
                       {'fresh_quality': dict(self.row['fresh_quality'], eval_data_sha256='a' * 64)},
                       {'fresh_quality': dict(self.row['fresh_quality'], final_eval={'step': 85864, 'pass': False})}):
            row = copy.deepcopy(self.row); row.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                watch.completion_verified(row, self.args)
        self.args.remote_handoff = None
        with self.assertRaisesRegex(ValueError, 'explicitly requested'):
            watch.completion_verified(self.row, self.args)

    def test_generic_cli_requires_fixed_source_and_exact_4000_window(self):
        new = self.argv + ['--expected-source-step', '81864', '--expected-target-step', '85864',
                           '--expected-source-cursor', '47062', '--expected-source-sha256', 'd' * 64]
        with patch.object(watch, 'supervise', return_value=0) as supervise:
            self.assertEqual(watch.main(new), 0)
        self.assertEqual(supervise.call_args.args[0].expected_source_cursor, 47062)
        for extra in (['--expected-source-step', '81864', '--expected-target-step', '85864'],
                      ['--expected-target-step', '81863'], ['--expected-source-cursor', '-1'],
                      ['--remote-handoff', '/remote/run/plan.json']):
            with self.subTest(extra=extra), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                watch.main(self.argv + extra)

    def test_fresh_verified_payload_cleanup_and_rerun_preserve_evidence_no_reupload(self):
        self.fresh_window()
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        self.assertEqual(watch.cleanup_verified_payload(self.args, prepared)['local_cleanup_status'], 'complete')
        for name in ('trainable.pt', 'weights-only.pt'):
            self.assertFalse((prepared / manifest['checkpoint_prefix'] / name).exists())
        self.assertTrue((prepared / manifest['checkpoint_prefix'] / 'evidence/data_handoff.json').is_file())
        with patch.object(watch, 'remote') as remote, patch.object(watch, 'local_publication') as publish, patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 0)
        remote.assert_not_called(); publish.assert_not_called()
        self.assertTrue(self.read_status()['publication_resumed_from_verified_receipt'])

    def test_fresh_resume_rejects_different_expected_plan_or_changed_archived_handoff(self):
        self.fresh_window()
        prepared, manifest = self.stage_payload()
        original = self.args.expected_handoff_sha256
        self.args.expected_handoff_sha256 = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'different source or handoff'):
            watch.prepared_resume(self.args)
        self.args.expected_handoff_sha256 = original
        (prepared / manifest['checkpoint_prefix'] / 'evidence/data_handoff.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'evidence SHA differs'):
            watch.prepared_resume(self.args)

    def test_original_published_81864_resume_without_new_fields_still_succeeds(self):
        prepared, manifest = self.stage_payload()
        proof_path = self.args.local_stage / 'watch-prepared.json'
        proof = json.loads(proof_path.read_text())
        for key in ('source_cursor', 'original_source_checkpoint_sha256', 'data_handoff_sha256'):
            proof.pop(key)
        watch.write_status(proof_path, proof)
        self.mark_published(prepared)
        watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(watch.prepared_resume(self.args), prepared)

    def test_remote_status_missing_complete_fresh_evidence_is_terminal_invalid_receipt_not_disconnect(self):
        self.fresh_window()
        run = self.root / 'remote-status-run'
        (run / 'continuation').mkdir(parents=True)
        plan_path = run / 'data_handoff.json'
        plan_path.write_bytes(self.fresh_plan_bytes)
        status = {'status': 'complete', 'data_handoff': self.fresh_plan,
                  'data_handoff_path': str(plan_path), 'data_handoff_sha256': self.args.expected_handoff_sha256}
        watch.write_status(run / 'status.json', status)
        command = [sys.executable, '-c', watch.REMOTE_STATUS, str(run), '/remote/source']
        missing = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 0, missing.stderr)
        receipt = json.loads(missing.stdout)
        self.assertIsNone(receipt['actual_data_handoff_sha256']); self.assertEqual(receipt['fresh_quality'], {})
        watch.write_status(run / 'continuation/fresh_quality.json', self.row['fresh_quality'])
        complete = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertEqual(json.loads(complete.stdout)['actual_data_handoff_sha256'], self.args.expected_handoff_sha256)
        plan_path.write_text('{}')
        changed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(changed.returncode, 0, changed.stderr)
        self.assertIsNone(json.loads(changed.stdout)['actual_data_handoff_sha256'])

    def test_once_waits_without_running_preparer_and_disconnect_is_sanitized(self):
        waiting = dict(self.row, status='running_continuation')
        with patch.object(watch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(waiting), '')), \
                patch.object(watch, 'publish_completed') as publish, patch.object(watch.time, 'sleep') as sleep, \
                patch('builtins.print'):
            self.assertEqual(watch.main(self.argv), 0)
        self.assertEqual(self.read_status()['status'], 'waiting_for_training')
        publish.assert_not_called(); sleep.assert_not_called()
        failure = subprocess.CompletedProcess([], 255, '', 'token=secret https://host/?signed=secret')
        with patch.object(watch.subprocess, 'run', return_value=failure), patch('builtins.print'):
            self.assertEqual(watch.main(self.argv), 0)
        self.assertEqual(self.read_status()['status'], 'waiting_for_remote')
        self.assertNotIn('secret', self.args.status_file.read_text())

    def test_training_failure_exits_without_preparation_or_publication(self):
        with patch.object(watch, 'remote', return_value=dict(self.row, status='failed')), \
                patch.object(watch, 'publish_completed') as publish, patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 1)
        self.assertEqual(self.read_status()['status'], 'training_failed')
        publish.assert_not_called()

    def test_wrong_run_target_quality_or_state_never_passes_completion(self):
        for change in ({'run_path': '/another/run'}, {'source_step': 77863}, {'target_step': 81865},
                       {'updates': 3999}, {'source_cursor': 43061}, {'final_adam_step': 47061},
                       {'optimizer_moments': 263}, {'final_verified': False}, {'final_quality_done': False},
                       {'final_tokens_in_phase': 355010559},
                       {'quality_final': dict(self.row['quality_final'], step=81863)}):
            row = copy.deepcopy(self.row); row.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                watch.completion_verified(row, self.args)
        self.assertTrue(watch.completion_verified(self.row, self.args))

    def test_remote_command_quotes_arguments_and_disables_interactive_authentication(self):
        words = ['/remote/python', '-c', 'literal $(secret) `secret`', '/run with spaces']
        with patch.object(watch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
            watch.remote(self.args, words)
        command = run.call_args.args[0]
        self.assertEqual(shlex.split(command[-1]), words)
        for flag in ('BatchMode=yes', 'PasswordAuthentication=no', 'KbdInteractiveAuthentication=no',
                     'ConnectTimeout=15', 'ServerAliveInterval=10', 'StrictHostKeyChecking=yes'):
            self.assertIn(flag, command)
        self.assertEqual(run.call_args.kwargs['stdin'], subprocess.DEVNULL)

    def test_complete_pipeline_prepares_remotely_but_publishes_only_locally(self):
        archive, _ = self.payload()
        commands = []
        def fake(command, **kwargs):
            commands.append(command)
            if command[0] == 'ssh':
                words = shlex.split(command[-1])
                if '--prepare-only' in words:
                    return subprocess.CompletedProcess(command, 0, '{"status":"prepared"}', '')
                if words[2] == watch.REMOTE_PACK:
                    receipt = {'status': 'packed', 'step': 81864, 'archive_path': '/remote/stage/payload.tar.gz',
                               'archive_sha256': watch.digest(archive), 'archive_bytes': archive.stat().st_size}
                    return subprocess.CompletedProcess(command, 0, json.dumps(receipt), '')
                return subprocess.CompletedProcess(command, 0, json.dumps(self.row), '')
            if command[0] == 'scp':
                shutil.copyfile(archive, command[-1])
                return subprocess.CompletedProcess(command, 0, '', '')
            self.assertEqual(command[0], '/local/venv/python')
            self.assertIn('--publish-prepared', command)
            self.mark_published(self.args.local_stage / 'prepared')
            return subprocess.CompletedProcess(command, 0, json.dumps({'status': 'verified', 'step': 81864,
                'repo_id': self.args.repo_id, 'commit': 'c' * 40, 'url': 'https://hf/?signed=secret'}), '')
        with patch.object(watch.subprocess, 'run', side_effect=fake), patch('builtins.print'):
            self.assertEqual(watch.main(self.argv), 0)
        status = self.read_status()
        self.assertEqual(status['status'], 'published_verified')
        self.assertEqual(status['commit'], 'c' * 40)
        self.assertNotIn('secret', self.args.status_file.read_text())
        remote_prepare = shlex.split(commands[1][-1])
        self.assertEqual(remote_prepare[remote_prepare.index('--template-release') + 1], '/remote/template/release.json')
        self.assertNotIn('--publish-prepared', remote_prepare)
        self.assertNotIn('--prepare-only', commands[-1])
        self.assertFalse((self.args.local_stage / 'incoming-payload.tar.gz').exists())
        prepared = self.args.local_stage / 'prepared'
        self.assertFalse((prepared / 'checkpoints/b0-rocm-realtext/step-81864/trainable.pt').exists())
        self.assertFalse((prepared / 'checkpoints/b0-rocm-realtext/step-81864/weights-only.pt').exists())
        self.assertTrue((prepared / 'publish.json').is_file())
        self.assertTrue((prepared / 'checkpoints/b0-rocm-realtext/step-81864/source/cat_yoko/model.py').is_file())
        self.assertEqual(status['local_cleanup_status'], 'complete')

    def test_archive_rejects_traversal_links_extra_binaries_duplicate_and_hash_mismatch(self):
        archive, manifest = self.payload()
        original = archive.read_bytes()
        for name, kind in (('../escaped.pt', tarfile.REGTYPE),
                           ('checkpoints/b0-rocm-realtext/step-81864/extra.pt', tarfile.REGTYPE),
                           ('prepared.json', tarfile.REGTYPE),
                           ('checkpoints/b0-rocm-realtext/step-81864/source/cat_yoko/x.py', tarfile.SYMTYPE)):
            archive.write_bytes(original)
            # Repack original members plus the adversarial entry.
            with tarfile.open(fileobj=io.BytesIO(original), mode='r:gz') as old:
                entries = [(member, old.extractfile(member).read()) for member in old.getmembers()]
            with tarfile.open(archive, 'w:gz') as output:
                for member, content in entries:
                    output.addfile(member, io.BytesIO(content))
                member = tarfile.TarInfo(name); member.type = kind; member.linkname = '/etc/passwd'
                content = b'forbidden'; member.size = len(content) if kind == tarfile.REGTYPE else 0
                output.addfile(member, io.BytesIO(content) if member.size else None)
            with self.subTest(name=name, kind=kind), self.assertRaises(ValueError):
                watch.extract_payload(archive, self.root / 'extract', step=81864, repo_id=self.args.repo_id)
        self.assertFalse((self.root / 'escaped.pt').exists())
        archive.write_bytes(original)
        bad = manifest['files']['checkpoints/b0-rocm-realtext/step-81864/trainable.pt']
        bad['sha256'] = 'f' * 64
        with tarfile.open(fileobj=io.BytesIO(original), mode='r:gz') as old, tarfile.open(archive, 'w:gz') as output:
            for member in old.getmembers():
                content = json.dumps(manifest).encode() if member.name == 'prepared.json' else old.extractfile(member).read()
                member.size = len(content); output.addfile(member, io.BytesIO(content))
        with self.assertRaisesRegex(ValueError, 'SHA'):
            watch.extract_payload(archive, self.root / 'extract', step=81864, repo_id=self.args.repo_id)

    def test_validated_archive_is_idempotent_but_refuses_changed_existing_payload(self):
        archive, _ = self.payload()
        ready = watch.extract_payload(archive, self.root / 'extract', step=81864, repo_id=self.args.repo_id)
        self.assertEqual(watch.extract_payload(archive, self.root / 'extract', step=81864, repo_id=self.args.repo_id), ready)
        (ready / 'checkpoints/b0-rocm-realtext/step-81864/trainable.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            watch.extract_payload(archive, self.root / 'extract', step=81864, repo_id=self.args.repo_id)

    def test_publish_failure_never_marks_success_or_persists_signed_exception_text(self):
        with patch.object(watch, 'remote', return_value=self.row), \
                patch.object(watch, 'publish_completed', side_effect=watch.PublicationFailed('https://host/?sig=secret')), \
                patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 1)
        self.assertEqual(self.read_status()['status'], 'publication_failed')
        self.assertEqual(self.read_status()['exception_type'], 'PublicationFailed')
        self.assertNotIn('secret', self.args.status_file.read_text())

    def test_bounded_retry_round_can_wait_for_remote_and_recover(self):
        self.args.once = False
        receipt = {'status': 'published_verified', 'repo_id': self.args.repo_id, 'step': 81864, 'commit': 'd' * 40}
        with patch.object(watch, 'remote', side_effect=[watch.RemoteUnavailable('no connection'), self.row]), \
                patch.object(watch, 'publish_completed', return_value=receipt), \
                patch.object(watch.time, 'sleep') as sleep, patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 0)
        sleep.assert_called_once_with(60)
        self.assertEqual(self.read_status()['status'], 'published_verified')

    def test_local_network_retries_are_bounded_and_do_not_repeat_remote_preparation(self):
        prepared, _ = self.stage_payload()
        self.mark_published(prepared, commit='e' * 40)
        success = subprocess.CompletedProcess([], 0, json.dumps({'status': 'verified', 'step': 81864,
            'repo_id': self.args.repo_id, 'commit': 'e' * 40}), '')
        retryable = subprocess.CompletedProcess([], 1, '{"status":"failed","error_type":"RuntimeError"}', '')
        with patch.object(watch.subprocess, 'run', side_effect=[retryable, retryable, success]) as run, \
                patch.object(watch.time, 'sleep') as sleep:
            receipt = watch.local_publication(self.args, prepared)
        self.assertEqual(receipt['publication_attempts'], 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(60,), (60,)])
        self.assertTrue(all(call.args[0][0] == '/local/venv/python' for call in run.call_args_list))
        with patch.object(watch.subprocess, 'run', return_value=retryable) as run, patch.object(watch.time, 'sleep') as sleep:
            with self.assertRaises(watch.PublicationFailed):
                watch.local_publication(self.args, prepared)
        self.assertEqual(run.call_count, 3); self.assertEqual(sleep.call_count, 2)

    def test_immutable_valueerror_or_unknown_failure_never_retried(self):
        for stderr, stdout in (('', '{"status":"failed","error_type":"ValueError"}'),
                               ('signed=secret', ''),
                               ('Completed-run publication failed: ValueError\n', '')):
            with self.subTest(stdout=stdout), patch.object(watch.subprocess, 'run',
                    return_value=subprocess.CompletedProcess([], 1, stdout, stderr)) as run, \
                    patch.object(watch.time, 'sleep') as sleep:
                with self.assertRaises(watch.PublicationFailed):
                    watch.local_publication(self.args, self.root / 'prepared')
            run.assert_called_once(); sleep.assert_not_called()

    def test_prepared_resume_verifies_binding_and_publishes_without_remote_contact(self):
        archive, _ = self.payload()
        prepared = watch.extract_payload(archive, self.args.local_stage, step=81864, repo_id=self.args.repo_id)
        proof = {'remote_run': self.args.remote_run, 'remote_source': self.args.remote_source,
                 'source_step': 77864, 'step': 81864, 'repo_id': self.args.repo_id,
                 'prepared_manifest_sha256': watch.digest(prepared / 'prepared.json'),
                 'archive_sha256': watch.digest(archive)}
        watch.write_status(self.args.local_stage / 'watch-prepared.json', proof)
        success = {'status': 'published_verified', 'step': 81864, 'repo_id': self.args.repo_id, 'commit': 'a' * 40}
        def publish(*unused):
            self.mark_published(prepared, commit='a' * 40)
            return success
        with patch.object(watch, 'remote') as remote, patch.object(watch, 'local_publication', side_effect=publish), \
                patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 0)
        remote.assert_not_called()
        self.assertEqual(self.read_status()['status'], 'published_verified')
        proof['step'] = 81865
        watch.write_status(self.args.local_stage / 'watch-prepared.json', proof)
        with self.assertRaisesRegex(ValueError, 'another'):
            watch.prepared_resume(self.args)

    def test_cleanup_validates_both_weights_before_deleting_either(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        prefix = manifest['checkpoint_prefix']
        full, light = (prepared / prefix / name for name in ('trainable.pt', 'weights-only.pt'))
        light.write_bytes(b'changed light weights')
        cleanup = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(cleanup['local_cleanup_status'], 'retained')
        self.assertTrue(full.is_file()); self.assertTrue(light.is_file())
        self.assertTrue((self.args.local_stage / 'incoming-payload.tar.gz').exists())

    def test_cleanup_requires_actual_verified_bound_receipt_and_pinned_lfs(self):
        mutations = ({'status': 'failed'}, {'repo_id': 'other/repo'}, {'step': 81865},
                     {'commit': 'invalid'}, {'checkpoint_prefix': 'another/prefix'},
                     {'checkpoint_lfs_verified': {}}, {'allowlist': {}}, {'download_verified': {}})
        for change in mutations:
            with self.subTest(change=change):
                # Reset this isolated temporary stage between mutation cases.
                shutil.rmtree(self.args.local_stage, ignore_errors=True)
                prepared, manifest = self.stage_payload()
                receipt = self.mark_published(prepared); receipt.update(change)
                watch.write_status(prepared / 'publish.json', receipt)
                with self.assertRaises(ValueError):
                    watch.cleanup_verified_payload(self.args, prepared)
                for name in ('trainable.pt', 'weights-only.pt'):
                    self.assertTrue((prepared / manifest['checkpoint_prefix'] / name).is_file())
                self.assertTrue((self.args.local_stage / 'incoming-payload.tar.gz').is_file())

    def test_cleanup_is_idempotent_and_preserves_unknown_files_login_and_evidence(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        prefix = manifest['checkpoint_prefix']
        unknown = prepared / prefix / 'unknown.pt'; unknown.write_bytes(b'leave untouched')
        login = self.args.local_stage / 'hf-login'; login.write_text('leave untouched')
        evidence = prepared / prefix / 'evidence/continuation/quality.json'
        evidence_sha = watch.digest(evidence)
        cleanup = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(cleanup['local_weight_files_deleted'], 2)
        self.assertEqual(cleanup['local_cleanup_status'], 'complete')
        repeated = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(repeated['local_weight_files_deleted'], 0)
        self.assertEqual(repeated['local_weight_files_already_missing'], 2)
        self.assertEqual(repeated['local_cleanup_status'], 'complete')
        self.assertTrue(unknown.is_file()); self.assertTrue(login.is_file())
        self.assertEqual(watch.digest(evidence), evidence_sha)
        self.assertTrue((prepared / 'prepared.json').is_file())
        self.assertTrue((prepared / 'publish.json').is_file())
        self.assertTrue((self.args.local_stage / 'local-cleanup.json').is_file())

    def test_symlink_checkpoint_or_parent_never_deletes_external_content(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        prefix = manifest['checkpoint_prefix']
        full = prepared / prefix / 'trainable.pt'
        outside = self.root / 'outside.pt'; outside.write_bytes(full.read_bytes())
        full.unlink(); full.symlink_to(outside)
        cleanup = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(cleanup['local_cleanup_status'], 'retained')
        self.assertTrue(outside.is_file()); self.assertTrue(full.is_symlink())
        self.assertTrue((prepared / prefix / 'weights-only.pt').is_file())

    def test_symlink_parent_rejects_receipt_before_any_weight_cleanup(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        directory = prepared / manifest['checkpoint_prefix']
        outside = self.root / 'external-step'
        directory.rename(outside); directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'parent'):
            watch.cleanup_verified_payload(self.args, prepared)
        self.assertTrue((outside / 'trainable.pt').is_file())
        self.assertTrue((outside / 'weights-only.pt').is_file())
        self.assertTrue((self.args.local_stage / 'incoming-payload.tar.gz').is_file())

    def test_failed_publication_retains_prepared_pair_transport_and_resume_proof(self):
        prepared, manifest = self.stage_payload()
        failure = subprocess.CompletedProcess([], 1, '{"status":"failed","error_type":"ValueError"}', '')
        with patch.object(watch.subprocess, 'run', return_value=failure), patch.object(watch, 'remote') as remote, \
                patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 1)
        remote.assert_not_called()
        self.assertEqual(self.read_status()['status'], 'publication_failed')
        for name in ('trainable.pt', 'weights-only.pt'):
            self.assertTrue((prepared / manifest['checkpoint_prefix'] / name).is_file())
        self.assertTrue((self.args.local_stage / 'incoming-payload.tar.gz').is_file())
        self.assertTrue((self.args.local_stage / 'watch-prepared.json').is_file())
        self.assertFalse((self.args.local_stage / 'local-cleanup.json').exists())

    def test_interrupted_cleanup_with_one_missing_weight_is_idempotent(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        (prepared / manifest['checkpoint_prefix'] / 'trainable.pt').unlink()
        result = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(result['local_cleanup_status'], 'complete')
        self.assertEqual(result['local_weight_files_already_missing'], 1)
        self.assertEqual(result['local_weight_files_deleted'], 1)

    def test_symlink_cleanup_receipt_never_overwrites_unrelated_file_or_deletes(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        outside = self.root / 'unrelated.json'; outside.write_text('do not overwrite')
        (self.args.local_stage / 'local-cleanup.json').symlink_to(outside)
        with self.assertRaises(ValueError):
            watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(outside.read_text(), 'do not overwrite')
        self.assertTrue((prepared / manifest['checkpoint_prefix'] / 'trainable.pt').is_file())

    def test_changed_or_symlink_transport_is_preserved_after_valid_weight_cleanup(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared)
        archive = self.args.local_stage / 'incoming-payload.tar.gz'
        archive.write_bytes(b'changed transport')
        cleanup = watch.cleanup_verified_payload(self.args, prepared)
        self.assertEqual(cleanup['local_cleanup_status'], 'retained')
        self.assertEqual(cleanup['local_weight_files_deleted'], 2)
        self.assertEqual(archive.read_bytes(), b'changed transport')

    def test_verified_rerun_after_cleanup_skips_remote_and_republication(self):
        prepared, manifest = self.stage_payload()
        self.mark_published(prepared, commit='d' * 40)
        watch.cleanup_verified_payload(self.args, prepared)
        self.args.status_file.write_text('{"status":"untrusted_stale_value"}')
        with patch.object(watch, 'remote') as remote, patch.object(watch, 'local_publication') as publish, \
                patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 0)
        remote.assert_not_called(); publish.assert_not_called()
        self.assertTrue(self.read_status()['publication_resumed_from_verified_receipt'])
        self.assertEqual(self.read_status()['commit'], 'd' * 40)

    def test_forged_watcher_status_or_stdout_never_substitutes_for_real_publication(self):
        prepared, manifest = self.stage_payload()
        self.args.status_file.write_text('{"status":"published_verified","commit":"' + 'c' * 40 + '"}')
        success = subprocess.CompletedProcess([], 0, json.dumps({'status': 'verified', 'step': 81864,
            'repo_id': self.args.repo_id, 'commit': 'c' * 40}), '')
        with patch.object(watch.subprocess, 'run', return_value=success), self.assertRaises(ValueError):
            watch.local_publication(self.args, prepared)
        for name in ('trainable.pt', 'weights-only.pt'):
            self.assertTrue((prepared / manifest['checkpoint_prefix'] / name).is_file())


if __name__ == '__main__':
    unittest.main()
