"""Local publication watcher tests; no SSH, Hugging Face or torch required."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
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
        prefix = 'checkpoints/b0-rocm-realtext/step-81864/'
        files = {prefix + name: value for name, value in {
            'trainable.pt': b'complete weights + Adam', 'weights-only.pt': b'132 BF16 weights',
            'release.json': b'{"step":81864}', 'README.md': b'verified checkpoint',
            'source/cat_yoko/model.py': b'previous accepted math',
            'evidence/continuation/quality.json': b'{"status":"completed"}',
            'evidence/continuation/train/metrics.jsonl': b'{"step":81864}\n'}.items()}
        directory = self.root / 'payload'
        directory.mkdir(exist_ok=True)
        for name, value in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        manifest = {'status': 'prepared', 'step': 81864, 'repo_id': self.args.repo_id,
                    'checkpoint_prefix': prefix[:-1], 'stage_dir': '/remote/stage',
                    'files': {name: {'bytes': len(value), 'sha256': watch.digest(directory / name)}
                              for name, value in files.items()}}
        (directory / 'prepared.json').write_text(json.dumps(manifest))
        archive = self.root / 'payload.tar.gz'
        with tarfile.open(archive, 'w:gz') as output:
            for name in ['prepared.json', *files]:
                output.add(directory / name, arcname=name, recursive=False)
        return archive, manifest

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
        success = subprocess.CompletedProcess([], 0, json.dumps({'status': 'verified', 'step': 81864,
            'repo_id': self.args.repo_id, 'commit': 'e' * 40}), '')
        retryable = subprocess.CompletedProcess([], 1, '{"status":"failed","error_type":"RuntimeError"}', '')
        with patch.object(watch.subprocess, 'run', side_effect=[retryable, retryable, success]) as run, \
                patch.object(watch.time, 'sleep') as sleep:
            receipt = watch.local_publication(self.args, self.root / 'prepared')
        self.assertEqual(receipt['publication_attempts'], 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(60,), (60,)])
        self.assertTrue(all(call.args[0][0] == '/local/venv/python' for call in run.call_args_list))
        with patch.object(watch.subprocess, 'run', return_value=retryable) as run, patch.object(watch.time, 'sleep') as sleep:
            with self.assertRaises(watch.PublicationFailed):
                watch.local_publication(self.args, self.root / 'prepared')
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
        with patch.object(watch, 'remote') as remote, patch.object(watch, 'local_publication', return_value=success), \
                patch('builtins.print'):
            self.assertEqual(watch.supervise(self.args), 0)
        remote.assert_not_called()
        self.assertEqual(self.read_status()['status'], 'published_verified')
        proof['step'] = 81865
        watch.write_status(self.args.local_stage / 'watch-prepared.json', proof)
        with self.assertRaisesRegex(ValueError, 'another'):
            watch.prepared_resume(self.args)


if __name__ == '__main__':
    unittest.main()
