"""Completed-run publication gates and immutable Hub commits without GPU/network."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operators.rocm import publish_completed_run as publisher


class HubError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)


class FakeHub:
    """Revision-pinned downloads and LFS metadata, including CAS interference."""
    def __init__(self, root, card, *, conflict=False, corrupt_lfs=False, corrupt_download=False):
        self.root = root
        self.revision = 'a' * 40
        self.revisions = {self.revision: {'README.md': card.encode()}}
        self.commit_calls = []
        self.download_calls = []
        self.conflict = conflict
        self.corrupt_lfs = corrupt_lfs
        self.corrupt_download = corrupt_download

    def model_info(self, repo_id, *, revision=None, **kwargs):
        commit = revision or self.revision
        siblings = []
        for name, data in self.revisions[commit].items():
            sha = publisher.hashlib.sha256(data).hexdigest()
            if name.endswith('.pt'):
                lfs = {'sha256': 'wrong' if self.corrupt_lfs and revision else sha, 'size': len(data)}
            else:
                lfs = None
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs))
        return SimpleNamespace(sha=commit, siblings=siblings)

    def download(self, *, repo_id, filename, revision):
        self.download_calls.append((filename, revision))
        path = self.root / ('download-' + str(len(self.download_calls)))
        data = self.revisions[revision][filename]
        if self.corrupt_download and revision != 'a' * 40 and filename.endswith('release.json'):
            data += b'corrupted'
        path.write_bytes(data)
        return str(path)

    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit):
        self.commit_calls.append((parent_commit, [operation.path_in_repo for operation in operations]))
        if self.conflict:
            self.conflict = False
            revised = dict(self.revisions[self.revision])
            revised['README.md'] += b'\nUpstream history retained.\n'
            self.revision = 'b' * 40
            self.revisions[self.revision] = revised
            raise HubError(409)
        if parent_commit != self.revision:
            raise HubError(409)
        revised = dict(self.revisions[self.revision])
        for operation in operations:
            revised[operation.path_in_repo] = Path(operation.path_or_fileobj).read_bytes()
        self.revision = 'c' * 40
        self.revisions[self.revision] = revised
        return SimpleNamespace(oid=self.revision)


class CompletedPublisherTests(unittest.TestCase):
    def fixture(self, root):
        args = SimpleNamespace(run_dir=root / 'run', source_dir=root / 'deployed', python='cpu-python',
                               base=root / 'base', data=root / 'train.bin', eval_data=root / 'eval.bin',
                               template_release=root / 'template.json', stage_dir=root / 'stage',
                               code_commit='d' * 40, repo_id='AvrovaDonz/CAT-YOKO')
        directory = args.run_dir / 'continuation'
        (directory / 'train').mkdir(parents=True)
        for name in ('cat_yoko/model.py', 'cat_yoko/optim.py', 'cat_yoko/trainer.py',
                     'operators/rocm/production_continuation.py', 'operators/rocm/packed_attention.py'):
            path = args.source_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# immutable deployed source\n')
        args.base.mkdir()
        for name in ('model.safetensors', 'config.json', 'tokenizer.json'):
            (args.base / name).write_bytes(name.encode())
        args.data.write_bytes(b'identical corpus'); args.eval_data.write_bytes(b'identical heldout')
        checkpoint = directory / 'train/trainable_step_77864.pt'
        checkpoint.write_bytes(b'full optimizer checkpoint')
        latest = directory / 'train/trainable.pt'; os.link(checkpoint, latest)
        fixed = args.run_dir / 'source_checkpoint/trainable.pt'
        fixed.parent.mkdir(); fixed.write_bytes(b'original fixed source')
        audit = {'source_step': 77864, 'source_stream_i': 43062, 'source_adam_step': 43062,
                 'source_data_rows': 19531, 'source_tokens_in_phase': 338626560, 'source_tokens_seen': 338626560,
                 'checkpoint_verified': True, 'optimizer_states': 132, 'optimizer_moments': 264,
                 'source_checkpoint': str(latest), 'trainable_names': [f'p{n}' for n in range(132)],
                 'trainable_shapes': {f'p{n}': [2] for n in range(132)}}
        status = {'status': 'complete', 'child_pid': None, 'source_dir': str(args.source_dir),
                  'source_checkpoint': str(fixed), 'final_checkpoint_verified': audit, 'updated_at': '2026-10-06T00:00:00Z',
                  'plan': {'target_cursor': 43062}, 'audit_max_cursor': 58593,
                  'checkpoint_every_seconds': 300, 'keep_last': 3, 'extra_heldout_start_row': 32}
        primary = {'pass': True, 'step': 77864, 'eval_nll': 7.2, 'eval_batches': 32, 'eval_valid_tokens': 130877}
        quality = {'status': 'completed', 'protocol': 'same_validation_corpus_disjoint_rows_not_external_benchmark',
                   'eval_data': str(args.eval_data), 'eval_data_sha256': publisher.digest(args.eval_data),
                   'initial_eval': {'step': 73864},
                   'final_eval': dict(primary, eval_nll=7.3, row_start=32, row_stop=64)}
        parity = {'gpu': 'RX7900XTX', 'torch': '2.9.1', 'hip': '6.4', 'data': str(args.data),
                  'eval_data': str(args.eval_data), 'final_eval': primary}
        operators = {'cached_cpu_adam_calls': {'state_loads': 1, 'parameter_updates': 528000, 'cache_reuses': 527868},
                     'split_attention_calls': {'split_optimized_calls': 12345}}
        for name, value in (('status.json', status), ('continuation/parity.json', parity),
                            ('continuation/operators.json', {}), ('continuation/round3_operators.json', operators),
                            ('continuation/quality.json', quality)):
            publisher.write_json(args.run_dir / name, value)
        (directory / 'train/metrics.jsonl').write_text('{"step":77864}\n')
        accepted = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': publisher.digest(checkpoint),
                    'source_update_params': {'updates': 4000, 'source_step': 73864},
                    'reference_backend': 'previous_packed_production'}
        extra = {'phase': 'B0', 'name': 'CAT-YOKO-12B', 'step': 77864,
                 'tokens_in_phase': 338626560, 'tokens_seen': 338626560, 'seq_len': 4096, 'seed': 0,
                 'cfg': {}, 'stream': {'kind': 'packed', 'i': 43062, 'nseq': 19531, 'stride': 1}}
        cpu = {'status': 'passed', 'extra': extra, 'optimizer_states': 132, 'optimizer_moments': 264,
               'trainable_tensors': 132, 'native_adam_export_byteexact': True, 'weights_only_byteexact': True}
        template = {'model_format': 'native overlay with MiniCPM5 frozen base', 'license': 'apache-2.0',
                    'tokenizer': {'files_sha256': {'tokenizer.json': publisher.digest(args.base / 'tokenizer.json')}},
                    'mixture': {'english_web': .6, 'chinese_web': .3, 'L2_math': .1, 'code': 0},
                    'data': {name: publisher.file_record(path) for name, path in (('train', args.data), ('eval', args.eval_data))},
                    'base_model': {'repo_id': 'openbmb/MiniCPM5-2B-Base',
                                   'model.safetensors': publisher.file_record(args.base / 'model.safetensors'),
                                   'auxiliary_sha256': {'config.json': publisher.digest(args.base / 'config.json')}},
                    'publication_source': {'training_continues': True, 'remaining_first_pass_updates': 1026},
                    'operator_adoption': {'updates_since_handoff': 164}, 'deployed_source_sha256': {'stale.py': 'old'}}
        publisher.write_json(args.template_release, template)
        return args, status, accepted, audit, quality, cpu

    def prepare_fixture(self, root):
        from operators.rocm import production_continuation, continuation_quality, run_operator_switch
        args, status, accepted, audit, quality, cpu = self.fixture(root)

        def export(_args, full, light):
            light.write_bytes(b'light overlay no optimizer')
            return cpu

        with patch.object(production_continuation, 'validate_accepted_run', return_value=accepted) as accepted_call, \
                patch.object(continuation_quality, 'validate_quality_report', return_value=quality), \
                patch.object(run_operator_switch, 'cpu_check', return_value=audit) as check, \
                patch.object(publisher, 'cpu_export', side_effect=export):
            prepared = publisher.prepare(args, status=status)
        return args, prepared, accepted_call, check

    def original_card(self):
        return ('---\nlicense: apache-2.0\nbase_model: openbmb/MiniCPM5-2B-Base\n---\n\n# CAT-YOKO\n\n'
                '## Spec\n\nOriginal specification retained.\n\n'
                '## Current B0 snapshot\n\nRecommended step 53307. Recovery1026 stale.\n\n'
                '## Current weights\n\nstep-53307 historical files.\n\n'
                '## Validation observation\n\nOld NLL data.\n\n## License\n\nLicense retained.\n')

    def test_completed_prepare_uses_real_counts_modulo_and_deployed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, prepared, accepted, check = self.prepare_fixture(Path(tmp))
            directory = args.stage_dir / prepared['checkpoint_prefix']
            release = publisher.read_json(directory / 'release.json')
            self.assertEqual(release['step'], 77864)
            self.assertEqual(release['stream']['next_unread_row'], 43062)
            self.assertEqual(release['stream']['next_row_mod_nseq'], 4000)
            self.assertEqual(release['real_text_tokens_processed'], 176381952)
            self.assertEqual(release['unique_training_tokens'], 79998976)
            self.assertEqual(release['completed_corpus_passes'], 2)
            self.assertEqual(release['operator_adoption']['verified_completed_updates'], 4000)
            self.assertEqual(release['operator_adoption']['counts']['cached_cpu_adam_calls']['parameter_updates'], 528000)
            self.assertNotIn('updates_since_handoff', release['operator_adoption'])
            self.assertNotIn('remaining_first_pass_updates', release['publication_source'])
            self.assertNotIn('stale.py', release['deployed_source_sha256'])
            self.assertEqual(release['latest_fixed_evaluation']['step'], 77864)
            self.assertEqual(release['completed_at_utc'], '2026-10-06T00:00:00Z')
            self.assertNotEqual(release['snapshot_at_utc'], release['completed_at_utc'])
            self.assertEqual(release['additional_fixed_evaluation']['eval_nll'], 7.3)
            self.assertTrue((directory / 'trainable.pt').samefile(Path(accepted.call_args.args[1])))
            self.assertEqual(accepted.call_count, 2)
            self.assertEqual(check.call_args.kwargs['max_cursor'], 58593)
            self.assertEqual(check.call_args.args[2], 4000)
            for name, file in prepared['files'].items():
                self.assertTrue(name.startswith(prepared['checkpoint_prefix'] + '/'))
                self.assertEqual(publisher.file_record(args.stage_dir / name), file)
            self.assertNotIn('cpu-check.log', prepared['files'])

    def test_prepare_retry_reuses_verified_inventory_without_reexporting(self):
        from operators.rocm import production_continuation, continuation_quality
        with tempfile.TemporaryDirectory() as tmp:
            args, prepared, first_validation, check = self.prepare_fixture(Path(tmp))
            status = publisher.read_json(args.run_dir / 'status.json')
            accepted = first_validation.return_value
            quality = publisher.read_json(args.run_dir / 'continuation/quality.json')
            with patch.object(production_continuation, 'validate_accepted_run', return_value=accepted), \
                    patch.object(continuation_quality, 'validate_quality_report', return_value=quality), \
                    patch.object(publisher, 'cpu_export') as export:
                self.assertEqual(publisher.prepare(args, status=status), prepared)
            export.assert_not_called()
            (args.stage_dir / prepared['checkpoint_prefix'] / 'trainable.pt').write_bytes(b'tampered')
            with patch.object(production_continuation, 'validate_accepted_run', return_value=accepted), \
                    patch.object(continuation_quality, 'validate_quality_report', return_value=quality), \
                    patch.object(publisher, 'cpu_export') as export, self.assertRaisesRegex(ValueError, 'changed'):
                publisher.prepare(args, status=status)
            export.assert_not_called()

    def test_standalone_json_reader_rejects_nonfinite_without_other_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'record.json'
            path.write_text('{"nested":{"values":[NaN]}}')
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                publisher.read_json(path)

    def test_unfinished_prior_or_failed_quality_never_stage_weights(self):
        from operators.rocm import production_continuation, continuation_quality
        for failure in ('unfinished', 'accepted', 'quality'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                args, status, accepted, audit, quality, cpu = self.fixture(Path(tmp))
                if failure == 'unfinished': status['status'] = 'running_continuation'
                with patch.object(production_continuation, 'validate_accepted_run', side_effect=ValueError('bad accepted') if failure == 'accepted' else None,
                                  return_value=accepted), \
                        patch.object(continuation_quality, 'validate_quality_report', side_effect=ValueError('bad quality') if failure == 'quality' else None,
                                     return_value=quality), patch.object(publisher, 'cpu_export') as export:
                    with self.assertRaises(ValueError): publisher.prepare(args, status=status)
                export.assert_not_called()
                self.assertFalse(args.stage_dir.exists())

    def test_wait_only_successful_complete_receipt_or_explicit_failure(self):
        pending = {'status': 'running_continuation', 'supervisor_pid': 10, 'child_pid': 11}
        complete = {'status': 'complete', 'supervisor_pid': 10, 'child_pid': None}
        with patch.object(publisher, 'read_json', side_effect=[pending, complete]):
            sleep = Mock()
            self.assertEqual(publisher.await_completion(Path('run'), wait=True, alive=lambda pid: True, sleep=sleep), complete)
            sleep.assert_called_once_with(30)
        for status in ({'status': 'failed'}, pending):
            with patch.object(publisher, 'read_json', return_value=status), self.assertRaises(ValueError):
                publisher.await_completion(Path('run'), wait=True, alive=lambda pid: False, sleep=Mock())
        with patch.object(publisher, 'read_json', return_value=pending), self.assertRaisesRegex(ValueError, 'not complete'):
            publisher.await_completion(Path('run'))
        with patch.object(publisher, 'read_json', return_value=pending), self.assertRaisesRegex(ValueError, 'timed out'):
            publisher.await_completion(Path('run'), wait=True, alive=lambda pid: True, clock=Mock(side_effect=[0, 2]), timeout_seconds=1)

    def test_root_card_replaces_stale_snapshot_and_preserves_yaml_spec_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, prepared, *_ = self.prepare_fixture(Path(tmp))
            release = publisher.read_json(args.stage_dir / prepared['checkpoint_prefix'] / 'release.json')
            old = self.original_card()
            card = publisher.latest_card(old, release)
            self.assertTrue(card.startswith(old.split('# CAT-YOKO')[0]))
            self.assertIn('Original specification retained.', card)
            self.assertIn('License retained.', card)
            self.assertIn('## Historical weights', card)
            self.assertIn('step-53307 historical files.', card)
            self.assertNotIn('Recovery1026 stale.', card)
            self.assertNotIn('Old NLL data.', card)
            self.assertIn('7.2', card); self.assertIn('7.3', card)
            self.assertEqual(publisher.latest_card(card, release), card)
            with self.assertRaisesRegex(ValueError, 'downgrade'):
                publisher.latest_card(old.replace('53307', '81864'), release)

    def test_publish_atomic_allowlist_pinned_verification_and_portable_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, prepared, *_ = self.prepare_fixture(root)
            prepared['stage_dir'] = '/remote/path/no/longer/accessible'
            hub = FakeHub(root, self.original_card())
            with patch.object(publisher, 'request', side_effect=lambda call: call()):
                result = publisher.publish(args, prepared, api=hub, download=hub.download, addition=SimpleNamespace)
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(result['commit'], 'c' * 40)
            self.assertEqual(len(hub.commit_calls), 1)
            parent, files = hub.commit_calls[0]
            self.assertEqual(parent, 'a' * 40)
            self.assertEqual(set(files), set(prepared['files']) | {'README.md'})
            self.assertEqual(publisher.read_json(args.stage_dir / 'publish.json')['status'], 'verified')
            self.assertIn((prepared['checkpoint_prefix'] + '/release.json', result['commit']), hub.download_calls)
            self.assertIn(('README.md', result['commit']), hub.download_calls)

    def test_cas_retry_reloads_root_card_preserving_upstream_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args, prepared, *_ = self.prepare_fixture(root)
            hub = FakeHub(root, self.original_card(), conflict=True)
            with patch.object(publisher, 'request', side_effect=lambda call: call()):
                result = publisher.publish(args, prepared, api=hub, download=hub.download, addition=SimpleNamespace)
            self.assertEqual([parent for parent, _ in hub.commit_calls], ['a' * 40, 'b' * 40])
            self.assertIn(b'Upstream history retained.', hub.revisions[result['commit']]['README.md'])

    def test_existing_immutable_checkpoint_mismatch_and_higher_step_reject(self):
        for mismatch in ('weights', 'manifest', 'higher'):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args, prepared, *_ = self.prepare_fixture(root)
                hub = FakeHub(root, self.original_card())
                if mismatch == 'weights':
                    name = prepared['checkpoint_prefix'] + '/trainable.pt'
                elif mismatch == 'manifest':
                    name = prepared['checkpoint_prefix'] + '/release.json'
                else:
                    name = publisher.FAMILY + '/step-81864/release.json'
                hub.revisions[hub.revision][name] = b'immutable other content'
                with patch.object(publisher, 'request', side_effect=lambda call: call()), self.assertRaises(ValueError):
                    publisher.publish(args, prepared, api=hub, download=hub.download, addition=SimpleNamespace)
                self.assertEqual(hub.commit_calls, [])

    def test_upstream_lfs_or_pinned_download_mismatch_never_verifies(self):
        for field in ('corrupt_lfs', 'corrupt_download'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args, prepared, *_ = self.prepare_fixture(root)
                hub = FakeHub(root, self.original_card(), **{field: True})
                with patch.object(publisher, 'request', side_effect=lambda call: call()), self.assertRaises(ValueError):
                    publisher.publish(args, prepared, api=hub, download=hub.download, addition=SimpleNamespace)
                self.assertFalse((args.stage_dir / 'publish.json').exists())

    def test_staged_tampering_or_escaping_allowlist_never_commits(self):
        for field in ('bytes', 'path'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); args, prepared, *_ = self.prepare_fixture(root)
                if field == 'bytes':
                    (args.stage_dir / prepared['checkpoint_prefix'] / 'trainable.pt').write_bytes(b'changed')
                else:
                    prepared['files']['elsewhere/secret'] = {'sha256': 'bad', 'bytes': 1}
                hub = FakeHub(root, self.original_card())
                with self.assertRaises(ValueError):
                    publisher.publish(args, prepared, api=hub, download=hub.download, addition=SimpleNamespace)
                self.assertEqual(hub.commit_calls, [])

    def test_publish_prepared_cli_requires_no_training_paths_and_sanitizes_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); args, prepared, *_ = self.prepare_fixture(root)
            with patch.object(publisher, 'publish', return_value={'status': 'verified', 'step': 77864,
                              'repo_id': args.repo_id, 'commit': 'c' * 40}) as publish, \
                    patch.object(publisher, 'prepare') as prepare, patch('builtins.print'):
                self.assertEqual(publisher.main(['--publish-prepared', '--stage-dir', str(args.stage_dir)]), 0)
            prepare.assert_not_called()
            self.assertEqual(publish.call_args.args[1], prepared)
            with patch.object(publisher, 'publish', side_effect=RuntimeError('signed_url=https://secret token=secret')), \
                    patch('builtins.print') as printed:
                self.assertEqual(publisher.main(['--publish-prepared', '--stage-dir', str(args.stage_dir)]), 1)
            self.assertNotIn('secret', str(printed.call_args))
            self.assertNotIn('secret', (args.stage_dir / 'publish.json').read_text())


if __name__ == '__main__':
    unittest.main()
