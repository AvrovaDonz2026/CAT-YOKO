"""Explicit repeated-pass bounds and live policy receipt checks."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operators.rocm import run_corpus_pass as controller
from operators.rocm.run_corpus_pass import completed_metrics, pass_plan


class CorpusPassTests(unittest.TestCase):
    def metadata(self, cursor=19531):
        return {'source_data_rows': 19531, 'source_stream_i': cursor,
                'source_adam_step': cursor, 'source_step': 34802 + cursor,
                'source_tokens_in_phase': 162244608 + cursor * 4096,
                'trainable_names': ['weight_' + str(index) for index in range(132)]}

    def test_second_pass_preserves_monotonic_counters(self):
        plan = pass_plan(self.metadata(), 39062)
        self.assertEqual(plan['updates'], 19531)
        self.assertEqual(plan['target_step'], 73864)
        self.assertEqual(plan['target_adam_step'], 39062)
        self.assertEqual(plan['target_tokens_in_phase'], 322242560)
        self.assertTrue(plan['repeated_corpus'])
        self.assertEqual(plan['start_next_row'], 0)
        self.assertEqual(plan['audit_max_cursor'], 39062)

    def test_recovery_inside_second_pass_reaches_same_boundary(self):
        plan = pass_plan(self.metadata(cursor=20000), 39062)
        self.assertEqual(plan['updates'], 19062)
        self.assertEqual(plan['target_step'], 73864)
        self.assertEqual(plan['start_next_row'], 469)

    def test_unbounded_or_wrong_pass_end_rejects(self):
        for target in (19531, 39061, 39063, 58593, -1, True, 39062.0):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pass_plan(self.metadata(), target)

    def test_explicit_4000_updates_keeps_whole_corpus_audit_cap(self):
        plan = pass_plan(self.metadata(cursor=39062), 43062, run_updates=4000)
        self.assertEqual(plan['updates'], 4000)
        self.assertEqual(plan['target_step'], 77864)
        self.assertEqual(plan['target_cursor'], 43062)
        self.assertEqual(plan['target_adam_step'], 43062)
        self.assertEqual(plan['target_tokens_in_phase'], 338626560)
        self.assertEqual(plan['audit_max_cursor'], 58593)
        self.assertEqual(plan['target_next_row'], 4000)

    def test_recovery_within_window_keeps_exact_requested_endpoint(self):
        plan = pass_plan(self.metadata(cursor=40000), 43062, run_updates=3062)
        self.assertEqual(plan['updates'], 3062)
        self.assertEqual(plan['target_step'], 77864)
        self.assertEqual(plan['target_adam_step'], 43062)
        self.assertEqual(plan['start_next_row'], 938)
        self.assertEqual(plan['audit_max_cursor'], 58593)

    def test_explicit_updates_require_positive_integer_and_bounded_endpoint(self):
        for updates in (0, -1, True, 4000.0, '4000', 19532):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                pass_plan(self.metadata(cursor=39062), 43062, run_updates=updates)
        for target in (43061, 43063, 58594, 43062.0, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pass_plan(self.metadata(cursor=39062), target, run_updates=4000)
        boundary = pass_plan(self.metadata(cursor=39062), 58593, run_updates=19531)
        self.assertEqual(boundary['target_cursor'], boundary['audit_max_cursor'])

    def test_default_still_requires_complete_next_pass(self):
        with self.assertRaisesRegex(ValueError, 'next exact'):
            pass_plan(self.metadata(cursor=39062), 43062)
        plan = pass_plan(self.metadata(cursor=39062), 58593)
        self.assertEqual(plan['updates'], 19531)
        self.assertEqual(plan['audit_max_cursor'], 58593)

    def test_counter_disagreement_and_b0_budget_reject(self):
        for change in ({'source_adam_step': 0}, {'source_tokens_in_phase': 8e9 - 4096}):
            metadata = self.metadata(); metadata.update(change)
            with self.assertRaises(ValueError):
                pass_plan(metadata, 39062)

    def row(self, step=54334):
        return {'step': step, 'tok_s': 750, 'loss': 7.8,
                'split_attention': True, 'cached_cpu_adam': True,
                'actual_optimizer_device': 'cpu', 'deterministic_training': False,
                'deterministic_algorithms': False}

    def supervisor_args(self, root, *, cursor=19531, target=39062, **options):
        out = root / 'run'
        (out / 'source_checkpoint').mkdir(parents=True)
        source = root / 'source'; source.mkdir()
        base = root / 'base'; base.mkdir()
        corpus = root / 'data'; corpus.mkdir()
        data = corpus / 'train.bin'
        with data.open('wb') as stream:
            stream.truncate(19531 * 4096 * 4)
        heldout = corpus / 'eval.bin'; heldout.write_bytes(b'heldout')
        resume = out / 'source_checkpoint/trainable.pt'; resume.write_bytes(b'fixed source')
        args = SimpleNamespace(source_dir=source, base=base, data=data, eval_data=heldout,
                               out=out, resume=resume, lock_file=root / 'training.lock',
                               target_cursor=target, run_updates=None, python='test-python',
                               reference_backend='native', native_failure_receipt=None,
                               accepted_run=None)
        args.__dict__.update(options)
        return args

    def accepted_receipt(self, root, args):
        prior = root / 'prior'; prior.mkdir()
        receipt = {'source_previous_status': 'complete'}
        for key, relative in (('status_path', 'status.json'), ('parity_path', 'parity.json'),
                              ('operators_path', 'operators.json'), ('round3_operators_path', 'round3_operators.json'),
                              ('checkpoint_path', 'trainable_73864.pt'),
                              ('prior_source_checkpoint', 'source_trainable.pt')):
            path = prior / relative
            path.write_text('{"status":"complete"}')
            receipt[key] = str(path)
        args.accepted_run = prior
        return receipt

    def test_live_reader_ignores_only_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.jsonl'
            path.write_text(json.dumps(self.row()) + '\n' + '{"step":54335')
            self.assertEqual(len(completed_metrics(path, 54333)), 1)

    def test_policy_change_or_skipped_update_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'metrics.jsonl'
            for mutation in ({'step': 54335}, {'split_attention': False},
                             {'cached_cpu_adam': False}, {'deterministic_algorithms': True},
                             {'actual_optimizer_device': 'cuda'}, {'loss': float('nan')}):
                row = self.row(); row.update(mutation)
                path.write_text(json.dumps(row) + '\n')
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    completed_metrics(path, 54333)

    def test_first_running_status_write_failure_cleans_up_created_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root)
            out = args.out
            child = Mock(pid=12345)
            original_write = controller.checks.write_json

            def write_status(path, value):
                if path.name == 'status.json' and value.get('status') == 'running_continuation':
                    raise OSError('injected first running status failure')
                return original_write(path, value)

            with patch.object(controller, 'protected_sources', return_value=(Mock(), 'fixed-sha')), \
                    patch.object(controller.checks, 'cpu_check', return_value=self.metadata()), \
                    patch.object(controller, 'wait_for_gpu_idle'), \
                    patch.object(controller.checks, 'write_json', side_effect=write_status), \
                    patch.object(controller.subprocess, 'Popen', return_value=child) as launch, \
                    patch.object(controller.round3, 'stop_owned_child') as cleanup, \
                    patch('builtins.print'):
                with self.assertRaisesRegex(OSError, 'first running status failure'):
                    controller.supervise(args)
            launch.assert_called_once()
            cleanup.assert_called_once_with(child)
            child.poll.assert_not_called()
            self.assertTrue((out / 'continuation.command.json').is_file())

    def test_window_supervisor_passes_whole_cap_and_accepted_run_to_entry(self):
        from operators.rocm import production_continuation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root, cursor=39062, target=43062, run_updates=4000,
                                        reference_backend='previous_packed_production')
            receipt = self.accepted_receipt(root, args)
            child = Mock(pid=12345); child.poll.return_value = 0; child.wait.return_value = 0
            final = args.out / 'continuation/train/trainable.pt'
            with patch.object(production_continuation, 'validate_accepted_run', return_value=receipt, create=True) as accepted, \
                    patch.object(controller.checks, 'cpu_check', side_effect=[self.metadata(39062), self.metadata(39062), self.metadata(43062)]) as cpu, \
                    patch.object(controller, 'wait_for_gpu_idle'), \
                    patch.object(controller.subprocess, 'Popen', return_value=child) as launch, \
                    patch.object(controller.round3, 'stop_owned_child'), \
                    patch.object(controller.checks, 'validate_run', return_value={'checkpoint': str(final), 'updates': 4000}), \
                    patch.object(controller.round3, 'validate_round3_report'), \
                    patch.object(controller.round3, 'validate_training_policy'), patch('builtins.print'):
                state = controller.supervise(args)
            accepted.assert_called_once_with(args.accepted_run, args.resume, args.source_dir)
            self.assertEqual([call.args[2] for call in cpu.call_args_list], [0, 4000, 4000])
            self.assertEqual([call.kwargs['max_cursor'] for call in cpu.call_args_list], [58593] * 3)
            self.assertEqual(state['status'], 'complete')
            self.assertEqual(state['target_cursor'], 43062)
            self.assertEqual(state['audit_max_cursor'], 58593)
            command = launch.call_args.args[0]
            self.assertEqual(command[2], str(args.source_dir / 'operators/rocm/production_continuation.py'))
            self.assertEqual(command[command.index('--run-steps') + 1], '4000')
            self.assertEqual(command[command.index('--accepted-run') + 1], str(args.accepted_run))
            self.assertNotIn('--native-failure-receipt', command)
            self.assertEqual(command[command.index('--save-every-seconds') + 1], '300')
            self.assertEqual(command[command.index('--keep-last') + 1], '3')

    def test_checkpoint_inside_window_uses_same_whole_corpus_cap(self):
        from operators.rocm import production_continuation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root, cursor=39062, target=43062, run_updates=4000,
                                        reference_backend='previous_packed_production')
            receipt = self.accepted_receipt(root, args)
            child = Mock(pid=12345); child.poll.side_effect = [None, 0]; child.wait.return_value = 0
            final = args.out / 'continuation/train/trainable.pt'

            def launch_child(*unused, **kwargs):
                final.parent.mkdir(parents=True)
                final.write_bytes(b'periodic snapshot')
                os.link(final, final.parent / 'trainable_step_73865.pt')
                (final.parent / 'metrics.jsonl').write_text(json.dumps(self.row(73865)) + '\n')
                return child

            with patch.object(production_continuation, 'validate_accepted_run', return_value=receipt, create=True), \
                    patch.object(controller.checks, 'cpu_check', side_effect=[self.metadata(39062), self.metadata(39062), self.metadata(39063), self.metadata(43062)]) as cpu, \
                    patch.object(controller, 'wait_for_gpu_idle'), patch.object(controller.time, 'sleep'), \
                    patch.object(controller.subprocess, 'Popen', side_effect=launch_child), \
                    patch.object(controller.round3, 'stop_owned_child'), \
                    patch.object(controller.checks, 'validate_run', return_value={'checkpoint': str(final), 'updates': 4000}), \
                    patch.object(controller.round3, 'validate_round3_report'), \
                    patch.object(controller.round3, 'validate_training_policy'), patch('builtins.print'):
                state = controller.supervise(args)
            self.assertEqual([call.args[2] for call in cpu.call_args_list], [0, 4000, 1, 4000])
            self.assertEqual([call.kwargs['max_cursor'] for call in cpu.call_args_list], [58593] * 4)
            self.assertTrue(state['first_checkpoint_verified'])
            self.assertEqual(state['latest_verified_checkpoint']['source_stream_i'], 39063)
            self.assertTrue((args.out / 'verified_checkpoint/latest_verified.pt').is_file())

    def test_extra_quality_is_forwarded_and_must_bind_this_completed_window(self):
        from operators.rocm import production_continuation
        for outcome in ('complete', 'rejected', 'wrong_step', 'wrong_corpus'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = self.supervisor_args(root, cursor=39062, target=43062, run_updates=4000,
                                            reference_backend='previous_packed_production',
                                            extra_heldout_start_row=32, extra_heldout_batches=32)
                receipt = self.accepted_receipt(root, args)
                quality = {'eval_data': str(args.eval_data),
                           'eval_data_sha256': controller.checks.digest_file(args.eval_data),
                           'initial_eval': {'step': 73864, 'pass': True},
                           'final_eval': {'step': 77864, 'pass': True}}
                if outcome == 'wrong_step':
                    quality['final_eval']['step'] += 1
                if outcome == 'wrong_corpus':
                    quality['eval_data_sha256'] = 'wrong-sha'
                child = Mock(pid=12345); child.poll.return_value = 0; child.wait.return_value = 0
                final = args.out / 'continuation/train/trainable.pt'
                quality_path = args.out / 'continuation/quality.json'

                def launch_child(*unused, **kwargs):
                    quality_path.parent.mkdir(parents=True)
                    quality_path.write_text(json.dumps(quality))
                    return child

                module = ModuleType('operators.rocm.continuation_quality')
                validate_quality = Mock(return_value=quality)
                if outcome == 'rejected':
                    validate_quality.side_effect = ValueError('quality module SHA mismatch or incomplete evaluation')
                module.validate_quality_report = validate_quality
                with patch.dict(sys.modules, {module.__name__: module}), \
                        patch.object(production_continuation, 'validate_accepted_run', return_value=receipt, create=True), \
                        patch.object(controller.checks, 'cpu_check', side_effect=[self.metadata(39062), self.metadata(39062), self.metadata(43062)]), \
                        patch.object(controller, 'wait_for_gpu_idle'), \
                        patch.object(controller.subprocess, 'Popen', side_effect=launch_child) as launch, \
                        patch.object(controller.round3, 'stop_owned_child'), \
                        patch.object(controller.checks, 'validate_run', return_value={'checkpoint': str(final), 'updates': 4000}), \
                        patch.object(controller.round3, 'validate_round3_report'), \
                        patch.object(controller.round3, 'validate_training_policy'), patch('builtins.print'):
                    if outcome == 'complete':
                        state = controller.supervise(args)
                        self.assertTrue(state['final_quality_done'])
                    else:
                        with self.assertRaises(ValueError):
                            controller.supervise(args)
                validate_quality.assert_called_once_with(quality_path.resolve(), source_dir=args.source_dir, start_row=32, batches=32)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index('--extra-heldout-start-row') + 1], '32')
                self.assertEqual(command[command.index('--extra-heldout-batches') + 1], '32')
                self.assertEqual(command[command.index('--eval-batches') + 1], '32')

    def test_invalid_extra_quality_options_reject_before_any_work(self):
        for start, batches, backend in ((31, 32, 'previous_packed_production'),
                                        (True, 32, 'previous_packed_production'),
                                        (32.0, 32, 'previous_packed_production'),
                                        (32, True, 'previous_packed_production'),
                                        (32, 31, 'previous_packed_production'),
                                        (32, 32, 'native')):
            with self.subTest(start=start, batches=batches, backend=backend), self.assertRaises(ValueError):
                controller.supervise(SimpleNamespace(reference_backend=backend, run_updates=4000,
                                                      extra_heldout_start_row=start, extra_heldout_batches=batches))

    def test_failed_prior_run_rejects_before_cpu_gpu_or_child(self):
        from operators.rocm import production_continuation
        with tempfile.TemporaryDirectory() as tmp:
            args = self.supervisor_args(Path(tmp), cursor=39062, target=43062, run_updates=4000,
                                        reference_backend='previous_packed_production')
            self.accepted_receipt(Path(tmp), args)
            with patch.object(production_continuation, 'validate_accepted_run', side_effect=ValueError('prior run is not complete'), create=True), \
                    patch.object(controller.checks, 'cpu_check') as cpu, \
                    patch.object(controller, 'wait_for_gpu_idle') as wait, \
                    patch.object(controller.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'prior run is not complete'):
                    controller.supervise(args)
            cpu.assert_not_called(); wait.assert_not_called(); launch.assert_not_called()

    def test_accepted_run_protects_reports_and_numbered_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root)
            args.accepted_run_receipt = self.accepted_receipt(root, args)
            for key in ('status_path', 'parity_path', 'operators_path', 'round3_operators_path',
                        'checkpoint_path', 'prior_source_checkpoint'):
                verify, _ = controller.protected_sources(args)
                path = Path(args.accepted_run_receipt[key])
                old = path.read_bytes(); path.write_bytes(old + b'changed')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'protected'):
                    verify()
                path.write_bytes(old)

    def test_accepted_run_protects_historical_math_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root)
            receipt = self.accepted_receipt(root, args)
            prior_source = root / 'prior_source'; prior_source.mkdir()
            module = prior_source / 'attention.py'; module.write_text('accepted_math = True\n')
            receipt.update(prior_source_dir=str(prior_source),
                           math_file_sha256={'attention.py': controller.checks.digest_file(module)})
            args.accepted_run_receipt = receipt
            verify, _ = controller.protected_sources(args)
            module.write_text('accepted_math = False\n')
            with self.assertRaisesRegex(ValueError, 'protected'):
                verify()

    def test_production_reference_requires_exactly_one_receipt(self):
        for options in ({'reference_backend': 'previous_packed_production'},
                        {'reference_backend': 'previous_packed_production', 'native_failure_receipt': Path('failure'),
                         'accepted_run': Path('prior')},
                        {'reference_backend': 'native', 'accepted_run': Path('prior')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                controller.supervise(SimpleNamespace(run_updates=4000, **options))

    def test_old_failure_receipt_cannot_bind_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.supervisor_args(root, cursor=39062, target=43062, run_updates=4000,
                                        reference_backend='previous_packed_production')
            old_source = root / 'old_source.pt'; old_source.write_bytes(b'old source 54333')
            failure = root / 'native_failure.json'
            failure.write_text(json.dumps({'status': 'parity_failure', 'updates': 0, 'source_step': 54333,
                                           'source_checkpoint': str(old_source),
                                           'events': [{'event': 'parity', 'pass': False}]}))
            args.native_failure_receipt = failure
            with patch.object(controller.checks, 'cpu_check') as cpu, \
                    patch.object(controller, 'wait_for_gpu_idle') as wait, patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'unchanged source'):
                    controller.supervise(args)
            cpu.assert_not_called(); wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
