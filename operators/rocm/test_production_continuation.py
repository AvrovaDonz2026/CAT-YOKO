"""Exercise oracle scope and failure preservation without a GPU runtime."""
from contextlib import contextmanager
import json
import copy
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import operators.rocm as rocm
from operators.rocm import production_continuation as continuation
from operators.rocm.test_round3_candidate_bench import runner_harness


class ProductionContinuationTests(unittest.TestCase):
    @contextmanager
    def harness(self, root, *, capture_failure=False, parity_failure=False):
        with runner_harness() as (entry, parent, model, trainer, state):
            calls = []

            def capture(_model, batch, *, device, offload):
                calls.append((offload, list(state.active)))
                if capture_failure:
                    raise RuntimeError('injected reference capture failure')
                if 'packed' in state.active:
                    state.packed['optimized_calls'] += 1
                return {'seq_len': 4096, 'gradient_tensors': 132}, {'gradients': 'unchanged snapshot'}

            model.capture = capture
            original_main, original_emit = model.main, model.emit

            def emit(path, report, event):
                if event.get('event') == 'parity_complete':
                    # The real model runner emits this after the installer;
                    # the lightweight inherited harness omits that event.
                    model.emit(path, report, {'event': 'moe_layout_installed', 'experimental_operators': {
                        'patches_installed_after_native_reference': True}})
                    model.capture(object(), {}, device='cuda', offload=False)
                original_emit(path, report, event)
                path.write_text(json.dumps(report))

            model.emit = emit

            def run(argv):
                args = model.build_parser().parse_args(argv)
                args.out.mkdir(parents=True, exist_ok=True)
                model.capture(object(), {}, device='cuda', offload=True)
                model.capture(object(), {}, device='cuda', offload=True)
                if parity_failure:
                    model.install_moe_layout(object(), args.moe_layout)
                    model.emit(args.out / 'parity.json', {'status': 'parity_failure'},
                               {'event': 'parity_complete', 'pass': False, 'failed_gradients': ['q_norm']})
                    return 1
                return original_main(argv)

            model.main = run
            with patch.object(rocm, 'round3_candidate_bench', entry, create=True):
                yield model, state, calls

    def argv(self, root):
        train, heldout = root / 'train.bin', root / 'eval.bin'
        train.write_bytes(b'train'); heldout.write_bytes(b'heldout')
        values = ['--base', str(root / 'base'), '--resume', str(root / 'source.pt'),
                  '--out', str(root / 'out'), '--data', str(train), '--eval-data', str(heldout),
                  '--moe-layout', 'shared-storage', '--packed-attention', '--deterministic-parity',
                  '--reference-repeat', '--save-optim', '--parity-seqs', '4096', '--run-steps', '2']
        (root / 'source.pt').write_bytes(b'complete immutable checkpoint')
        return values + ['--split-attention', '--cached-cpu-adam']

    def test_reference_scope_restores_before_candidate_and_records_truthful_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.argv(root)
            with self.harness(root) as (model, state, calls):
                original_capture, original_emit = model.capture, model.emit
                self.assertEqual(continuation.main(argv), 0)
                self.assertIs(model.capture, original_capture)
                self.assertIs(model.emit, original_emit)
                self.assertEqual(calls, [(True, ['packed']), (True, ['packed']),
                                         (False, ['packed', 'split', 'cached'])])
                self.assertEqual(state.active, [])
                self.assertEqual(state.creations, [['packed', 'split', 'cached']])
                self.assertEqual(state.run_args.grad_relative_l2, 0.05)
                self.assertEqual(state.run_args.loss_atol, 0.02)
                self.assertEqual(state.run_args.output_relative_l2, 0.05)
                layout = next(event for event in state.events if isinstance(event, dict)
                              and event.get('event') == 'moe_layout_installed')
                self.assertFalse(layout['experimental_operators']['patches_installed_after_native_reference'])
                self.assertTrue(layout['experimental_operators']['patches_installed_after_packed_production_reference'])
            for name in ('operators.json', 'round3_operators.json'):
                report = json.loads((root / 'out' / name).read_text())
                self.assertEqual(report['status'], 'completed')
                self.assertFalse(report['patches_installed_after_native_reference'])
                self.assertTrue(report['patches_installed_after_packed_production_reference'])
                self.assertEqual(report['reference_backend'], continuation.REFERENCE_BACKEND)
                self.assertEqual(len(report['production_reference_captures']), 2)
                self.assertIn(continuation.WRAPPER_SOURCE, report['file_sha256'])
                continuation.validate_reference_evidence(report, source_dir=continuation.Path(__file__).resolve().parents[2],
                                                          source=root / 'source.pt')

    def test_reference_exception_restores_capture_context_and_never_installs_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.argv(root)
            with self.harness(root, capture_failure=True) as (model, state, calls):
                original_capture, original_emit = model.capture, model.emit
                with self.assertRaisesRegex(RuntimeError, 'reference capture failure'):
                    continuation.main(argv)
                self.assertIs(model.capture, original_capture)
                self.assertIs(model.emit, original_emit)
                self.assertEqual(state.active, [])
                self.assertEqual(state.creations, [])
                self.assertEqual(calls, [(True, ['packed'])])

    def test_failed_numeric_gate_and_original_dense_receipt_are_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = self.argv(root)
            receipt = root / 'dense-failure.json'
            payload = json.dumps({'status': 'parity_failure', 'failed_gradients': ['q_norm', 'k_norm']}).encode()
            receipt.write_bytes(payload)
            with self.harness(root, parity_failure=True):
                self.assertEqual(continuation.main(argv + ['--native-failure-receipt', str(receipt)]), 1)
            report = json.loads((root / 'out/parity.json').read_text())
            self.assertEqual(report['status'], 'parity_failure')
            self.assertIs(report['events'][-1]['pass'], False)
            self.assertEqual(receipt.read_bytes(), payload)
            self.assertEqual(report['native_failure_receipt']['sha256'], continuation.digest(receipt))

    def test_reference_repeat_and_nested_source_metadata_use_actual_oracle(self):
        model = types.SimpleNamespace(capture=lambda *a, **k: None,
                                      emit=lambda path, report, event: event)
        metadata = {'patches_installed_after_packed_production_reference': True}
        report = {'source_code_version': {'experimental_operators': {
            'patches_installed_after_native_reference': True}, 'round3_operators': {'split_attention': True}}}
        with continuation.reference_context(model, metadata, tile_size=256, packed_context=None):
            event = model.emit(Path('unused'), report, {'event': 'reference_repeat', 'moe_layout': 'native-reference-repeat'})
        self.assertEqual(event['moe_layout'], 'previous-packed-production-reference-repeat')
        for value in report['source_code_version'].values():
            self.assertEqual(value['reference_backend'], continuation.REFERENCE_BACKEND)
            self.assertFalse(value['patches_installed_after_native_reference'])
            self.assertTrue(value['patches_installed_after_packed_production_reference'])

    def test_accepted_provenance_is_forwarded_as_proof_without_fabricating_dense_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = {'source_previous_status': 'complete', 'checkpoint_sha256': 'verified-state-sha',
                     'source_update_params': {'updates': 19531, 'final_step': 73864, 'final_cursor': 39062}}
            with self.harness(root) as (_, state, _), \
                    patch.object(continuation, 'validate_accepted_run', return_value=proof) as validate:
                self.assertEqual(continuation.main(self.argv(root) + ['--accepted-run', str(root / 'prior')]), 0)
            validate.assert_called_once_with(root / 'prior', root / 'source.pt', Path(continuation.__file__).resolve().parents[2])
            report = json.loads((root / 'out/parity.json').read_text())
            self.assertEqual(report['accepted_run'], proof)
            self.assertNotIn('native_failure_receipt', report)
            self.assertTrue(any(event.get('event') == 'accepted_run_verified' for event in report['events']))
            self.assertFalse(any(isinstance(event, dict) and event.get('event') == 'native_failure'
                                 for event in state.events))

    def test_optional_quality_context_is_separate_from_legacy_args_and_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, activity = Path(tmp), []

            @contextmanager
            def quality(model, out, *, eval_row_start, eval_batches):
                activity.append(('enter', out, eval_row_start, eval_batches))
                try:
                    yield
                finally:
                    activity.append(('exit',))

            module = types.SimpleNamespace(quality_context=quality)
            original_digest = continuation.digest
            def digest(filename):
                return 'quality-source-sha' if str(filename).endswith(continuation.QUALITY_SOURCE) else original_digest(filename)
            with self.harness(root) as (_, state, _), \
                    patch.dict('sys.modules', {'operators.rocm.continuation_quality': module}), \
                    patch.object(continuation, 'digest', side_effect=digest):
                argv = self.argv(root) + ['--extra-heldout-start-row=32', '--extra-heldout-batches', '32']
                self.assertEqual(continuation.main(argv), 0)
                self.assertFalse(hasattr(state.run_args, 'extra_heldout_start_row'))
            self.assertEqual(activity, [('enter', root / 'out', 32, 32), ('exit',)])
            report = json.loads((root / 'out/round3_operators.json').read_text())
            self.assertEqual(report['quality_file_sha256'], {continuation.QUALITY_SOURCE: 'quality-source-sha'})
            self.assertEqual(set(report['reference_file_sha256']), {continuation.WRAPPER_SOURCE, continuation.PACKED_SOURCE})
            self.assertNotIn(continuation.QUALITY_SOURCE, report['file_sha256'])


class AcceptedProductionRunTests(unittest.TestCase):
    """Use the real stdlib validators on a full completed 19,531-update receipt."""
    def setUp(self):
        from operators.rocm import run_round3_switch as round3
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.prior_source, self.current_source = self.root / 'old-source', self.root / 'new-source'
        self.run = self.root / 'accepted'
        self.directory = self.run / 'continuation'
        self.train = self.directory / 'train'
        self.train.mkdir(parents=True)
        core = {'cat_yoko/' + name for name in ('model.py', 'trainer.py', 'checkpoint.py', 'optim.py',
                                                'moe.py', 'data.py', 'attention.py', 'loss.py', 'c.py')}
        for relative in core | continuation.MATH_OPERATOR_SOURCES | {continuation.WRAPPER_SOURCE}:
            for source in (self.prior_source, self.current_source):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('accepted mathematics: ' + relative)
        # A future provenance/audit wrapper may evolve, never the math source.
        (self.current_source / continuation.WRAPPER_SOURCE).write_text('new audit wrapper')
        self.original = self.run / 'source_checkpoint/trainable.pt'
        self.original.parent.mkdir()
        self.original.write_bytes(b'prior complete initial state')
        self.fixed = self.root / 'new-fixed.pt'
        self.fixed.write_bytes(b'accepted complete final state')
        self.numbered = self.train / 'trainable_step_73864.pt'
        self.numbered.write_bytes(self.fixed.read_bytes())
        os.link(self.numbered, self.train / 'trainable.pt')
        self.names = [f'p{index}' for index in range(132)]
        self.updates = 19531
        def metadata(path, step, cursor, tokens):
            return {'source_checkpoint': str(path), 'source_step': step, 'source_stream_i': cursor,
                    'source_adam_step': cursor, 'source_data_rows': self.updates,
                    'source_tokens_in_phase': tokens, 'source_tokens_seen': tokens,
                    'optimizer_states': 132, 'optimizer_moments': 264, 'checkpoint_verified': True,
                    'trainable_names': self.names, 'trainable_shapes': {name: [2, 2] for name in self.names}}
        previous = metadata(self.original, 54333, 19531, 242243584)
        final = metadata(self.train / 'trainable.pt', 73864, 39062, 322242560)
        self.status = {'status': 'complete', 'child_pid': None, 'reference_backend': continuation.REFERENCE_BACKEND,
                       'phase': 'B0', 'cursor_reset': False, 'optimizer_reset': False, 'rng_reset': False,
                       'source_dir': str(self.prior_source), 'source_checkpoint': str(self.original),
                       'source_sha256': continuation.digest(self.original), 'source_metadata': previous,
                       'final_checkpoint_verified': final, 'target_step': 73864, 'target_cursor': 39062,
                       'plan': {'updates': self.updates, 'target_step': 73864, 'target_cursor': 39062,
                                'target_adam_step': 39062, 'target_tokens_in_phase': 322242560}}
        def hashes(names):
            return {name: continuation.digest(self.prior_source / name) for name in names}
        reference = {'reference_backend': continuation.REFERENCE_BACKEND,
                     'patches_installed_after_native_reference': False,
                     'patches_installed_after_packed_production_reference': True,
                     'production_reference_context_restored': True,
                     'reference_file_sha256': hashes({continuation.WRAPPER_SOURCE, continuation.PACKED_SOURCE}),
                     'reference_checkpoint_sha256': continuation.digest(self.original),
                     'production_reference_captures': [dict(status='completed', seq_len=4096,
                         offload_blocks=True, gradient_tensors=132, packed_attention_calls={'optimized_calls': 42}) for _ in range(2)]}
        comparison = {'seq_len': 4096, 'pass': True, 'failed_gradients': [], 'loss_abs': 0.0,
                      'global_gradient_relative_l2': 0.0, 'gradients': {name: {'pass': True, 'finite': True,
                          'relative_l2': 0.0} for name in self.names},
                      'selected_logits': {'finite': True, 'relative_l2': 0.0},
                      'final_hidden': {'finite': True, 'relative_l2': 0.0}}
        native_inventory = core - {'cat_yoko/model.py', 'cat_yoko/c.py'} | {
            'operators/rocm/model_bench.py', 'operators/rocm/shared_storage_moe.py'}
        self.parity = {**reference, 'status': 'training_complete', 'source_checkpoint': str(self.original),
                       'source_step': 54333, 'updates': self.updates, 'data': str(self.root / 'train.bin'),
                       'eval_data': str(self.root / 'eval.bin'), 'eos_id': 1, 'eval_eos_id': 1, 'dtype': 'bf16',
                       'moe_layout': 'shared-storage', 'parity_sequences': [4096], 'source_stream_kind': 'packed',
                       'eval_batches': 32, 'source_optimizer_present': True, 'save_optimizer': True,
                       'deterministic_parity': True, 'reference_repeat': True, 'original_deterministic_algorithms': False,
                       'source_code_version': {'file_sha256': hashes(native_inventory)},
                       'initial_eval': {'pass': True, 'eval_nll': 7.87}, 'final_eval': {'pass': True, 'eval_nll': 7.49},
                       'thresholds': {'loss_atol': 0.02, 'gradient_relative_l2_per_tensor_and_global': 0.05,
                                      'selected_output_relative_l2': 0.05},
                       'events': [{**comparison, 'event': 'reference_repeat'}, {**comparison, 'event': 'parity'},
                                  {'event': 'training_complete', 'step': 73864, 'updates': self.updates,
                                   'optimizer_restored': True, 'source_optimizer_present': True, 'save_optimizer': True,
                                   'deterministic_training': False, 'deterministic_algorithms': False}]}
        self.operators = {**reference, 'status': 'completed', 'packed_attention': True, 'batched_grad_norm': False,
                          'packed_attention_parity_calls': {'optimized_calls': 42},
                          'packed_attention_calls': {'optimized_calls': 1000},
                          'file_sha256': hashes({'operators/rocm/candidate_bench.py', continuation.PACKED_SOURCE,
                                                continuation.WRAPPER_SOURCE})}
        self.round3 = {**reference, 'status': 'completed', 'split_attention': True, 'cached_cpu_adam': True,
                       'sync_update_timing': False, 'full_model_parity_passed': True,
                       'deterministic_training': False, 'deterministic_algorithms': False,
                       'file_sha256': hashes(round3.ROUND3_SOURCES | {
                           'operators/rocm/split_attention.py', 'operators/rocm/cpu_adam_cached.py', continuation.WRAPPER_SOURCE}),
                       'split_attention_parity_calls': {'split_optimized_calls': 1},
                       'cached_cpu_adam_calls': {'state_loads': 1, 'gpu_parameter_updates': 132 * self.updates,
                           'parameter_updates': 132 * self.updates, 'fallback_parameter_updates': 0,
                           'cache_reuses': 132 * (self.updates - 1), 'checkpoint_format_changed': False,
                           'moment_dtype': 'float32'}}
        self.metrics = [dict(step=54334 + i, tok_s=750, deterministic_training=False,
                             deterministic_algorithms=False) for i in range(self.updates)]
        self.publish()

    def publish(self):
        for path, value in ((self.run / 'status.json', self.status), (self.directory / 'parity.json', self.parity),
                            (self.directory / 'operators.json', self.operators),
                            (self.directory / 'round3_operators.json', self.round3)):
            path.write_text(json.dumps(value))
        (self.train / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.metrics))

    def validate(self):
        return continuation.validate_accepted_run(self.run, self.fixed, self.current_source)

    def test_complete_production_run_binds_real_validators_final_sha_and_unchanged_math(self):
        before = {path: path.read_bytes() for path in self.directory.glob('*.json')}
        receipt = self.validate()
        self.assertEqual(receipt['source_previous_status'], 'complete')
        self.assertEqual(receipt['source_update_params']['updates'], 19531)
        self.assertEqual(receipt['source_update_params']['final_cursor'], 39062)
        self.assertEqual(receipt['checkpoint_sha256'], continuation.digest(self.fixed))
        self.assertEqual(receipt['training_policy']['consecutive_metric_updates'], 19531)
        self.assertEqual(receipt['checkpoint_path'], str(self.numbered))
        self.assertNotIn(continuation.WRAPPER_SOURCE, receipt['math_file_sha256'])
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        json.dumps(receipt, allow_nan=False)

    def test_different_state_or_math_never_accepted_even_when_prior_reports_pass(self):
        self.fixed.write_bytes(b'another state')
        with self.assertRaisesRegex(ValueError, 'state SHA'):
            self.validate()
        self.fixed.write_bytes(self.numbered.read_bytes())
        for name in ('cat_yoko/model.py', 'operators/rocm/split_attention.py'):
            filename = self.current_source / name
            original = filename.read_bytes()
            filename.write_bytes(b'changed math')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'mathematics'):
                self.validate()
            filename.write_bytes(original)

    def test_incomplete_or_changed_gates_and_optimizer_counts_reject(self):
        original = copy.deepcopy((self.status, self.parity, self.round3, self.metrics))
        mutations = [lambda: self.status.update(status='running_continuation'),
                     lambda: self.status['final_checkpoint_verified'].update(optimizer_moments=263),
                     lambda: self.status['final_checkpoint_verified'].update(source_adam_step=39061),
                     lambda: self.parity['events'][1]['gradients'].pop('p131'),
                     lambda: self.parity['events'][1]['gradients']['p0'].update(relative_l2=0.051),
                     lambda: self.round3['cached_cpu_adam_calls'].update(parameter_updates=132 * 19531 - 1),
                     lambda: self.metrics.pop(),
                     lambda: self.metrics[-1].update(deterministic_algorithms=True)]
        for mutation in mutations:
            self.status, self.parity, self.round3, self.metrics = copy.deepcopy(original)
            mutation(); self.publish()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate()


if __name__ == '__main__':
    unittest.main()
