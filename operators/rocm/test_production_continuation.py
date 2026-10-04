"""Exercise oracle scope and failure preservation without a GPU runtime."""
from contextlib import contextmanager
import json
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


if __name__ == '__main__':
    unittest.main()
