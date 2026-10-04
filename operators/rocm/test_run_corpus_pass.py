"""Explicit repeated-pass bounds and live policy receipt checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from operators.rocm import run_corpus_pass as controller
from operators.rocm.run_corpus_pass import completed_metrics, pass_plan


class CorpusPassTests(unittest.TestCase):
    def metadata(self, cursor=19531):
        return {'source_data_rows': 19531, 'source_stream_i': cursor,
                'source_adam_step': cursor, 'source_step': 34802 + cursor,
                'source_tokens_in_phase': 162244608 + cursor * 4096}

    def test_second_pass_preserves_monotonic_counters(self):
        plan = pass_plan(self.metadata(), 39062)
        self.assertEqual(plan['updates'], 19531)
        self.assertEqual(plan['target_step'], 73864)
        self.assertEqual(plan['target_adam_step'], 39062)
        self.assertEqual(plan['target_tokens_in_phase'], 322242560)
        self.assertTrue(plan['repeated_corpus'])
        self.assertEqual(plan['start_next_row'], 0)

    def test_recovery_inside_second_pass_reaches_same_boundary(self):
        plan = pass_plan(self.metadata(cursor=20000), 39062)
        self.assertEqual(plan['updates'], 19062)
        self.assertEqual(plan['target_step'], 73864)
        self.assertEqual(plan['start_next_row'], 469)

    def test_unbounded_or_wrong_pass_end_rejects(self):
        for target in (19531, 39061, 39063, 58593, -1, True, 39062.0):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pass_plan(self.metadata(), target)

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
            out = root / 'run'
            (out / 'source_checkpoint').mkdir(parents=True)
            args = SimpleNamespace(source_dir=root / 'source', base=root / 'base',
                                   data=root / 'data/train.bin', eval_data=root / 'data/eval.bin',
                                   out=out, resume=out / 'source_checkpoint/trainable.pt',
                                   lock_file=root / 'training.lock', target_cursor=39062,
                                   python='test-python')
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


if __name__ == '__main__':
    unittest.main()
