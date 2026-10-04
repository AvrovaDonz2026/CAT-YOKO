"""Acceptance and process-lifecycle regressions for the isolated switch."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from operators.rocm import run_round3_switch as switch

class Round3SwitchTests(unittest.TestCase):
    def test_production_retains_data_optimizer_and_save_policy(self):
        args=SimpleNamespace(base=Path('/base'),data=Path('/train.bin'),eval_data=Path('/eval.bin'))
        flags=switch.common(args,Path('/source.pt'),Path('/run'),1024,production=True)
        for flag,value in {'--run-steps':'1024','--save-every-seconds':'300','--keep-last':'3','--data':'/train.bin','--eval-data':'/eval.bin','--eval-batches':'32'}.items():
            self.assertEqual(flags[flags.index(flag)+1],value)
        self.assertIn('--save-optim',flags)
        self.assertNotIn('--max-hours',flags)
        self.assertNotIn('--deterministic-training',flags)

    def test_live_child_cleanup_reaps_only_supplied_child(self):
        child=Mock();child.poll.return_value=None
        switch.stop_owned_child(child)
        child.terminate.assert_called_once_with();child.wait.assert_called_once_with(timeout=30);child.kill.assert_not_called()

    def test_live_child_cleanup_escalates_after_timeout(self):
        child=Mock();child.poll.return_value=None;child.wait.side_effect=[subprocess.TimeoutExpired('own',30),-9]
        switch.stop_owned_child(child)
        child.terminate.assert_called_once_with();child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count,2)

    def test_exited_child_is_not_signaled(self):
        child=Mock();child.poll.return_value=0
        switch.stop_owned_child(child)
        child.terminate.assert_not_called();child.kill.assert_not_called()

    def test_numbered_lookup_ignores_concurrent_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);saved=directory/'trainable_step_100.pt';saved.write_bytes(b'checkpoint');stable=directory/'pending.pt';os.link(saved,stable)
            vanished=directory/'trainable_step_99.pt'
            with patch.object(Path,'glob',return_value=iter([vanished,saved])):
                self.assertEqual(switch.numbered_step(directory,stable),100)

    def report_fixture(self,root):
        enabled=switch.ROUND3_SOURCES|{'operators/rocm/split_attention.py','operators/rocm/cpu_adam_cached.py','operators/rocm/update_timing.py'}
        return {'status':'completed','patches_installed_after_native_reference':True,'full_model_parity_passed':True,
                'split_attention':True,'cached_cpu_adam':True,'sync_update_timing':True,
                'file_sha256':dict.fromkeys(enabled,'sha'),
                'split_attention_parity_calls':{'split_optimized_calls':1},
                'cached_cpu_adam_calls':{'state_loads':1,'gpu_parameter_updates':2640,'parameter_updates':2640,
                    'fallback_parameter_updates':0,'cache_reuses':2508,'checkpoint_format_changed':False,'moment_dtype':'float32'}}

    def check_report(self,mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=self.report_fixture(root)
            if mutate:mutate(report)
            (root/'round3_operators.json').write_text(json.dumps(report))
            with patch.object(switch.helpers,'digest_file',return_value='sha'):
                return switch.validate_round3_report(SimpleNamespace(source_dir=root),root,'combined',20,timing=True)

    def test_valid_complete_installation_accepts(self):
        self.assertEqual(self.check_report()['cached_cpu_adam_calls']['parameter_updates'],2640)

    def test_missing_source_inventory_rejects(self):
        with self.assertRaises(ValueError):
            self.check_report(lambda d:d['file_sha256'].pop('cat_yoko/optim.py'))

    def test_unexercised_optimizer_rejects(self):
        with self.assertRaises(ValueError):
            self.check_report(lambda d:d['cached_cpu_adam_calls'].update(gpu_parameter_updates=0))

    def test_parameter_offload_fallback_rejects(self):
        with self.assertRaises(ValueError):
            self.check_report(lambda d:d['cached_cpu_adam_calls'].update(fallback_parameter_updates=1))

    def test_changed_precision_rejects(self):
        with self.assertRaises(ValueError):
            self.check_report(lambda d:d['cached_cpu_adam_calls'].update(moment_dtype='bfloat16'))

    def test_packed_production_reference_is_explicit_and_binds_wrapper_inventory(self):
        from operators.rocm.production_continuation import REFERENCE_BACKEND, WRAPPER_SOURCE, PACKED_SOURCE
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.pt';source.write_bytes(b'fixed source')
            names=switch.ROUND3_SOURCES|{'operators/rocm/split_attention.py','operators/rocm/cpu_adam_cached.py',
                                         'operators/rocm/update_timing.py',WRAPPER_SOURCE}
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen source')
            report=self.report_fixture(root)
            report.update(reference_backend=REFERENCE_BACKEND,patches_installed_after_native_reference=False,
                          patches_installed_after_packed_production_reference=True,production_reference_context_restored=True,
                          reference_checkpoint_sha256=switch.helpers.digest_file(source),
                          reference_file_sha256={name:switch.helpers.digest_file(root/name) for name in (WRAPPER_SOURCE,PACKED_SOURCE)},
                          production_reference_captures=[dict(status='completed',seq_len=4096,offload_blocks=True,
                              gradient_tensors=132,packed_attention_calls={'optimized_calls':42}) for _ in range(2)],
                          file_sha256={name:switch.helpers.digest_file(root/name) for name in names})
            path=root/'round3_operators.json';path.write_text(json.dumps(report))
            args=SimpleNamespace(source_dir=root,resume=source)
            self.assertTrue(switch.validate_round3_report(args,root,'combined',20,timing=True,
                                                         reference_backend=REFERENCE_BACKEND)['full_model_parity_passed'])
            with self.assertRaises(ValueError):switch.validate_round3_report(args,root,'combined',20,timing=True)
            for mutate in (lambda d:d['file_sha256'].pop(WRAPPER_SOURCE),
                           lambda d:d.update(reference_checkpoint_sha256='wrong'),
                           lambda d:d['production_reference_captures'][0]['packed_attention_calls'].update(optimized_calls=0),
                           lambda d:d['cached_cpu_adam_calls'].update(gpu_parameter_updates=2639)):
                changed=json.loads(json.dumps(report));mutate(changed);path.write_text(json.dumps(changed))
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                    switch.validate_round3_report(args,root,'combined',20,timing=True,reference_backend=REFERENCE_BACKEND)

    def policy_fixture(self,root,*,updates=2,deterministic=True,legacy=False):
        root.mkdir(parents=True,exist_ok=True)
        (root/'train').mkdir(exist_ok=True)
        flags={} if legacy else {'deterministic_training':deterministic,'deterministic_algorithms':deterministic}
        operators=dict(flags)
        rows=[{'step':53144+i,'tok_s':100.0,**flags} for i in range(updates)]
        parity={'original_deterministic_algorithms':deterministic,
                'events':[{'event':'training_complete','updates':updates,**flags}]}
        self.write_policy(root,operators,parity,rows)
        return operators,parity,rows

    def write_policy(self,root,operators,parity,rows):
        (root/'round3_operators.json').write_text(json.dumps(operators))
        (root/'parity.json').write_text(json.dumps(parity))
        (root/'train/metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_controlled_policy_requires_two_consecutive_real_update_receipts(self):
        corruptions=[lambda rows:rows.clear(),lambda rows:rows.pop(),
                     lambda rows:rows.append(dict(rows[-1])),lambda rows:rows.reverse(),
                     lambda rows:rows[1].update(step=53146),
                     lambda rows:rows[1].pop('deterministic_algorithms'),
                     lambda rows:rows[1].update(deterministic_training=False),
                     lambda rows:rows[1].update(deterministic_algorithms=False)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.policy_fixture(root)
            proof=switch.validate_training_policy(root,source_step=53143,updates=2,deterministic_training=True)
            self.assertEqual(proof['consecutive_metric_updates'],2)
            for corruption in corruptions:
                with self.subTest(corruption=corruption):
                    operators,parity,rows=self.policy_fixture(root)
                    corruption(rows);self.write_policy(root,operators,parity,rows)
                    with self.assertRaises(ValueError):
                        switch.validate_training_policy(root,source_step=53143,updates=2,deterministic_training=True)

    def test_legacy_timing_requires_recorded_native_false_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            operators,parity,rows=self.policy_fixture(root,updates=20,deterministic=False,legacy=True)
            proof=switch.validate_training_policy(root,source_step=53143,updates=20,deterministic_training=False,allow_legacy=True)
            self.assertEqual(proof['policy_evidence'],'source-bound native original policy')
            for original in (None,True):
                with self.subTest(original=original):
                    parity['original_deterministic_algorithms']=original
                    self.write_policy(root,operators,parity,rows)
                    with self.assertRaises(ValueError):
                        switch.validate_training_policy(root,source_step=53143,updates=20,deterministic_training=False,allow_legacy=True)
            operators,parity,rows=self.policy_fixture(root,updates=20,deterministic=False,legacy=True)
            operators['deterministic_training']=True
            self.write_policy(root,operators,parity,rows)
            with self.assertRaises(ValueError):
                switch.validate_training_policy(root,source_step=53143,updates=20,deterministic_training=False,allow_legacy=True)

    def test_timing_reuse_runs_the_policy_gate_before_checkpoint_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.policy_fixture(root,updates=20,deterministic=False,legacy=True)
            args=SimpleNamespace(resume=root/'source.pt',source_dir=root/'code',out=root)
            metadata={'source_step':53143,'trainable_names':['trainable']}
            with patch.object(switch.helpers,'validate_run',return_value={'checkpoint':str(root/'train/trainable.pt')}), \
                 patch.object(switch,'validate_timing',return_value={}), \
                 patch.object(switch,'validate_round3_report'), \
                 patch.object(switch.helpers,'cpu_check',return_value={'checkpoint_verified':True}) as cpu_check:
                accepted=switch.validate_trial(args,root,metadata,'baseline')
                self.assertFalse(accepted['training_policy']['deterministic_algorithms'])
                self.policy_fixture(root,updates=20,deterministic=True,legacy=True)
                with self.assertRaisesRegex(ValueError,'different training policy'):
                    switch.validate_trial(args,root,metadata,'baseline')
                self.assertEqual(cpu_check.call_count,1)

    def test_production_requires_explicit_requested_actual_and_final_false_policy(self):
        mutations=[lambda o,p,r:o.pop('deterministic_training'),
                   lambda o,p,r:o.pop('deterministic_algorithms'),
                   lambda o,p,r:o.update(deterministic_training=True),
                   lambda o,p,r:o.update(deterministic_algorithms=True),
                   lambda o,p,r:p.update(original_deterministic_algorithms=True),
                   lambda o,p,r:p['events'].clear(),
                   lambda o,p,r:p['events'][-1].pop('deterministic_algorithms'),
                   lambda o,p,r:p['events'][-1].update(deterministic_algorithms=True)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.policy_fixture(root,deterministic=False)
            self.assertFalse(switch.validate_training_policy(root,source_step=53143,updates=2,
                                                           deterministic_training=False)['deterministic_algorithms'])
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    operators,parity,rows=self.policy_fixture(root,deterministic=False)
                    mutate(operators,parity,rows);self.write_policy(root,operators,parity,rows)
                    with self.assertRaises(ValueError):
                        switch.validate_training_policy(root,source_step=53143,updates=2,deterministic_training=False)
            self.policy_fixture(root,deterministic=False,legacy=True)
            with self.assertRaises(ValueError):
                switch.validate_training_policy(root,source_step=53143,updates=2,deterministic_training=False)

    def confirmation_fixture(self,root):
        prior=root/'prior';prior.mkdir()
        out=root/'confirmation';out.mkdir()
        source=root/'source.pt';source.write_bytes(b'fixed source')
        old_source=root/'old-source.pt';old_source.write_bytes(source.read_bytes())
        status={'comparison_failure':'ValueError: full-update weights/moments changed: native_repeat',
                'source_dir':str(root/'old-code'),'source_checkpoint':str(old_source),
                'source_sha256':switch.helpers.digest_file(source)}
        (prior/'status.json').write_text(json.dumps(status))
        (prior/'native_repeat.terminal_compare.json').write_text(json.dumps({'pass':False}))
        args=SimpleNamespace(confirm_from=prior,out=out,resume=source,source_dir=root/'new-code',
                             base=root/'base',data=root/'train.bin',eval_data=root/'eval.bin')
        return args,status,old_source

    def test_confirmation_binds_recorded_and_both_live_checkpoint_hashes_before_children(self):
        for change in ('recorded','old','new'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                args,status,old_source=self.confirmation_fixture(Path(tmp))
                if change=='recorded':
                    status['source_sha256']='unrelated prior source'
                    (args.confirm_from/'status.json').write_text(json.dumps(status))
                elif change=='old':old_source.write_bytes(b'changed previous source')
                else:args.resume.write_bytes(b'changed confirmation source')
                child=Mock()
                with patch.object(switch,'validate_trial') as validate_trial, self.assertRaisesRegex(ValueError,'checkpoint hashes'):
                    switch.confirm_nondeterministic_trials(args,{'source_step':53143},child)
                child.assert_not_called();validate_trial.assert_not_called()

    def test_confirmation_wires_strict_two_updates_and_reuses_default_policy_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,_,_=self.confirmation_fixture(Path(tmp))
            metadata={'source_step':53143,'trainable_names':['trainable']}
            speeds={'baseline_before':100.,'combined':108.,'baseline_after':102.}
            def trial(_args,directory,_metadata,_variant):
                return {key:speeds[directory.name] for key in ('median_completed_update_tokens_s',
                         'aggregate_completed_update_tokens_s','median_tokens_per_second')}
            def child(name,flags):
                self.assertIn('--deterministic-training',flags)
                self.assertEqual(flags[flags.index('--run-steps')+1],'2')
                self.policy_fixture(args.out/name)
            def validate(directory,**kwargs):
                self.assertIsNone(kwargs['steps']);self.assertEqual(kwargs['max_updates'],2)
                return {'updates':2,'checkpoint':str(directory/'train/trainable.pt')}
            with patch.object(switch,'validate_trial',side_effect=trial) as timing_trials, \
                 patch.object(switch.helpers,'validate_run',side_effect=validate), \
                 patch.object(switch,'validate_round3_report'), \
                 patch.object(switch.helpers,'cpu_check',return_value={'checkpoint_verified':True}), \
                 patch.object(switch,'compare',return_value={'pass':True,'weights_bitwise':True,'moments_bitwise':True}):
                decision=switch.confirm_nondeterministic_trials(args,metadata,child)
            self.assertEqual(timing_trials.call_count,3)
            self.assertEqual(decision['controlled_updates'],2)
            self.assertFalse(decision['production_deterministic_training'])
            self.assertEqual(decision['production_policy_timing_runs'],str(args.confirm_from.resolve()))
            self.assertEqual(decision['source_sha256'],switch.helpers.digest_file(args.resume))
            self.assertAlmostEqual(decision['conservative_gains']['median_completed_update_tokens_s'],108/102-1)

if __name__=='__main__':unittest.main()
