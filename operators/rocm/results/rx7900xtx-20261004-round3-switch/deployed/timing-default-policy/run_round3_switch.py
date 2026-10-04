#!/usr/bin/env python3
"""Checkpoint-first, process-local round-three acceptance and continuation."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import run_operator_switch as helpers
from operators.rocm.run_next_operator_round import validate_timing
from operators.rocm.gpu_wait import wait_for_gpu_idle

ROUND3_FLAGS = ['--split-attention', '--cached-cpu-adam']
ROUND3_SOURCES = {'operators/rocm/round3_candidate_bench.py','operators/rocm/candidate_bench.py',
                 'operators/rocm/model_bench.py','operators/rocm/packed_attention.py',
                 'operators/rocm/shared_storage_moe.py','cat_yoko/optim.py','cat_yoko/trainer.py',
                 'cat_yoko/checkpoint.py'}

COMPARE_TERMINALS = r'''
import json, sys, torch
left, right = [torch.load(p, map_location='cpu', weights_only=False) for p in sys.argv[1:]]
assert list(left['trainable']) == list(right['trainable'])
weights = all(torch.equal(left['trainable'][n].contiguous().view(torch.uint8), right['trainable'][n].contiguous().view(torch.uint8)) for n in left['trainable'])
a,b=left['optimizer'],right['optimizer'];assert a['param_groups']==b['param_groups'] and list(a['state'])==list(b['state'])
bitwise=True;close=True;maximum=0.0
for pid in a['state']:
 assert a['state'][pid]['step']==b['state'][pid]['step']
 for name in ('exp_avg','exp_avg_sq'):
  x,y=a['state'][pid][name],b['state'][pid][name]
  assert x.dtype==y.dtype==torch.float32 and x.device.type==y.device.type=='cpu' and x.shape==y.shape
  assert torch.isfinite(x).all() and torch.isfinite(y).all()
  bitwise &= torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
  close &= torch.allclose(x,y,atol=1e-8,rtol=1e-6)
  maximum=max(maximum,float((x-y).abs().max()))
for key in ('step','stream','tokens_in_phase','tokens_seen','cfg','phase','seed'):
 assert left['extra'][key]==right['extra'][key],key
print(json.dumps({'pass':bool(weights and close),'weights_bitwise':weights,'moments_bitwise':bitwise,'moments_close':close,'moment_max_abs':maximum,'parameters':132,'moments':264,'moment_atol':1e-8,'moment_rtol':1e-6}))
'''

def need(value, message):
    if not value: raise ValueError(message)

def read(path):
    d=json.loads(path.read_text());helpers.require_finite(d);return d

def stop_owned_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()

def numbered_step(directory, stable):
    saved=stable.stat()
    for path in directory.glob('trainable_step_*.pt'):
        try:
            current=path.stat()
        except FileNotFoundError:
            continue
        if (current.st_dev,current.st_ino)==(saved.st_dev,saved.st_ino):
            return int(path.stem.rsplit('_',1)[1])
    raise ValueError('fixed checkpoint has no retained numbered publication')

def compare(args, left, right, name):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONOPTIMIZE='0')
    r=subprocess.run([args.python,'-c',COMPARE_TERMINALS,str(left),str(right)],capture_output=True,text=True,env=env,timeout=300)
    (args.out/(name+'.terminal_compare.log')).write_text(r.stdout+r.stderr)
    need(r.returncode==0,'terminal comparison failed: '+name)
    d=json.loads(r.stdout.splitlines()[-1]);helpers.write_json(args.out/(name+'.terminal_compare.json'),d)
    need(d['pass'],'full-update weights/moments changed: '+name)
    return d

def common(args, resume, directory, updates, *, production=False):
    flags=['--base',str(args.base),'--resume',str(resume),'--out',str(directory),
           '--data',str(args.data),'--eval-data',str(args.eval_data),'--eos-id','1',
           '--moe-layout','shared-storage','--packed-attention','--parity-seqs','4096','--seq-len','4096',
           '--run-steps',str(updates),'--save-every','0','--save-every-seconds','300' if production else '0',
           '--keep-last','3' if production else '1','--eval-every','250' if production else '0',
           '--eval-batches','32' if production else '2','--save-optim','--deterministic-parity','--reference-repeat',
           '--loss-atol','0.02','--grad-relative-l2','0.05','--output-relative-l2','0.05','--wait-gpu-idle',
           '--gpu-idle-max-wait','3600']
    return flags

def validate_round3_report(args, directory, variant, updates, *, timing):
    report=read(directory/'round3_operators.json')
    enabled=variant=='combined'
    need(report.get('status')=='completed' and report.get('patches_installed_after_native_reference') is True,
         'round3 installation did not complete')
    need(report.get('split_attention') is enabled and report.get('cached_cpu_adam') is enabled
         and report.get('sync_update_timing') is timing and report.get('full_model_parity_passed') is True,
         'requested round3 flags differ from actual installation')
    hashes=report.get('file_sha256',{})
    required=ROUND3_SOURCES|({'operators/rocm/split_attention.py','operators/rocm/cpu_adam_cached.py'} if enabled else set())
    if timing: required=required|{'operators/rocm/update_timing.py'}
    need(set(hashes)==required and all(helpers.digest_file(args.source_dir/name)==sha for name,sha in hashes.items()),
         'round3 code changed or source inventory incomplete')
    if enabled:
        need(report.get('split_attention_parity_calls',{}).get('split_optimized_calls',0)>0,'split parity not exercised')
        calls=report.get('cached_cpu_adam_calls',{})
        need(calls.get('state_loads',0)>=1 and calls.get('gpu_parameter_updates')==132*updates
             and calls.get('parameter_updates')==132*updates and calls.get('fallback_parameter_updates')==0,
             'cached Adam load/update count mismatch')
        need(calls.get('cache_reuses',0)>=132*max(0,updates-1),'cached Adam shadow was not reused')
        need(calls.get('checkpoint_format_changed') is False and calls.get('moment_dtype')=='float32',
             'optimizer format or precision changed')
    return report

def validate_trial(args, directory, metadata, variant):
    result=helpers.validate_run(directory,source=args.resume,source_step=metadata['source_step'],steps=20,
            variant='attention',source_dir=args.source_dir,expected_names=metadata['trainable_names'],args=args)
    result.update(validate_timing(directory/'update_timing.json',20))
    validate_round3_report(args,directory,variant,20,timing=True)
    result['checkpoint_metadata']=helpers.cpu_check(args,Path(result['checkpoint']),20,args.out/(directory.name+'.checkpoint_check.log'))
    return result

def supervise(args):
    args.source_dir,args.out,args.resume,args.base,args.data,args.eval_data=[getattr(args,n).resolve() for n in ('source_dir','out','resume','base','data','eval_data')]
    need(args.out.is_dir() and (args.out/'source_checkpoint/trainable.pt').resolve()==args.resume,
         'output must be the separate run holding its fixed recovery source')
    need(not any(args.out==path or args.out.is_relative_to(path) or path.is_relative_to(args.out)
                 for path in (args.source_dir,args.base,args.data.parent,args.eval_data.parent)),
         'output must be separate from code, base and corpus')
    state={'status':'preflight','supervisor_pid':os.getpid(),'source_checkpoint':str(args.resume),
           'source_sha256':helpers.digest_file(args.resume),'source_dir':str(args.source_dir),
           'user_requested_round3_switch':True,'controls_existing_processes':False,
           'benchmark_updates_count_toward_continuation':False,'checkpoint_every_seconds':300,'keep_last':3,
           'operator_configuration':'packed FP32 split attention/shared-storage MoE/cached BF16 CPU shadow/native FP32 Adam'}
    def update(**values):
        state.update(values,updated_at=helpers.utc_now().isoformat());helpers.write_json(args.out/'status.json',state)
        print(json.dumps(values,allow_nan=False),flush=True)
    paths=[args.resume,args.data,args.eval_data,*args.source_dir.glob('cat_yoko/*.py'),*args.source_dir.glob('operators/rocm/*.py'),
           *([args.base] if args.base.is_file() else args.base.rglob('*.safetensors')),
           *([] if args.base.is_file() else args.base.glob('*.json'))]
    fingerprints={str(p):helpers.fingerprint(p) for p in paths}
    hashes={str(p):helpers.digest_file(p) for p in paths if p.suffix=='.py'}
    def integrity():
        need(all(helpers.fingerprint(Path(p))==s for p,s in fingerprints.items()),'protected source changed')
        need(all(helpers.digest_file(Path(p))==s for p,s in hashes.items()),'protected code changed')
        need(helpers.digest_file(args.resume)==state['source_sha256'],'fixed checkpoint changed')
    def child(name,flags,script='round3_candidate_bench.py',timeout=2400):
        integrity();update(status='waiting_for_gpu',stage=name)
        wait_for_gpu_idle(max_wait_seconds=3600,on_event=lambda e:update(gpu_wait=e))
        command=[args.python,'-u',str(args.source_dir/'operators/rocm'/script),*flags]
        helpers.write_json(args.out/(name+'.command.json'),{'command':command})
        with (args.out/(name+'.log')).open('w') as log:
            p=subprocess.Popen(command,cwd=args.source_dir,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONOPTIMIZE='0'))
            update(status='running',stage=name,child_pid=p.pid)
            try: code=p.wait(timeout=timeout)
            except BaseException:
                stop_owned_child(p);raise
        update(child_pid=None);integrity();need(code==0,name+' failed with exit '+str(code))
    metadata=helpers.cpu_check(args,args.resume,20,args.out/'source_preflight.log')
    update(source_metadata=metadata,target_step=metadata['source_step']+metadata['source_data_rows']-metadata['source_stream_i'],
           target_cursor=metadata['source_data_rows'])
    selected='baseline'
    try:
        child('long_attention_micro',['--data',str(args.data),'--rows','12000','17478','18000',str(metadata['source_stream_i']),
              '--warmup','0','--repeats','1','--output',str(args.out/'long_attention_micro.jsonl')],script='split_attention_bench.py')
        micro=[json.loads(x) for x in (args.out/'long_attention_micro.jsonl').read_text().splitlines()]
        summary=micro[-1]
        need(summary.get('event')=='summary' and summary.get('failed')==summary.get('errors')==summary.get('fixtures_skipped_memory')==0
             and summary.get('passed')==14,'long-document/single-document GPU microchecks failed or skipped')
        results={}
        for name,variant in [('baseline_before','baseline'),('combined','combined'),('baseline_after','baseline')]:
            flags=common(args,args.resume,args.out/name,20)+['--sync-update-timing']+(ROUND3_FLAGS if variant=='combined' else [])
            child(name,flags);results[name]=validate_trial(args,args.out/name,metadata,variant)
            helpers.write_json(args.out/(name+'.validation.json'),results[name])
        compare(args,Path(results['baseline_before']['checkpoint']),Path(results['baseline_after']['checkpoint']),'native_repeat')
        terminal=compare(args,Path(results['baseline_before']['checkpoint']),Path(results['combined']['checkpoint']),'combined')
        keys=['median_completed_update_tokens_s','aggregate_completed_update_tokens_s','median_tokens_per_second']
        gains={key:results['combined'][key]/max(results[name][key] for name in ('baseline_before','baseline_after'))-1 for key in keys}
        decision={'selected':'combined','selected_flags':ROUND3_FLAGS,'selection_mode':'user-requested after numerical and recovery acceptance',
                  'automatic_five_percent_gate_passed':all(v>=0.05 for v in gains.values()),'conservative_gains':gains,
                  'baselines':[results['baseline_before'],results['baseline_after']],'candidate':results['combined'],'terminal':terminal}
        # The user explicitly requested this backend. Numerical acceptance stays strict;
        # report the 5% automatic selection gate separately, including measured noise.
        helpers.write_json(args.out/'decision.json',decision);selected='combined';update(status='accepted',decision=decision)
    except Exception as error:
        update(comparison_failure=type(error).__name__+': '+str(error),fallback_reason='retain original operators after failed acceptance')
        helpers.write_json(args.out/'decision.json',{'selected':'baseline','error':str(error)})
    integrity()
    remaining=metadata['source_data_rows']-metadata['source_stream_i']
    resume=args.resume;verified_meta=metadata
    # Launch from the fixed source, never from discarded benchmark trajectories.
    flags=common(args,resume,args.out/'continuation',remaining,production=True)+(ROUND3_FLAGS if selected=='combined' else [])
    wait_for_gpu_idle(max_wait_seconds=3600,on_event=lambda e:update(gpu_wait=e))
    command=[args.python,'-u',str(args.source_dir/'operators/rocm/round3_candidate_bench.py'),*flags]
    helpers.write_json(args.out/'continuation.command.json',{'command':command})
    with (args.out/'continuation.log').open('w') as log:
        p=subprocess.Popen(command,cwd=args.source_dir,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONOPTIMIZE='0'))
        update(status='running_continuation',stage='continuation',child_pid=p.pid,selected_operators=selected)
        try:
            seen=set()
            while p.poll() is None:
                current=args.out/'continuation/train/trainable.pt'
                if current.is_file() and helpers.fingerprint(current) not in seen:
                    verification=args.out/'verified_checkpoint';verification.mkdir(exist_ok=True)
                    stable=verification/'pending.pt';stable.unlink(missing_ok=True);os.link(current,stable)
                    try:
                        fingerprint=helpers.fingerprint(stable)
                        ckstep=numbered_step(current.parent,stable)
                        checked=helpers.cpu_check(args,stable,ckstep-metadata['source_step'],args.out/'latest_checkpoint_check.log')
                        stable.replace(verification/'latest_verified.pt');seen.add(fingerprint);verified_meta=checked
                        update(first_checkpoint_verified=True,latest_verified_checkpoint={**checked,'file':str(verification/'latest_verified.pt')},
                               latest_verified_at=helpers.utc_now().isoformat())
                    finally:
                        stable.unlink(missing_ok=True)
                time.sleep(1)
            code=p.wait()
        finally:
            stop_owned_child(p)
    need(code==0,'continuation exited unsuccessfully; latest verified recovery is retained')
    result=helpers.validate_run(args.out/'continuation',source=args.resume,source_step=metadata['source_step'],steps=remaining,
                  variant='attention',source_dir=args.source_dir,expected_names=metadata['trainable_names'],args=args,eval_batches=32)
    validate_round3_report(args,args.out/'continuation',selected,remaining,timing=False)
    final=helpers.cpu_check(args,Path(result['checkpoint']),remaining,args.out/'final_checkpoint_check.log')
    need(final['source_stream_i']==metadata['source_data_rows'] and final['source_step']==state['target_step'],'wrong final corpus boundary')
    integrity();update(status='complete',child_pid=None,final_checkpoint_verified=final)
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','resume','base','data','eval-data','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args()
    try: supervise(args)
    except BaseException as error:
        p=args.out/'status.json'
        d=read(p) if p.exists() else {};d.update(status='failed',error=type(error).__name__+': '+str(error))
        helpers.write_json(p,d);raise
if __name__=='__main__':main()
