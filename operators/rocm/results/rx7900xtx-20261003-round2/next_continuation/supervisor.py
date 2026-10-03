"""Resume the verified fixed checkpoint through the remaining packed corpus."""
import datetime,json,os,pathlib,subprocess,sys,time
from types import SimpleNamespace
root=pathlib.Path('/home/donz/cat-yoko-rocm-20261002')
source_dir=root/'source-operators-round2-20261003T0903'
previous=root/'runs/operators-round2-20261003T0903/comparison'
out=root/'runs/longtrain-45024-20261003T1520'
python='/home/donz/revelation-rocm-venv/bin/python'
sys.path.insert(0,str(source_dir))
from operators.rocm import run_operator_switch as helper
from operators.rocm.gpu_wait import wait_for_gpu_idle

def write(path,value):helper.write_json(path,value)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
if out.exists():raise RuntimeError('refuse existing continuation output')
old=json.loads((previous/'status.json').read_text())
if old.get('status')!='complete' or old.get('continuation',{}).get('final_step')!=45024 or old.get('continuation',{}).get('verified_checkpoint_metadata',{}).get('checkpoint_verified') is not True:raise RuntimeError('previous continuation not verified complete')
out.mkdir();(out/'source_checkpoint').mkdir()
source=out/'source_checkpoint/trainable.pt'
os.link(previous/'continuation/train/trainable_step_45024.pt',source)
args=SimpleNamespace(python=python,resume=source.resolve(),source_dir=source_dir,data=root/'data/phase-b-real-20261002/train.bin')
source_sha=helper.digest_file(source)
state={'status':'source_preflight','supervisor_pid':os.getpid(),'source_checkpoint':str(source),'source_sha256':source_sha,'source_step':45024,'source_dir':str(source_dir),'previous_run':str(previous),'started_at':now(),'child_pid':None,'checkpoint_every_seconds':300,'keep_last':3,'time_limit_hours':None,'completion_bound':'remaining single-pass packed corpus; no wrap or new corpus','operator_configuration':'validated packed attention/shared-storage MoE/CPU FP32 Adam','controls_existing_processes':False}
write(out/'status.json',state)
protected=[source,args.data,root/'data/phase-b-real-20261002/eval.bin',*(source_dir/'cat_yoko').glob('*.py'),*(source_dir/'operators/rocm').glob('*.py'),*(root/'hf/MiniCPM5-2B-Base').rglob('*.safetensors')]
fingerprints={str(p):helper.fingerprint(p) for p in protected}
hashes={str(p):helper.digest_file(p) for p in protected if p.suffix=='.py'}
def assert_source():
 if any(helper.fingerprint(pathlib.Path(p))!=v for p,v in fingerprints.items()) or any(helper.digest_file(pathlib.Path(p))!=v for p,v in hashes.items()) or helper.digest_file(source)!=source_sha:raise RuntimeError('fixed source/code/base/corpus changed')
def update(**values):
 state.update(values,updated_at=now());write(out/'status.json',state);print(json.dumps(values),flush=True)
child=None
try:
 metadata=helper.cpu_check(args,source,1,out/'source_preflight.log')
 remaining=metadata['source_data_rows']-metadata['source_stream_i']
 if metadata['source_step']!=45024 or metadata['source_adam_step']!=10222 or metadata['source_stream_i']!=10222 or remaining!=9309:raise RuntimeError('unexpected source or corpus cursor')
 helper.cpu_check(args,source,remaining,out/'corpus_bound_check.log')
 update(status='waiting_for_gpu',source_metadata=metadata,remaining_updates=remaining,target_step=45024+remaining,target_cursor=metadata['source_data_rows'])
 wait_for_gpu_idle(on_event=lambda event:update(gpu_wait=event))
 assert_source()
 command=[python,'-u',str(source_dir/'operators/rocm/next_candidate_bench.py'),'--base',str(root/'hf/MiniCPM5-2B-Base'),'--resume',str(source),'--out',str(out/'continuation'),'--data',str(args.data),'--eval-data',str(root/'data/phase-b-real-20261002/eval.bin'),'--eos-id','1','--moe-layout','shared-storage','--packed-attention','--parity-seqs','4096','--seq-len','4096','--run-steps',str(remaining),'--save-every','0','--save-every-seconds','300','--keep-last','3','--eval-every','250','--eval-batches','32','--save-optim','--deterministic-parity','--reference-repeat','--loss-atol','0.02','--grad-relative-l2','0.05','--output-relative-l2','0.05','--wait-gpu-idle']
 write(out/'continuation.command.json',{'command':command,'started_at':now(),'source_sha256':source_sha})
 begin=time.monotonic()
 with (out/'continuation.log').open('w') as log:
  child=subprocess.Popen(command,cwd=source_dir,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1'})
  update(status='running',child_pid=child.pid)
  code=child.wait()
 write(out/'continuation.receipt.json',{'exit_code':code,'elapsed_s':time.monotonic()-begin,'ended_at':now()})
 update(child_pid=None)
 assert_source()
 if code!=0:raise RuntimeError('training child exited '+str(code)+'; preserve checkpoints for recovery')
 result=helper.validate_run(out/'continuation',source=source,source_step=45024,steps=None,variant='attention',source_dir=source_dir,expected_names=metadata['trainable_names'],args=args,max_updates=remaining,eval_batches=32)
 if result['updates']!=remaining or result['final_step']!=54333:raise RuntimeError('corpus continuation did not finish requested updates')
 result['verified_checkpoint_metadata']=helper.cpu_check(args,pathlib.Path(result['checkpoint']),remaining,out/'final_checkpoint_check.log')
 assert_source();write(out/'continuation.validation.json',result)
 update(status='complete',continuation=result)
except BaseException as error:
 if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
 update(status='failed',error=type(error).__name__+': '+str(error))
 raise
finally:write(out/'summary.json',state)
