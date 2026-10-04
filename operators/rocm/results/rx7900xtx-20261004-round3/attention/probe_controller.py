import argparse,datetime,hashlib,json,os,pathlib,signal,subprocess,sys,time
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--source',required=True);args=p.parse_args()
root=pathlib.Path('/home/donz/cat-yoko-rocm-20261002')
r=root/'runs/longtrain-45024-20261003T1520';out=root/'runs/operators-split-20261004T0320';out.mkdir(exist_ok=False)
src=pathlib.Path(args.source);python='/home/donz/revelation-rocm-venv/bin/python'
sys.path.insert(0,str(src))
from operators.rocm import run_operator_switch as helpers
receipt={'status':'waiting_for_verified_checkpoint','controller_pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_dir':str(src)}
def write():
 t=out/'probe_receipt.json.tmp';t.write_text(json.dumps(receipt,indent=2)+'\n');t.replace(out/'probe_receipt.json')
def identity(pid):
 proc=pathlib.Path('/proc')/str(pid)
 stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
 return {'pid':pid,'uid':proc.stat().st_uid,'start_ticks':int(stat[19]),'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),'state':stat[0]}
def same(pid,expected):
 try:current=identity(pid)
 except OSError:return False
 return all(current[k]==expected[k] for k in ('pid','uid','start_ticks','command'))
def interrupted(signum,frame):raise RuntimeError('controller interrupted by signal '+str(signum))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
write();begin=time.monotonic()
while True:
 s=json.loads((r/'status.json').read_text())
 if s.get('status')!='running' or s.get('source_dir')!=str(root/'source-operators-round2-20261003T0903'):raise RuntimeError('continuation not running')
 worker=int(s['child_pid']);expected=identity(worker)
 if expected['uid']!=os.getuid() or '--out '+str(r/'continuation') not in expected['command'] or str(root/'source-operators-round2-20261003T0903/operators/rocm/next_candidate_bench.py') not in expected['command']:raise RuntimeError('not the owned continuation worker')
 checkpoints=sorted((r/'continuation/train').glob('trainable_step_*.pt'))
 if checkpoints:break
 if time.monotonic()-begin>900:raise TimeoutError('no periodic checkpoint within 15 minutes')
 time.sleep(5)
original_checkpoint=checkpoints[-1];step=int(original_checkpoint.stem.rsplit('_',1)[1]);updates=step-45024
checkpoint=out/'training_checkpoint.pt';os.link(original_checkpoint,checkpoint)
checkargs=SimpleNamespace(python=python,resume=(r/'source_checkpoint/trainable.pt').resolve(),data=root/'data/phase-b-real-20261002/train.bin',source_dir=src)
verified=helpers.cpu_check(checkargs,checkpoint,updates,out/'continuation_checkpoint_check.log')
receipt.update(checkpoint=str(checkpoint),checkpoint_sha256=helpers.digest_file(checkpoint),checkpoint_verified=verified,worker=expected)
other=[int(x.name) for x in pathlib.Path('/sys/class/kfd/kfd/proc').iterdir() if x.name.isdigit() and int(x.name)!=worker]
if other:raise RuntimeError('unrelated GPU processes present: '+str(other))
memcheck=subprocess.run([python,'-c','import torch,json;print(json.dumps(torch.cuda.mem_get_info()))'],capture_output=True,text=True,timeout=30)
if memcheck.returncode:raise RuntimeError(memcheck.stderr)
free,total=json.loads(memcheck.stdout.splitlines()[-1]);receipt.update(free_gpu_bytes=free,total_gpu_bytes=total);write()
if free < 2*2**30:raise RuntimeError('less than 2 GiB free for memory-filtered split microcheck')
if not same(worker,expected) or identity(worker)['state']=='T':raise RuntimeError('worker identity/state changed')
watchcode=r"""import pathlib,os,signal,sys,time,json
pid=int(sys.argv[1]);expected=json.loads(sys.argv[2]);time.sleep(420)
def matches(identity):
 try:
  p=pathlib.Path('/proc')/str(identity['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split();command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
  return p.stat().st_uid==identity['uid'] and int(s[19])==identity['start_ticks'] and command==identity['command']
 except OSError:return False
try:
 receipt=json.loads(pathlib.Path(sys.argv[3]).read_text());micro=receipt.get('micro_identity')
 if micro and matches(micro):
  os.kill(micro['pid'],signal.SIGKILL);time.sleep(2)
except (OSError,ValueError,KeyError):pass
if matches(expected):os.kill(pid,signal.SIGCONT)
"""
watch=subprocess.Popen([python,'-c',watchcode,str(worker),json.dumps(expected),str(out/'probe_receipt.json')],start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
paused=False;child=None
try:
 os.kill(worker,signal.SIGSTOP);paused=True
 for _ in range(100):
  if identity(worker)['state']=='T':break
  time.sleep(.05)
 else:raise RuntimeError('worker did not stop')
 other=[int(x.name) for x in pathlib.Path('/sys/class/kfd/kfd/proc').iterdir() if x.name.isdigit() and int(x.name)!=worker]
 if other:raise RuntimeError('GPU ownership changed')
 receipt.update(status='isolated_gpu_micro_running',paused_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),watchdog_pid=watch.pid,automatic_resume_after_s=420);write()
 commands=[('split_attention',[python,'-u',str(src/'operators/rocm/split_attention_bench.py'),'--data',str(checkargs.data),'--warmup','2','--repeats','7','--output',str(out/'split_attention.jsonl')])]
 receipt['commands']=[{'name':name,'argv':command} for name,command in commands];receipt['results']=[];write()
 probe_started=time.monotonic()
 for name,command in commands:
  with (out/(name+'.log')).open('w') as log:
   child=subprocess.Popen(command,cwd=src,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   receipt['micro_pid']=child.pid;receipt['micro_identity']=identity(child.pid);receipt['stage']=name;write()
   try:code=child.wait(timeout=max(1,360-(time.monotonic()-probe_started)))
   except subprocess.TimeoutExpired:
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
    raise TimeoutError('isolated probes exceeded total 360 seconds')
  receipt['results'].append({'name':name,'exit_code':code});write()
  if code != 0:raise RuntimeError(name+' failed correctness or execution gate: exit '+str(code))
 receipt.update(status='micro_complete',total_probe_seconds=time.monotonic()-probe_started,micro_completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
except BaseException as error:
 receipt.update(status='probe_failed',error=type(error).__name__+': '+str(error));raise
finally:
 try:
  if child is not None and child.poll() is None:
   child.kill();child.wait(timeout=10)
 except (OSError,subprocess.TimeoutExpired) as cleanup_error:
  receipt['cleanup_error']=str(cleanup_error)
 finally:
  if paused and same(worker,expected):
   os.kill(worker,signal.SIGCONT)
   receipt['resumed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
   for _ in range(100):
    if identity(worker)['state'] not in ('T','t'):break
    time.sleep(.05)
   receipt['worker_resumed']=identity(worker)['state'] not in ('T','t','Z','X')
  if not paused or receipt.get('worker_resumed'):
   watch.terminate();watch.wait(timeout=10)
  write()
print(json.dumps({k:v for k,v in receipt.items() if k!='checkpoint_verified'}),flush=True)
