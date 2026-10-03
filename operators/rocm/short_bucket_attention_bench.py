#!/usr/bin/env python3
"""Copy-inclusive short-document crossover checks; no training or installation.

Default matrix: lengths64/128/256 x counts4/8/16, both production QKV layouts.
With --data, also compare the actual 4096-token row8701. Operator speed does not
qualify a candidate for full-model continuation or change the132-gradient gate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch

import cat_yoko.attention as native
from operators.rocm import packed_attention as packed
from operators.rocm.attention_bench import _emit,_errors,_is_oom,_measure
from operators.rocm.gpu_wait import wait_for_gpu_idle
from operators.rocm.packed_attention_bench import (
    TOLERANCES,dense_oracle,input_values,load_documents,pid_snapshot,snapshot,
)
from operators.rocm.short_bucket_attention import (
    ShortBucketConfig,clear_short_plan_cache,new_stats,short_bucket_window_sdpa,
)


def clear_plans():
    packed.clear_doc_plan_cache()
    clear_short_plan_cache()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--counts',type=int,nargs='+',default=[4,8,16])
    parser.add_argument('--layouts',nargs='+',choices=('packed_qkv','cross_cache'),default=['packed_qkv','cross_cache'])
    parser.add_argument('--dtypes',nargs='+',choices=tuple(TOLERANCES),default=['bf16'])
    parser.add_argument('--oracle-device',choices=('cpu','cuda'),default='cuda',
                        help='CPU oracle avoids a full dense reference graph in GPU memory')
    parser.add_argument('--data',type=Path)
    parser.add_argument('--data-row',type=int,default=8701)
    parser.add_argument('--eos-id',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--wait-gpu-idle',action='store_true')
    parser.add_argument('--gpu-wait-max-seconds',type=float)
    args=parser.parse_args(argv)
    if (min(args.lengths)<1 or min(args.counts)<1 or max(args.lengths)*max(args.counts)>4096
            or min(args.repeats, args.eos_id)<1 or args.warmup<0 or args.data_row<0):
        parser.error('positive shapes/repeats/eos, sequence<=4096 and nonnegative warmup/row required')
    if args.data is not None and not args.data.is_file():parser.error('--data must be an existing packed bin')
    if args.output is not None:args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.wait_gpu_idle:
        wait_for_gpu_idle(max_wait_seconds=args.gpu_wait_max_seconds,on_event=lambda event:_emit(event,args.output))
    if not torch.cuda.is_available():parser.error('CUDA/HIP GPU required')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    config=ShortBucketConfig()
    root=Path(__file__).resolve().parents[2]
    dependencies=[root/'operators/rocm'/name for name in (
        'short_bucket_attention_bench.py','short_bucket_attention.py',
        'bucketed_attention.py','packed_attention.py','packed_attention_bench.py',
        'attention_bench.py','gpu_wait.py')]
    dependencies += [root/'cat_yoko'/name for name in ('attention.py','config.py','rope.py','ops.py')]
    source_hashes={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in dependencies}
    _emit({'event':'environment','torch':torch.__version__,'hip':torch.version.hip,
        'gpu':torch.cuda.get_device_name(),'heads':16,'kv_heads':2,'head_dim':128,'window':8192,
        'config':asdict(config),'tolerances':TOLERANCES,'warmup':args.warmup,'repeats':args.repeats,
        'oracle_device':args.oracle_device,'file_sha256':source_hashes,
        'gate':'finite output/dQ/dK/dV; elementwise atol/rtol AND relative-L2 thresholds unchanged',
        'oracle_scope':'reference fixture transfers and computation excluded from GPU candidate timing',
        'merge_heads_included':True,'scope':'QKV casts, padding, concatenation, GQA, SDPA, fragment assembly and merge_heads; projections and fixture creation excluded',
        'cold_scope':'both plan caches cleared per invocation, includes CPU plan transfer and Python planning',
        'wait_does_not_reserve_gpu':True,**pid_snapshot()},args.output)
    fixtures=[]
    for length in sorted(set(args.lengths)):
        for count in sorted(set(args.counts)):
            docs=torch.arange(count).repeat_interleave(length).unsqueeze(0)
            fixtures.append((docs,{'kind':'uniform_synthetic','length':length,'count':count}))
    if args.data is not None:
        docs,info=load_documents(args.data,seq=4096,batch=1,row=args.data_row,eos_id=args.eos_id,synthetic_doc_len=64)
        fixtures.append((docs,info))
    totals={'passed':0,'failed':0,'errors':0,'short_exercised':0}
    for docs_cpu,provenance in fixtures:
        seq=docs_cpu.size(-1);docs=docs_cpu.to('cuda')
        for name in args.dtypes:
            dtype=torch.bfloat16 if name=='bf16' else torch.float32
            for layout in args.layouts:
                values=input_values(seq,1,dtype,layout,args.seed+seq)
                dy=torch.randn(values[0].shape,device='cuda',dtype=dtype,
                    generator=torch.Generator(device='cuda').manual_seed(args.seed+10000+seq))
                dy=native.merge_heads(dy.transpose(1,2).contiguous().transpose(1,2))
                base={'event':'candidate','seq':seq,'window':8192,'dtype':name,'layout':layout,
                    'documents':provenance,'input_strides':[list(v.stride()) for v in values],
                    'dy_stride':list(dy.stride()),'oracle_device':args.oracle_device,
                    'gpu_processes_before':pid_snapshot()}
                try:
                    oracle_values=tuple(value.detach().cpu() for value in values) if args.oracle_device=='cpu' else values
                    oracle_dy=dy.detach().cpu() if args.oracle_device=='cpu' else dy
                    oracle_docs=docs_cpu if args.oracle_device=='cpu' else docs
                    reference,ref_grads=snapshot(
                        lambda q,k,v:native.merge_heads(dense_oracle(q,k,v,8192,oracle_docs)),
                        oracle_values,oracle_dy)
                    del oracle_values,oracle_dy
                except Exception as error:
                    _emit({**base,'candidate':'oracle','status':'error','error':str(error),'oom':_is_oom(error)},args.output)
                    totals['errors']+=1;continue
                for candidate in ('packed','short_bucketed'):
                    clear_plans();stats=new_stats()
                    function=packed.packed_window_sdpa if candidate=='packed' else short_bucket_window_sdpa

                    def operation(q,k,v):
                        return native.merge_heads(function(q,k,v,8192,docs,stats=stats))

                    row={**base,'candidate':candidate}
                    try:
                        actual,grads=snapshot(operation,values,dy)
                        row['output']=_errors(actual.cpu(),reference.cpu(),TOLERANCES[name])
                        row['gradients']={label:_errors(a.cpu(),b.cpu(),TOLERANCES[name])
                            for label,a,b in zip(('dq','dk','dv'),grads,ref_grads)}
                        passed=row['output']['pass'] and all(x['pass'] for x in row['gradients'].values())
                        row['status']='pass' if passed else 'numerical_failure'
                        row['correctness_path_counts']=json.loads(json.dumps(stats))
                        totals['short_exercised']+=int(candidate=='short_bucketed' and stats['short_bucketed_calls']>0)
                        del actual,grads
                        if passed:
                            for mode in ('warm','cold'):
                                def forward():
                                    if mode=='cold':clear_plans()
                                    with torch.no_grad():return operation(*values)

                                def forward_backward():
                                    if mode=='cold':clear_plans()
                                    qkv=tuple(v.detach().requires_grad_() for v in values)
                                    return torch.autograd.grad(operation(*qkv),qkv,dy)

                                row[mode+'_forward']=_measure(forward,args.warmup,args.repeats)
                                gc.collect();torch.cuda.empty_cache()
                                row[mode+'_forward_backward']=_measure(forward_backward,args.warmup,args.repeats)
                        totals['passed' if passed else 'failed']+=1
                    except Exception as error:
                        row.update(status='error',error=str(error),oom=_is_oom(error));totals['errors']+=1
                    row['gpu_processes_after']=pid_snapshot();_emit(row,args.output)
                    gc.collect();torch.cuda.empty_cache()
                del values,dy,reference,ref_grads
        del docs
    _emit({'event':'summary',**totals},args.output)
    return int(bool(totals['failed'] or totals['errors']))


if __name__=='__main__':
    raise SystemExit(main())
