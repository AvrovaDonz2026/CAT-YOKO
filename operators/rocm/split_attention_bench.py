#!/usr/bin/env python3
"""Isolated B/C/B split-backward attention measurements on production layouts.

No process is paused or patched by this entry point. Compare against the existing
packed FP32 MATH implementation, with byte-exact output and all input gradients.
Independent dense-mask correctness is covered by test_split_attention.py. QKV
values are synthetic; document boundaries come from the specified real corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

import cat_yoko.attention as native
from operators.rocm import packed_attention as packed
from operators.rocm.attention_bench import _emit, _measure
from operators.rocm.packed_attention_bench import input_values, load_documents, pid_snapshot, snapshot
from operators.rocm.split_attention import new_stats, split_window_sdpa


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def graph_nodes(output):
    pending, seen, counts = [output.grad_fn], set(), Counter()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(value for value, _ in node.next_functions)
    return dict(sorted(counts.items()))


def exact(actual, reference):
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    bitwise = torch.equal(actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
    return {'finite': finite, 'bitwise': bitwise,
            'max_abs': float((actual.float() - reference.float()).abs().max()) if finite else None,
            'pass': finite and bitwise}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[8701, 10222, 12000, 14000, 16000, 17478, 18000, 19530])
    parser.add_argument('--eos-id', type=int, default=1)
    parser.add_argument('--layouts', nargs='+', choices=('packed_qkv', 'cross_cache'), default=['packed_qkv', 'cross_cache'])
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory-fraction', type=float, default=0.8,
                        help='skip fixtures whose estimated attention peak exceeds this share of currently free VRAM')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if (not args.data.is_file() or min(args.rows) < 0 or args.warmup < 0 or args.repeats < 1
            or not 0 < args.memory_fraction <= 0.8):
        parser.error('existing data, nonnegative rows/warmup and positive repeats required')
    if args.output.exists():
        parser.error('refusing to append to an existing benchmark output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        parser.error('CUDA/HIP GPU required')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    memory_budget = int(free_bytes * args.memory_fraction)
    root = Path(__file__).resolve().parents[2]
    dependencies = ['operators/rocm/' + name for name in (
        'split_attention_bench.py', 'split_attention.py', 'packed_attention.py',
        'packed_attention_bench.py', 'attention_bench.py', 'gpu_wait.py')]
    dependencies += ['cat_yoko/' + name for name in ('attention.py', 'config.py', 'rope.py', 'ops.py')]
    _emit({'event': 'environment', 'torch': torch.__version__, 'hip': torch.version.hip,
           'gpu': torch.cuda.get_device_name(), 'shape': [1, 16, 4096, 128], 'kv_heads': 2,
           'window': 8192, 'dtype': 'bf16', 'warmup': args.warmup, 'repeats': args.repeats,
           'free_gpu_bytes': free_bytes, 'total_gpu_bytes': total_bytes,
           'memory_budget_bytes': memory_budget,
           'memory_estimate': '4 * FP32 sum(document score elements) + 256MiB for QKV, gradients and assembly; conservative prefilter, not a reservation',
           'file_sha256': {name: digest(root / name) for name in dependencies},
           'data_sha256': digest(args.data), 'reference': 'existing packed FP32 MATH SDPA',
           'gate': 'finite and byte-exact output/dQ/dK/dV against existing packed implementation',
           'timing_scope': 'casts, GQA, fragment assembly, merge_heads and complete input backward; excludes projections and fixture creation',
           'cold_scope': 'document-plan cache cleared each invocation, includes host planning',
           'comparison': 'baseline_before / split / baseline_after, identical fixtures',
           'input_layout_scope': 'Q/K materialized to dense BSHD as after qk_norm/RoPE; V retains fused QKV or KV-cache row stride',
           'full_model_gate_passed': False, **pid_snapshot()}, args.output)
    fixtures = []
    for row in args.rows:
        fixtures.append(load_documents(args.data, seq=4096, batch=1, row=row,
                                      eos_id=args.eos_id, synthetic_doc_len=64))
    for length in (64, 1024, 4096):
        fixtures.append((torch.arange(4096).div(length, rounding_mode='floor').unsqueeze(0),
                         {'kind': 'uniform_synthetic', 'length': length, 'count': 4096 // length}))
    totals = {'passed': 0, 'failed': 0, 'errors': 0, 'split_exercised': 0, 'fixtures_skipped_memory': 0}
    for docs_cpu, provenance in fixtures:
        plan = packed._document_plan(docs_cpu, None)
        lengths = [end - start for start, end in plan[0]]
        estimated_peak = 4 * 16 * sum(length * length for length in lengths) * 4 + (256 << 20)
        if estimated_peak > memory_budget:
            _emit({'event': 'skipped_fixture', 'reason': 'insufficient_free_gpu_memory',
                   'documents': provenance, 'document_lengths': lengths,
                   'estimated_peak_bytes': estimated_peak, 'memory_budget_bytes': memory_budget}, args.output)
            totals['fixtures_skipped_memory'] += 1
            continue
        # Match production: a single document needs no document mask.
        docs = docs_cpu.to('cuda') if len(lengths) > 1 else None
        for layout in args.layouts:
            packed.clear_doc_plan_cache()
            values = input_values(4096, 1, torch.bfloat16, layout, args.seed)
            # QK normalization and RoPE materialize Q/K before production SDPA.
            # Keep the original V stride; fixture preparation is outside timing.
            values = tuple(v.transpose(1, 2).contiguous().transpose(1, 2) if i < 2 else v
                           for i, v in enumerate(values))
            dy = native.merge_heads(torch.randn(values[0].shape, device='cuda', dtype=torch.bfloat16,
                                   generator=torch.Generator(device='cuda').manual_seed(args.seed + 10000)))
            baseline_stats, candidate_stats = packed.new_stats(), new_stats()

            def baseline(q, k, v):
                return native.merge_heads(packed.packed_window_sdpa(q, k, v, 8192, docs, stats=baseline_stats))

            def candidate(q, k, v):
                return native.merge_heads(split_window_sdpa(q, k, v, 8192, docs, stats=candidate_stats))

            record = {'event': 'comparison', 'layout': layout, 'documents': provenance,
                      'document_lengths': lengths, 'input_strides': [list(v.stride()) for v in values],
                      'dy_stride': list(dy.stride()), 'gpu_processes_before': pid_snapshot()}
            try:
                reference, ref_grads = snapshot(baseline, values, dy)
                actual, grads = snapshot(candidate, values, dy)
                record['output'] = exact(actual, reference)
                record['gradients'] = {label: exact(a, b) for label, a, b in zip(('dq', 'dk', 'dv'), grads, ref_grads)}
                passed = record['output']['pass'] and all(g['pass'] for g in record['gradients'].values())
                record['correctness_path_counts'] = json.loads(json.dumps(candidate_stats))
                totals['split_exercised'] += int(candidate_stats['split_optimized_calls'] > 0)
                record['status'] = 'pass' if passed else 'numerical_failure'
                del actual, grads, reference, ref_grads
                if passed:
                    # Verify that the autograd structure changed, without retaining a graph during timing.
                    for label, operation in (('baseline', baseline), ('split', candidate)):
                        qkv = tuple(v.detach().requires_grad_() for v in values)
                        output = operation(*qkv)
                        record[label + '_graph_nodes'] = graph_nodes(output)
                        del output, qkv
                    for mode in ('warm', 'cold'):
                        measurements = {}
                        for label, operation in (('baseline_before', baseline), ('split', candidate), ('baseline_after', baseline)):
                            def forward_backward():
                                if mode == 'cold':
                                    packed.clear_doc_plan_cache()
                                qkv = tuple(v.detach().requires_grad_() for v in values)
                                return torch.autograd.grad(operation(*qkv), qkv, dy)

                            measurements[label] = _measure(forward_backward, args.warmup, args.repeats)
                        for metric in ('wall_ms', 'gpu_ms'):
                            fastest = min(measurements[label][metric] for label in ('baseline_before', 'baseline_after'))
                            measurements['speedup_' + metric] = fastest / measurements['split'][metric]
                            measurements['latency_reduction_' + metric] = 1 - measurements['split'][metric] / fastest
                        record[mode] = measurements
                    totals['passed' if passed else 'failed'] += 1
                else:
                    totals['failed'] += 1
            except Exception as error:
                record.update(status='error', error=type(error).__name__ + ': ' + str(error))
                totals['errors'] += 1
            record['gpu_processes_after'] = pid_snapshot()
            _emit(record, args.output)
            del values, dy
            gc.collect()
            torch.cuda.empty_cache()
        del docs
    _emit({'event': 'summary', **totals}, args.output)
    return int(bool(totals['failed'] or totals['errors'] or not totals['split_exercised']))


if __name__ == '__main__':
    raise SystemExit(main())
