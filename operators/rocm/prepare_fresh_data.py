"""Prepare fresh B0 text from new pinned shards, excluding all old documents.

Run only on the training machine. Resolve public metadata into a source lock
first, then prepare with the existing local MiniCPM5 tokenizer. No torch imports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Mapping

from operators.rocm import prepare_real_data as base

WEIGHTS = {'en': 6, 'zh': 3, 'math': 1}
HASH = re.compile(r'[0-9a-f]{64}')


def need(value, message):
    if not value:
        raise ValueError(message)


def read_old(directory):
    directory = Path(directory).resolve()
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    expected = base.make_plan()
    need(manifest.get('status') == 'complete' and manifest.get('seq_len') == 4096
         and manifest.get('vocab_size') == base.VOCAB_SIZE and manifest.get('eos_id') == base.EOS_ID
         and manifest.get('normalization') == expected['normalization']
         and manifest.get('split_rule') == expected['split_rule'], 'old corpus is not the verified B0 text format')
    need([source.get('key') for source in manifest.get('sources', [])] == list(WEIGHTS), 'old source mixture differs')
    for source in manifest['sources']:
        need(source.get('weight') == WEIGHTS[source['key']] / 10
             and re.fullmatch(r'[0-9a-f]{40}', source.get('revision', '')),
             'old source mixture or revision is not pinned')
    selected, evidence = {}, {}
    for split in ('train', 'eval'):
        filename = directory / (split + '.docs.sha256')
        record = manifest.get('hashes', {}).get(split, {})
        need(filename.is_file() and record.get('path') == filename.name
             and base.file_digest(filename) == record.get('sha256'), 'old document hash evidence SHA differs')
        lines = filename.read_text(encoding='ascii').splitlines()
        need(all(HASH.fullmatch(line) for line in lines)
             and len(lines) == len(set(lines)) == record.get('documents'), 'old document hash evidence is incomplete')
        selected[split] = set(lines)
        evidence[split] = {'path': str(filename), 'sha256': record['sha256'], 'documents': len(lines)}
    need(not selected['train'] & selected['eval'], 'old training and validation documents overlap')
    return manifest, selected, {'manifest_path': str(path), 'manifest_sha256': base.file_digest(path),
                                'document_hashes': evidence}


def next_paths(source, count=2):
    need(type(count) is int and 0 < count <= 8, 'choose between one and eight bounded fresh shards')
    files = source.get('files', [])
    need(files, 'old source has no explicit shard inventory')
    expression = re.compile(r'^(.*-part-)(\d+)(-of-)(\d+)(\.parquet)$')
    matches = [expression.fullmatch(item['path']) for item in files]
    need(all(matches), 'unsupported source shard naming')
    first = matches[0]
    need(all((m[1], m[3], m[4], len(m[2])) == (first[1], first[3], first[4], len(first[2])) for m in matches),
         'source shard inventory has inconsistent naming')
    start = max(int(m[2]) for m in matches) + 1
    need(start + count - 1 <= int(first[4]), 'fresh shard request exceeds the source')
    return [first[1] + str(n).zfill(len(first[2])) + first[3] + first[4] + first[5]
            for n in range(start, start + count)]


def official_metadata(repo, revision, paths):
    """Public official metadata only; fall back to official resolve headers."""
    from huggingface_hub import HfApi, get_hf_file_metadata
    records = {}
    try:
        entries = HfApi(endpoint='https://huggingface.co', token=False).get_paths_info(
            repo_id=repo, paths=paths, revision=revision, repo_type='dataset', token=False)
        for entry in entries:
            lfs = getattr(entry, 'lfs', None)
            sha = lfs.get('sha256') if isinstance(lfs, Mapping) else getattr(lfs, 'sha256', None)
            if HASH.fullmatch(str(sha)) and type(entry.size) is int and entry.size > 0:
                records[entry.path] = {'bytes': entry.size, 'official_sha256': sha, 'metadata_method': 'official_paths_info'}
    except Exception:
        # The fallback still requires an official size and full SHA256.
        pass
    for path in paths:
        if path not in records:
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}'
            entry = get_hf_file_metadata(url, token=False)
            sha = str(entry.etag).strip('"')
            need(HASH.fullmatch(sha) and type(entry.size) is int and entry.size > 0,
                 'official source size/SHA is unavailable; refusing unlocked text')
            records[path] = {'bytes': entry.size, 'official_sha256': sha, 'metadata_method': 'official_resolve_headers'}
    need(set(records) == set(paths), 'official metadata omitted requested shards')
    return records


def resolve_lock(old, old_evidence, *, metadata=official_metadata, shard_count=2):
    sources = []
    for source in old['sources']:
        paths = next_paths(source, shard_count)
        records = metadata(source['repo'], source['revision'], paths)
        need(set(records) == set(paths), 'source resolver omitted a fresh shard')
        files = []
        for path in paths:
            entry = records[path]
            need(type(entry.get('bytes')) is int and entry['bytes'] > 0
                 and HASH.fullmatch(str(entry.get('official_sha256', ''))), 'new shard is missing official size/SHA')
            files.append({'path': path, **entry,
                          'url': f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{path}"})
        sources.append({'key': source['key'], 'repo': source['repo'], 'revision': source['revision'],
                        'weight': source['weight'], 'files': files})
    return {'schema_version': 1, 'status': 'source_locked', 'old_manifest_sha256': old_evidence['manifest_sha256'],
            'sources': sources, 'metadata_endpoint': 'https://huggingface.co', 'authentication': False,
            'eligible_shard_bytes': sum(f['bytes'] for s in sources for f in s['files']),
            'freshness_scope': 'New named shards plus exact full-document SHA exclusion; near duplicates are not detected.'}


def validate_lock(lock, old, old_evidence):
    need(lock.get('status') == 'source_locked' and lock.get('old_manifest_sha256') == old_evidence['manifest_sha256'],
         'source lock belongs to a different old corpus')
    need([s.get('key') for s in lock.get('sources', [])] == list(WEIGHTS), 'source lock mixture is incomplete')
    sources, sha = [], {}
    for current, previous in zip(lock['sources'], old['sources']):
        need(all(current.get(name) == previous.get(name) for name in ('key', 'repo', 'revision', 'weight')),
             'source lock changed the accepted mixture or revision')
        files = current.get('files', [])
        need(files and len(files) <= 8, 'source lock has an unbounded shard list')
        eligible = next_paths(previous, len(files))
        need([item.get('path') for item in files] == eligible, 'source lock reuses old or nonconsecutive shards')
        for item in files:
            path = PurePosixPath(item['path'])
            need(not path.is_absolute() and '..' not in path.parts and '\\' not in item['path']
                 and type(item.get('bytes')) is int and item['bytes'] > 0
                 and HASH.fullmatch(str(item.get('official_sha256', ''))), 'invalid locked source identity')
            url = f"https://huggingface.co/datasets/{current['repo']}/resolve/{current['revision']}/{item['path']}"
            need(item.get('url') == url, 'source lock changed the official URL')
            sha[item['path']] = item['official_sha256']
        sources.append(base.Source(current['key'], current['repo'], current['revision'], WEIGHTS[current['key']],
                                   tuple((item['path'], item['bytes']) for item in files)))
    return tuple(sources), sha


def iter_fresh_rows(source, sha, *, endpoint, source_dir=None, log=base.emit):
    from pyarrow import parquet
    for (name, size), official in zip(source.files, source.urls):
        event = {'event': 'open_file', 'source': source.key, 'url': official,
                 'expected_bytes': size, 'official_sha256': sha[name], 'full_file_verified': False}
        if source_dir is not None:
            path = Path(source_dir) / name
            if not path.is_file():
                path = Path(source_dir) / Path(name).name
            need(path.is_file() and path.stat().st_size == size and base.file_digest(path) == sha[name],
                 'local fresh shard does not match official size/SHA')
            log({**event, 'full_file_verified': True, 'local_path': str(path.resolve())})
            with path.open('rb') as handle:
                for batch in parquet.ParquetFile(handle).iter_batches(batch_size=128, columns=['content']):
                    yield from ({'content': value} for value in batch.column(0).to_pylist())
        else:
            import fsspec
            url = base.normalize_endpoint(endpoint) + official.removeprefix('https://huggingface.co')
            log({**event, 'download_url': url})
            with fsspec.open(url, mode='rb', block_size=16 << 20, cache_type='readahead', timeout=30) as handle:
                need(handle.size == size, 'fresh range source size differs from official lock')
                for batch in parquet.ParquetFile(handle).iter_batches(batch_size=128, columns=['content']):
                    yield from ({'content': value} for value in batch.column(0).to_pylist())


def prepare_fresh(*, exclude_data_dir, source_lock, tokenizer, tokenizer_metadata, out_dir,
                  train_tokens=80_000_000, eval_tokens=1_000_000, seq_len=4096, seed=20261007,
                  endpoint='https://huggingface.co', source_dir=None, row_loader=None, log=base.emit):
    old, excluded, old_evidence = read_old(exclude_data_dir)
    source_lock = Path(source_lock).resolve()
    lock_hash = base.file_digest(source_lock)
    lock = json.loads(source_lock.read_text())
    sources, sha = validate_lock(lock, old, old_evidence)
    need(tokenizer_metadata.get('files_sha256') == old['tokenizer']['files_sha256'], 'fresh tokenizer differs from old corpus')
    out_dir = Path(out_dir).resolve()
    need(not out_dir.exists() or not any(out_dir.iterdir()), 'fresh output directory must be empty')
    need(not out_dir.is_relative_to(Path(exclude_data_dir).resolve()), 'fresh outputs must not overwrite old corpus')
    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_json(out_dir / 'status.json', {'status': 'running', 'stage': 'fresh_text_preparation'})
    counters = {s.key: {'old_train_excluded': 0, 'old_eval_excluded': 0} for s in sources}
    input_files = []
    def record(event):
        if event.get('event') == 'open_file': input_files.append(event)
        log({key: value for key, value in event.items() if key != 'error'})
    def filtered(source):
        rows = iter(row_loader(source) if row_loader else iter_fresh_rows(
            source, sha, endpoint=endpoint, source_dir=source_dir, log=record))
        try:
            for row in rows:
                text = row.get('content')
                if isinstance(text, str) and text.strip():
                    hashed = base.document_hash(text)
                    if hashed in excluded['train']:
                        counters[source.key]['old_train_excluded'] += 1; continue
                    if hashed in excluded['eval']:
                        counters[source.key]['old_eval_excluded'] += 1; continue
                yield row
        finally:
            close = getattr(rows, 'close', None)
            if close is not None: close()
    try:
        with tempfile.TemporaryDirectory(prefix='.fresh-prepare-', dir=out_dir.parent) as temporary:
            working = Path(temporary) / 'corpus'
            manifest = base.prepare_corpus(out_dir=working, tokenizer=tokenizer, tokenizer_metadata=tokenizer_metadata,
                train_tokens=train_tokens, eval_tokens=eval_tokens, seq_len=seq_len, seed=seed,
                sources=sources, endpoint=endpoint, source_dir=source_dir, row_loader=filtered, log=record)
            selected = {split: set((working / (split + '.docs.sha256')).read_text().splitlines()) for split in ('train', 'eval')}
            intersections = {f'new_{new}_old_{prior}': len(selected[new] & excluded[prior])
                             for new in selected for prior in excluded}
            need(not any(intersections.values()) and not selected['train'] & selected['eval'],
                 'fresh document selection overlaps prior data or its validation')
            _, _, rechecked = read_old(exclude_data_dir)
            need(rechecked == old_evidence and base.file_digest(source_lock) == lock_hash,
                 'source lock or old exclusion evidence changed during preparation')
            manifest.update(schema_version=2, dataset_kind='fresh_exact_text_excluded',
                fresh_preparer_sha256=base.file_digest(Path(__file__)),
                old_corpus_exclusion=old_evidence, old_hash_intersections=intersections,
                excluded_source_stats=counters, source_lock={'path': str(source_lock), 'sha256': lock_hash},
                input_files_read=input_files,
                freshness_scope=lock['freshness_scope'],
                validation_protocol='New eval full-text hashes are disjoint from new train and both old splits; old eval remains a separate baseline.')
            for info, locked in zip(manifest['sources'], lock['sources']): info['files'] = locked['files']
            base.write_json(working / 'manifest.json', manifest)
            base.write_json(working / 'plan.json', {**manifest, 'status': 'planned'})
            for path in working.iterdir():
                if path.name not in ('manifest.json', 'status.json'): os.replace(path, out_dir / path.name)
            os.replace(working / 'manifest.json', out_dir / 'manifest.json')
            base.write_json(out_dir / 'status.json', {'status': 'complete', 'outputs': manifest['outputs'],
                'hash_intersection': 0, 'old_hash_intersections': intersections, 'excluded_source_stats': counters})
            log({'event': 'fresh_complete', 'outputs': manifest['outputs'], 'old_hash_intersections': intersections})
            return manifest
    except Exception as error:
        (out_dir / 'manifest.json').unlink(missing_ok=True)
        base.write_json(out_dir / 'status.json', {'status': 'failed', 'error_type': type(error).__name__})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exclude-data-dir', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    parser.add_argument('--resolve-only', action='store_true')
    parser.add_argument('--shards-per-source', type=int, default=2)
    parser.add_argument('--tokenizer-dir', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--train-tokens', type=int, default=80_000_000)
    parser.add_argument('--eval-tokens', type=int, default=1_000_000)
    parser.add_argument('--seq-len', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--source-dir', type=Path, help='optional complete local shards, verified against official size/SHA')
    args = parser.parse_args(argv)
    if args.resolve_only:
        old, _, evidence = read_old(args.exclude_data_dir)
        need(not args.source_lock.exists(), 'refusing to overwrite an existing source lock')
        lock = resolve_lock(old, evidence, shard_count=args.shards_per_source)
        args.source_lock.parent.mkdir(parents=True, exist_ok=True)
        base.write_json(args.source_lock, lock)
        base.emit({'status': 'source_locked', 'source_lock': str(args.source_lock),
                   'eligible_shard_bytes': lock['eligible_shard_bytes']})
        return 0
    if args.tokenizer_dir is None or args.out_dir is None:
        parser.error('--tokenizer-dir and --out-dir are required for preparation')
    if args.seq_len != 4096 or not 50_000_000 <= args.train_tokens <= 100_000_000:
        parser.error('fresh production preparation requires seq4096 and a bounded 50M–100M training budget')
    need(not args.out_dir.exists() or not any(args.out_dir.iterdir()), 'fresh output directory must be empty')
    need(not args.out_dir.resolve().is_relative_to(args.exclude_data_dir.resolve()), 'fresh output must be separate from old corpus')
    try:
        tokenizer, metadata = base.load_local_tokenizer(args.tokenizer_dir)
        prepare_fresh(exclude_data_dir=args.exclude_data_dir, source_lock=args.source_lock,
            tokenizer=tokenizer, tokenizer_metadata=metadata, out_dir=args.out_dir,
            train_tokens=args.train_tokens, eval_tokens=args.eval_tokens, seq_len=args.seq_len,
            seed=args.seed, endpoint=base.normalize_endpoint(args.endpoint), source_dir=args.source_dir)
    except Exception as error:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(args.out_dir / 'status.json', {'status': 'failed', 'error_type': type(error).__name__})
        raise
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1)
