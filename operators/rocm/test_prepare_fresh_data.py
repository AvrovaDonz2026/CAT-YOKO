"""Fresh text exclusion, fixed-shard locks and packing without data downloads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from operators.rocm import prepare_fresh_data as fresh
from operators.rocm import prepare_real_data as base
from operators.rocm.test_prepare_real_data import FakeTokenizer, matching_documents


class FreshDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.old = self.root / 'old'; self.old.mkdir()
        self.old_train = matching_documents('oldtrain', 'train', 2)
        self.old_eval = matching_documents('oldeval', 'eval', 2)
        self.tokenizer_metadata = {'files_sha256': {'tokenizer.json': 'f' * 64}, 'add_special_tokens': False}
        manifest = base.make_plan()
        manifest.update(status='complete', tokenizer=self.tokenizer_metadata, hashes={})
        for split, texts in (('train', self.old_train), ('eval', self.old_eval)):
            filename = self.old / (split + '.docs.sha256')
            filename.write_text(''.join(base.document_hash(text) + '\n' for text in texts))
            manifest['hashes'][split] = {'path': filename.name, 'sha256': base.file_digest(filename), 'documents': len(texts)}
        base.write_json(self.old / 'manifest.json', manifest)
        self.manifest, _, self.evidence = fresh.read_old(self.old)
        def metadata(repo, revision, paths):
            return {path: {'bytes': 1024, 'official_sha256': base.document_hash(repo + path),
                           'metadata_method': 'test_official_metadata'} for path in paths}
        self.lock = fresh.resolve_lock(self.manifest, self.evidence, metadata=metadata)
        self.lock_path = self.root / 'source-lock.json'; base.write_json(self.lock_path, self.lock)

    def rows(self):
        rows = {}
        for key, quota in (('en', 6), ('zh', 3), ('math', 1)):
            selected = matching_documents('fresh-' + key, 'train', quota) + matching_documents('fresh-' + key, 'eval', quota)
            rows[key] = [{'content': text} for text in selected]
            rows[key][:0] = [{'content': '  ' + self.old_train[0] + '\n'}, {'content': self.old_eval[0]}]
        rows['zh'].insert(2, rows['en'][2])
        return rows

    def prepare(self, *, rows=None, metadata=None):
        tokenizer = FakeTokenizer(); rows = self.rows() if rows is None else rows
        result = fresh.prepare_fresh(exclude_data_dir=self.old, source_lock=self.lock_path,
            tokenizer=tokenizer, tokenizer_metadata=metadata or self.tokenizer_metadata,
            out_dir=self.root / 'fresh', train_tokens=80, eval_tokens=80, seq_len=8,
            row_loader=lambda source: iter(rows[source.key]), log=lambda event: None)
        return result, tokenizer

    def test_new_shards_keep_original_revision_mixture_and_need_official_sha(self):
        sources, _ = fresh.validate_lock(self.lock, self.manifest, self.evidence)
        self.assertEqual([s.weight for s in sources], [6, 3, 1])
        for current, old in zip(sources, self.manifest['sources']):
            self.assertEqual(current.revision, old['revision'])
            self.assertFalse({path for path, _ in current.files} & {item['path'] for item in old['files']})
            self.assertEqual([int(path.split('-part-')[1].split('-')[0]) for path, _ in current.files], [3, 4])
        for mutation in (lambda lock: lock['sources'][0]['files'][0].update(official_sha256=None),
                         lambda lock: lock['sources'][0].update(revision='e' * 40),
                         lambda lock: lock['sources'][0].update(weight=0.5),
                         lambda lock: lock['sources'][0]['files'][0].update(path=self.manifest['sources'][0]['files'][0]['path'])):
            value = copy.deepcopy(self.lock); mutation(value)
            with self.assertRaises(ValueError): fresh.validate_lock(value, self.manifest, self.evidence)

    def test_old_train_and_eval_excluded_before_tokenizing_with_all_intersections_zero(self):
        result, tokenizer = self.prepare()
        self.assertEqual(result['dataset_kind'], 'fresh_exact_text_excluded')
        self.assertEqual(result['schema_version'], 2)
        self.assertEqual(result['hash_intersection'], 0)
        self.assertEqual(result['old_hash_intersections'], {
            'new_train_old_train': 0, 'new_train_old_eval': 0,
            'new_eval_old_train': 0, 'new_eval_old_eval': 0})
        self.assertFalse(set(tokenizer.seen) & set(self.old_train + self.old_eval))
        self.assertEqual(len(tokenizer.seen), 20)
        self.assertEqual(result['source_stats']['zh']['duplicate_documents'], 1)
        for source in result['excluded_source_stats'].values():
            self.assertEqual(source, {'old_train_excluded': 1, 'old_eval_excluded': 1})
        self.assertEqual(result['outputs']['train']['sequences'], 10)
        self.assertEqual(result['outputs']['eval']['sequences'], 10)
        self.assertEqual(result['outputs']['train']['bytes'], 320)
        self.assertEqual(json.loads((self.root / 'fresh/status.json').read_text())['status'], 'complete')
        self.assertTrue(all(file['official_sha256'] for source in result['sources'] for file in source['files']))

    def test_exclusion_hash_tampering_and_tokenizer_change_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'tokenizer'):
            self.prepare(metadata={'files_sha256': {'tokenizer.json': 'e' * 64}})
        old = self.old / 'eval.docs.sha256'; old.write_text('e' * 64 + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            fresh.read_old(self.old)

    def test_exhaustion_does_not_publish_complete_bins_or_manifest_and_old_files_untouched(self):
        before = {p.name: p.read_bytes() for p in self.old.iterdir()}
        rows = {key: [{'content': self.old_train[0]}, {'content': self.old_eval[0]}] for key in fresh.WEIGHTS}
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            self.prepare(rows=rows)
        out = self.root / 'fresh'
        self.assertFalse((out / 'manifest.json').exists())
        self.assertFalse((out / 'train.bin').exists())
        self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'failed')
        self.assertEqual({p.name: p.read_bytes() for p in self.old.iterdir()}, before)

    def test_old_exclusion_change_during_preparation_prevents_publication(self):
        rows = self.rows()
        def loader(source):
            if source.key == 'math':
                with (self.old / 'train.docs.sha256').open('a') as file: file.write('a' * 64 + '\n')
            return iter(rows[source.key])
        with self.assertRaises(ValueError):
            fresh.prepare_fresh(exclude_data_dir=self.old, source_lock=self.lock_path,
                tokenizer=FakeTokenizer(), tokenizer_metadata=self.tokenizer_metadata, out_dir=self.root / 'fresh',
                train_tokens=80, eval_tokens=80, seq_len=8, row_loader=loader, log=lambda event: None)
        self.assertFalse((self.root / 'fresh/manifest.json').exists())

    def test_http_reader_records_pinned_sha_without_claiming_full_file_verification(self):
        source = fresh.validate_lock(self.lock, self.manifest, self.evidence)[0][0]
        sha = {file['path']: file['official_sha256'] for file in self.lock['sources'][0]['files']}
        events, handles, calls = [], [], []
        import io
        def open_file(url, **kwargs):
            file = io.BytesIO(); file.size = 1024; handles.append(file); calls.append((url, kwargs)); return file
        class ParquetFile:
            def __init__(self, handle): pass
            def iter_batches(self, **kwargs):
                column = types.SimpleNamespace(to_pylist=lambda: ['fresh sample'])
                yield types.SimpleNamespace(column=lambda n: column)
        with patch.dict('sys.modules', {'fsspec': types.SimpleNamespace(open=open_file),
            'pyarrow': types.SimpleNamespace(parquet=types.SimpleNamespace(ParquetFile=ParquetFile))}):
            reader = fresh.iter_fresh_rows(source, sha, endpoint='https://hf-mirror.com', log=events.append)
            self.assertEqual(next(reader), {'content': 'fresh sample'}); reader.close()
        self.assertTrue(handles[0].closed)
        self.assertTrue(calls[0][0].startswith('https://hf-mirror.com/'))
        self.assertTrue(events[0]['url'].startswith('https://huggingface.co/'))
        self.assertFalse(events[0]['full_file_verified'])
        self.assertEqual(events[0]['official_sha256'], sha[source.files[0][0]])

    def test_official_metadata_uses_public_api_and_falls_back_to_sha256_headers(self):
        requested = ['part3.parquet', 'part4.parquet']; calls = []
        class Api:
            def __init__(self, **kwargs): calls.append(kwargs)
            def get_paths_info(self, **kwargs):
                calls.append(kwargs)
                return [types.SimpleNamespace(path=requested[0], size=99, lfs={'sha256': 'a' * 64})]
        fallback = lambda url, **kwargs: types.SimpleNamespace(size=88, etag='b' * 64)
        with patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(HfApi=Api,get_hf_file_metadata=fallback)}):
            records = fresh.official_metadata('official/repo', 'c' * 40, requested)
        self.assertIs(calls[0]['token'], False); self.assertIs(calls[1]['token'], False)
        self.assertEqual(records[requested[0]]['metadata_method'], 'official_paths_info')
        self.assertEqual(records[requested[1]]['metadata_method'], 'official_resolve_headers')
        with patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(HfApi=Api,
                get_hf_file_metadata=lambda *a, **kw: types.SimpleNamespace(size=88, etag='short-git-oid'))}):
            with self.assertRaises(ValueError): fresh.official_metadata('official/repo', 'c' * 40, requested)

    def test_cli_tokenizer_failure_writes_failed_status_without_complete_manifest(self):
        out = self.root / 'fresh'
        with patch.object(base, 'load_local_tokenizer', side_effect=ImportError('missing dependency')):
            with self.assertRaises(ImportError):
                fresh.main(['--exclude-data-dir', str(self.old), '--source-lock', str(self.lock_path),
                            '--tokenizer-dir', str(self.root / 'tokenizer'), '--out-dir', str(out)])
        self.assertEqual(json.loads((out / 'status.json').read_text()), {'status': 'failed', 'error_type': 'ImportError'})
        self.assertFalse((out / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
