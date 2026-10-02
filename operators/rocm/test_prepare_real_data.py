"""Corpus split/packing/provenance regressions with no external dependencies."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from operators.rocm.prepare_real_data import (
    EOS_ID, SOURCES, Source, document_hash, document_split, main,
    iter_source_rows, make_plan, normalize_endpoint, prepare_corpus, row_quotas, validate_tokenizer,
)


class FakeTokenizer:
    vocab_size, eos_id = 130560, EOS_ID

    def __init__(self):
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        marker = 10 + int(document_hash(text)[:6], 16) % 100000
        # Both already-terminated and unterminated documents must pack exactly
        # one row without an accidental second EOS or automatic BOS.
        return [marker] * 7 + ([EOS_ID] if int(document_hash(text)[-1], 16) % 2 else [])


def matching_documents(prefix, split, count):
    output, i = [], 0
    while len(output) < count:
        text = f"{prefix} document {i}"
        if document_split(document_hash(text)) == split:
            output.append(text)
        i += 1
    return output


class PrepareRealDataTests(unittest.TestCase):
    def test_tokenizer_checks_added_tokens_against_embedding_vocabulary(self):
        class MiniCPMTokenizer:
            vocab_size = 130072
            eos_token_id = 1

            def __len__(self):
                return 130560

            def get_vocab(self):
                return {str(i): i for i in range(130560)}

        tokenizer = MiniCPMTokenizer()
        self.assertEqual(validate_tokenizer(tokenizer),
                         {"base_vocab_size": 130072, "total_vocab_size": 130560})
        tokenizer.eos_token_id = 2
        with self.assertRaises(ValueError):
            validate_tokenizer(tokenizer)

    def test_default_pinned_plan_and_whole_row_budget(self):
        plan = make_plan()
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["target_tokens"], {"train": 79998976, "eval": 999424})
        self.assertEqual(row_quotas(80_000_000, 4096), {"en": 11719, "zh": 5859, "math": 1953})
        self.assertEqual(row_quotas(1_000_000, 4096), {"en": 147, "zh": 73, "math": 24})
        for source, entry in zip(SOURCES, plan["sources"]):
            self.assertEqual(len(source.revision), 40)
            self.assertEqual(len(entry["files"]), 2)
            self.assertTrue(all(source.revision in item["url"] for item in entry["files"]))
            self.assertTrue(all(len(item["official_sha256"]) == 64 for item in entry["files"]))
        self.assertEqual(document_hash(" hello \n"), document_hash("hello"))
        self.assertNotEqual(document_hash("a b"), document_hash("a  b"))

    def _prepare(self, directory, seed=33):
        sources = tuple(Source(key, "fake/repo", "f" * 40, weight, ())
                        for key, weight in (("en", 6), ("zh", 3), ("math", 1)))
        token_budget, seq_len = 80, 8
        quotas = row_quotas(token_budget, seq_len, sources)
        rows = {}
        all_train, all_eval = [], []
        # Train arrives first, then many unusable extra train docs, then eval.
        # The reader must continue after the train quota is reached.
        for source in sources:
            train = matching_documents(source.key, "train", quotas[source.key] + 3)
            validation = matching_documents(source.key, "eval", quotas[source.key])
            rows[source.key] = [{"content": "  " + text + "\n"} for text in train + validation]
            all_train.extend(train[:quotas[source.key]])
            all_eval.extend(validation)
        # Repeat both selected and quota-discarded documents across sources.
        rows["zh"].insert(0, rows["en"][0])
        rows["zh"].insert(1, rows["en"][quotas["en"]])
        rows["math"].insert(0, {"content": "   "})
        rows["math"].insert(1, {"content": None})
        tokenizer = FakeTokenizer()
        manifest = prepare_corpus(out_dir=directory, tokenizer=tokenizer,
                                  train_tokens=token_budget, eval_tokens=token_budget,
                                  seq_len=seq_len, seed=seed, sources=sources,
                                  row_loader=lambda source: iter(rows[source.key]), log=lambda event: None)
        return manifest, tokenizer, all_train, all_eval

    def test_disjoint_hashes_cross_source_dedup_eos_and_exact_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest, tokenizer, train_docs, eval_docs = self._prepare(path)
            train_hashes = set((path / "train.docs.sha256").read_text().splitlines())
            eval_hashes = set((path / "eval.docs.sha256").read_text().splitlines())
            self.assertFalse(train_hashes & eval_hashes)
            self.assertEqual(train_hashes, {document_hash(text) for text in train_docs})
            self.assertEqual(eval_hashes, {document_hash(text) for text in eval_docs})
            self.assertEqual(len(tokenizer.seen), 20)
            self.assertTrue(all(text == text.strip() for text in tokenizer.seen))
            self.assertEqual(manifest["hash_intersection"], 0)
            self.assertEqual(manifest["source_stats"]["zh"]["duplicate_documents"], 2)
            self.assertEqual(manifest["source_stats"]["en"]["quota_skipped_documents"], 3)
            self.assertEqual(manifest["source_stats"]["math"]["empty_documents"], 2)
            for split in ("train", "eval"):
                tokens = struct.unpack("<" + "i" * 80, (path / f"{split}.bin").read_bytes())
                for row in range(10):
                    chunk = tokens[row * 8:(row + 1) * 8]
                    self.assertEqual(chunk[-1], EOS_ID)
                    self.assertNotIn(EOS_ID, chunk[:-1])
                    self.assertGreaterEqual(min(chunk[:-1]), 10)
                sidecar = json.loads((path / f"{split}.bin.meta.json").read_text())
                self.assertEqual(sidecar["eos_id"], EOS_ID)
                self.assertEqual(sidecar["tokens"], 80)
                self.assertEqual(sidecar["source_row_quotas"], {"en": 6, "zh": 3, "math": 1})
            self.assertEqual(json.loads((path / "status.json").read_text())["status"], "complete")

    def test_shuffled_mixture_is_reproducible_and_seed_changes_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, seed in (("one", 9), ("two", 9), ("three", 10)):
                self._prepare(root / name, seed)
            for split in ("train", "eval"):
                first = (root / "one" / f"{split}.bin").read_bytes()
                self.assertEqual(first, (root / "two" / f"{split}.bin").read_bytes())
                self.assertNotEqual(first, (root / "three" / f"{split}.bin").read_bytes())

    def test_exhaustion_and_row_loader_failure_never_create_complete_manifest(self):
        source = Source("one", "fake/repo", "f" * 40, 1, ())
        for kind in ("exhaustion", "exception"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)

                def loader(_):
                    if kind == "exception":
                        raise OSError("simulated network failure")
                    return iter([{"content": matching_documents("short", "train", 1)[0]}])

                with self.assertRaises((OSError, RuntimeError)):
                    prepare_corpus(out_dir=path, tokenizer=FakeTokenizer(), train_tokens=16,
                                   eval_tokens=8, seq_len=8, sources=(source,), row_loader=loader,
                                   log=lambda event: None)
                self.assertFalse((path / "manifest.json").exists())
                self.assertFalse((path / "train.bin").exists())
                self.assertEqual(json.loads((path / "status.json").read_text())["status"], "failed")

    def test_manifest_only_avoids_loading_tokenizer_or_data(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("operators.rocm.prepare_real_data.load_local_tokenizer", side_effect=AssertionError), \
                patch("operators.rocm.prepare_real_data.iter_source_rows", side_effect=AssertionError), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--out-dir", directory, "--manifest-only"]), 0)
            path = Path(directory)
            self.assertTrue((path / "plan.json").exists())
            self.assertEqual(json.loads((path / "status.json").read_text()),
                             {"status": "planned", "downloaded": False})
            self.assertFalse((path / "train.bin").exists())

    def test_refuses_replacing_existing_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self._prepare(path)
            before = (path / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self._prepare(path)
            self.assertEqual(before, (path / "manifest.json").read_bytes())

    def test_tokenizer_failure_replaces_previous_planned_status(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            main(["--out-dir", directory, "--manifest-only"])
            with patch("operators.rocm.prepare_real_data.load_local_tokenizer",
                       side_effect=ImportError("missing transformers")):
                with self.assertRaises(ImportError):
                    main(["--out-dir", directory, "--tokenizer-dir", directory])
            status = json.loads((Path(directory) / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error_type"], "ImportError")

    def test_mirror_plan_retains_official_provenance_and_no_network(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            main(["--out-dir", directory, "--manifest-only", "--endpoint", "https://hf-mirror.com/"])
            plan = json.loads((Path(directory) / "plan.json").read_text())
            self.assertEqual(plan["download_endpoint"], "https://hf-mirror.com")
            self.assertEqual(plan["reader"], "http")
            self.assertTrue(all(item["url"].startswith("https://huggingface.co/")
                                for source in plan["sources"] for item in source["files"]))
        for invalid in ("https://user:password@hf-mirror.com", "https://hf-mirror.com/path", "file:///tmp"):
            with self.assertRaises(ValueError):
                normalize_endpoint(invalid)

    def test_http_reader_uses_range_url_and_closes_on_early_quota_stop(self):
        calls, events, handles = [], [], []

        def open_range(url, **kwargs):
            calls.append((url, kwargs))
            handle = io.BytesIO()
            handle.size = SOURCES[0].files[0][1]
            handles.append(handle)
            return handle

        class ParquetFile:
            def __init__(self, handle):
                self.handle = handle

            def iter_batches(self, **kwargs):
                self.asserted_kwargs = kwargs
                if kwargs != {"batch_size": 128, "columns": ["content"]}:
                    raise AssertionError(kwargs)
                column = types.SimpleNamespace(to_pylist=lambda: ["first document", "second document"])
                yield types.SimpleNamespace(column=lambda index: column)

        modules = {"fsspec": types.SimpleNamespace(open=open_range),
                   "pyarrow": types.SimpleNamespace(parquet=types.SimpleNamespace(ParquetFile=ParquetFile))}
        with patch.dict("sys.modules", modules):
            rows = iter_source_rows(SOURCES[0], events.append, endpoint="https://hf-mirror.com")
            self.assertEqual(next(rows), {"content": "first document"})
            rows.close()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0].startswith("https://hf-mirror.com/datasets/"))
        self.assertIn(SOURCES[0].revision, calls[0][0])
        self.assertEqual(calls[0][1]["block_size"], 16 << 20)
        self.assertEqual(calls[0][1]["timeout"], 30)
        self.assertTrue(handles[0].closed)
        self.assertTrue(events[0]["url"].startswith("https://huggingface.co/"))
        self.assertFalse(events[0]["full_file_verified"])


if __name__ == "__main__":
    unittest.main()
