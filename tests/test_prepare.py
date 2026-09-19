#!/usr/bin/env python3
"""OpenBMB prepare → mmap PackedBinStream → tiny C1 step. No HuggingFace download."""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import (
    PackedBinStream,
    doc_ids_from_eos,
    labels_with_doc_boundaries,
    open_stream,
    resolve_eos,
    sidecar_path,
)
from cat_yoko.hf_minicpm import _unwrap_state, load_minicpm_state
from cat_yoko.prepare import iter_hf_texts, iter_local_jsonl, main as prepare_main, mix_documents, prepare
from cat_yoko.recipe import MIXES, PHASE_B, PHASE_B_WITH_CODE, PHASE_D, PHASE_E, default_mix, mix_named
from cat_yoko.tokenizer import HashTokenizer, load_tokenizer
from cat_yoko.train import main as train_main
from cat_yoko.trainer import Trainer
from cat_yoko.upcycle import dummy_minicpm_state


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _long_text(n: int = 8) -> str:
    return ("the quick brown fox jumps over the lazy dog. " * n).strip()


class RecipeTests(unittest.TestCase):
    def test_phase_b_is_all_openbmb_and_sums_to_one(self) -> None:
        self.assertAlmostEqual(sum(s.weight for s in PHASE_B), 1.0)
        self.assertTrue(all(s.repo.startswith("openbmb/") for s in PHASE_B))
        keys = {s.key for s in PHASE_B}
        self.assertEqual(keys, {"ultrafineweb-en", "ultrafineweb-zh", "ultradata-math"})

    def test_code_mix_is_optional_extra(self) -> None:
        self.assertAlmostEqual(sum(s.weight for s in PHASE_B_WITH_CODE), 1.0)
        self.assertTrue(any(s.repo == "bigcode/starcoderdata" for s in PHASE_B_WITH_CODE))
        self.assertIn("phase-b", MIXES)
        self.assertIn("phase-b-code", MIXES)
        self.assertIn("phase-c", MIXES)
        self.assertIn("phase-f", MIXES)
        self.assertEqual(MIXES["phase-c"], PHASE_B)
        self.assertEqual(MIXES["phase-d"], PHASE_D)
        self.assertAlmostEqual(sum(s.weight for s in PHASE_D), 1.0)
        self.assertTrue(any(s.key == "starcoder" for s in PHASE_D))
        self.assertIn("phase-e", MIXES)
        self.assertIn("phase-g", MIXES)
        self.assertNotEqual(MIXES["phase-e"], PHASE_B)
        self.assertAlmostEqual(sum(s.weight for s in PHASE_E), 1.0)
        e_keys = {s.key for s in PHASE_E}
        self.assertTrue({"ultradata-math", "starcoder", "ultrachat"} <= e_keys)
        self.assertEqual(default_mix("C-index"), "phase-c")
        self.assertEqual(default_mix("D-128k"), "phase-d")
        self.assertEqual(default_mix("E"), "phase-e")
        self.assertEqual(default_mix("F"), "phase-f")
        self.assertEqual(default_mix("B0"), "phase-b")

    def test_sft_turns_ultrachat_and_alpaca(self) -> None:
        from cat_yoko.sft import encode_sft, sft_turns, turns_to_text

        chat = sft_turns({"data": ["hello there", "hi, how can I help"]})
        self.assertEqual(chat[0][0], "user")
        self.assertEqual(chat[1][0], "assistant")
        self.assertIn("user:", turns_to_text(chat))
        alpaca = sft_turns({"instruction": "add", "input": "1+1", "output": "2"})
        self.assertEqual([r for r, _ in alpaca], ["user", "assistant"])
        tok = HashTokenizer(64)
        packed = encode_sft(tok, chat, seq_len=32)
        self.assertIsNotNone(packed)
        ids, labels = packed
        self.assertEqual(len(ids), len(labels))
        self.assertIn(-100, labels)
        self.assertTrue(any(x != -100 for x in labels))
        short = encode_sft(tok, chat, seq_len=8)
        self.assertIsNotNone(short)
        self.assertTrue(any(x != -100 for x in short[1]))

    def test_pack_sft_pairs_fills_seq_len(self) -> None:
        from cat_yoko.sft import pack_sft_pairs

        a = ([1, 2, 3], [-100, -100, 3])
        b = ([4, 5], [-100, 5])
        packed = list(pack_sft_pairs(iter([a, b]), seq_len=8))
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0][0], [1, 2, 3, 4, 5])
        self.assertEqual(packed[0][1], [-100, -100, 3, -100, 5])

    def test_mix_named_rejects_unknown(self) -> None:
        with self.assertRaises(KeyError):
            mix_named("ultra-chat")


class TokenizerTests(unittest.TestCase):
    def test_hash_appends_eos_and_avoids_eos_in_body(self) -> None:
        tok = HashTokenizer(128, eos_id=2)
        ids = tok.encode("abc")
        self.assertEqual(ids[-1], 2)
        self.assertNotIn(2, ids[:-1])

    def test_load_dummy_does_not_need_transformers(self) -> None:
        cfg = CATYokoConfig.tiny()
        tok = load_tokenizer("dummy", cfg)
        self.assertIsInstance(tok, HashTokenizer)
        self.assertEqual(tok.vocab_size, cfg.vocab_size)


class MixTests(unittest.TestCase):
    def test_mix_documents_drops_exhausted_source(self) -> None:
        iters = {"a": iter(["a1", "a2"]), "b": iter(["b1"])}
        out = list(mix_documents(iters, {"a": 0.5, "b": 0.5}, random.Random(0)))
        texts = [t for _, t in out]
        self.assertEqual(sorted(texts), ["a1", "a2", "b1"])

    def test_iter_local_prefers_content_and_skips_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "docs.jsonl"
            _write_jsonl(
                path,
                [
                    {"content": "math-row"},
                    {"tokens": [1, 2, 3], "text": "should-skip"},
                    {"text": "web-row"},
                ],
            )
            self.assertEqual(list(iter_local_jsonl(path)), ["math-row", "web-row"])

    def test_hf_stream_requires_data_extra(self) -> None:
        try:
            import datasets  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                next(iter_hf_texts(PHASE_B[0]))
            return
        self.skipTest("datasets installed; skip to avoid Hub download")


class PrepareTrainTests(unittest.TestCase):
    def test_local_prepare_writes_mmap_bin_and_meta(self) -> None:
        cfg = CATYokoConfig.tiny()
        tok = HashTokenizer(cfg.vocab_size)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "docs.jsonl"
            _write_jsonl(src, [{"text": _long_text()} for _ in range(4)])
            out = td / "t.bin"
            meta = prepare(
                mix="local",
                out=out,
                max_tokens=64,
                seq_len=cfg.seq_len,
                tokenizer=tok,
                local_jsonl=src,
            )
            self.assertEqual(meta["tokens"], 64)
            self.assertEqual(meta["sequences"], 4)
            self.assertEqual(meta["eos_id"], tok.eos_id)
            self.assertTrue(out.is_file())
            sidecar = sidecar_path(out)
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar, Path(str(out) + ".meta.json"))
            self.assertEqual(resolve_eos(out, None), tok.eos_id)
            stream = open_stream(out, cfg.vocab_size, cfg.seq_len)
            self.assertIsInstance(stream, PackedBinStream)
            batch = stream.batch(2, "cpu")
            self.assertEqual(tuple(batch["input_ids"].shape), (2, cfg.seq_len))
            self.assertTrue((batch["labels"] != -100).any())

    def test_prepare_cli_then_trainer_step(self) -> None:
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "docs.jsonl"
            _write_jsonl(src, [{"text": _long_text(12)} for _ in range(3)])
            out = td / "t.bin"
            code = prepare_main(
                [
                    "--mix",
                    "local",
                    "--local",
                    str(src),
                    "--out",
                    str(out),
                    "--tokenizer",
                    "dummy",
                    "--config",
                    "tiny",
                    "--max-tokens",
                    "64",
                ]
            )
            self.assertEqual(code, 0)
            tr = Trainer(cfg, "B0", "cpu", steps=1, micro_batch=2, accum=1, data=out)
            result = tr.run()
            self.assertEqual(result.step, 1)
            self.assertGreater(result.nll, 0)

    def test_local_sft_jsonl_masks_prompt_and_trains_f(self) -> None:
        cfg = CATYokoConfig.tiny()
        tok = HashTokenizer(cfg.vocab_size)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "chat.jsonl"
            _write_jsonl(
                src,
                [
                    {"data": ["what is 2+2?", "4. the sum of two and two."]},
                    {
                        "messages": [
                            {"role": "user", "content": "name a color"},
                            {"role": "assistant", "content": "blue is a color"},
                        ]
                    },
                ]
                * 4,
            )
            out = td / "sft.bin"
            meta = prepare(
                mix="local",
                out=out,
                max_tokens=256,
                seq_len=64,
                tokenizer=tok,
                local_jsonl=src,
                sft=True,
            )
            dest = Path(meta["out"])
            self.assertEqual(meta["format"], "sft_jsonl")
            self.assertTrue(dest.is_file())
            self.assertTrue(str(dest).endswith(".jsonl"))
            self.assertGreater(meta["sequences"], 0)
            row = json.loads(dest.read_text().splitlines()[0])
            self.assertIn(-100, row["labels"])
            self.assertTrue(any(x != -100 for x in row["labels"]))
            nll = Trainer(
                cfg, "F", "cpu", steps=1, accum=1, micro_batch=1, data=dest, seq_len=16
            ).run().nll
            self.assertTrue(math.isfinite(nll))

    def test_train_cli_on_prepared_bin_and_upcycle_pt(self) -> None:
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "docs.jsonl"
            _write_jsonl(src, [{"text": _long_text(12)} for _ in range(3)])
            bin_path = td / "t.bin"
            prepare_main(
                [
                    "--mix",
                    "local",
                    "--local",
                    str(src),
                    "--out",
                    str(bin_path),
                    "--tokenizer",
                    "dummy",
                    "--config",
                    "tiny",
                    "--max-tokens",
                    "64",
                ]
            )
            weights = td / "minicpm5.pt"
            torch.save(dummy_minicpm_state(cfg), weights)
            code = train_main(
                [
                    "--config",
                    "tiny",
                    "--phase",
                    "B0",
                    "--steps",
                    "1",
                    "--accum",
                    "1",
                    "--data",
                    str(bin_path),
                    "--upcycle",
                    str(weights),
                    "--dummy-teacher",
                ]
            )
            self.assertEqual(code, 0)

    def test_upcycle_and_upcycle_hf_conflict(self) -> None:
        with self.assertRaises(SystemExit):
            train_main(
                [
                    "--config",
                    "tiny",
                    "--dummy-upcycle",
                    "--upcycle-hf",
                    "openbmb/MiniCPM5-2B-Base",
                    "--steps",
                    "1",
                ]
            )


class SidecarAndDocTests(unittest.TestCase):
    def test_doc_ids_increase_after_eos(self) -> None:
        ids = torch.tensor([[7, 2, 8, 9]])
        docs = doc_ids_from_eos(ids, eos_id=2)
        self.assertEqual(docs.tolist(), [[0, 0, 1, 1]])

    def test_labels_mask_last_dim_not_batch(self) -> None:
        ids = torch.tensor([[7, 2, 8, 9], [1, 1, 1, 1]])
        docs = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 0]])
        labels = labels_with_doc_boundaries(ids, docs)
        self.assertEqual(labels[0].tolist(), [7, 2, -100, 9])
        self.assertEqual(labels[1].tolist(), [1, 1, 1, 1])
        self.assertEqual(labels_with_doc_boundaries(ids[0], docs[0]).tolist(), [7, 2, -100, 9])
        row = labels_with_doc_boundaries(ids[:1], docs[:1])
        self.assertEqual(row.tolist(), [[7, 2, -100, 9]])

    def test_packed_bin_micro_batch_one_masks_eos_boundary(self) -> None:
        import struct

        cfg = CATYokoConfig.tiny()
        seq = [7, 2, 8, 9] + [1] * (cfg.seq_len - 4)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            path.write_bytes(struct.pack("<" + "i" * len(seq), *seq))
            stream = PackedBinStream(path, cfg.seq_len, eos_id=2)
            batch = stream.batch(1, "cpu")
            self.assertEqual(int(batch["labels"][0, 2].item()), -100)
            self.assertNotEqual(int(batch["labels"][0, 1].item()), -100)

    def test_explicit_eos_wins_over_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.bin"
            path.write_bytes(b"\x00" * 16)
            (Path(str(path) + ".meta.json")).write_text(json.dumps({"eos_id": 2}))
            self.assertEqual(resolve_eos(path, 9), 9)
            self.assertEqual(resolve_eos(path, None), 2)


class HfMinicpmTests(unittest.TestCase):
    def test_unwrap_state_dict_and_pt_roundtrip(self) -> None:
        cfg = CATYokoConfig.tiny()
        src = dummy_minicpm_state(cfg)
        wrapped = _unwrap_state({"state_dict": src})
        self.assertIn("model.embed_tokens.weight", wrapped)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            torch.save(src, path)
            loaded = load_minicpm_state(path)
            self.assertEqual(loaded["model.embed_tokens.weight"].shape, src["model.embed_tokens.weight"].shape)

    def test_hub_load_requires_transformers(self) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                load_minicpm_state("openbmb/MiniCPM5-2B-Base")
            return
        self.skipTest("transformers installed; skip to avoid Hub download")


if __name__ == "__main__":
    unittest.main()
