"""Candidates must never contaminate native parity or leak process patches."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import cat_yoko.attention as attention
import cat_yoko.offload as offload
import cat_yoko.trainer as trainer
from operators.rocm import model_bench
from operators.rocm.candidate_bench import (
    build_parser, candidate_context, native_argv, validate_candidates, PackedParityStream,
)


class CandidateIsolationTests(unittest.TestCase):
    def options(self, **changes):
        return Namespace(packed_attention=True, packed_attention_tile=256,
                         batched_grad_norm=True, **changes)

    def test_candidates_install_after_reference_and_restore_on_exception(self):
        window = attention._window_sdpa
        norm = trainer.clip_grad_norm_mixed
        with patch.object(model_bench, "install_moe_layout", return_value={"layers": 42}) as install:
            wrapper = model_bench.install_moe_layout
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                with candidate_context(self.options()) as report:
                    self.assertIs(attention._window_sdpa, window)
                    self.assertIs(trainer.clip_grad_norm_mixed, norm)
                    self.assertFalse(report["patches_installed_after_native_reference"])
                    result = model_bench.install_moe_layout(object(), "shared-storage")
                    self.assertEqual(result["layers"], 42)
                    self.assertIsNot(attention._window_sdpa, window)
                    self.assertIsNot(trainer.clip_grad_norm_mixed, norm)
                    self.assertIs(trainer.clip_grad_norm_mixed, offload.clip_grad_norm_mixed)
                    self.assertTrue(report["patches_installed_after_native_reference"])
                    raise RuntimeError("injected failure")
            self.assertIs(model_bench.install_moe_layout, wrapper)
            install.assert_called_once()
        self.assertIs(attention._window_sdpa, window)
        self.assertIs(trainer.clip_grad_norm_mixed, norm)
        self.assertEqual(report["packed_attention_calls"]["tiled_calls"], 0)

    def test_unused_packed_candidate_cannot_proceed_to_updates(self):
        with patch.object(model_bench, "install_moe_layout", return_value={}), \
                patch.object(model_bench, "emit") as emit:
            with candidate_context(self.options()) as report:
                model_bench.install_moe_layout(object(), "shared-storage")
                full_report = {}
                with self.assertRaisesRegex(RuntimeError, "not exercised"):
                    model_bench.emit(Path("unused"), full_report,
                                     {"event": "parity_complete", "pass": True})
                self.assertEqual(full_report["status"], "candidate_not_exercised")
                self.assertFalse(emit.call_args.args[2]["pass"])
                self.assertEqual(report["packed_attention_parity_calls"]["tiled_calls"], 0)

    def test_cli_preserves_native_arguments_and_requires_full_parity(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["--base", str(root), "--resume", str(root / "source.pt"),
                    "--out", str(root / "out"), "--data", str(root / "train.bin"),
                    "--packed-attention", "--packed-attention-tile=128",
                    "--batched-grad-norm", "--moe-layout", "shared-storage",
                    "--deterministic-parity", "--reference-repeat"]
            args = parser.parse_args(argv)
            validate_candidates(parser, args)
            forwarded = native_argv(argv)
            self.assertNotIn("--packed-attention", forwarded)
            native = model_bench.build_parser().parse_args(forwarded)
            self.assertEqual(native.data, args.data)
            self.assertEqual(native.resume, args.resume)
            self.assertTrue(native.reference_repeat)
            for key, value in (("moe_layout", "compact"), ("data", None),
                               ("deterministic_parity", False), ("reference_repeat", False),
                               ("parity_seqs", "64,256"), ("packed_attention_tile", 4096)):
                previous = getattr(args, key)
                setattr(args, key, value)
                with self.subTest(key=key), patch("sys.stderr"), self.assertRaises(SystemExit):
                    validate_candidates(parser, args)
                setattr(args, key, previous)

    def test_parity_search_uses_real_rows_and_is_bounded(self):
        class Stream:
            def __init__(self, rows):
                self.rows, self.calls = rows, 0

            def batch(self, micro_batch, device):
                self.calls += 1
                return {"doc_ids": self.rows[(self.calls - 1) % len(self.rows)]}

        single = torch.zeros(1, 8, dtype=torch.long)
        multiple = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2]])
        stream, info = Stream([single, single, multiple]), {}
        actual = PackedParityStream(stream, info).batch(1, "cpu")
        self.assertIs(actual["doc_ids"], multiple)
        self.assertEqual(info["packed_parity_scanned_batches"], 3)
        self.assertEqual(info["packed_parity_document_boundaries"], 2)
        stream = Stream([single])
        with self.assertRaisesRegex(RuntimeError, "no multi-document"):
            PackedParityStream(stream, {}, max_batches=2).batch(1, "cpu")
        self.assertEqual(stream.calls, 2)


if __name__ == "__main__":
    unittest.main()
