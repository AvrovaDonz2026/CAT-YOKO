"""Real-data continuation must validate its masks and preserve evaluation isolation."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.data import DummyStream, open_stream
from cat_yoko.freeze import apply_freeze, set_gate
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import MoE
from cat_yoko.trainer import Trainer
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm
from operators.rocm.model_bench import (
    build_parser, evaluate_heldout, open_parity_stream, validate_args,
)
from operators.rocm.shared_storage_moe import share_frozen_moe_storage


class RealDataContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "train.bin"
        tokens = [3, 4, 2, 5, 6, 2, 7, 8, 9, 10, 2, 11, 12, 2, 13, 14]
        self.data.write_bytes(struct.pack("<16i", *tokens))
        self.data.with_suffix(".bin.meta.json").write_text(json.dumps({"eos_id": 2, "seq_len": 8}))
        self.eval_data = self.root / "validation.bin"
        self.eval_data.write_bytes(struct.pack("<8i", 15, 16, 2, 17, 18, 2, 19, 20))
        self.eval_data.with_suffix(".bin.meta.json").write_text(json.dumps({"eos_id": 2, "seq_len": 8}))
        self.cfg = CATYokoConfig.tiny()

    def parsed(self, *extra):
        parser = build_parser()
        args = parser.parse_args(["--base", str(self.root), "--resume", str(self.root / "source.pt"),
                                  "--out", str(self.root / "out"), *extra])
        with patch("sys.stderr"):
            sequences = validate_args(parser, args)
        return args, sequences

    def test_real_training_options_and_shared_storage_gate(self):
        args, sequences = self.parsed(
            "--data", str(self.data), "--eval-data", str(self.eval_data),
            "--moe-layout", "shared-storage", "--parity-seqs", "8", "--seq-len", "8",
            "--eval-every", "10", "--eval-batches", "3", "--save-optim", "--max-hours", "24",
        )
        self.assertEqual(sequences, [8])
        self.assertEqual(args.eval_every, 10)
        self.assertEqual(args.eval_batches, 3)
        self.assertTrue(args.save_optim)
        self.assertEqual(args.max_hours, 24)
        with self.assertRaises(SystemExit):
            self.parsed("--data", str(self.data))
        with self.assertRaises(SystemExit):
            self.parsed("--run-steps", "1000")
        with self.assertRaises(SystemExit):
            self.parsed("--data", str(self.data), "--moe-layout", "shared-storage", "--seq-len", "8")
        with self.assertRaises(SystemExit):
            self.parsed("--data", str(self.data), "--eval-data", str(self.data))
        # The previous bounded DummyStream experiment and compact parity remain available.
        self.assertEqual(self.parsed()[0].moe_layout, "compact")
        self.parsed("--data", str(self.data), "--parity-only", "--moe-layout", "compact")

    def test_reject_invalid_eval_and_time_limits_before_gpu_setup(self):
        for flags in (("--eval-every", "-1"), ("--eval-batches", "0"),
                      ("--eval-every", "10"), ("--eos-id", "-1"),
                      ("--max-hours", "0"), ("--max-hours", "nan"),
            ("--max-hours", "inf"), ("--data", str(self.root / "missing.bin"))):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                self.parsed(*flags)

    def test_timed_checkpoint_interval_validation(self):
        args, _ = self.parsed("--save-every", "0", "--save-every-seconds", "300", "--keep-last", "3")
        self.assertEqual(args.save_every_seconds, 300)
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parsed("--save-every-seconds", value)

    def test_parity_uses_real_document_masks_and_ignores_dummy_cursor(self):
        dummy = DummyStream(self.cfg.vocab_size, 8, 11)
        dummy.batch(4, "cpu")
        stream, info = open_parity_stream(
            self.cfg, 8, 11, data=self.data, source_state=dummy.state_dict(),
        )
        batch = stream.batch(1, "cpu")
        expected = open_stream(self.data, self.cfg.vocab_size, 8, seed=11).batch(1, "cpu")
        self.assertEqual(info, {"kind": "packed", "source_cursor_restored": False})
        for key in expected:
            self.assertTrue(torch.equal(batch[key], expected[key]), key)
        self.assertGreater(int(batch["doc_ids"].max()), 0)
        self.assertEqual(batch["labels"][0, 3].item(), -100)
        self.assertEqual(batch["labels"][0, 6].item(), -100)

    def test_same_kind_cursor_restores_but_stride_mismatch_starts_new_stream(self):
        original = open_stream(self.data, self.cfg.vocab_size, 8)
        original.batch(1, "cpu")
        source_state = original.state_dict()
        restored, info = open_parity_stream(self.cfg, 8, 0, data=self.data, source_state=source_state)
        self.assertTrue(info["source_cursor_restored"])
        self.assertTrue(torch.equal(restored.batch(1, "cpu")["input_ids"],
                                    original.batch(1, "cpu")["input_ids"]))
        incompatible = {**source_state, "stride": 2}
        stream, info = open_parity_stream(self.cfg, 8, 0, data=self.data, source_state=incompatible)
        self.assertFalse(info["source_cursor_restored"])
        self.assertIn("stride", info["source_cursor_restore_error"])
        self.assertEqual(stream.batch(1, "cpu")["input_ids"][0, 0].item(), 3)

    def test_dummy_parity_retains_legacy_random_only_setting(self):
        original = DummyStream(self.cfg.vocab_size, 8, 19, code_frac=0)
        original.batch(3, "cpu")
        restored, info = open_parity_stream(self.cfg, 8, 19, source_state=original.state_dict())
        self.assertTrue(info["source_cursor_restored"])
        self.assertEqual(restored.code_frac, 0)
        self.assertTrue(torch.equal(restored.batch(1, "cpu")["input_ids"],
                                    original.batch(1, "cpu")["input_ids"]))

    def test_heldout_eval_preserves_training_cursor_weights_and_cleans_statistics(self):
        torch.manual_seed(19)
        model = CATYokoForCausalLM(self.cfg)
        upcycle_from_minicpm(model, dummy_minicpm_state(self.cfg))
        apply_freeze(model, "B0")
        set_gate(model, 0.25)
        share_frozen_moe_storage(model)
        model.train()
        trainer = Trainer(self.cfg, "B0", "cpu", data=self.data, eval_data=self.eval_data,
                          seq_len=8, micro_batch=1, eval_batches=2)
        training_stream = trainer._open(self.data, trainer.seed)
        training_stream.batch(1, "cpu")
        cursor = training_stream.state_dict()
        saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
        result = evaluate_heldout(trainer, model, event="initial_eval", step=34802)
        self.assertTrue(result["pass"])
        self.assertGreater(result["eval_valid_tokens"], 0)
        self.assertEqual(result["step"], 34802)
        self.assertEqual(cursor, training_stream.state_dict())
        self.assertTrue(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        for module in model.modules():
            if isinstance(module, MoE):
                self.assertIsNone(module.last_load)
                self.assertEqual(module._load_n, 0)
                self.assertIsNone(module.last_aux)

    def test_invalid_heldout_result_is_reported_without_nan_json(self):
        model = CATYokoForCausalLM(self.cfg)
        trainer = Trainer(self.cfg, "B0", "cpu", eval_data=self.data)
        for weighted, count in ((0.0, 0.0), (float("nan"), 8.0), (float("inf"), 8.0)):
            with patch.object(trainer, "_eval_nll_stats", return_value=(weighted, count)):
                result = evaluate_heldout(trainer, model, event="initial_eval", step=0)
            self.assertFalse(result["pass"])
            self.assertIsNone(result["eval_nll"])
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
