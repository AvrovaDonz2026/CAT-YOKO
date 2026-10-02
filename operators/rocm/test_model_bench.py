"""Parity must leave the continuation's policy, weights, and statistics clean."""

from contextlib import ExitStack, nullcontext
import unittest
from unittest.mock import patch

import torch

from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze, set_gate
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import MoE
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm
from operators.rocm.model_bench import capture, parity_determinism
from operators.rocm.shared_storage_moe import share_frozen_moe_storage


class ParityIsolationTests(unittest.TestCase):
    def test_policy_restores_after_success_and_exception(self):
        original = torch.are_deterministic_algorithms_enabled()
        original_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            for enabled, warn_only in ((False, False), (True, True)):
                for fail in (False, True):
                    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                    try:
                        with parity_determinism(True):
                            self.assertTrue(torch.are_deterministic_algorithms_enabled())
                            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                            if fail:
                                raise RuntimeError("injected parity failure")
                    except RuntimeError as exc:
                        self.assertEqual(str(exc), "injected parity failure")
                    self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)
                    self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), warn_only)
        finally:
            torch.use_deterministic_algorithms(original, warn_only=original_warn)

    def test_capture_clears_pending_loads_without_changing_weights(self):
        torch.manual_seed(19)
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        upcycle_from_minicpm(model, dummy_minicpm_state(cfg))
        apply_freeze(model, "B0")
        set_gate(model, 0.25)
        share_frozen_moe_storage(model)
        saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
        ids = torch.randint(cfg.vocab_size, (1, cfg.seq_len))
        batch = {"input_ids": ids, "labels": ids, "doc_ids": None}
        for fail in (False, True):
            for module in model.modules():
                if isinstance(module, MoE):
                    module.last_load = torch.ones(module.n_routed)
                    module._load_n = 99
                    module.last_aux = torch.ones(())
            with ExitStack() as stack:
                # Run the complete tiny CPU graph; mock only CUDA accounting
                # and autocast so no GPU is needed for cleanup/error coverage.
                for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
                    stack.enter_context(patch(f"torch.cuda.{name}"))
                for name in ("memory_allocated", "max_memory_allocated"):
                    stack.enter_context(patch(f"torch.cuda.{name}", return_value=0))
                stack.enter_context(patch("torch.autocast", side_effect=lambda *a, **k: nullcontext()))
                if fail:
                    stack.enter_context(patch.object(model, "forward", side_effect=RuntimeError("injected")))
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        capture(model, batch, device=torch.device("cpu"), offload=False)
                else:
                    row, snapshot = capture(model, batch, device=torch.device("cpu"), offload=False)
                    self.assertGreater(row["gradient_tensors"], 0)
                    self.assertTrue(snapshot["gradients"])
            self.assertFalse(model.norm._forward_hooks)
            for module in model.modules():
                if isinstance(module, MoE):
                    self.assertIsNone(module.last_load)
                    self.assertEqual(module._load_n, 0)
                    self.assertIsNone(module.last_aux)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, saved[name]), name)


if __name__ == "__main__":
    unittest.main()
