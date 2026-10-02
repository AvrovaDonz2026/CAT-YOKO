"""Native arithmetic, B0 graph, overlays, and compact storage regressions."""

from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from cat_yoko.checkpoint import load_trainable_state, trainable_state_dict
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze, set_gate
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import MoE
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm
from operators.rocm.shared_storage_moe import SharedStorageFrozenMoE, share_frozen_moe_storage


def identical_moe(hash_route=False):
    cfg = CATYokoConfig.tiny()
    model = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec, hash_route=hash_route)
    for expert in model.experts[1:]:
        expert.load_state_dict(model.experts[0].state_dict())
    model.requires_grad_(False)
    model.grouped_experts = False
    return model


class SharedStorageTests(unittest.TestCase):
    def test_preflight_is_atomic_and_refuses_trainable_or_changed_experts(self):
        valid, invalid = identical_moe(), identical_moe()
        invalid.experts[-1].gate_proj.weight.requires_grad_(True)
        model = nn.ModuleList([valid, invalid])
        with self.assertRaisesRegex(ValueError, "fully frozen"):
            share_frozen_moe_storage(model)
        self.assertIs(model[0], valid)
        invalid.requires_grad_(False)
        with torch.no_grad():
            invalid.experts[-1].up_proj.weight.add_(0.01)
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            share_frozen_moe_storage(model)
        self.assertIs(model[0], valid)
        with self.assertRaisesRegex(ValueError, "only supported in B0"):
            share_frozen_moe_storage(model, phase="B1")

    def test_cache_uses_one_expert_storage_and_preserves_full_state_keys(self):
        native = identical_moe()
        keys_before = set(native.state_dict())
        shared = SharedStorageFrozenMoE(native)
        shared(torch.randn(1, 8, 64))
        self.assertEqual(len(shared.experts), shared.n_routed)
        self.assertTrue(all(expert is shared.experts[0] for expert in shared.experts))
        self.assertEqual(set(shared.state_dict()), keys_before)
        gate, up, down, nv = shared._swiglu_w
        self.assertFalse(nv)
        for view, name in zip((gate, up, down), ("gate_proj", "up_proj", "down_proj")):
            weight = getattr(shared.experts[0], name).weight
            self.assertEqual(view.stride(0), 0)
            self.assertEqual(view.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())
            self.assertEqual(view.untyped_storage().nbytes(), weight.numel() * weight.element_size())
        gu = shared._swiglu_gu
        self.assertEqual(gu.stride(0), 0)
        self.assertEqual(gu.untyped_storage().nbytes(), (gate[0].numel() + up[0].numel()) * gate.element_size())
        shared.to(dtype=torch.float64)
        self.assertIsNone(shared._swiglu_gu)
        shared(torch.randn(1, 8, 64, dtype=torch.float64))
        self.assertEqual(shared._swiglu_gu.dtype, torch.float64)
        # Full native frozen state loads into a reconstructed model, while
        # all repeated expert keys share one saved tensor storage.
        restored = identical_moe()
        restored.load_state_dict(shared.float().state_dict(), strict=True)

    def _same_input(self, device, dtype):
        torch.manual_seed(33)
        for hash_route in (False, True):
            for batched in (False, True):
                for tracking in (False, True):
                    with self.subTest(hash_route=hash_route, batched=batched, tracking=tracking):
                        native, source = identical_moe(hash_route), identical_moe(hash_route)
                        source.load_state_dict(native.state_dict())
                        native.batched_experts = source.batched_experts = batched
                        native.track_load = source.track_load = tracking
                        shared = SharedStorageFrozenMoE(source)
                        native.to(device=device, dtype=dtype)
                        shared.to(device=device, dtype=dtype)
                        self.assertEqual(shared.batched_experts, batched)
                        self.assertEqual(shared.track_load, tracking)
                        x = torch.randn(1, 16, 64, device=device, dtype=dtype, requires_grad=True)
                        xs = x.detach().clone().requires_grad_(True)
                        ids = torch.randint(128, (1, 16), device=device)
                        probe = torch.randn(x.shape, device=device, dtype=torch.float32)
                        context = (torch.autocast(device_type=device, dtype=dtype)
                                   if dtype == torch.bfloat16 else nullcontext())
                        with context:
                            yn, ys = native(x, ids), shared(xs, ids)
                        (yn.float() * probe).sum().backward()
                        (ys.float() * probe).sum().backward()
                        for actual, expected in ((ys, yn), (xs.grad, x.grad)):
                            if dtype == torch.float32:
                                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                            else:
                                relative = ((actual.float() - expected.float()).norm() /
                                            expected.float().norm().clamp_min(1e-8)).item()
                                self.assertLessEqual(relative, 0.015)
                        self.assertEqual(shared._load_n, native._load_n)
                        if native.last_load is None:
                            self.assertIsNone(shared.last_load)
                        else:
                            torch.testing.assert_close(shared.last_load, native.last_load, atol=0, rtol=0)

    def test_cpu_fp32_native_arithmetic_flags_and_statistics(self):
        self._same_input("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_bf16_native_arithmetic_flags_and_statistics(self):
        self._same_input("cuda", torch.bfloat16)

    def _whole_graph(self, device, dtype):
        torch.manual_seed(19)
        cfg = CATYokoConfig.tiny()
        native, shared = CATYokoForCausalLM(cfg), CATYokoForCausalLM(cfg)
        upcycle_from_minicpm(native, dummy_minicpm_state(cfg))
        shared.load_state_dict(native.state_dict())
        for model in (native, shared):
            apply_freeze(model, "B0")
            set_gate(model, 0.25)
            for block in (*model.encoder, *model.decoder):
                block.mlp.grouped_experts = False
        overlay = trainable_state_dict(shared)
        full_keys = set(shared.state_dict())
        report = share_frozen_moe_storage(shared)
        self.assertEqual(report["trainable_keys"], len(overlay))
        self.assertGreater(report["removed_parameter_bytes"], 0)
        self.assertEqual(set(shared.state_dict()), full_keys)
        self.assertEqual(set(trainable_state_dict(shared)), set(overlay))
        native.to(device=device, dtype=dtype)
        shared.to(device=device, dtype=dtype)
        # Test the actual resume helper after installation and GPU staging.
        load_trainable_state(shared, overlay)
        apply_freeze(shared, "B0")
        for model in (native, shared):
            model.grad_checkpoint = True
            model.train()
        ids = torch.randint(cfg.vocab_size, (1, cfg.seq_len), device=device)
        context = (torch.autocast(device_type=device, dtype=dtype)
                   if dtype == torch.bfloat16 else nullcontext())
        with context:
            reference, actual = native(ids, labels=ids, doc_ids=None), shared(ids, labels=ids, doc_ids=None)
        reference["loss"].backward()
        actual["loss"].backward()
        limit = 0.015 if dtype == torch.bfloat16 else 1e-5
        self.assertLessEqual(abs(actual["loss"].item() - reference["loss"].item()), limit)
        expected_params = dict(native.named_parameters())
        for name, parameter in shared.named_parameters():
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                continue
            expected = expected_params[name].grad
            self.assertIsNotNone(parameter.grad, name)
            error = ((parameter.grad.float() - expected.float()).norm() /
                     expected.float().norm().clamp_min(1e-8)).item()
            self.assertLessEqual(error, limit, name)
        self.assertEqual(set(trainable_state_dict(shared)), set(overlay))

    def test_cpu_whole_b0_graph_checkpoint_and_overlay(self):
        self._whole_graph("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_bf16_whole_b0_graph_checkpoint_and_overlay(self):
        self._whole_graph("cuda", torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
