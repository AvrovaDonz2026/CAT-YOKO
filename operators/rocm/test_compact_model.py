"""Numerical, B0 training, statistics, and overlay checks for compact experts."""

from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from cat_yoko.checkpoint import trainable_state_dict
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze, set_gate
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.moe import MoE
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm
from operators.rocm.compact_model import CompactFrozenMoE, compact_frozen_moes


def _identical_moe(hash_route=False):
    cfg = CATYokoConfig.tiny()
    moe = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec, hash_route=hash_route)
    for expert in moe.experts[1:]:
        expert.load_state_dict(moe.experts[0].state_dict())
    moe.requires_grad_(False)
    moe.grouped_experts = False
    return moe


class CompactModelTests(unittest.TestCase):
    def test_preflight_rejects_without_partial_replacement(self):
        valid = _identical_moe()
        invalid = _identical_moe()
        with torch.no_grad():
            invalid.experts[-1].down_proj.weight.add_(0.01)
        model = nn.ModuleList([valid, invalid])
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            compact_frozen_moes(model)
        self.assertIs(model[0], valid)
        self.assertIs(model[1], invalid)
        with self.assertRaisesRegex(ValueError, "only supported in B0"):
            compact_frozen_moes(model, phase="B1")

    def test_frozen_guard_rejects_updates_and_training(self):
        compact = CompactFrozenMoE(_identical_moe())
        x = torch.randn(1, 8, 64)
        compact.experts[0].gate_proj.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "became trainable"):
            compact(x)
        compact.experts[0].gate_proj.weight.requires_grad_(False)
        with torch.no_grad():
            compact.experts[0].down_proj.weight.add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "parameters changed"):
            compact(x)

    def test_projection_guard_rejects_shared_parameter_quantization(self):
        from cat_yoko.nvfp4_linear import Nvfp4Linear

        compact = CompactFrozenMoE(_identical_moe())
        original = compact.experts[0].gate_proj
        replacement = Nvfp4Linear.from_linear(original)
        self.assertIs(replacement.weight, original.weight)
        compact.experts[0].gate_proj = replacement
        with self.assertRaisesRegex(RuntimeError, "projection modules changed"):
            compact(torch.randn(1, 8, 64))

    def test_routing_stats_hash_and_input_gradients(self):
        self._routing_stats_hash_and_input_gradients("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_bf16_same_input_routing_stats_and_input_gradients(self):
        self._routing_stats_hash_and_input_gradients("cuda", torch.bfloat16)

    def _routing_stats_hash_and_input_gradients(self, device, dtype):
        torch.manual_seed(31)
        for hash_route in (False, True):
            with self.subTest(hash_route=hash_route):
                native = _identical_moe(hash_route)
                copied = _identical_moe(hash_route)
                copied.load_state_dict(native.state_dict())
                compact = CompactFrozenMoE(copied)
                native.to(device=device, dtype=dtype)
                compact.to(device=device, dtype=dtype)
                ids = torch.randint(128, (1, 8), device=device)
                for _ in range(2):
                    x = torch.randn(1, 8, 64, device=device, dtype=dtype, requires_grad=True)
                    x2 = x.detach().clone().requires_grad_(True)
                    context = (torch.autocast(device_type=device, dtype=dtype)
                               if dtype == torch.bfloat16 else nullcontext())
                    with context:
                        y = native(x, ids)
                        y2 = compact(x2, ids)
                    y.float().square().sum().backward()
                    y2.float().square().sum().backward()
                    if dtype == torch.bfloat16:
                        for actual, expected in ((y2, y), (x2.grad, x.grad)):
                            error = ((actual.float() - expected.float()).norm() /
                                     expected.float().norm().clamp_min(1e-8)).item()
                            self.assertLess(error, 0.015)
                    else:
                        torch.testing.assert_close(y2, y, atol=1e-6, rtol=1e-5)
                        torch.testing.assert_close(x2.grad, x.grad, atol=1e-6, rtol=1e-5)
                    self.assertEqual(compact._load_n, native._load_n)
                    if hash_route:
                        self.assertEqual(compact.last_aux.item(), 0.0)
                        self.assertIsNone(compact.last_load)
                    else:
                        self.assertIsNone(compact.last_aux)
                        # Identical inputs must yield identical router gates
                        # and statistics even when expert GEMMs reassociate.
                        torch.testing.assert_close(compact.last_load, native.last_load, rtol=0, atol=0)
                native.step_router_bias()
                compact.step_router_bias()
                torch.testing.assert_close(compact.e_score_correction_bias,
                                           native.e_score_correction_bias)
                compact.eval()
                compact(torch.randn(1, 8, 64, device=device, dtype=dtype), ids)
                self.assertIsNone(compact.last_load)
                self.assertEqual(compact._load_n, 0)

    def _whole_b0_graph(self, device, dtype, *, grad_checkpoint=False):
        torch.manual_seed(25)
        cfg = CATYokoConfig.tiny()
        native = CATYokoForCausalLM(cfg)
        upcycle_from_minicpm(native, dummy_minicpm_state(cfg))
        compact = CATYokoForCausalLM(cfg)
        compact.load_state_dict(native.state_dict())
        for model in (native, compact):
            apply_freeze(model, "B0")
            set_gate(model, 0.25)
            for blk in (*model.encoder, *model.decoder):
                blk.mlp.grouped_experts = False
        overlay_before = trainable_state_dict(compact)
        report = compact_frozen_moes(compact)
        overlay_after = trainable_state_dict(compact)
        self.assertEqual(overlay_before.keys(), overlay_after.keys())
        for name in overlay_before:
            torch.testing.assert_close(overlay_after[name], overlay_before[name], rtol=0, atol=0)
        self.assertEqual(report["compacted_moes"], cfg.encoder_layers + cfg.decoder_layers)
        self.assertGreater(report["removed_parameter_bytes"], 0)
        for blk in (*compact.encoder, *compact.decoder):
            self.assertIsInstance(blk.mlp, MoE)
            self.assertEqual(len(blk.mlp.experts), 1)
            self.assertEqual(blk.mlp.n_routed, cfg.n_routed_dec)
        native.to(device=device, dtype=dtype)
        compact.to(device=device, dtype=dtype)
        # B0 freeze may be reapplied by Trainer after operator installation.
        apply_freeze(compact, "B0")
        native.train()
        compact.train()
        native.grad_checkpoint = grad_checkpoint
        compact.grad_checkpoint = grad_checkpoint
        block_calls = {}
        hooks = []
        if grad_checkpoint:
            for label, model in (("native", native), ("compact", compact)):
                for stack_name in ("encoder", "decoder"):
                    for index, block in enumerate(getattr(model, stack_name)):
                        key = (label, stack_name, index)
                        block_calls[key] = 0

                        def count_none_docs(module, args, key=key):
                            self.assertIsNone(args[-1], key)
                            block_calls[key] += 1

                        hooks.append(block.register_forward_pre_hook(count_none_docs))
        ids = torch.randint(cfg.vocab_size, (1, cfg.seq_len), device=device)
        native_params = {n: p for n, p in native.named_parameters() if p.requires_grad}
        compact_params = {n: p for n, p in compact.named_parameters() if p.requires_grad}
        self.assertEqual(native_params.keys(), compact_params.keys())
        native_opt = torch.optim.SGD(native_params.values(), lr=1e-3)
        compact_opt = torch.optim.SGD(compact_params.values(), lr=1e-3)
        for _ in range(2):
            native_opt.zero_grad(set_to_none=True)
            compact_opt.zero_grad(set_to_none=True)
            calls_before = block_calls.copy()
            context = (torch.autocast(device_type=device, dtype=dtype)
                       if dtype == torch.bfloat16 else nullcontext())
            with context:
                reference = native(input_ids=ids, labels=ids, doc_ids=None)
                result = compact(input_ids=ids, labels=ids, doc_ids=None)
            reference["loss"].backward()
            result["loss"].backward()
            if grad_checkpoint:
                for key, calls in block_calls.items():
                    # Frozen B0 encoder needs one no-grad forward; trainable
                    # decoder cross-attention must trigger recomputation with
                    # the optional None document argument preserved.
                    expected_calls = 1 if key[1] == "encoder" else 2
                    self.assertEqual(calls - calls_before[key], expected_calls, key)
            limit = 0.02 if dtype == torch.bfloat16 else 1e-4
            self.assertLess(abs(result["loss"].item() - reference["loss"].item()), limit)
            diff = (result["logits"].float() - reference["logits"].float()).norm()
            self.assertLess((diff / reference["logits"].float().norm()).item(), limit)
            for name, param in compact_params.items():
                self.assertIsNotNone(param.grad, name)
                actual = param.grad.float()
                expected = native_params[name].grad.float()
                relative = ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item()
                self.assertLess(relative, limit, name)
            for a, b in zip((*native.encoder, *native.decoder), (*compact.encoder, *compact.decoder)):
                self.assertEqual(a.mlp._load_n, b.mlp._load_n)
                if a.mlp.last_load is None:
                    self.assertIsNone(b.mlp.last_load)
                elif dtype == torch.bfloat16:
                    # Previous layers reassociate BF16 expert arithmetic, so
                    # downstream near-tied top-k decisions can change buckets.
                    # Check their conserved statistics here; the isolated MoE
                    # test above verifies bit-exact stats for identical inputs.
                    for moe in (a.mlp, b.mlp):
                        load = moe.last_load.float()
                        self.assertTrue(torch.isfinite(load).all().item())
                        self.assertTrue((load >= 0).all().item())
                        self.assertEqual(load.numel(), moe.n_routed)
                        mean = moe.mean_pending_load().float()
                        self.assertAlmostEqual(mean.sum().item(), 1.0, delta=0.02)
                else:
                    torch.testing.assert_close(b.mlp.last_load, a.mlp.last_load)
            native.step_router_bias()
            compact.step_router_bias()
            native_opt.step()
            compact_opt.step()
        # Frozen full-state keys change, but trainable overlay round-trip keys
        # remain exactly the B0 checkpoint interface after training.
        self.assertEqual(trainable_state_dict(compact).keys(), overlay_before.keys())
        for hook in hooks:
            hook.remove()

    def test_cpu_whole_b0_graph_training_and_overlay(self):
        self._whole_b0_graph("cpu", torch.float32)

    def test_cpu_checkpoint_whole_b0_graph_none_docs(self):
        self._whole_b0_graph("cpu", torch.float32, grad_checkpoint=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_bf16_whole_b0_graph_training_and_overlay(self):
        self._whole_b0_graph("cuda", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_bf16_checkpoint_whole_b0_graph_none_docs(self):
        self._whole_b0_graph("cuda", torch.bfloat16, grad_checkpoint=True)


if __name__ == "__main__":
    unittest.main()
