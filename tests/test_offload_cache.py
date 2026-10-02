"""Offloading must release unregistered frozen-weight caches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cat_yoko.offload import move_module, offload_checkpoint_block


class OffloadCacheTests(unittest.TestCase):
    def test_device_move_clears_nested_transient_caches(self) -> None:
        block = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))
        for child in (block, block[0], block[1][0]):
            child._swiglu_w = (torch.ones(2, 4, 4), False)
            child._swiglu_gu = torch.ones(2, 8, 4)
            child._wq_cache = torch.ones(4, 4)
            child._wq_ver = 2
            child._bf16_fused_cat = ((), torch.ones(8, 4))
            child._te_fused_cat = [(), nn.Linear(4, 8), object(), object()]
            child._te_grouped = ("fused_gu", nn.Linear(4, 8), nn.Linear(4, 4))
        # Simulate crossing devices without requiring CUDA on CPU workers.
        with patch("cat_yoko.offload.module_device", return_value=torch.device("cuda:0")):
            move_module(block, "cpu")
        for child in (block, block[0], block[1][0]):
            for name in (
                "_swiglu_w", "_swiglu_gu", "_wq_cache", "_wq_ver",
                "_bf16_fused_cat", "_te_fused_cat", "_te_grouped",
            ):
                self.assertIsNone(getattr(child, name), name)

    def test_same_device_keeps_cached_weight(self) -> None:
        block = nn.Linear(4, 4)
        cached = torch.ones(4, 4)
        block._wq_cache = cached
        move_module(block, "cpu")
        self.assertIs(block._wq_cache, cached)

    def test_clear_preserves_saved_tensor_and_trainable_te_aliases(self) -> None:
        block = nn.Linear(4, 4)
        cached = torch.randn(4, 4)
        block._bf16_fused_cat = ((), cached)
        te_pack = [block, object(), object()]
        grouped = ("split", block, block, block)
        block._te_pack = te_pack
        block._te_grouped = grouped
        x = torch.randn(2, 4, requires_grad=True)
        y = nn.functional.linear(x, cached)
        expected = torch.ones_like(y) @ cached
        with patch("cat_yoko.offload.module_device", return_value=torch.device("cuda:0")):
            move_module(block, "cpu")
        self.assertIsNone(block._bf16_fused_cat)
        self.assertIs(block._te_pack, te_pack)
        self.assertIs(block._te_grouped, grouped)
        y.sum().backward()
        torch.testing.assert_close(x.grad, expected)

    def _check_b0_blocks_with_none_docs(self, device: str) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.freeze import apply_freeze
        from cat_yoko.model import CATYokoForCausalLM

        cfg = CATYokoConfig.tiny()
        torch.manual_seed(12)
        eager = CATYokoForCausalLM(cfg)
        offloaded = CATYokoForCausalLM(cfg)
        offloaded.load_state_dict(eager.state_dict())
        for model in (eager, offloaded):
            apply_freeze(model, "B0")
            model.eval()
            model.decoder[-1].gate.fill_(0.5)
            for blk in (*model.encoder, *model.decoder):
                blk.mlp.grouped_experts = False
        eager.to(device)
        x = torch.randn(1, cfg.seq_len, cfg.hidden_size, device=device)
        ids = torch.randint(cfg.vocab_size, (1, cfg.seq_len), device=device)
        # B0 encoder is frozen: this exercises the direct offload branch.
        expected_encoder = eager.encoder[0](x, ids, None)
        actual_encoder, aux = offload_checkpoint_block(offloaded.encoder[0], x, ids, None)
        torch.testing.assert_close(actual_encoder, expected_encoder)
        self.assertFalse(actual_encoder.requires_grad)
        self.assertEqual(aux.item(), 0.0)

        shape = (1, cfg.seq_len, cfg.kv_dim)
        k = torch.randn(shape, device=device, requires_grad=True)
        v = torch.randn(shape, device=device, requires_grad=True)
        expected = eager.decoder[-1](expected_encoder.detach(), k, v, None, None)
        expected.square().sum().backward()
        k_off = k.detach().clone().requires_grad_(True)
        v_off = v.detach().clone().requires_grad_(True)
        # Optional arguments after differentiable K/V must be restored as
        # None by _Fn.none_mask during backward recomputation.
        actual, aux = offload_checkpoint_block(
            offloaded.decoder[-1], actual_encoder.detach(), k_off, v_off, None, None
        )
        self.assertEqual(type(actual.grad_fn).__name__, "_FnBackward")
        (actual.square().sum() + aux).backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(k_off.grad, k.grad)
        torch.testing.assert_close(v_off.grad, v.grad)
        self.assertGreater(k_off.grad.norm().item(), 0.0)
        expected_params = dict(eager.decoder[-1].named_parameters())
        for name, p in offloaded.decoder[-1].named_parameters():
            if not p.requires_grad:
                self.assertIsNone(p.grad, name)
                continue
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, expected_params[name].grad.cpu())
        for blk in (offloaded.encoder[0], offloaded.decoder[-1]):
            self.assertTrue(all(p.device.type == "cpu" for p in blk.parameters()))
            if device == "cuda":
                self.assertIsNone(blk.mlp._swiglu_w)
                self.assertIsNone(blk.mlp._swiglu_gu)

    def test_cpu_b0_encoder_decoder_none_docs(self) -> None:
        self._check_b0_blocks_with_none_docs("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_b0_encoder_decoder_none_docs(self) -> None:
        self._check_b0_blocks_with_none_docs("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or ROCm")
    def test_gpu_frozen_moe_checkpoint_offload_releases_caches(self) -> None:
        from cat_yoko.config import CATYokoConfig
        from cat_yoko.moe import MoE

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                cfg = CATYokoConfig.tiny()
                self.mlp = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec)
                self.mlp.grouped_experts = False
                self.requires_grad_(False)
                self.eval()

            def forward(self, x):
                return self.mlp(x)

        torch.manual_seed(4)
        eager = Block().cuda()
        offloaded = Block()
        offloaded.load_state_dict(eager.state_dict())
        x = torch.randn(1, 16, 64, device="cuda", requires_grad=True)
        reference = eager(x)
        reference.square().sum().backward()
        expected_grad = x.grad.clone()
        expected_output = reference.detach().clone()
        self.assertIsNotNone(eager.mlp._swiglu_w)
        self.assertTrue(eager.mlp._swiglu_gu.is_cuda)
        del eager, reference

        for _ in range(2):
            value = x.detach().clone().requires_grad_(True)
            y, aux = offload_checkpoint_block(offloaded, value)
            self.assertIsNone(offloaded.mlp._swiglu_w)
            self.assertIsNone(offloaded.mlp._swiglu_gu)
            self.assertTrue(all(p.device.type == "cpu" for p in offloaded.parameters()))
            (y.square().sum() + aux).backward()
            torch.testing.assert_close(y, expected_output)
            torch.testing.assert_close(value.grad, expected_grad)
            self.assertIsNone(offloaded.mlp._swiglu_w)
            self.assertIsNone(offloaded.mlp._swiglu_gu)
            for child in offloaded.modules():
                self.assertIsNone(getattr(child, "_bf16_fused_cat", None))


if __name__ == "__main__":
    unittest.main()
