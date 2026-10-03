"""CPU dense oracle, native parity and packed tile-boundary regressions."""

from __future__ import annotations

import math
import json
import struct
import tempfile
import unittest
import gc
import weakref
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import torch

import cat_yoko.attention as native
from cat_yoko.config import CATYokoConfig
from operators.rocm.packed_attention import (
    clear_doc_plan_cache, doc_plan_cache_size, install_packed_attention,
    new_stats, packed_attention_context, packed_window_sdpa,
)


def dense_oracle(q, k, v, window, docs):
    """Independent explicit FP32 reference; no candidate/native mask helpers."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        repeats = q.size(1) // k.size(1)
        qf, kf, vf = q.float(), k.float().repeat_interleave(repeats, 1), v.float().repeat_interleave(repeats, 1)
        scores = qf @ kf.transpose(-1, -2) / math.sqrt(q.size(-1))
        qi = torch.arange(q.size(-2), device=q.device)[:, None]
        ki = torch.arange(k.size(-2), device=q.device)[None, :]
        keep = (ki <= qi) & ((qi - ki) < window)
        keep = keep.unsqueeze(0) & (docs[:, :, None] == docs[:, None, :])
        probabilities = scores.masked_fill(~keep[:, None], torch.finfo(torch.float32).min).softmax(-1)
        return (probabilities @ vf).to(q.dtype)


def inputs(batch, seq, dtype, layout):
    generator = torch.Generator().manual_seed(seq + batch)
    values = [torch.randn(batch, heads, seq, 16, generator=generator, dtype=dtype)
              for heads in (8, 2, 2)]
    if layout == "bshd":
        values = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in values]
    elif layout == "packed_qkv":
        packed = torch.cat([x.transpose(1, 2).reshape(batch, seq, -1) for x in values], -1)
        values = [x.reshape(batch, seq, heads, 16).transpose(1, 2)
                  for x, heads in zip(packed.split((128, 32, 32), -1), (8, 2, 2))]
    return tuple(x.detach().requires_grad_() for x in values)


def documents(batch, seq):
    result = torch.zeros(batch, seq, dtype=torch.long)
    # Boundaries immediately before/at/after a tile start, and a long document
    # spanning multiple tiles. The second batch uses different boundaries.
    for row in range(batch):
        for boundary in (3 + row, 7 + row, 16 + row, 17 + row, 31 + row):
            result[row, boundary:] += 1
    return result


class PackedAttentionTests(unittest.TestCase):
    def setUp(self):
        clear_doc_plan_cache()

    def _parity(self, batch, seq, window, tile, dtype, layout):
        qkv = inputs(batch, seq, dtype, layout)
        docs = documents(batch, seq)
        dy = torch.randn(qkv[0].shape, generator=torch.Generator().manual_seed(777), dtype=dtype)
        # Exercise noncontiguous upstream gradients with identical logical values.
        dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
        ref = dense_oracle(*qkv, window, docs)
        reference_grad = torch.autograd.grad(ref, qkv, dy)
        actual = packed_window_sdpa(*qkv, window, docs, tile_size=tile)
        actual_grad = torch.autograd.grad(actual, qkv, dy)
        tolerance = (2e-5, 2e-5) if dtype == torch.float32 else (0.015, 0.015)
        for name, a, b in zip(("output", "dq", "dk", "dv"), (actual, *actual_grad), (ref, *reference_grad)):
            self.assertTrue(torch.isfinite(a).all(), name)
            torch.testing.assert_close(a, b, atol=tolerance[0], rtol=tolerance[1], msg=name)
            relative = ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
            self.assertLessEqual(relative, 2e-5 if dtype == torch.float32 else 0.005, name)
        baseline = native._window_sdpa(*qkv, window, docs)
        torch.testing.assert_close(actual, baseline, atol=tolerance[0], rtol=tolerance[1])

    def test_fp32_dense_oracle_all_gradients_layouts_padding_and_batch(self):
        for layout in ("bhsd", "bshd", "packed_qkv"):
            for batch, seq, window, tile in ((1, 41, 5, 8), (2, 35, 9, 8), (2, 33, 1, 16), (2, 65, 17, 8)):
                with self.subTest(layout=layout, batch=batch, seq=seq, window=window, tile=tile):
                    self._parity(batch, seq, window, tile, torch.float32, layout)

    def test_cpu_bf16_math_all_gradients(self):
        for layout in ("bshd", "packed_qkv"):
            with self.subTest(layout=layout):
                self._parity(2, 43, 7, 16, torch.bfloat16, layout)

    def test_dense_packed_oracle_variable_lengths_and_all_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("bhsd", "bshd", "packed_qkv"):
                with self.subTest(dtype=dtype, layout=layout):
                    self._parity(2, 43, 8192, 16, dtype, layout)

    def test_dense_packed_4096_context_low_heads(self):
        generator = torch.Generator().manual_seed(900)
        values = tuple(torch.randn(1, h, 4096, 8, generator=generator).requires_grad_() for h in (2, 1, 1))
        docs = torch.arange(4096).div(517, rounding_mode="floor").unsqueeze(0)
        with packed_attention_context(min_seq_len=2048) as installation:
            result = native._window_sdpa(*values, 8192, docs)
            grads = torch.autograd.grad(result.sum(), values)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        report = installation.report()
        self.assertEqual(report["optimized_calls"], 1)
        self.assertEqual(report["dense_packed_calls"], 1)
        self.assertEqual(report["host_plan_builds"], 1)
        self.assertEqual(report["doc_fragment_calls"], 8)
        self.assertEqual(report["max_mask_elements"], 0)
        self.assertLess(report["max_fragment_score_elements"], 2 * 4096 ** 2)

    def test_dense_repeated_document_ids_in_separated_runs_keep_native_links(self):
        qkv = inputs(1, 35, torch.float32, "bshd")
        docs = torch.zeros(1, 35, dtype=torch.long)
        docs[:, 7:14] = 1
        baseline = native._window_sdpa(*qkv, 8192, docs)
        with packed_attention_context(min_seq_len=0) as installation:
            actual = native._window_sdpa(*qkv, 8192, docs)
            again = native._window_sdpa(*qkv, 8192, docs)
        torch.testing.assert_close(actual, baseline, atol=0, rtol=0)
        torch.testing.assert_close(again, baseline, atol=0, rtol=0)
        self.assertEqual(installation.report()["optimized_calls"], 0)
        self.assertEqual(installation.report()["host_plan_builds"], 1)
        self.assertEqual(installation.report()["fallback_calls"], 2)

    def test_document_cache_identity_mutation_weak_lifetime_and_rng(self):
        qkv = inputs(2, 35, torch.float32, "bshd")
        docs = documents(2, 35)
        before_rng = torch.get_rng_state().clone()
        with packed_attention_context(min_seq_len=0) as installation:
            native._window_sdpa(*qkv, 8192, docs)
            native._window_sdpa(*qkv, 8192, docs)
            self.assertEqual(installation.report()["host_plan_builds"], 1)
            docs[0, :3] = -1
            actual = native._window_sdpa(*qkv, 8192, docs)
            torch.testing.assert_close(actual, dense_oracle(*qkv, 8192, docs), atol=2e-5, rtol=2e-5)
            self.assertEqual(installation.report()["host_plan_builds"], 2)
            copied = docs.clone()
            native._window_sdpa(*qkv, 8192, copied)
            self.assertEqual(installation.report()["host_plan_builds"], 3)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(doc_plan_cache_size(), 2)
        reference = weakref.ref(copied)
        del copied
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(doc_plan_cache_size(), 1)

    def test_cross_native_weights_outputs_input_and_projection_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                cfg = replace(CATYokoConfig.tiny(), qk_norm=True)
                module = native.CrossAttention(cfg).to(dtype=dtype)
                generator = torch.Generator().manual_seed(333)
                batch, seq = 2, 37
                x = torch.randn(batch, seq, cfg.hidden_size, generator=generator, dtype=dtype).requires_grad_()
                kv = torch.randn(batch, seq, 2 * cfg.kv_dim, generator=generator, dtype=dtype)
                k, v = (value.detach().requires_grad_() for value in kv.split(cfg.kv_dim, dim=-1))
                docs = documents(batch, seq)
                params = tuple(module.parameters())
                dy = torch.randn(x.shape, generator=generator, dtype=dtype)
                reference = module(x, k, v, docs)
                reference_grads = torch.autograd.grad(reference, (x, k, v, *params), dy)
                with packed_attention_context(min_seq_len=0) as installation:
                    actual = module(x, k, v, docs)
                    actual_grads = torch.autograd.grad(actual, (x, k, v, *params), dy)
                    self.assertEqual(installation.report()["cross_attention_calls"], 1)
                    self.assertEqual(installation.report()["optimized_calls"], 1)
                tolerance = 3e-5 if dtype == torch.float32 else 0.02
                for a, b in zip((actual, *actual_grads), (reference, *reference_grads)):
                    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)

    def test_no_cross_document_or_future_leakage_at_tile_boundary(self):
        q, k, v = inputs(2, 35, torch.float32, "bshd")
        docs = torch.zeros(2, 35, dtype=torch.long)
        docs[0, 16:], docs[1, 17:] = 1, 1
        before = packed_window_sdpa(q, k, v, 9, docs, tile_size=8)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[0, :, :16], changed_v[0, :, :16] = 90, -90
        changed_k[1, :, :17], changed_v[1, :, :17] = -90, 90
        after = packed_window_sdpa(q, changed_k, changed_v, 9, docs, tile_size=8)
        torch.testing.assert_close(before[0, :, 16:], after[0, :, 16:], atol=0, rtol=0)
        torch.testing.assert_close(before[1, :, 17:], after[1, :, 17:], atol=0, rtol=0)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[:, :, 25:], changed_v[:, :, 25:] = 100, -100
        after = packed_window_sdpa(q, changed_k, changed_v, 9, docs, tile_size=8)
        torch.testing.assert_close(before[:, :, :25], after[:, :, :25], atol=0, rtol=0)

    def test_single_doc_extra_bias_and_minimum_sequence_use_original_fallback(self):
        qkv = inputs(1, 35, torch.float32, "bhsd")
        docs = documents(1, 35)
        sentinel = object()
        fallback = Mock(return_value=sentinel)
        for kwargs in ({"doc_ids": None}, {"doc_ids": docs, "extra_bias": torch.zeros(35, 35)},
                       {"doc_ids": docs, "min_seq_len": 64}):
            self.assertIs(packed_window_sdpa(*qkv, 7, tile_size=8, fallback=fallback, **kwargs), sentinel)
        self.assertEqual(fallback.call_count, 3)

    def test_install_context_restores_after_exception_and_nested_order(self):
        original = native._window_sdpa
        original_cross = native.CrossAttention.forward
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with packed_attention_context(tile_size=8, min_seq_len=0):
                self.assertIsNot(native._window_sdpa, original)
                raise RuntimeError("injected")
        self.assertIs(native._window_sdpa, original)
        self.assertIs(native.CrossAttention.forward, original_cross)
        first = install_packed_attention(tile_size=8, min_seq_len=0)
        second = install_packed_attention(tile_size=16, min_seq_len=0)
        try:
            with self.assertRaisesRegex(RuntimeError, "reverse"):
                first.remove()
        finally:
            second.remove()
            first.remove()
        first.remove()
        self.assertIs(native._window_sdpa, original)
        self.assertIs(native.CrossAttention.forward, original_cross)

    def test_context_reports_tiled_calls_and_dense_fallback(self):
        qkv = inputs(1, 35, torch.float32, "bhsd")
        docs = documents(1, 35)
        native.reset_sdpa_counts()
        with packed_attention_context(tile_size=8, min_seq_len=0) as installation:
            native._window_sdpa(*qkv, 7, docs)
            self.assertEqual(native.sdpa_counts()["math_fp32"], 1)
            native._window_sdpa(*qkv, 7, None)
            # Direct use under an installed context must not recurse on fallback.
            packed_window_sdpa(*qkv, 7, None, tile_size=8)
            report = installation.report()
        self.assertEqual(report["tiled_calls"], 1)
        self.assertEqual(report["fallback_calls"], 1)
        self.assertEqual(report["query_tokens"], 35)
        self.assertEqual(report["tile_batches"], 5)
        self.assertEqual(report["max_mask_elements"], 5 * 8 * 14)
        report["tiled_calls"] = 90
        self.assertEqual(installation.report()["tiled_calls"], 1)

    def test_benchmark_loads_real_packed_document_boundaries(self):
        from operators.rocm.packed_attention_bench import load_documents

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(struct.pack("<16i", 5, 6, 1, 7, 8, 9, 1, 10,
                                          5, 1, 7, 8, 1, 10, 11, 12))
            path.with_name("data.bin.meta.json").write_text(json.dumps({"eos_id": 1, "seq_len": 8}))
            docs, provenance = load_documents(path, seq=8, batch=2, row=0,
                                              eos_id=None, synthetic_doc_len=3)
        self.assertEqual(docs.tolist(), [[0, 0, 0, 1, 1, 1, 1, 2], [0, 0, 1, 1, 1, 2, 2, 2]])
        self.assertEqual(provenance["eos_id"], 1)

    def test_invalid_docs_and_parameters_are_rejected(self):
        qkv = inputs(2, 35, torch.float32, "bhsd")
        with self.assertRaisesRegex(ValueError, "doc_ids"):
            packed_window_sdpa(*qkv, 7, torch.zeros(35), tile_size=8)
        with self.assertRaisesRegex(ValueError, "tile_size"):
            packed_window_sdpa(*qkv, 7, documents(2, 35), tile_size=0)


if __name__ == "__main__":
    unittest.main()
