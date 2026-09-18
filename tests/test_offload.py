#!/usr/bin/env python3
"""CPU tests for encoder/block offload and CPU-offload AdamW."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.blocks import EncoderBlock
from cat_yoko.config import CATYokoConfig
from cat_yoko.freeze import apply_freeze
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.offload import auto_offload_flags, clip_grad_norm_mixed, move_module, offload_checkpoint_block
from cat_yoko.optim import CPUOffloadAdamW, build_optimizer
from cat_yoko.train import main
from cat_yoko.trainer import Trainer, run_c1_chain, train_loop


class AutoFlagsTests(unittest.TestCase):
    def test_12b_cuda_defaults(self) -> None:
        e, b, o = auto_offload_flags(
            phase="B1",
            cfg_name="CAT-YOKO-12B",
            device="cuda",
            fsdp=False,
            ddp=False,
            offload_encoder=None,
            offload_blocks=None,
            optim_cpu=None,
        )
        self.assertTrue(e)
        self.assertFalse(b)
        self.assertTrue(o)
        e2, b2, o2 = auto_offload_flags(
            phase="B2",
            cfg_name="CAT-YOKO-12B",
            device="cuda",
            fsdp=False,
            ddp=False,
            offload_encoder=None,
            offload_blocks=None,
            optim_cpu=None,
        )
        self.assertFalse(e2)
        self.assertTrue(b2)
        self.assertTrue(o2)

    def test_tiny_cpu_stays_off(self) -> None:
        e, b, o = auto_offload_flags(
            phase="B2",
            cfg_name="tiny",
            device="cpu",
            fsdp=False,
            ddp=False,
            offload_encoder=None,
            offload_blocks=None,
            optim_cpu=None,
        )
        self.assertFalse(e or b or o)


class OffloadBlockTests(unittest.TestCase):
    def test_checkpoint_offload_matches_eager(self) -> None:
        cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)
        a = EncoderBlock(cfg, kind="sliding")
        torch.manual_seed(0)
        b = EncoderBlock(cfg, kind="sliding")
        x = torch.randn(2, cfg.seq_len, cfg.hidden_size, requires_grad=True)
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        doc = torch.zeros_like(ids)
        y1 = a(x, ids, doc)
        aux1 = a.mlp.last_aux
        (y1.sum() + aux1).backward()
        y2, aux2 = offload_checkpoint_block(b, x.detach().requires_grad_(True), ids, doc)
        (y2.sum() + aux2).backward()
        self.assertTrue(torch.allclose(y1, y2, atol=1e-5, rtol=1e-4))
        g1 = next(a.parameters()).grad
        g2 = next(b.parameters()).grad
        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)
        self.assertTrue(torch.allclose(g1, g2, atol=1e-4, rtol=1e-3))

    def test_move_module_is_noop_on_same_device(self) -> None:
        cfg = CATYokoConfig.tiny()
        m = EncoderBlock(cfg)
        p = next(m.parameters())
        move_module(m, "cpu")
        self.assertIs(next(m.parameters()), p)


class RouterBiasOffloadTests(unittest.TestCase):
    def test_step_router_bias_casts_load_to_buffer(self) -> None:
        from cat_yoko.moe import MoE

        cfg = CATYokoConfig.tiny()
        moe = MoE(cfg, cfg.n_routed_dec, cfg.top_k_dec)
        moe.train()
        moe.last_load = torch.ones(cfg.n_routed_dec, dtype=torch.float64)
        before = moe.e_score_correction_bias.clone()
        moe.step_router_bias()
        self.assertFalse(torch.equal(before, moe.e_score_correction_bias))
        self.assertIsNone(moe.last_load)


class CpuAdamTests(unittest.TestCase):
    def test_cpu_adam_changes_trainable(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B0")
        opt = build_optimizer(model, cfg, cpu_offload=True)
        self.assertIsInstance(opt, CPUOffloadAdamW)
        before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        model(input_ids=ids, labels=ids)["loss"].backward()
        opt.step()
        after = next(p for p in model.parameters() if p.requires_grad)
        self.assertFalse(torch.equal(before, after))

    def test_cpu_adam_fp16_and_ephemeral(self) -> None:
        cfg = CATYokoConfig.tiny()
        ids = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
        for kwargs in (
            {"cpu_offload": True, "state_dtype": torch.float16, "retain_state": True},
            {"cpu_offload": True, "state_dtype": torch.float32, "retain_state": False},
        ):
            model = CATYokoForCausalLM(cfg)
            apply_freeze(model, "B1")
            opt = build_optimizer(model, cfg, **kwargs)
            before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
            model(input_ids=ids, labels=ids)["loss"].backward()
            opt.step()
            after = next(p for p in model.parameters() if p.requires_grad)
            self.assertFalse(torch.equal(before, after), msg=kwargs)

    def test_clip_mixed_scales(self) -> None:
        p = torch.nn.Parameter(torch.ones(4))
        p.grad = torch.ones(4)
        n = clip_grad_norm_mixed([p], 1.0)
        self.assertAlmostEqual(n, 2.0, places=5)
        self.assertAlmostEqual(float(p.grad.norm()), 1.0, places=5)


class TrainerOffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = CATYokoConfig.tiny()
        torch.manual_seed(0)

    def test_encoder_offload_and_cpu_adam(self) -> None:
        nll = train_loop(
            self.cfg,
            "B1",
            steps=1,
            device="cpu",
            accum=1,
            offload_encoder=True,
            optim_cpu=True,
        )
        self.assertGreater(nll, 0)

    def test_block_offload_b2(self) -> None:
        nll = train_loop(
            self.cfg,
            "B2",
            steps=1,
            device="cpu",
            accum=1,
            offload_blocks=True,
            optim_cpu=True,
        )
        self.assertGreater(nll, 0)

    def test_block_offload_logs_grad_norm(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            Trainer(
                self.cfg,
                "B2",
                "cpu",
                steps=1,
                accum=1,
                offload_blocks=True,
                optim_cpu=True,
                log_path=log,
            ).run()
            row = json.loads(log.read_text().splitlines()[0])
            self.assertGreater(row["grad_norm"], 0)
            self.assertEqual(row["adam"], "ephemeral")

    def test_block_offload_rejects_accum(self) -> None:
        with self.assertRaises(RuntimeError):
            train_loop(
                self.cfg,
                "B2",
                steps=1,
                device="cpu",
                accum=2,
                offload_blocks=True,
                optim_cpu=True,
            )

    def test_c1_chain(self) -> None:
        out = run_c1_chain(self.cfg, "cpu", steps=1, accum=1, micro_batch=2)
        self.assertEqual(set(out), {"B0", "B1", "B2"})
        for phase, row in out.items():
            self.assertEqual(row.step, 1, msg=phase)
            self.assertGreater(row.nll, 0, msg=phase)

    def test_reuse_model_keeps_object(self) -> None:
        model = CATYokoForCausalLM(self.cfg)
        Trainer(
            self.cfg, "B0", "cpu", steps=1, accum=1, reuse_model=model
        ).run()
        Trainer(
            self.cfg, "B1", "cpu", steps=1, accum=1, reuse_model=model
        ).run()
        self.assertTrue(next(model.decoder[0].self_attn.parameters()).requires_grad)
        self.assertFalse(next(model.encoder.parameters()).requires_grad)


class CliOffloadTests(unittest.TestCase):
    def test_tiny_c1_smoke(self) -> None:
        code = main(["--config", "tiny", "--c1-smoke", "--steps", "1", "--accum", "1"])
        self.assertEqual(code, 0)

    def test_12b_c1_smoke_cpu_refuses(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "12b", "--c1-smoke", "--steps", "1"])

    def test_c1_smoke_rejects_tokens(self) -> None:
        with self.assertRaises(SystemExit):
            main(["--config", "tiny", "--c1-smoke", "--tokens", "64"])


if __name__ == "__main__":
    unittest.main()
