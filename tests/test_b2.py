#!/usr/bin/env python3
"""C1 B2: 15B, train all, detach=False, nvfp4 wrap, block offload accum=1."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from cat_yoko.attention import CrossAttention, WindowAttention
from cat_yoko.checkpoint import load_checkpoint, load_trainable_state, trainable_state_dict
from cat_yoko.config import C1_SPLIT, CATYokoConfig
from cat_yoko.freeze import apply_freeze, gate_schedule
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.nvfp4_linear import Nvfp4Linear, apply_nvfp4, nvfp4_module_names
from cat_yoko.optim import wsd_lr
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.phases import PHASES, TRY_STEPS
from cat_yoko.trainer import Trainer, train_loop
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm


def _nv_tiny() -> CATYokoConfig:
    return replace(CATYokoConfig.tiny(), use_nvfp4=True)


class PublishedB2Tests(unittest.TestCase):
    def test_envelope_and_flags(self) -> None:
        ph = PHASES["B2"]
        self.assertEqual(ph.tokens, C1_SPLIT["B2"])
        self.assertEqual(ph.tokens, 15e9)
        self.assertEqual(ph.student, "nvfp4")
        self.assertFalse(ph.detach)
        self.assertEqual(ph.gate_start, 1.0)
        self.assertEqual(ph.gate_end, 1.0)
        self.assertTrue(ph.offload_blocks)
        self.assertTrue(ph.optim_cpu)
        self.assertFalse(ph.offload_encoder)
        self.assertEqual(gate_schedule("B2", 0.0), 1.0)
        self.assertEqual(gate_schedule("B2", 1.0), 1.0)

    def test_try_resume_argv(self) -> None:
        argv = build_phase_argv(
            "B2",
            ["--try", "--resume", "/root/autodl-tmp/runs/b1", "--save-dir", "/root/autodl-tmp/runs/b2"],
        )
        self.assertEqual(argv[argv.index("--phase") + 1], "B2")
        self.assertEqual(argv[argv.index("--resume") + 1], "/root/autodl-tmp/runs/b1")
        self.assertEqual(argv[argv.index("--save-dir") + 1], "/root/autodl-tmp/runs/b2")
        self.assertIn("--offload-blocks", argv)
        self.assertIn("--optim-cpu", argv)
        self.assertNotIn("--offload-encoder", argv)
        self.assertEqual(argv[argv.index("--accum") + 1], "1")
        self.assertEqual(argv[argv.index("--steps") + 1], str(TRY_STEPS))
        self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
        self.assertEqual(argv[argv.index("--save-every") + 1], "8")
        self.assertIn("--save-trainable", argv)
        self.assertIn("--no-save-full", argv)
        self.assertIn("--no-save-optim", argv)
        self.assertIn("--dummy-upcycle", argv)
        self.assertNotIn("--tokens", argv)

    def test_try_keeps_explicit_upcycle(self) -> None:
        argv = build_phase_argv(
            "B2",
            [
                "--try",
                "--resume",
                "/root/autodl-tmp/runs/b1",
                "--upcycle-hf",
                "/root/autodl-tmp/hf/MiniCPM5-2B-Base",
            ],
        )
        self.assertIn("--upcycle-hf", argv)
        self.assertEqual(
            argv[argv.index("--upcycle-hf") + 1],
            "/root/autodl-tmp/hf/MiniCPM5-2B-Base",
        )
        self.assertNotIn("--dummy-upcycle", argv)
        self.assertEqual(argv[argv.index("--accum") + 1], "1")
        self.assertIn("--offload-blocks", argv)

    def test_wsd_uses_lr_b2_after_warmup(self) -> None:
        cfg = CATYokoConfig.tiny()
        self.assertAlmostEqual(wsd_lr(8e9 + 27e9, cfg, "B2"), cfg.lr_b2)


class FreezeB2Tests(unittest.TestCase):
    def test_all_requires_grad_and_no_detach(self) -> None:
        model = CATYokoForCausalLM(_nv_tiny())
        apply_freeze(model, "B2")
        self.assertFalse(model.detach_cache)
        for name, p in model.named_parameters():
            self.assertTrue(p.requires_grad, msg=name)
        self.assertTrue(model.embed.weight.requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)
        self.assertTrue(model.norm.weight.requires_grad)
        self.assertTrue(next(model.encoder.parameters()).requires_grad)
        self.assertTrue(next(model.decoder.parameters()).requires_grad)

    def test_encoder_gets_grad(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B2")
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        model(input_ids=ids, labels=ids)["loss"].backward()
        enc = next(model.encoder.parameters())
        self.assertIsNotNone(enc.grad)
        self.assertTrue(torch.isfinite(enc.grad).all())
        self.assertIsNotNone(model.embed.weight.grad)


class WrapB2Tests(unittest.TestCase):
    def test_wraps_encoder_decoder_lm_head_cache_skips_router(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B2")
        keys_before = set(model.state_dict().keys())
        n = apply_nvfp4(model, "B2", enabled=True)
        names = nvfp4_module_names(model)
        self.assertGreater(n, 0)
        self.assertEqual(set(model.state_dict().keys()), keys_before)
        enc = model.encoder[0].attn
        dec = model.decoder[0].self_attn
        for proj in (enc.q_proj, enc.k_proj, enc.v_proj, enc.o_proj):
            self.assertIsInstance(proj, Nvfp4Linear)
        for proj in (dec.q_proj, dec.k_proj, dec.v_proj, dec.o_proj):
            self.assertIsInstance(proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].mlp.experts[0].gate_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.experts[0].gate_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].cross_attn.q_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].cross_attn.o_proj, Nvfp4Linear)
        self.assertIsInstance(model.cache_k, Nvfp4Linear)
        self.assertIsInstance(model.cache_v, Nvfp4Linear)
        self.assertIsInstance(model.lm_head, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].mlp.experts[0].up_proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].mlp.experts[0].down_proj, Nvfp4Linear)
        self.assertIsInstance(model.encoder[0].mlp.shared[0].gate_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.experts[0].up_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.experts[0].down_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.shared[0].down_proj, Nvfp4Linear)
        leftover = [
            n
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not isinstance(m, Nvfp4Linear)
        ]
        self.assertTrue(leftover)
        self.assertTrue(all(n.endswith("router") for n in leftover), msg=leftover)
        self.assertFalse(any(n.endswith("router") for n in names))
        self.assertNotIsInstance(model.encoder[0].mlp.router, Nvfp4Linear)
        self.assertNotIsInstance(model.decoder[0].mlp.router, Nvfp4Linear)
        self.assertIsInstance(model.embed, nn.Embedding)
        self.assertIsInstance(enc, WindowAttention)
        self.assertIsInstance(dec, WindowAttention)
        self.assertIsInstance(model.decoder[0].cross_attn, CrossAttention)
        self.assertEqual(apply_nvfp4(model, "B2", enabled=True), 0)


class OverlayResumeTests(unittest.TestCase):
    def test_b1_overlay_loads_into_b2(self) -> None:
        cfg = _nv_tiny()
        torch.manual_seed(0)
        src = dummy_minicpm_state(cfg)
        with tempfile.TemporaryDirectory() as td:
            b1 = Path(td) / "b1"
            Trainer(
                cfg,
                "B1",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                save_dir=b1,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                upcycle_src=src,
                seed=1,
            ).run()
            overlay = load_checkpoint(b1 / "trainable.pt")
            keys = set(overlay["trainable"])
            self.assertTrue(any(k.startswith("decoder.") for k in keys))
            self.assertTrue(any(k.startswith("lm_head") for k in keys))
            self.assertFalse(any(k.startswith("encoder.") for k in keys))
            self.assertNotIn("embed.weight", keys)

            dst = CATYokoForCausalLM(cfg)
            torch.manual_seed(99)
            upcycle_from_minicpm(dst, src, cfg)
            enc_ref = next(dst.encoder.parameters()).detach().clone()
            embed_ref = dst.embed.weight.detach().clone()
            apply_freeze(dst, "B2")
            apply_nvfp4(dst, "B2", enabled=True)
            load_trainable_state(dst, overlay["trainable"])
            current = dst.state_dict()
            for name, tensor in overlay["trainable"].items():
                self.assertTrue(torch.equal(current[name].cpu(), tensor.cpu()), msg=name)
            self.assertTrue(torch.equal(next(dst.encoder.parameters()).cpu(), enc_ref.cpu()))
            self.assertTrue(torch.equal(dst.embed.weight.cpu(), embed_ref.cpu()))
            self.assertFalse(dst.detach_cache)
            for name, p in dst.named_parameters():
                self.assertTrue(p.requires_grad, msg=name)

            b2 = Path(td) / "b2"
            log = Path(td) / "b2.jsonl"
            out = Trainer(
                cfg,
                "B2",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                resume=b1,
                save_dir=b2,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                log_path=log,
                offload_blocks=True,
                optim_cpu=True,
                upcycle_src=src,
                seed=1,
            ).run()
            self.assertEqual(out.phase, "B2")
            self.assertEqual(out.step, 1)
            self.assertTrue(math.isfinite(out.nll) and out.nll > 0)
            b2_overlay = load_checkpoint(b2 / "trainable.pt")
            self.assertTrue(any(k.startswith("encoder.") for k in b2_overlay["trainable"]))
            self.assertIn("embed.weight", b2_overlay["trainable"])
            row = json.loads(log.read_text().splitlines()[0])
            self.assertTrue(row["nvfp4"])
            self.assertGreater(row["nvfp4_n"], 0)
            self.assertTrue(row["offload_blocks"])
            self.assertEqual(row["gate"], 1.0)
            self.assertEqual(row["accum"], 1)
            self.assertEqual(row["phase"], "B2")


class OffloadAndStepTests(unittest.TestCase):
    def test_offload_blocks_rejects_accum_gt_1(self) -> None:
        cfg = _nv_tiny()
        with self.assertRaises(RuntimeError) as ctx:
            train_loop(
                cfg,
                "B2",
                steps=1,
                device="cpu",
                accum=2,
                micro_batch=1,
                offload_blocks=True,
                optim_cpu=True,
            )
        msg = str(ctx.exception)
        self.assertIn("offload-blocks", msg)
        self.assertIn("accum", msg)
        self.assertIn("accum 1", msg)

    def test_finite_nll_one_step(self) -> None:
        cfg = _nv_tiny()
        torch.manual_seed(0)
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B2")
        apply_nvfp4(model, "B2", enabled=True)
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "m.jsonl"
            tr = Trainer(
                cfg,
                "B2",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                reuse_model=model,
                offload_blocks=True,
                optim_cpu=True,
                log_path=log,
            )
            out = tr.run()
            self.assertGreater(tr.nvfp4_n, 0)
            self.assertTrue(math.isfinite(out.nll) and out.nll > 0)
            self.assertFalse(model.detach_cache)
            for name, p in model.named_parameters():
                self.assertTrue(p.requires_grad, msg=name)
            row = json.loads(log.read_text().splitlines()[0])
            self.assertTrue(row["nvfp4"])
            self.assertGreater(row["nvfp4_n"], 0)
            self.assertTrue(row["offload_blocks"])
            self.assertEqual(row["gate"], 1.0)

    def test_b2_overlay_is_full_graph(self) -> None:
        cfg = CATYokoConfig.tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B2")
        overlay = trainable_state_dict(model)
        self.assertEqual(set(overlay), {n for n, _ in model.named_parameters()})
        self.assertIn("embed.weight", overlay)
        self.assertTrue(any(k.startswith("encoder.") for k in overlay))


class B2ScriptAndHubTests(unittest.TestCase):
    def test_run_b2_try_autodl_script(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "run_b2_try_autodl.sh"
        self.assertTrue(path.is_file())
        self.assertTrue(path.stat().st_mode & 0o111)
        text = path.read_text(encoding="utf-8")
        self.assertIn("--try", text)
        self.assertIn("/root/autodl-tmp/runs/b1", text)
        self.assertIn("/root/autodl-tmp/runs/b2", text)
        self.assertIn("/root/autodl-tmp/hf/MiniCPM5-2B-Base", text)
        self.assertIn("--upcycle-hf", text)
        self.assertIn("encoder+embed", text)
        self.assertIn("cat_yoko.b2", text)
        self.assertIn("autodl_env.sh", text)
        self.assertNotIn("git fetch", text)
        self.assertNotIn("git pull", text)
        self.assertNotIn("Ultra-FineWeb", text)
        self.assertNotIn("--save-full", text)
        self.assertNotIn("westc", text)
        self.assertNotIn("BEGIN OPENSSH", text)

    def test_hub_and_artifact_pointers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        hub = root / "checkpoints" / "b2" / "README.md"
        art = root / "artifacts" / "autodl-rtx6000d" / "b2" / "README.md"
        self.assertTrue(hub.is_file(), hub)
        self.assertTrue(art.is_file(), art)
        hub_txt = hub.read_text(encoding="utf-8")
        art_txt = art.read_text(encoding="utf-8")
        self.assertIn("huggingface.co/AvrovaDonz/CAT-YOKO", hub_txt)
        self.assertIn("checkpoints/b2", hub_txt)
        self.assertIn("15e9", hub_txt)
        self.assertIn("--try", hub_txt)
        self.assertIn("runs/b1", hub_txt)
        self.assertIn("huggingface.co/AvrovaDonz/CAT-YOKO", art_txt)
        self.assertIn("run_b2_try_autodl.sh", art_txt)
        self.assertIn("offload-blocks", art_txt)
        self.assertIn("accum=1", art_txt)
        self.assertNotIn("BEGIN OPENSSH", hub_txt + art_txt)


if __name__ == "__main__":
    unittest.main()
