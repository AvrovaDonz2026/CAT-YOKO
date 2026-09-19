#!/usr/bin/env python3
"""C1 B1 CPU tiny: freeze names, NVFP4 wrap, B0 overlay resume, one-step NLL."""

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
from cat_yoko.checkpoint import is_trainable_ckpt, load_checkpoint, load_trainable_state
from cat_yoko.config import CATYokoConfig, C1_SPLIT
from cat_yoko.freeze import apply_freeze, gate_schedule, trainable_names
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.nvfp4_linear import Nvfp4Linear, apply_nvfp4, nvfp4_module_names
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.phases import PHASES, PUBLISHED_SAVE_EVERY, PUBLISHED_SEQ, TRY_STEPS
from cat_yoko.rope import RMSNorm
from cat_yoko.trainer import Trainer
from cat_yoko.upcycle import dummy_minicpm_state, upcycle_from_minicpm


def _nv_tiny() -> CATYokoConfig:
    return replace(CATYokoConfig.tiny(), use_nvfp4=True)


class B1PublishedSpecTests(unittest.TestCase):
    def test_envelope_flags(self) -> None:
        ph = PHASES["B1"]
        self.assertEqual(ph.tokens, C1_SPLIT["B1"])
        self.assertEqual(ph.tokens, 27e9)
        self.assertEqual(ph.student, "nvfp4")
        self.assertTrue(ph.detach)
        self.assertEqual(ph.gate_start, 0.3)
        self.assertEqual(ph.gate_end, 1.0)
        self.assertTrue(ph.offload_encoder)
        self.assertTrue(ph.optim_cpu)
        self.assertFalse(ph.offload_blocks)


class B1FreezeTests(unittest.TestCase):
    def test_b1_freeze_names(self) -> None:
        model = CATYokoForCausalLM(CATYokoConfig.tiny())
        apply_freeze(model, "B1")
        names = trainable_names(model)
        self.assertTrue(model.detach_cache)
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertFalse(any(n == "embed.weight" or n.startswith("embed.") for n in names))
        self.assertFalse(any(n.startswith("encoder.") for n in names))
        self.assertFalse(next(model.encoder.parameters()).requires_grad)
        self.assertTrue(model.lm_head.weight.requires_grad)
        self.assertIn("lm_head.weight", names)
        self.assertTrue(model.norm.weight.requires_grad)
        self.assertIn("norm.weight", names)
        self.assertTrue(model.cache_k.weight.requires_grad)
        self.assertTrue(model.cache_v.weight.requires_grad)
        self.assertTrue(any(n.startswith("cache_k") for n in names))
        self.assertTrue(any(n.startswith("cache_v") for n in names))
        self.assertTrue(any("self_attn" in n for n in names))
        self.assertTrue(any("cross_attn" in n for n in names))
        self.assertTrue(any("ln_cross" in n for n in names))
        self.assertTrue(any(".mlp." in n for n in names))
        self.assertTrue(any(n.endswith("ln1.weight") and "decoder" in n for n in names))
        self.assertTrue(any(n.endswith("ln2.weight") and "decoder" in n for n in names))
        self.assertTrue(model.decoder[0].mlp.router.weight.requires_grad)
        self.assertTrue(any(n.endswith("mlp.router.weight") and "decoder" in n for n in names))
        self.assertAlmostEqual(gate_schedule("B1", 0.0), 0.3)
        self.assertAlmostEqual(gate_schedule("B1", 1.0), 1.0)
        self.assertAlmostEqual(gate_schedule("B1", 0.5), 0.65)


class B1Nvfp4WrapTests(unittest.TestCase):
    def test_b1_apply_nvfp4_wraps_cache_cross_lm_head_skips_router(self) -> None:
        cfg = _nv_tiny()
        model = CATYokoForCausalLM(cfg)
        apply_freeze(model, "B1")
        n = apply_nvfp4(model, "B1", enabled=True)
        names = nvfp4_module_names(model)
        self.assertGreater(n, 0)
        self.assertIsInstance(model.cache_k, Nvfp4Linear)
        self.assertIsInstance(model.cache_v, Nvfp4Linear)
        self.assertIsInstance(model.lm_head, Nvfp4Linear)
        cross = model.decoder[0].cross_attn
        self.assertIsInstance(cross, CrossAttention)
        self.assertIsInstance(cross.q_proj, Nvfp4Linear)
        self.assertIsInstance(cross.o_proj, Nvfp4Linear)
        dec = model.decoder[0].self_attn
        self.assertIsInstance(dec, WindowAttention)
        for proj in (dec.q_proj, dec.k_proj, dec.v_proj, dec.o_proj):
            self.assertIsInstance(proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.experts[0].gate_proj, Nvfp4Linear)
        self.assertIsInstance(model.decoder[0].mlp.shared[0].down_proj, Nvfp4Linear)
        self.assertFalse(any(n.endswith("router") for n in names))
        self.assertNotIsInstance(model.decoder[0].mlp.router, Nvfp4Linear)
        self.assertNotIsInstance(model.encoder[0].mlp.router, Nvfp4Linear)
        leftover = [
            n
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not isinstance(m, Nvfp4Linear)
        ]
        self.assertTrue(leftover)
        self.assertTrue(all(n.endswith("router") for n in leftover), msg=leftover)
        self.assertIsInstance(model.embed, nn.Embedding)
        self.assertNotIsInstance(model.embed, Nvfp4Linear)
        self.assertIsInstance(dec.q_norm, RMSNorm)
        self.assertIsInstance(dec.k_norm, RMSNorm)
        self.assertIsInstance(cross.q_norm, RMSNorm)
        self.assertNotIsInstance(dec.q_norm, Nvfp4Linear)
        self.assertNotIsInstance(cross.q_norm, Nvfp4Linear)


class B1ResumeHandoffTests(unittest.TestCase):
    def test_b0_to_b1_resume_handoff(self) -> None:
        cfg = _nv_tiny()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b0_dir = td / "b0"
            b1_dir = td / "b1"
            log = b1_dir / "metrics.jsonl"
            Trainer(
                cfg,
                "B0",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                seed=0,
                save_dir=b0_dir,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                upcycle_src=dummy_minicpm_state(cfg),
            ).run()
            overlay = b0_dir / "trainable.pt"
            self.assertTrue(overlay.is_file())
            ckpt = load_checkpoint(overlay)
            self.assertTrue(is_trainable_ckpt(ckpt))
            keys = set(ckpt["trainable"])
            self.assertIn("cache_k.weight", keys)
            self.assertIn("cache_v.weight", keys)
            self.assertTrue(any("cross_attn" in k for k in keys))
            self.assertNotIn("lm_head.weight", keys)
            self.assertFalse(any(k.startswith("encoder.") for k in keys))
            self.assertFalse(any(k.startswith("embed.") for k in keys))
            cache_k = ckpt["trainable"]["cache_k.weight"].clone()

            handed = CATYokoForCausalLM(cfg)
            torch.manual_seed(99)
            load_trainable_state(handed, ckpt["trainable"])
            apply_freeze(handed, "B1")
            self.assertTrue(torch.equal(handed.cache_k.weight.detach().cpu(), cache_k.cpu()))
            self.assertTrue(handed.lm_head.weight.requires_grad)
            self.assertFalse(handed.embed.weight.requires_grad)
            self.assertTrue(handed.detach_cache)

            out = Trainer(
                cfg,
                "B1",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                seed=1,
                resume=b0_dir,
                save_dir=b1_dir,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                log_path=log,
                upcycle_src=dummy_minicpm_state(cfg),
            ).run()
            self.assertEqual(out.step, 1)
            self.assertEqual(out.phase, "B1")
            self.assertTrue(math.isfinite(out.nll))
            self.assertGreater(out.nll, 0.0)
            b1_overlay = b1_dir / "trainable.pt"
            self.assertTrue(b1_overlay.is_file())
            b1_ckpt = load_checkpoint(b1_overlay)
            self.assertTrue(is_trainable_ckpt(b1_ckpt))
            self.assertEqual(b1_ckpt["extra"]["phase"], "B1")
            self.assertEqual(b1_ckpt["extra"]["step"], 1)
            b1_keys = set(b1_ckpt["trainable"])
            self.assertIn("lm_head.weight", b1_keys)
            self.assertIn("norm.weight", b1_keys)
            self.assertTrue(any("self_attn" in k for k in b1_keys))
            self.assertFalse(any(k.startswith("encoder.") for k in b1_keys))
            self.assertFalse(any(k.startswith("embed.") for k in b1_keys))

    def test_minicpm_upcycle_then_b0_overlay_then_wrap(self) -> None:
        """B1 GPU path: same MiniCPM5 dummy → B0 overlay cache/cross → wrap student GEMMs."""
        cfg = _nv_tiny()
        src = dummy_minicpm_state(cfg)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            b0_dir = td / "b0"
            Trainer(
                cfg,
                "B0",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                seed=0,
                save_dir=b0_dir,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                upcycle_src=src,
            ).run()
            overlay = load_checkpoint(b0_dir / "trainable.pt")["trainable"]
            cache_k = overlay["cache_k.weight"].clone()

            handed = CATYokoForCausalLM(cfg)
            upcycle_from_minicpm(handed, src, cfg)
            embed = handed.embed.weight.detach().clone()
            enc0 = next(handed.encoder.parameters()).detach().clone()
            load_trainable_state(handed, overlay)
            apply_freeze(handed, "B1")
            n = apply_nvfp4(handed, "B1", enabled=True)
            self.assertGreater(n, 0)
            self.assertTrue(torch.equal(handed.embed.weight.detach().cpu(), embed.cpu()))
            self.assertTrue(torch.equal(next(handed.encoder.parameters()).detach().cpu(), enc0.cpu()))
            self.assertTrue(torch.equal(handed.cache_k.weight.detach().cpu(), cache_k.cpu()))
            self.assertFalse(handed.embed.weight.requires_grad)
            self.assertFalse(next(handed.encoder.parameters()).requires_grad)
            self.assertTrue(handed.lm_head.weight.requires_grad)
            self.assertTrue(handed.norm.weight.requires_grad)
            self.assertTrue(handed.detach_cache)
            self.assertIsInstance(handed.cache_k, Nvfp4Linear)
            self.assertIsInstance(handed.cache_v, Nvfp4Linear)
            self.assertIsInstance(handed.lm_head, Nvfp4Linear)
            self.assertIsInstance(handed.decoder[0].cross_attn.q_proj, Nvfp4Linear)
            self.assertIsInstance(handed.decoder[0].self_attn.q_proj, Nvfp4Linear)
            self.assertIsInstance(handed.decoder[0].mlp.experts[0].gate_proj, Nvfp4Linear)
            self.assertNotIsInstance(handed.decoder[0].mlp.router, Nvfp4Linear)
            self.assertNotIsInstance(handed.encoder[0].mlp.router, Nvfp4Linear)


class B1TrainStepTests(unittest.TestCase):
    def test_finite_nll_one_step(self) -> None:
        cfg = _nv_tiny()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            log = td / "metrics.jsonl"
            save = td / "b1"
            tr = Trainer(
                cfg,
                "B1",
                "cpu",
                steps=1,
                accum=1,
                micro_batch=1,
                save_dir=save,
                save_every=1,
                save_full=False,
                save_trainable=True,
                save_optim=False,
                log_path=log,
            )
            out = tr.run()
            self.assertTrue(math.isfinite(out.nll))
            self.assertGreater(out.nll, 0.0)
            self.assertGreater(tr.nvfp4_n, 0)
            self.assertTrue((save / "trainable.pt").is_file())
            self.assertFalse((save / "latest.pt").is_file())
            row = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(row["phase"], "B1")
            self.assertTrue(row["nvfp4"])
            self.assertGreater(row["nvfp4_n"], 0)
            self.assertTrue(math.isfinite(row["nll"]))
            self.assertAlmostEqual(row["gate"], 1.0)

    def test_gate_ramps_from_b0_end(self) -> None:
        cfg = _nv_tiny()
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "metrics.jsonl"
            Trainer(
                cfg,
                "B1",
                "cpu",
                steps=2,
                accum=1,
                micro_batch=1,
                log_path=log,
            ).run()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[0]["gate"], 0.3 + 0.7 * 0.5)
            self.assertAlmostEqual(rows[1]["gate"], 1.0)
            self.assertTrue(rows[0]["nvfp4"])


class B1ArgvTests(unittest.TestCase):
    def test_b1_try_resume_argv(self) -> None:
        argv = build_phase_argv(
            "B1",
            ["--try", "--resume", "/root/autodl-tmp/runs/b0", "--save-dir", "/root/autodl-tmp/runs/b1"],
        )
        self.assertEqual(argv[argv.index("--phase") + 1], "B1")
        self.assertEqual(argv[argv.index("--resume") + 1], "/root/autodl-tmp/runs/b0")
        self.assertEqual(argv[argv.index("--save-dir") + 1], "/root/autodl-tmp/runs/b1")
        self.assertEqual(argv[argv.index("--steps") + 1], str(TRY_STEPS))
        self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
        self.assertEqual(argv[argv.index("--save-every") + 1], "8")
        self.assertIn("--offload-encoder", argv)
        self.assertIn("--optim-cpu", argv)
        self.assertIn("--no-save-full", argv)
        self.assertIn("--save-trainable", argv)
        self.assertIn("--no-save-optim", argv)
        self.assertIn("--dummy-upcycle", argv)
        self.assertNotIn("--offload-blocks", argv)
        self.assertNotIn("--tokens", argv)
        log = argv[argv.index("--log") + 1]
        self.assertTrue(log.endswith("metrics.jsonl"))

    def test_b1_try_keeps_explicit_upcycle(self) -> None:
        argv = build_phase_argv(
            "B1",
            [
                "--try",
                "--resume",
                "/tmp/b0",
                "--upcycle-hf",
                "/root/autodl-tmp/hf/MiniCPM5-2B-Base",
            ],
        )
        self.assertIn("--upcycle-hf", argv)
        self.assertNotIn("--dummy-upcycle", argv)

    def test_b1_envelope_argv(self) -> None:
        from unittest.mock import patch

        with patch("cat_yoko.phase_train._tight_gpu", return_value=False):
            argv = build_phase_argv("B1", ["--save-dir", "/tmp/b1", "--resume", "/tmp/b0"])
        self.assertEqual(argv[argv.index("--phase") + 1], "B1")
        self.assertEqual(argv[argv.index("--tokens") + 1], str(C1_SPLIT["B1"]))
        self.assertEqual(C1_SPLIT["B1"], 27e9)
        self.assertIn("--offload-encoder", argv)
        self.assertIn("--optim-cpu", argv)
        self.assertIn("--save-trainable", argv)
        self.assertIn("--no-save-full", argv)
        self.assertNotIn("--offload-blocks", argv)
        self.assertNotIn("--dummy-upcycle", argv)
        self.assertEqual(argv[argv.index("--seq-len") + 1], str(PUBLISHED_SEQ))
        self.assertEqual(argv[argv.index("--save-every") + 1], str(PUBLISHED_SAVE_EVERY))


class B1ScriptTests(unittest.TestCase):
    def test_run_b1_try_autodl_script(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "run_b1_try_autodl.sh"
        self.assertTrue(path.is_file())
        self.assertTrue(path.stat().st_mode & 0o111)
        text = path.read_text(encoding="utf-8")
        self.assertIn("--try", text)
        self.assertIn("/root/autodl-tmp/runs/b1", text)
        self.assertIn("/root/autodl-tmp/runs/b0-full", text)
        self.assertIn("/root/autodl-tmp/runs/b0", text)
        self.assertIn("HF_ENDPOINT", text)
        self.assertIn("https://hf-mirror.com", text)
        self.assertIn("cat_yoko.b1", text)
        self.assertIn("--upcycle-hf", text)
        self.assertIn("MiniCPM5-2B-Base", text)
        self.assertIn("download_minicpm5", text)
        self.assertNotIn("westc", text)
        self.assertNotIn("Ultra-FineWeb", text)
        self.assertNotIn("BEGIN OPENSSH", text)
        self.assertNotIn("git fetch", text)
        self.assertNotIn("git pull", text)
        self.assertNotIn("PRIVATE KEY", text)
        self.assertNotRegex(text.lower(), r"password\s*=")

    def test_hub_pointer_and_artifacts_readme(self) -> None:
        root = Path(__file__).resolve().parents[1]
        hub = root / "checkpoints" / "b1" / "README.md"
        art = root / "artifacts" / "autodl-rtx6000d" / "b1" / "README.md"
        self.assertTrue(hub.is_file())
        self.assertTrue(art.is_file())
        body = hub.read_text(encoding="utf-8")
        self.assertIn("huggingface.co/AvrovaDonz/CAT-YOKO", body)
        self.assertIn("checkpoints/b1", body)
        self.assertIn("trainable.pt", body)
        self.assertIn("不进 GitHub", body)
        self.assertNotIn("BEGIN OPENSSH", body)
        art_body = art.read_text(encoding="utf-8")
        self.assertIn("--try", art_body)
        self.assertIn("27e9", art_body)
        self.assertIn("hf-mirror", art_body)
        self.assertIn("MiniCPM5", art_body)
        self.assertIn("offload-encoder", art_body)
        self.assertIn("optim-cpu", art_body)
        self.assertNotIn("Ultra-FineWeb", art_body)
        self.assertFalse((root / "checkpoints" / "b1" / "trainable.pt").exists())


if __name__ == "__main__":
    unittest.main()
