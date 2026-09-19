#!/usr/bin/env python3
"""Phase C–G training entries: argv, freeze, indexer KL, SFT, GRPO, no CSA kernel."""

from __future__ import annotations

import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cat_yoko.config import CATYokoConfig, KEEP_HIGH_PREC
from cat_yoko.freeze import apply_freeze, trainable_names
from cat_yoko.indexer import (
    LightningIndexer,
    ensure_indexers,
    indexer_param_names,
    phase_needs_indexer,
)
from cat_yoko.model import CATYokoForCausalLM
from cat_yoko.optim import wsd_lr
from cat_yoko.phase_train import build_phase_argv, main_c, main_d, main_e, main_f, main_g
from cat_yoko.phases import C_STAGES, D_STAGES, PHASES, TRY_STEPS, c_chain, resolve_phase_spec
from cat_yoko.trainer import Trainer


class PhaseEnvelopeTests(unittest.TestCase):
    def test_b_through_g_registered(self) -> None:
        for name in (
            "B0",
            "B1",
            "B2",
            "C-kda",
            "C-index",
            "C-topk",
            "C-hca",
            "C-win",
            "D-8k",
            "D-32k",
            "D-128k",
            "E",
            "F",
            "G",
            "G-dpo",
        ):
            self.assertIn(name, PHASES)
        self.assertEqual(PHASES["C-index"].loss, "indexer_kl")
        self.assertEqual(PHASES["C-index"].student, "bf16")
        self.assertTrue(PHASES["C-index"].align_indexer)
        self.assertEqual(PHASES["C-topk"].sparse, "topk")
        self.assertEqual(PHASES["C-hca"].sparse, "hca")
        self.assertEqual(PHASES["C-win"].seq_len, 8192)
        self.assertEqual(PHASES["C-win"].sparse, "hca")
        self.assertEqual(PHASES["D-128k"].seq_len, 131072)
        self.assertEqual(PHASES["E"].lr_mode, "decay")
        self.assertEqual(PHASES["F"].loss, "sft")
        self.assertEqual(PHASES["G"].loss, "grpo")
        self.assertEqual(PHASES["G-dpo"].loss, "dpo")
        self.assertEqual(PHASES["C-index"].tokens, 10e9)
        self.assertEqual(PHASES["C-kda"].sparse, "kda")
        self.assertEqual(PHASES["C-kda"].tokens, 5e9)
        self.assertEqual(PHASES["C-kda"].freeze, "none")
        self.assertEqual(PHASES["C-kda"].student, "nvfp4")
        self.assertFalse(PHASES["C-kda"].align_indexer)
        self.assertEqual(c_chain(), ("C-index", "C-topk", "C-hca", "C-win"))
        self.assertEqual(
            c_chain(use_kda=True),
            ("C-kda", "C-index", "C-topk", "C-hca", "C-win"),
        )
        kda_c = sum(resolve_phase_spec(n, use_kda=True).tokens for n in c_chain(use_kda=True))
        plain_c = sum(PHASES[n].tokens for n in c_chain())
        self.assertEqual(kda_c, 25e9)
        self.assertEqual(plain_c, 25e9)
        self.assertEqual(resolve_phase_spec("C-index", use_kda=True).tokens, 5e9)
        self.assertEqual(resolve_phase_spec("C-index", use_kda=True).sparse, "kda")
        self.assertEqual(resolve_phase_spec("D-8k", use_kda=True).sparse, "hca")
        self.assertEqual(resolve_phase_spec("D-8k", use_kda=False).sparse, "window")
        self.assertLessEqual(PHASES["C-index"].tokens + PHASES["C-topk"].tokens, 50e9)
        self.assertIn("indexer", KEEP_HIGH_PREC)
        self.assertIn("kda_gate", KEEP_HIGH_PREC)

    def test_c_try_argv(self) -> None:
        argv = build_phase_argv("C-index", ["--try", "--save-dir", "/tmp/c"])
        self.assertEqual(argv[argv.index("--phase") + 1], "C-index")
        self.assertEqual(argv[argv.index("--steps") + 1], str(TRY_STEPS))
        self.assertEqual(argv[argv.index("--seq-len") + 1], "64")
        self.assertIn("--dummy-upcycle", argv)
        self.assertIn("--no-save-full", argv)
        self.assertIn("--save-trainable", argv)
        self.assertNotIn("--tokens", argv)
        self.assertNotIn("--offload-encoder", argv)

    def test_c_published_tokens(self) -> None:
        with patch("cat_yoko.phase_train._tight_gpu", return_value=False):
            argv = build_phase_argv("C-index", ["--save-dir", "/tmp/c"])
        self.assertEqual(argv[argv.index("--tokens") + 1], str(10e9))
        self.assertEqual(argv[argv.index("--seq-len") + 1], "4096")

    def test_d_seq_curriculum_argv(self) -> None:
        with patch("cat_yoko.phase_train._tight_gpu", return_value=False):
            a8 = build_phase_argv("D-8k", ["--save-dir", "/tmp/d"])
            a32 = build_phase_argv("D-32k", ["--save-dir", "/tmp/d"])
            a128 = build_phase_argv("D-128k", ["--save-dir", "/tmp/d"])
        self.assertEqual(a8[a8.index("--seq-len") + 1], "8192")
        self.assertEqual(a32[a32.index("--seq-len") + 1], "32768")
        self.assertEqual(a128[a128.index("--seq-len") + 1], "131072")

    def test_g_steps_not_tokens(self) -> None:
        with patch("cat_yoko.phase_train._tight_gpu", return_value=False):
            argv = build_phase_argv("G", ["--save-dir", "/tmp/g"])
        self.assertEqual(argv[argv.index("--steps") + 1], "10000")
        self.assertNotIn("--tokens", argv)

    def test_stage_wrappers(self) -> None:
        self.assertEqual(C_STAGES["indexer"], "C-index")
        self.assertEqual(C_STAGES["kda"], "C-kda")
        self.assertEqual(D_STAGES["128k"], "D-128k")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            self.assertEqual(main_c(["--stage", "topk", "--try"]), 0)
            run.assert_called_once()
            self.assertEqual(run.call_args[0][0], "C-topk")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            self.assertEqual(main_c(["--stage", "kda", "--try"]), 0)
            self.assertEqual(run.call_args[0][0], "C-kda")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            main_d(["--stage", "32k", "--try"])
            self.assertEqual(run.call_args[0][0], "D-32k")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            main_e(["--try"])
            self.assertEqual(run.call_args[0][0], "E")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            main_f(["--try"])
            self.assertEqual(run.call_args[0][0], "F")
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            main_g(["--algo", "dpo", "--try"])
            self.assertEqual(run.call_args[0][0], "G-dpo")

    def test_c_chain_runs_four_stages(self) -> None:
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            self.assertEqual(main_c(["--chain", "--try", "--save-dir", "/tmp/c"]), 0)
        self.assertEqual(run.call_count, 4)
        names = [c.args[0] for c in run.call_args_list]
        self.assertEqual(names, ["C-index", "C-topk", "C-hca", "C-win"])
        second = run.call_args_list[1].args[1]
        self.assertIn("--resume", second)

    def test_c_chain_use_kda_lights_kda_first(self) -> None:
        with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
            self.assertEqual(
                main_c(["--chain", "--use-kda", "--try", "--save-dir", "/tmp/c"]),
                0,
            )
        names = [c.args[0] for c in run.call_args_list]
        self.assertEqual(names, ["C-kda", "C-index", "C-topk", "C-hca", "C-win"])
        self.assertIn("--use-kda", run.call_args_list[0].args[1])
        self.assertLess(names.index("C-kda"), names.index("C-index"))
        self.assertLess(names.index("C-topk"), names.index("C-hca"))

    def test_c_index_use_kda_carves_tokens(self) -> None:
        with patch("cat_yoko.phase_train._tight_gpu", return_value=False):
            argv = build_phase_argv("C-index", ["--use-kda", "--save-dir", "/tmp/c"])
        self.assertEqual(argv[argv.index("--tokens") + 1], str(5e9))
        self.assertIn("--use-kda", argv)

    def test_c_chain_rejects_late_kda(self) -> None:
        with patch(
            "cat_yoko.checkpoint.peek_checkpoint_extra",
            return_value={"use_kda": False},
        ):
            with self.assertRaises(SystemExit) as ctx:
                main_c(["--chain", "--use-kda", "--try", "--resume", "/tmp/b2"])
        self.assertIn("implement-then-light", str(ctx.exception))

    def test_c_chain_inherits_kda_from_resume(self) -> None:
        with patch(
            "cat_yoko.checkpoint.peek_checkpoint_extra",
            return_value={"use_kda": True},
        ):
            with patch("cat_yoko.phase_train.run_phase", return_value=0) as run:
                self.assertEqual(
                    main_c(["--chain", "--try", "--resume", "/tmp/b2", "--save-dir", "/tmp/c"]),
                    0,
                )
        names = [c.args[0] for c in run.call_args_list]
        self.assertEqual(names, ["C-kda", "C-index", "C-topk", "C-hca", "C-win"])
        self.assertIn("--use-kda", run.call_args_list[0].args[1])

    def test_c_chain_rejects_stage(self) -> None:
        with self.assertRaises(SystemExit):
            main_c(["--chain", "--stage", "topk"])

    def test_no_vast_ip_in_new_scripts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("train_c.py", "train_d.py", "train_e.py", "train_f.py", "train_g.py"):
            text = (root / "scripts" / name).read_text()
            self.assertNotIn("137.175.", text)
            self.assertNotIn("Ultra-FineWeb", text)
            self.assertNotIn("BEGIN OPENSSH", text)


class IndexerFreezeTests(unittest.TestCase):
    def test_default_graph_has_no_indexer(self) -> None:
        model = CATYokoForCausalLM(CATYokoConfig.tiny())
        self.assertFalse(any("indexer" in n for n, _ in model.named_modules()))
        self.assertFalse(phase_needs_indexer("B0"))
        self.assertTrue(phase_needs_indexer("C-index"))
        self.assertFalse(phase_needs_indexer("C-kda"))
        self.assertTrue(phase_needs_indexer("C-topk"))
        self.assertEqual(model.encoder[0].kind, "sliding")
        self.assertEqual(model.encoder[1].kind, "csa")

    def test_c_index_trains_only_indexer(self) -> None:
        model = CATYokoForCausalLM(CATYokoConfig.tiny())
        ensure_indexers(model, model.cfg)
        apply_freeze(model, "C-index")
        names = trainable_names(model)
        self.assertTrue(names)
        self.assertTrue(all("indexer" in n for n in names))
        self.assertFalse(any("cross_indexer" in n for n in names))
        self.assertFalse(any("cross_indexer" in n for n, _ in model.named_modules()))
        self.assertFalse(model.embed.weight.requires_grad)
        self.assertFalse(next(model.encoder.parameters()).requires_grad)
        self.assertTrue(model.detach_cache)
        self.assertEqual(set(names), set(indexer_param_names(model)))

    def test_indexer_not_in_attention_module(self) -> None:
        import cat_yoko.attention as attn_mod

        src = inspect.getsource(attn_mod)
        self.assertNotIn("class CSA", src)
        self.assertNotIn("LightningIndexer", src)
        self.assertIn("class LightningIndexer", inspect.getsource(LightningIndexer))


class IndexerTrainTests(unittest.TestCase):
    def test_c_index_one_step_finite_kl(self) -> None:
        cfg = CATYokoConfig.tiny()
        with tempfile.TemporaryDirectory() as td:
            save = Path(td)
            log = save / "metrics.jsonl"
            out = Trainer(
                cfg,
                "C-index",
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
            ).run()
            self.assertEqual(out.phase, "C-index")
            self.assertTrue(math.isfinite(out.nll))
            self.assertGreaterEqual(out.nll, 0.0)
            self.assertLess(out.nll, 80.0)
            overlay = torch.load(save / "trainable.pt", map_location="cpu", weights_only=False)
            keys = set(overlay["trainable"])
            self.assertTrue(any("indexer" in k for k in keys))
            self.assertFalse(any(k.startswith("embed.") for k in keys))
            row = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(row["phase"], "C-index")
            self.assertEqual(row["loss_mode"], "indexer_kl")
            self.assertIn("indexer_recall", row)
            self.assertGreaterEqual(row["indexer_recall"], 0.0)
            self.assertLessEqual(row["indexer_recall"], 1.0)

    def test_c_topk_and_hca_ce_step(self) -> None:
        cfg = CATYokoConfig.tiny()
        for phase in ("C-topk", "C-hca", "C-win"):
            nll = Trainer(cfg, phase, "cpu", steps=1, accum=1, micro_batch=1).run().nll
            self.assertTrue(math.isfinite(nll), msg=phase)


class LaterPhaseTests(unittest.TestCase):
    def test_d_e_f_steps(self) -> None:
        cfg = CATYokoConfig.tiny()
        for phase in ("D-8k", "E", "F"):
            tr = Trainer(cfg, phase, "cpu", steps=1, accum=1, micro_batch=1, seq_len=16)
            out = tr.run()
            self.assertTrue(math.isfinite(out.nll), msg=phase)
            if phase == "F":
                self.assertEqual(tr.loss_mode, "sft")

    def test_wsd_decay_drops(self) -> None:
        cfg = CATYokoConfig.tiny()
        early = wsd_lr(0, cfg, "E", tokens_in_phase=0.0, phase_budget=1000)
        late = wsd_lr(0, cfg, "E", tokens_in_phase=1000.0, phase_budget=1000)
        self.assertGreater(early, late)
        self.assertAlmostEqual(late / cfg.lr, cfg.wsd_decay_min_ratio, places=5)

    def test_sft_dummy_masks_prompt(self) -> None:
        from cat_yoko.data import DummyStream

        stream = DummyStream(32, 8, seed=0, response_only=True, prompt_frac=0.5)
        batch = stream.batch(1, "cpu")
        self.assertTrue((batch["labels"][0, :4] == -100).all())
        self.assertFalse((batch["labels"][0, 4:] == -100).all())

    def test_g_grpo_and_dpo_finite(self) -> None:
        cfg = CATYokoConfig.tiny()
        g = Trainer(cfg, "G", "cpu", steps=1, accum=1, micro_batch=1, seq_len=16).run()
        self.assertTrue(math.isfinite(g.nll))
        d = Trainer(cfg, "G-dpo", "cpu", steps=1, accum=1, micro_batch=1, seq_len=16).run()
        self.assertTrue(math.isfinite(d.nll))

    def test_sft_prompt_ids_jsonl(self) -> None:
        from cat_yoko.data import FileStream

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt_ids": [1, 2, 3, 4], "response_ids": [5, 6, 7, 8]}) + "\n"
            )
            batch = FileStream(path, 8).batch(1, "cpu")
            self.assertTrue((batch["labels"][0, :4] == -100).all())
            self.assertEqual(batch["labels"][0, 4:].tolist(), [5, 6, 7, 8])


class TheoremBMaskTests(unittest.TestCase):
    def test_compressed_keep_excludes_own_block(self) -> None:
        from cat_yoko.sparse import compressed_keep_matrix, own_block_token_keep, window_keep_matrix

        s, m = 32, 4
        comp = compressed_keep_matrix(s, m, torch.device("cpu"))
        own = own_block_token_keep(s, m, torch.device("cpu"))
        win = window_keep_matrix(s, 8, torch.device("cpu"))
        self.assertFalse((comp & own).any())
        t, p = 13, 12  # own block 3 = tokens 12..15
        self.assertFalse(bool(comp[t, p]))
        self.assertTrue(bool(own[t, p]))
        self.assertTrue(bool(win[t, p]))
        self.assertTrue(bool(win[t, t]))
        self.assertFalse(bool(win[t, t + 1]))

    def test_hca_slot_excludes_own(self) -> None:
        from cat_yoko.sparse import hca_slot_keep

        keep = hca_slot_keep(16, 4, 4, torch.device("cpu"))
        self.assertFalse(bool(keep[13, 3]))
        self.assertTrue(bool(keep[13, 2]))

    def test_indexer_topk_is_subset_of_compressed(self) -> None:
        from cat_yoko.indexer import LightningIndexer, indexer_compressed_keep
        from cat_yoko.sparse import compressed_keep_matrix

        cfg = CATYokoConfig.tiny()
        idx = LightningIndexer(cfg)
        x = torch.randn(2, 16, cfg.hidden_size)
        keep = indexer_compressed_keep(idx, x, group=4, topk=2)
        comp = compressed_keep_matrix(16, 4, x.device)
        self.assertFalse((keep & ~comp.unsqueeze(0)).any())


class DocsRangeTests(unittest.TestCase):
    def test_frozen_spec_lists_cg_entries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        spec = (root / "docs" / "FROZEN_SPEC.md").read_text()
        self.assertIn("python3 -m cat_yoko.c", spec)
        self.assertNotIn("Phase C indexer 训练循环", spec)
        plan = (root / "docs" / "TRAINING_PLAN.md").read_text()
        self.assertNotIn("本仓库不实现 CSA / Phase C 训练循环", plan)
        status = (root / "docs" / "STATUS.md").read_text()
        self.assertNotIn("Phase C indexer", status)
        self.assertIn("CSA CUDA kernel", status)


if __name__ == "__main__":
    unittest.main()
