#!/usr/bin/env python3
"""DeepSpeed ZeRO mapping: JSON/CLI/source contracts. Does not install DeepSpeed."""

from __future__ import annotations

import inspect
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.deepspeed_zero import (
    AMPERE_48GIB_ARGV,
    DEEPSPEED,
    DeepSpeedNotInstalled,
    dump_zero_config,
    import_deepspeed,
    is_deepspeed_engine,
    is_zero_partitioned,
    resolve_zero_stage,
    zero_config,
)
from cat_yoko.offload import auto_offload_flags
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.train import main
from cat_yoko.trainer import Trainer


FORBIDDEN = (
    "137.175.",
    "westc.seetacloud",
    "weste.seetacloud",
    "westd.seetacloud",
)
ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_default_stage_two_no_offload(self) -> None:
        cfg = zero_config()
        self.assertEqual(cfg["zero_optimization"]["stage"], 2)
        self.assertNotIn("offload_optimizer", cfg["zero_optimization"])
        self.assertNotIn("offload_param", cfg["zero_optimization"])
        self.assertTrue(cfg["bf16"]["enabled"])
        self.assertFalse(cfg["fp16"]["enabled"])
        self.assertFalse(cfg["zero_force_ds_cpu_optimizer"])
        self.assertFalse(cfg["activation_checkpointing"]["partition_activations"])

    def test_offload_param_forces_stage_three(self) -> None:
        self.assertEqual(resolve_zero_stage(1, offload_param=True), 3)
        cfg = zero_config(stage=1, offload_param=True)
        z = cfg["zero_optimization"]
        self.assertEqual(z["stage"], 3)
        self.assertEqual(z["offload_param"]["device"], "cpu")
        self.assertEqual(z["offload_optimizer"]["device"], "cpu")
        self.assertTrue(z["stage3_gather_16bit_weights_on_model_save"])

    def test_json_roundtrip(self) -> None:
        json.dumps(zero_config(stage=3, offload_optimizer=True, offload_param=True))

    def test_ampere_argv(self) -> None:
        self.assertIn("--backend", AMPERE_48GIB_ARGV)
        self.assertIn("deepspeed", AMPERE_48GIB_ARGV)
        self.assertIn("--zero-offload-param", AMPERE_48GIB_ARGV)
        self.assertNotIn("--save-full", AMPERE_48GIB_ARGV)


class DumpCliTests(unittest.TestCase):
    def test_dump_cli_zero(self) -> None:
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            code = main(
                [
                    "--config",
                    "12b",
                    "--dump-deepspeed",
                    "--zero",
                    "3",
                    "--zero-offload",
                    "--zero-offload-param",
                    "--phase",
                    "B0",
                ]
            )
        finally:
            sys.stdout = old
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["upstream"], DEEPSPEED)
        self.assertEqual(payload["backend"], "deepspeed")
        self.assertEqual(payload["zero"]["zero_optimization"]["stage"], 3)
        self.assertIn("ampere_48gib_argv", payload)
        self.assertTrue(any("CSA" in n or "csa" in n.lower() for n in payload["notes"]))

    def test_dump_does_not_import_deepspeed(self) -> None:
        dump_zero_config(stage=2, file=io.StringIO())
        self.assertNotIn("deepspeed", sys.modules)

    def test_backend_deepspeed_cpu_refuses(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--config",
                        "tiny",
                        "--backend",
                        "deepspeed",
                        "--phase",
                        "B0",
                        "--steps",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
        finally:
            sys.stderr = old
        self.assertIn("cuda", buf.getvalue())

    def test_zero_without_backend_errors(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                main(["--config", "tiny", "--zero", "2", "--steps", "1"])
        finally:
            sys.stderr = old
        self.assertIn("backend deepspeed", buf.getvalue())

    def test_fsdp_plus_deepspeed_errors(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--config",
                        "tiny",
                        "--backend",
                        "deepspeed",
                        "--fsdp",
                        "--dump-deepspeed",
                    ]
                )
        finally:
            sys.stderr = old
        self.assertIn("DDP/FSDP", buf.getvalue())

    def test_c1_plus_deepspeed_errors(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--config",
                        "tiny",
                        "--c1",
                        "--backend",
                        "deepspeed",
                        "--steps",
                        "1",
                    ]
                )
        finally:
            sys.stderr = old
        self.assertIn("resume", buf.getvalue())

    def test_megatron_plus_zero_errors(self) -> None:
        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit):
                main(["--config", "tiny", "--backend", "megatron", "--zero", "2"])
        finally:
            sys.stderr = old
        self.assertIn("deepspeed", buf.getvalue())

    def test_import_deepspeed_raises_without_install(self) -> None:
        try:
            import deepspeed  # noqa: F401
        except ImportError:
            with self.assertRaises(DeepSpeedNotInstalled) as ctx:
                import_deepspeed()
            self.assertIn("github.com/microsoft/DeepSpeed", str(ctx.exception))
        else:
            self.skipTest("deepspeed is installed in this env")


class OffloadMutexTests(unittest.TestCase):
    def test_deepspeed_disables_native_offload(self) -> None:
        e, b, o = auto_offload_flags(
            phase="B1",
            cfg_name="CAT-YOKO-12B",
            device="cuda",
            fsdp=False,
            ddp=False,
            deepspeed=True,
            offload_encoder=None,
            offload_blocks=None,
            optim_cpu=None,
        )
        self.assertFalse(e or b or o)

    def test_deepspeed_rejects_explicit_encoder_offload(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-offload-param"):
            auto_offload_flags(
                phase="B1",
                cfg_name="CAT-YOKO-12B",
                device="cuda",
                fsdp=False,
                ddp=False,
                deepspeed=True,
                offload_encoder=True,
                offload_blocks=None,
                optim_cpu=None,
            )

    def test_deepspeed_rejects_explicit_optim_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero-offload"):
            auto_offload_flags(
                phase="B1",
                cfg_name="CAT-YOKO-12B",
                device="cuda",
                fsdp=False,
                ddp=False,
                deepspeed=True,
                offload_encoder=None,
                offload_blocks=None,
                optim_cpu=True,
            )


class PhaseArgvTests(unittest.TestCase):
    def test_b0_deepspeed_drops_native_offload(self) -> None:
        argv = build_phase_argv(
            "B0",
            [
                "--try",
                "--save-dir",
                "/tmp/b0-ds",
                "--backend",
                "deepspeed",
                "--zero",
                "3",
                "--zero-offload",
                "--zero-offload-param",
            ],
        )
        self.assertIn("--backend", argv)
        self.assertEqual(argv[argv.index("--backend") + 1], "deepspeed")
        self.assertEqual(argv[argv.index("--zero") + 1], "3")
        self.assertIn("--zero-offload-param", argv)
        self.assertIn("--no-offload-encoder", argv)
        self.assertNotIn("--offload-encoder", argv)
        self.assertIn("--no-save-full", argv)
        self.assertIn("--save-trainable", argv)


class SourceContractTests(unittest.TestCase):
    def test_freeze_before_wrap_deepspeed(self) -> None:
        src = inspect.getsource(Trainer.run)
        wrap_src = inspect.getsource(Trainer._wrap_and_optim)
        self.assertLess(src.index("apply_freeze"), src.index("_wrap_and_optim"))
        self.assertIn("wrap_deepspeed", wrap_src)
        self.assertIn("wrap_distributed", wrap_src)
        self.assertIn("gradient_accumulation_steps=1", wrap_src)

    def test_zero3_gather_before_rank0_save(self) -> None:
        src = inspect.getsource(Trainer._maybe_save)
        self.assertLess(
            src.index("gathered_trainable_state_dict"), src.index("is_rank0")
        )
        self.assertIn("stage3", inspect.getsource(zero_config))

    def test_not_a_megatron_loop_or_csa_kernel(self) -> None:
        text = (ROOT / "cat_yoko" / "deepspeed_zero.py").read_text(encoding="utf-8")
        self.assertIn("Not Megatron EP/TP", text)
        self.assertIn("Not a CSA kernel", text)
        self.assertNotIn("class CSA", text)

    def test_plain_module_is_not_engine(self) -> None:
        import torch

        m = torch.nn.Linear(4, 4)
        self.assertFalse(is_deepspeed_engine(m))
        self.assertFalse(is_zero_partitioned(m))


class SecretScanTests(unittest.TestCase):
    def test_new_files_have_no_hosts(self) -> None:
        paths = [
            ROOT / "cat_yoko" / "deepspeed_zero.py",
            ROOT / "docs" / "DEEPSPEED_ZERO.md",
            ROOT / "cat_yoko" / "train.py",
            ROOT / "cat_yoko" / "trainer.py",
            ROOT / "cat_yoko" / "phase_train.py",
        ]
        for path in paths:
            self.assertTrue(path.is_file(), msg=str(path))
            text = path.read_text(encoding="utf-8")
            for tok in FORBIDDEN:
                self.assertNotIn(tok, text, msg=f"{path} contains {tok}")


if __name__ == "__main__":
    unittest.main()
