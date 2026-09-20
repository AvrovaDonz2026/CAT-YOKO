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

from cat_yoko.config import CATYokoConfig
from cat_yoko.deepspeed_zero import (
    AMPERE_48GIB_ARGV,
    DEEPSPEED,
    DeepSpeedNotInstalled,
    dump_zero_config,
    import_deepspeed,
    is_deepspeed_engine,
    is_zero_partitioned,
    resolve_zero_stage,
    seed_single_process_rank_env,
    wrap_deepspeed,
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
        self.assertGreaterEqual(z["stage3_prefetch_bucket_size"], 400_000_000)
        self.assertGreaterEqual(z["reduce_bucket_size"], 400_000_000)
        # 2048×2048 = 4_194_304 must stay below the threshold (5e6 OOM'd).
        self.assertGreaterEqual(z["stage3_param_persistence_threshold"], 1_000_000)
        self.assertLess(z["stage3_param_persistence_threshold"], 4_194_304)
        self.assertGreaterEqual(z["stage3_max_live_parameters"], 2_000_000_000)
        self.assertGreaterEqual(z["stage3_max_reuse_distance"], 2_000_000_000)
        self.assertEqual(z["offload_param"]["buffer_count"], 8)
        self.assertEqual(z["offload_optimizer"]["buffer_count"], 8)
        self.assertTrue(z["round_robin_gradients"])
        self.assertEqual(
            z["leaf_module"]["classes"],
            ["MoE", "EncoderBlock", "DecoderBlock"],
        )

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

    def test_phase_argv_forwards_no_nvfp4(self) -> None:
        argv = build_phase_argv(
            "B0",
            ["--try", "--no-nvfp4", "--save-dir", "/tmp/b0-ds", "--device", "cpu"],
        )
        self.assertIn("--no-nvfp4", argv)
        self.assertIn("--no-save-full", argv)
        self.assertTrue(CATYokoConfig.middle_12b().use_nvfp4)


class SourceContractTests(unittest.TestCase):
    def test_phase_argv_forwards_more_steps(self) -> None:
        argv = build_phase_argv(
            "B0",
            ["--more-steps", "8", "--save-dir", "/tmp/b0-ds", "--device", "cpu"],
        )
        self.assertIn("--more-steps", argv)
        self.assertEqual(argv[argv.index("--more-steps") + 1], "8")
        self.assertIn("--tokens", argv)

    def test_freeze_before_wrap_deepspeed(self) -> None:
        src = inspect.getsource(Trainer.run)
        wrap_src = inspect.getsource(Trainer._wrap_and_optim)
        self.assertLess(src.index("apply_freeze"), src.index("_wrap_and_optim"))
        self.assertIn("wrap_deepspeed", wrap_src)
        self.assertIn("wrap_distributed", wrap_src)
        self.assertIn("gradient_accumulation_steps=1", wrap_src)
        self.assertIn("seed_single_process_rank_env", inspect.getsource(wrap_deepspeed))

    def test_single_process_seeds_local_rank(self) -> None:
        import os

        src = inspect.getsource(seed_single_process_rank_env)
        self.assertIn('setdefault("LOCAL_RANK"', src)
        old = {k: os.environ.get(k) for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")}
        for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(k, None)
        try:
            seed_single_process_rank_env()
            self.assertEqual(os.environ["LOCAL_RANK"], "0")
            self.assertEqual(os.environ["WORLD_SIZE"], "1")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_zero3_gather_before_rank0_save(self) -> None:
        src = inspect.getsource(Trainer._maybe_save)
        self.assertLess(
            src.index("gathered_trainable_state_dict"), src.index("is_rank0")
        )
        self.assertIn("stage3", inspect.getsource(zero_config))

    def test_overlay_gather_skips_frozen_12b(self) -> None:
        import torch

        from cat_yoko.deepspeed_zero import (
            gathered_state_dict,
            gathered_trainable_state_dict,
        )

        overlay = inspect.getsource(gathered_trainable_state_dict)
        full = inspect.getsource(gathered_state_dict)
        self.assertIn("GatheredParameters", overlay)
        self.assertIn("requires_grad", overlay)
        self.assertNotIn("_zero3_consolidated_16bit_state_dict", overlay)
        self.assertIn("_zero3_consolidated_16bit_state_dict", full)
        self.assertIn("exclude_frozen_parameters=False", full)
        self.assertIsNone(gathered_trainable_state_dict(torch.nn.Linear(4, 4)))

    def test_not_a_megatron_loop_or_csa_kernel(self) -> None:
        text = (ROOT / "cat_yoko" / "deepspeed_zero.py").read_text(encoding="utf-8")
        self.assertIn("Not Megatron EP/TP", text)
        self.assertIn("Not a CSA kernel", text)
        self.assertNotIn("class CSA", text)

    def test_occupancy_hides_host_syncs(self) -> None:
        import inspect

        from cat_yoko.moe import _kick_max_count
        from cat_yoko.optim import _deepspeed_cpu_adam, build_optimizer
        from cat_yoko.trainer import Trainer, quiet_inductor

        run_src = inspect.getsource(Trainer.run)
        extra_src = inspect.getsource(Trainer._extra)
        self.assertIn("will_save", run_src)
        self.assertIn("_prefetch_batch", run_src)
        self.assertIn("include_rng", extra_src)
        self.assertIn("CUDA_DEVICE_MAX_CONNECTIONS", (ROOT / "cat_yoko" / "trainer.py").read_text(encoding="utf-8"))
        self.assertIn("TORCH_COMPILE_DISABLE", inspect.getsource(quiet_inductor))
        self.assertIn("copy_stream", inspect.getsource(_kick_max_count))
        wrap_src = inspect.getsource(Trainer._wrap_and_optim)
        self.assertIn("cpu_adam_fast", wrap_src)
        self.assertIn("ninja", wrap_src)
        self.assertIn("stage3_max_live_parameters", inspect.getsource(zero_config))
        self.assertIn("leaf_module", inspect.getsource(zero_config))
        z3 = zero_config(stage=3, offload_param=True)["zero_optimization"]
        self.assertEqual(
            z3["leaf_module"]["classes"],
            ["MoE", "EncoderBlock", "DecoderBlock"],
        )
        from cat_yoko.deepspeed_zero import (
            ZERO3_MAX_ONGOING_FETCH_EVENTS,
            freeze_host_gc_after_zero_init,
            gathered_trainable_state_dict,
            tune_zero3_prefetch_overlap,
            wrap_deepspeed,
        )

        self.assertGreaterEqual(ZERO3_MAX_ONGOING_FETCH_EVENTS, 8)
        self.assertEqual(tune_zero3_prefetch_overlap(object()), 0)

        class _Coord:
            _PartitionedParameterCoordinator__max_ongoing_fetch_events = 2

        coord = _Coord()

        class _Off:
            param_coordinator = coord

        class _Eng:
            optimizer = type("O", (), {"parameter_offload": _Off()})()

        self.assertEqual(tune_zero3_prefetch_overlap(_Eng()), 8)
        self.assertEqual(coord._PartitionedParameterCoordinator__max_ongoing_fetch_events, 8)
        freeze_host_gc_after_zero_init()
        self.assertIn("freeze_host_gc_after_zero_init", inspect.getsource(Trainer.run))
        self.assertIn("tune_zero3_prefetch_overlap", inspect.getsource(wrap_deepspeed))

        overlay = inspect.getsource(gathered_trainable_state_dict)
        self.assertIn("GatheredParameters", overlay)
        self.assertNotIn("_zero3_consolidated_16bit_state_dict", overlay)
        self.assertIn("DeepSpeedCPUAdam", inspect.getsource(_deepspeed_cpu_adam))
        self.assertIn("_warmup_zero", run_src)
        self.assertIn("warmup_zero3", inspect.getsource(Trainer._warmup_zero))
        warm = inspect.getsource(Trainer._warmup_zero)
        self.assertNotIn("model.step(", warm)
        self.assertNotIn("engine.step", warm)
        self.assertIn("set_rng_state", warm)
        self.assertNotIn("stream.batch", inspect.getsource(Trainer._synthetic_lm_batch))
        peak = inspect.getsource(Trainer._begin_step_peak)
        self.assertIn("if not self.deepspeed", peak)

    def test_plain_module_is_not_engine(self) -> None:
        import torch

        from cat_yoko.deepspeed_zero import warmup_zero3

        m = torch.nn.Linear(4, 4)
        self.assertFalse(is_deepspeed_engine(m))
        self.assertFalse(is_zero_partitioned(m))
        self.assertFalse(warmup_zero3(m, torch.zeros(())))


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
