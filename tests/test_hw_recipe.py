#!/usr/bin/env python3
"""Unknown-GPU B0 recipe: SM + VRAM → argv. No live GPU required."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_yoko.hw_recipe import (
    COMFORTABLE_GIB,
    HUB_OVERLAY,
    PUBLISHED_SEQ,
    TIGHT_12B_GPU_GIB,
    WIDE_MB2_GIB,
    B0Recipe,
    arch_kind,
    detect_snapshot,
    main as hw_main,
    parse_cap,
    recipe_for,
    recipe_payload,
    snapshot,
)
from cat_yoko.nvfp4_hw import compute_family
from cat_yoko.phase_train import build_phase_argv
from cat_yoko.phases import TIGHT_GPU_SEQ, TRY_STEPS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_b0_next.sh"


def _snap(
    cap: tuple[int, int] | None,
    gib: float,
    *,
    te_fprop: bool | None = None,
    env_te: str | None = None,
    name: str | None = None,
) -> HwSnapshot:
    return snapshot(
        cap=cap,
        gpu_gib=gib,
        family=compute_family(cap),
        name=name,
        te_nvfp4_fprop=te_fprop,
        env_te=env_te,
    )


class ParseTests(unittest.TestCase):
    def test_parse_cap_and_arch(self) -> None:
        self.assertEqual(parse_cap("10.0"), (10, 0))
        self.assertEqual(parse_cap("10.3"), (10, 3))
        self.assertEqual(parse_cap((9, 0)), (9, 0))
        self.assertEqual(arch_kind((10, 0)), "sm100")
        self.assertEqual(arch_kind((10, 3)), "sm103")
        self.assertEqual(arch_kind((12, 0)), "sm120")
        self.assertEqual(arch_kind((9, 0)), "hopper")
        self.assertEqual(arch_kind((8, 9)), "ada")
        self.assertEqual(arch_kind((8, 0)), "ampere")
        self.assertEqual(arch_kind(None), "cpu")
        self.assertLess(TIGHT_12B_GPU_GIB, COMFORTABLE_GIB)
        self.assertLess(COMFORTABLE_GIB, WIDE_MB2_GIB)


class RecipeTableTests(unittest.TestCase):
    def test_sm100_b200_wide(self) -> None:
        rec = recipe_for(_snap((10, 0), 183.0, te_fprop=True, name="NVIDIA B200"))
        self.assertEqual(rec.profile, "sm100_b200")
        self.assertTrue(rec.published)
        self.assertTrue(rec.launch)
        self.assertFalse(rec.try_run)
        self.assertEqual(rec.seq_len, PUBLISHED_SEQ)
        self.assertEqual(rec.micro_batch, 2)
        self.assertFalse(rec.offload_encoder)
        self.assertFalse(rec.grad_ckpt)
        self.assertEqual(rec.nvfp4_path, "te")
        self.assertIsNone(rec.te_env)
        self.assertFalse(rec.megatron)
        self.assertFalse(rec.save_full)
        self.assertEqual(rec.resume_hub, HUB_OVERLAY)
        self.assertEqual(
            rec.argv,
            (
                "--seq-len",
                "4096",
                "--micro-batch",
                "2",
                "--no-offload-encoder",
                "--no-grad-ckpt",
            ),
        )
        self.assertNotIn("--try", rec.argv)
        self.assertNotIn("--save-full", rec.argv)

    def test_sm103_wide(self) -> None:
        rec = recipe_for(_snap((10, 3), 192.0))
        self.assertEqual(rec.profile, "sm103_b200")
        self.assertEqual(rec.micro_batch, 2)
        self.assertEqual(rec.nvfp4_path, "te")

    def test_sm100_comfortable_mb1(self) -> None:
        rec = recipe_for(_snap((10, 0), 96.0))
        self.assertEqual(rec.profile, "sm100")
        self.assertTrue(rec.published)
        self.assertEqual(rec.micro_batch, 1)
        self.assertFalse(rec.offload_encoder)
        self.assertFalse(rec.grad_ckpt)
        self.assertEqual(rec.nvfp4_path, "te")

    def test_sm120_6000d(self) -> None:
        rec = recipe_for(_snap((12, 0), 96.0))
        self.assertEqual(rec.profile, "sm120_6000d")
        self.assertTrue(rec.published)
        self.assertEqual(rec.seq_len, PUBLISHED_SEQ)
        self.assertEqual(rec.micro_batch, 1)
        self.assertFalse(rec.offload_encoder)
        self.assertTrue(rec.grad_ckpt)
        self.assertEqual(rec.nvfp4_path, "emu")
        self.assertIn("--grad-ckpt", rec.argv)
        self.assertNotIn("--try", rec.argv)

    def test_hopper_80gib_offload(self) -> None:
        rec = recipe_for(_snap((9, 0), 80.0))
        self.assertEqual(rec.profile, "hopper_h100")
        self.assertTrue(rec.published)
        self.assertEqual(rec.micro_batch, 1)
        self.assertTrue(rec.offload_encoder)
        self.assertTrue(rec.grad_ckpt)
        self.assertEqual(rec.nvfp4_path, "emu")
        self.assertIn("FP8", rec.notes)
        self.assertIn("Megatron", rec.notes)

    def test_h200_comfortable_no_offload(self) -> None:
        rec = recipe_for(_snap((9, 0), 141.0))
        self.assertEqual(rec.profile, "hopper_h100")
        self.assertTrue(rec.published)
        self.assertFalse(rec.offload_encoder)
        self.assertEqual(rec.micro_batch, 1)
        self.assertNotEqual(rec.micro_batch, 2)

    def test_ampere_a100(self) -> None:
        rec = recipe_for(_snap((8, 0), 80.0))
        self.assertEqual(rec.profile, "ampere_a100")
        self.assertTrue(rec.published)
        self.assertTrue(rec.offload_encoder)
        self.assertEqual(rec.nvfp4_path, "emu")

    def test_ada_tight_try(self) -> None:
        rec = recipe_for(_snap((8, 9), 32.0))
        self.assertEqual(rec.profile, "ada_tight")
        self.assertFalse(rec.published)
        self.assertTrue(rec.try_run)
        self.assertTrue(rec.launch)
        self.assertEqual(rec.seq_len, TIGHT_GPU_SEQ)
        self.assertIn("--try", rec.argv)
        self.assertEqual(rec.refuse_reason, "tight_vram")

    def test_cpu_no_launch(self) -> None:
        rec = recipe_for(_snap(None, 0.0))
        self.assertEqual(rec.profile, "cpu")
        self.assertFalse(rec.launch)
        self.assertFalse(rec.published)
        self.assertEqual(rec.nvfp4_path, "off")
        self.assertEqual(rec.refuse_reason, "cpu")
        self.assertFalse(rec.megatron)

    def test_te_env_off_forces_emu_on_sm100(self) -> None:
        rec = recipe_for(_snap((10, 0), 183.0, env_te="0"))
        self.assertEqual(rec.nvfp4_path, "emu")
        self.assertEqual(rec.te_env, "0")
        self.assertTrue(rec.published)
        self.assertEqual(rec.profile, "sm100_b200")

    def test_te_fprop_miss_emulates(self) -> None:
        rec = recipe_for(_snap((10, 0), 183.0, te_fprop=False))
        self.assertEqual(rec.nvfp4_path, "emu")
        self.assertEqual(rec.te_env, "0")

    def test_force_try_on_wide_card(self) -> None:
        rec = recipe_for(_snap((10, 0), 183.0), force_try=True)
        self.assertTrue(rec.try_run)
        self.assertFalse(rec.published)
        self.assertEqual(rec.seq_len, TIGHT_GPU_SEQ)
        self.assertIn("--try", rec.argv)
        self.assertEqual(rec.refuse_reason, "force_try")


class ArgvComposeTests(unittest.TestCase):
    def test_phase_train_keeps_no_save_full(self) -> None:
        rec = recipe_for(_snap((10, 0), 183.0))
        built = build_phase_argv(
            "B0",
            list(rec.argv) + ["--device", "cpu", "--dummy-upcycle", "--save-dir", "/tmp/b0-hw"],
        )
        self.assertIn("--no-save-full", built)
        self.assertNotIn("--save-full", built)
        self.assertEqual(built[built.index("--seq-len") + 1], "4096")
        self.assertEqual(built[built.index("--micro-batch") + 1], "2")
        self.assertIn("--no-offload-encoder", built)
        self.assertIn("--no-grad-ckpt", built)
        self.assertIn("--tokens", built)
        self.assertEqual(built[built.index("--tokens") + 1], "8000000000.0")

    def test_try_argv_seq_64(self) -> None:
        rec = recipe_for(_snap((8, 9), 32.0))
        built = build_phase_argv(
            "B0",
            list(rec.argv) + ["--device", "cpu", "--dummy-upcycle"],
        )
        self.assertIn("--try", rec.argv)
        self.assertEqual(built[built.index("--seq-len") + 1], str(TIGHT_GPU_SEQ))
        self.assertEqual(built[built.index("--steps") + 1], str(TRY_STEPS))
        self.assertNotIn("--save-full", built)


class CliTests(unittest.TestCase):
    def test_json_cli(self) -> None:
        buf = __import__("io").StringIO()
        with patch("sys.stdout", buf):
            code = hw_main(["--family", "sm100", "--gib", "183", "--te-fprop", "1"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["recipe"]["profile"], "sm100_b200")
        self.assertEqual(payload["recipe"]["micro_batch"], 2)
        self.assertEqual(payload["snapshot"]["family"], "sm100")
        self.assertFalse(payload["recipe"]["megatron"])
        self.assertFalse(payload["recipe"]["save_full"])

    def test_argv_cli(self) -> None:
        buf = __import__("io").StringIO()
        with patch("sys.stdout", buf):
            code = hw_main(["--argv", "--cap", "12.0", "--gib", "96"])
        self.assertEqual(code, 0)
        text = buf.getvalue().strip()
        self.assertIn("--seq-len 4096", text)
        self.assertIn("--micro-batch 1", text)
        self.assertIn("--no-offload-encoder", text)
        self.assertIn("--grad-ckpt", text)
        self.assertNotIn("--try", text)

    def test_shell_cli(self) -> None:
        buf = __import__("io").StringIO()
        with patch("sys.stdout", buf):
            code = hw_main(["--shell", "--cap", "9.0", "--gib", "80"])
        self.assertEqual(code, 0)
        text = buf.getvalue()
        self.assertIn("CAT_YOKO_HW_PROFILE=hopper_h100", text)
        self.assertIn("CAT_YOKO_HW_LAUNCH=1", text)
        self.assertIn("CAT_YOKO_HW_TRY=0", text)
        self.assertIn("CAT_YOKO_HW_NVFP4=emu", text)

    def test_cpu_cli_launch_false(self) -> None:
        buf = __import__("io").StringIO()
        with patch("sys.stdout", buf):
            code = hw_main(["--family", "cpu", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["recipe"]["launch"])

    def test_detect_snapshot_cpu_without_cuda(self) -> None:
        with patch("cat_yoko.hw_recipe.compute_capability", return_value=None):
            with patch("cat_yoko.hw_recipe.compute_family", return_value="cpu"):
                snap = detect_snapshot()
        self.assertEqual(snap.family, "cpu")
        self.assertEqual(snap.gpu_gib, 0.0)


class LauncherDocTests(unittest.TestCase):
    def test_run_b0_next_script(self) -> None:
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue(SCRIPT.stat().st_mode & stat.S_IXUSR)
        text = SCRIPT.read_text()
        self.assertIn("cat_yoko.hw_recipe", text)
        self.assertIn("cat_yoko.b0", text)
        self.assertIn("download_hub_overlay.py", text)
        self.assertIn("download_minicpm5.py", text)
        self.assertIn("b0-full", text)
        self.assertIn("Does not download Ultra-FineWeb", text)
        self.assertIn("Not Megatron", text)
        self.assertIn("Do not pass --save-full", text)
        self.assertNotRegex(text, r"-m cat_yoko\.b0[\s\S]*--save-full")
        self.assertNotIn("git fetch", text)
        self.assertNotIn("westc.seetacloud", text)
        self.assertNotIn("137.175.", text)
        self.assertNotRegex(text, r"31jEePeb|vDw8xU9c|PRIVATE KEY|dfghjkl")
        self.assertIn("never Hub checkpoints/b0", text)
        self.assertIn("--name \"${CAT_YOKO_HW_RESUME_HUB:-b0-full}\"", text)
        self.assertNotIn("download_hub_overlay.py --name b0 ", text)

    def test_status_points_at_dispatcher(self) -> None:
        status = (ROOT / "docs" / "STATUS.md").read_text()
        self.assertIn("run_b0_next.sh", status)
        self.assertIn("hw_recipe", status)
        self.assertIn("26940", status)
        root = (ROOT / "README.md").read_text()
        self.assertIn("run_b0_next.sh", root)
        self.assertIn("hw_recipe", root)
        b200 = (ROOT / "docs" / "B200_TRAIN.md").read_text()
        self.assertIn("run_b0_next.sh", b200)
        self.assertIn("hw_recipe", b200)

    def test_module_entrypoint(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "cat_yoko.hw_recipe",
                "--family",
                "sm120",
                "--gib",
                "96",
                "--argv",
            ],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self.assertIn("--no-offload-encoder", proc.stdout)
        self.assertNotIn("Ultra-FineWeb", proc.stdout)


class PayloadContractTests(unittest.TestCase):
    def test_payload_jsonable(self) -> None:
        snap = _snap((10, 0), 183.0)
        rec = recipe_for(snap)
        payload = recipe_payload(snap, rec)
        dumped = json.dumps(payload)
        self.assertIn("sm100_b200", dumped)
        self.assertIsInstance(rec, B0Recipe)
        self.assertEqual(payload["recipe"]["save_every"], 20)


if __name__ == "__main__":
    unittest.main()
