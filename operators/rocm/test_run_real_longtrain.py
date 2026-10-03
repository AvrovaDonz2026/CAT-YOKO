"""Supervisor regression checks with tiny files and fake children, without torch."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from operators.rocm.run_real_longtrain import (
    CHECKPOINT_CHECK, build_parser, file_digest, run_child, supervise,
    training_command, validate_data_manifest, wait_for_data, write_json,
)


class LongtrainSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        outputs, hashes = {}, {}
        for split, rows, digest in (("train", 8, "a" * 64), ("eval", 1, "b" * 64)):
            path = self.data / f"{split}.bin"
            path.write_bytes(bytes(rows * 4096 * 4))
            outputs[split] = {"path": path.name, "sequences": rows, "tokens": rows * 4096,
                              "bytes": path.stat().st_size, "sha256": file_digest(path)}
            write_json(path.with_suffix(".bin.meta.json"), {**outputs[split], "seq_len": 4096,
                       "eos_id": 1, "vocab_size": 130560, "split": split})
            evidence = self.data / f"{split}.docs.sha256"
            evidence.write_text(digest + "\n")
            hashes[split] = {"path": evidence.name, "documents": 1, "sha256": file_digest(evidence)}
        self.manifest = {"status": "complete", "hash_intersection": 0, "seq_len": 4096,
                         "eos_id": 1, "vocab_size": 130560, "outputs": outputs, "hashes": hashes,
                         "target_tokens": {key: value["tokens"] for key, value in outputs.items()}}
        write_json(self.data / "manifest.json", self.manifest)
        write_json(self.data / "status.json", {"status": "complete"})
        (self.root / "base").mkdir()
        (self.root / "base/model.safetensors").write_bytes(b"base")
        (self.root / "source.pt").write_bytes(b"overlay")
        (self.root / "operators/rocm").mkdir(parents=True)
        (self.root / "operators/rocm/model_bench.py").write_text("# fake launcher\n")

    def args(self):
        return build_parser().parse_args([
            "--work-dir", str(self.root), "--source-dir", str(self.root), "--python", sys.executable,
            "--data-dir", str(self.data), "--base", str(self.root / "base"),
            "--resume", str(self.root / "source.pt"), "--out", str(self.root / "out"),
        ])

    def test_verified_files_and_disjoint_document_hashes(self):
        corpus = validate_data_manifest(self.data)
        self.assertEqual(corpus["splits"]["train"]["sequences"], 8)
        self.assertEqual(corpus["hash_intersection"], 0)
        (self.data / "train.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_data_manifest(self.data)

    def test_claimed_zero_intersection_is_recomputed(self):
        path = self.data / "eval.docs.sha256"
        path.write_text("a" * 64 + "\n")
        self.manifest["hashes"]["eval"]["sha256"] = file_digest(path)
        write_json(self.data / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_data_manifest(self.data)

    def test_wait_completes_and_failed_preparation_never_retries(self):
        write_json(self.data / "status.json", {"status": "running"})
        events = []
        def finish(_seconds):
            write_json(self.data / "status.json", {"status": "complete"})
        with patch("operators.rocm.run_real_longtrain.time.sleep", side_effect=finish):
            corpus = wait_for_data(self.data, events.append)
        self.assertEqual(len(events), 1)
        self.assertEqual(corpus["hash_intersection"], 0)
        write_json(self.data / "status.json", {"status": "failed", "error": "reader failed"})
        with patch("operators.rocm.run_real_longtrain.time.sleep", side_effect=AssertionError), \
                self.assertRaisesRegex(RuntimeError, "reader failed"):
            wait_for_data(self.data, events.append)

    def test_commands_preserve_exact_caps_and_real_layout(self):
        args = self.args()
        corpus = validate_data_manifest(self.data)
        smoke = training_command(args, corpus, stage="smoke", resume=args.resume, updates=5)
        long = training_command(args, corpus, stage="long", resume=self.root / "smoke.pt", updates=3)
        def option(command, flag):
            return command[command.index(flag) + 1]
        self.assertEqual(option(smoke, "--parity-seqs"), "64,4096")
        self.assertEqual(option(smoke, "--max-hours"), "0.25")
        self.assertEqual(option(long, "--run-steps"), "3")
        self.assertEqual(option(long, "--eval-every"), "250")
        self.assertEqual(option(long, "--eval-batches"), "32")
        self.assertEqual(option(long, "--max-hours"), "24")
        self.assertEqual(option(long, "--save-every-seconds"), "300")
        self.assertEqual(option(long, "--save-every"), "0")
        self.assertEqual(option(long, "--keep-last"), "3")
        self.assertEqual(option(smoke, "--save-every-seconds"), "0")
        for command in (smoke, long):
            self.assertEqual(option(command, "--moe-layout"), "shared-storage")
            self.assertIn("--save-optim", command)
            self.assertIn("--wait-gpu-idle", command)
        args.save_every_seconds = 0
        legacy = training_command(args, corpus, stage="long", resume=args.resume, updates=3)
        self.assertEqual(option(legacy, "--save-every"), "100")

    def test_real_stdlib_child_logs_pid_and_propagates_failure(self):
        events = []
        log = self.root / "child.log"
        run_child([sys.executable, "-c", "print('fake child')"], stage="fake", log=log,
                  work_dir=self.root, on_status=events.append)
        self.assertIn("fake child", log.read_text())
        self.assertGreater(events[0]["child_pid"], 0)
        self.assertIsNone(events[-1]["child_pid"])
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            run_child([sys.executable, "-c", "raise SystemExit(7)"], stage="fake", log=log,
                      work_dir=self.root, on_status=events.append)

    def test_fake_smoke_failure_blocks_long_child_and_writes_status(self):
        args = self.args()
        stages = []
        def child(command, *, stage, **kwargs):
            stages.append(stage)
            if stage == "smoke":
                raise RuntimeError("smoke failed")
        with patch("operators.rocm.run_real_longtrain.run_child", side_effect=child), \
             patch("operators.rocm.run_real_longtrain.verify_checkpoint", return_value={"step": 34802}), \
             redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "smoke failed"):
            supervise(args)
        self.assertEqual(stages, ["smoke"])
        self.assertEqual(json.loads((args.out / "status.json").read_text())["status"], "failed")

    def test_fake_pipeline_resumes_smoke_and_limits_to_one_pass(self):
        args = self.args()
        commands = []
        def child(command, *, stage, **kwargs):
            commands.append((stage, command))
        def verify(_args, path, *, mode, **kwargs):
            updates = {"source": 0, "smoke": 5, "long": 8}[mode]
            return {"step": 34802 + updates, "updates": updates}
        def report(path, **kwargs):
            return {"updates": 5 if "smoke" in str(path) else 3,
                    "completion": {"stop_reason": "steps"}}
        with patch("operators.rocm.run_real_longtrain.run_child", side_effect=child), \
             patch("operators.rocm.run_real_longtrain.verify_checkpoint", side_effect=verify), \
             patch("operators.rocm.run_real_longtrain.verified_training_report", side_effect=report), \
             redirect_stdout(io.StringIO()):
            summary = supervise(args)
        self.assertEqual([stage for stage, _ in commands], ["smoke", "long_training"])
        long = commands[-1][1]
        self.assertEqual(long[long.index("--run-steps") + 1], "3")
        self.assertEqual(Path(long[long.index("--resume") + 1]), args.out / "smoke/train/trainable.pt")
        self.assertEqual(summary["total_updates"], 8)
        self.assertEqual(json.loads((args.out / "status.json").read_text())["status"], "complete")

    def test_checkpoint_child_program_and_time_argument_validation(self):
        compile(CHECKPOINT_CHECK, "checkpoint-check", "exec")
        args = self.args()
        args.max_hours = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            supervise(args)
        for key, value in (("save_every_seconds", -1), ("save_every_seconds", float("nan")),
                           ("keep_last", 0)):
            args = self.args()
            setattr(args, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                supervise(args)


if __name__ == "__main__":
    unittest.main()
