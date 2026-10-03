"""Reject unverified optimizer installation and preserve legacy CLI values."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch

from operators.rocm import next_candidate_bench as entry
from operators.rocm.gpu_adam_bench import main as adam_benchmark


class NextCandidateTests(unittest.TestCase):
    def test_native_micro_report_binds_the_implementation_and_reference_sources(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "native.json"
            self.assertEqual(adam_benchmark(["--device", "cpu", "--json", str(output),
                                            "--steps", "5", "--warmup", "0", "--repeats", "1"]), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["arithmetic_variant"], "native-order")
            expected = {"operators/rocm/gpu_adam.py", "operators/rocm/gpu_adam_bench.py", "cat_yoko/optim.py"}
            self.assertEqual(set(report["optimizer_source_sha256"]), expected)
            source_root = Path(entry.__file__).resolve().parents[2]
            self.assertEqual(report["optimizer_source_sha256"],
                             {path: entry.digest(source_root / path) for path in expected})

    def test_bucket_parity_search_uses_only_its_independent_stream_and_is_bounded(self):
        class Stream:
            def __init__(self, batches):
                self.batches, self.cursor = batches, 0

            def batch(self, micro_batch, device):
                value = self.batches[min(self.cursor, len(self.batches) - 1)]
                self.cursor += 1
                return value

        single = {"doc_ids": torch.zeros(1, 64, dtype=torch.long)}
        packed = {"doc_ids": torch.arange(4).repeat_interleave(16).unsqueeze(0)}
        info, stream = {}, Stream([single, packed])
        result = entry.BucketParityStream(stream, info).batch(1, "cpu")
        self.assertIs(result, packed)
        self.assertEqual(stream.cursor, 2)
        self.assertEqual(info["bucketed_parity_scanned_batches"], 2)
        bad = Stream([single])
        with self.assertRaises(RuntimeError):
            entry.BucketParityStream(bad, {}, max_batches=2).batch(1, "cpu")
        self.assertEqual(bad.cursor, 2)

    def test_extra_options_do_not_reach_native_parser(self):
        argv = ["--resume", "/tmp/a.pt", "--gpu-fp32-adam", "--bucketed-attention",
                "--packed-attention", "--gpu-adam-gate", "/tmp/gate.json",
                "--sync-update-timing", "--data", "/tmp/train.bin"]
        self.assertEqual(entry.legacy_argv(argv), ["--resume", "/tmp/a.pt", "--packed-attention",
                                                "--data", "/tmp/train.bin"])
        self.assertEqual(entry.legacy_argv(["--gpu-adam-gate=/tmp/gate.json", "--save-optim"]),
                         ["--save-optim"])

    def test_gpu_optimizer_requires_same_source_and_strict_independent_gate(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / "source.pt"
            source.write_bytes(b"source-checkpoint-for-gate-unit-test")
            gate_path = root / "gate.json"
            args = argparse.Namespace(packed_attention=True, moe_layout="shared-storage",
                bucketed_attention=False, reference_repeat=True, gpu_fp32_adam=True,
                save_optim=True, gpu_adam_gate=gate_path, sync_update_timing=True,
                parity_only=False, resume=source)
            sample = {"pass": True, "parameters": 132, "all_weights_bitwise": True,
                      "gate": {"fp32_moment_atol": 1e-8, "fp32_moment_rtol": 1e-6}}
            gate = {"status": "complete", "device": "cuda",
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "arithmetic_variant": "native-order",
                    "optimizer_source_sha256": {name: entry.digest(Path(entry.__file__).resolve().parents[2] / name)
                                                for name in entry.NATIVE_ADAM_SOURCES},
                    "checkpoint_moments_cpu_fp32": True, "persistent_fp32_master": False,
                    "parity": [copy.deepcopy(sample) for _ in range(5)], "terminal": copy.deepcopy(sample)}
            gate_path.write_text(json.dumps(gate))
            entry.validate_extra_args(argparse.ArgumentParser(), args)
            for key, value in (("source_sha256", "wrong-source"), ("device", "cpu"),
                               ("checkpoint_moments_cpu_fp32", False), ("persistent_fp32_master", True)):
                bad = {**gate, key: value}
                gate_path.write_text(json.dumps(bad))
                with self.assertRaises(SystemExit):
                    entry.validate_extra_args(argparse.ArgumentParser(), args)
            bad = {**gate, "parity": [dict(sample, all_weights_bitwise=False)] + gate["parity"][1:]}
            gate_path.write_text(json.dumps(bad))
            with self.assertRaises(SystemExit):
                entry.validate_extra_args(argparse.ArgumentParser(), args)

            # An ordered or stale implementation's otherwise-passing report
            # must never authorize the native GPU implementation installed by
            # this entry. Check each dependency, including the CPU reference.
            source_mutations = [
                lambda report: report.update(arithmetic_variant="ordered-fp32"),
                lambda report: report.pop("arithmetic_variant"),
                lambda report: report.pop("optimizer_source_sha256"),
                lambda report: report.update(optimizer_source_sha256=None),
                lambda report: report["optimizer_source_sha256"].pop("cat_yoko/optim.py"),
                lambda report: report["optimizer_source_sha256"].update({"operators/rocm/gpu_adam_ordered.py": "unused"}),
            ]
            source_mutations.extend(
                lambda report, name=name: report["optimizer_source_sha256"].update({name: "0" * 64})
                for name in entry.NATIVE_ADAM_SOURCES)
            for mutate in source_mutations:
                with self.subTest(mutation=mutate):
                    bad = copy.deepcopy(gate)
                    mutate(bad)
                    gate_path.write_text(json.dumps(bad))
                    with self.assertRaises(SystemExit):
                        entry.validate_extra_args(argparse.ArgumentParser(), args)

            # Passing intermediate steps cannot hide a failed final update:
            # terminal must carry the same complete gate, not just pass=True.
            for terminal in ({"pass": True}, {},
                             {**sample, "pass": False}, dict(sample, parameters=131),
                             dict(sample, all_weights_bitwise=False),
                             {**sample, "gate": {"fp32_moment_atol": 1e-7, "fp32_moment_rtol": 1e-6}},
                             {**sample, "gate": {"fp32_moment_atol": 1e-8, "fp32_moment_rtol": 1e-5}}):
                with self.subTest(terminal=terminal):
                    bad = {**gate, "terminal": terminal}
                    gate_path.write_text(json.dumps(bad))
                    with self.assertRaises(SystemExit):
                        entry.validate_extra_args(argparse.ArgumentParser(), args)


if __name__ == "__main__":
    unittest.main()
