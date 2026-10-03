#!/usr/bin/env python3
"""Run the unchanged optimizer numerical gate with ordered GPU arithmetic."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operators.rocm import gpu_adam_bench
from operators.rocm.gpu_adam_ordered import use_ordered_gpu_fp32_adam


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    sources = ("operators/rocm/gpu_adam_ordered_bench.py", "operators/rocm/gpu_adam_ordered.py",
               "operators/rocm/gpu_adam.py", "operators/rocm/gpu_adam_bench.py", "cat_yoko/optim.py")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    write = gpu_adam_bench.write_report

    def write_ordered(path, report):
        report.update(arithmetic_variant="ordered-fp32", ordered_source_sha256=hashes,
                      optimizer_source_sha256=hashes,
                      ordered_numerical_gate="unchanged BF16 byte equality and FP32 moment atol=1e-8/rtol=1e-6")
        return write(path, report)

    with patch.object(gpu_adam_bench, "use_gpu_fp32_adam", use_ordered_gpu_fp32_adam), \
            patch.object(gpu_adam_bench, "write_report", write_ordered):
        return gpu_adam_bench.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
