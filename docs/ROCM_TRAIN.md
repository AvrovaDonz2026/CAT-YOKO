# ROCm B0 continuation

`scripts/run_b0_rocm.py` resumes BF16 B0 on a small-memory AMD GPU. It builds
the 12B model directly in CPU BF16, reconstructs frozen parameters from the
MiniCPM5 base, loads a B0 trainable overlay, and moves individual blocks to the
GPU for forward/backward. Adam state stays on CPU. AMD `gfx` capability values
must not be interpreted as NVIDIA SM versions.

The available BF16 Hub branch is:

| Field | Value |
| --- | --- |
| Path | `checkpoints/b0-3090-bf16/trainable.pt` |
| Source step | 33800 |
| Source tokens in phase | 158,140,416 |
| SHA256 | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| Origin | BF16 same-phase continuation from B0-full step 26940 |

Write new outputs to a separate directory. The B0-full published pointer
remains documented in [STATUS.md](STATUS.md).

```bash
YOKO_WORK=/path/to/large/workspace
HF_ENDPOINT=https://hf-mirror.com python scripts/download_minicpm5.py \
  --local-dir "$YOKO_WORK/hf/MiniCPM5-2B-Base" --check-hash
HF_ENDPOINT=https://hf-mirror.com python scripts/download_hub_overlay.py \
  --name b0-3090-bf16 --out-dir "$YOKO_WORK/hf/b0-3090-bf16"

python scripts/run_b0_rocm.py \
  --base "$YOKO_WORK/hf/MiniCPM5-2B-Base" \
  --resume "$YOKO_WORK/hf/b0-3090-bf16/trainable.pt" \
  --save-dir "$YOKO_WORK/runs/b0-rocm" \
  --run-steps 1000 --seq-len 4096 --save-every 10
```

Requires ROCm PyTorch, transformers, safetensors, huggingface_hub, and enough
host RAM for approximately 23 GiB of BF16 parameters plus loading/optimizer
overhead. Sequence 4096 is the starting configuration. If memory requires a
shorter sequence, record it as a different run configuration.

`--run-steps N` counts up to N additional updates after restoring the checkpoint
and retains the `--tokens 8e9` gate/LR schedule. `--steps` remains an absolute
end step. The absolute 32-step limit of `--try` is unsuitable for resuming step
33800.

Each save contains only B0's **132 trainable tensors**. Metrics go to
`metrics.jsonl`; a completed run writes `result.json`. `trainable.pt` points to
the latest completed overlay save. The original Hub input is preserved. Full
model checkpoints and automatic uploading are not part of this launcher.

The RX 7900 XTX baseline has saved **step 33850**, with **158,345,216 tokens in
phase**. The baseline process was stopped after this completed save to release
host memory; another GPU job had occupied approximately 22 GiB during setup.
This is a recorded state, not a guarantee of current process or GPU availability.
Verified baseline results are in [the run artifacts](../artifacts/rocm-rx7900xtx/README.md).

This still uses the original DummyStream and provides no language-quality
claim. The source overlay lacks Adam state, so moments restart. Block offload
uses the existing per-block gradient clipping. CPU initialization does not
guarantee reproduction of the previous CUDA frozen-router random values;
experts are copied from the same base, with routing differences reflected in
low-precision summation. The verified attention path uses FP32 math when fused
ROCm kernels are unsupported, and there is no hardware NVFP4 acceleration.

## Isolated operator and compact-model experiments

[operators/rocm](../operators/rocm/README.md) keeps candidate operators separate
from the baseline training code. The attention layout ledger contains 72 rows:
45 passed, 18 failed numerically, and 9 were unsupported. Native Efficient GQA
is unsupported; repeated-KV Efficient attention fails dQ checks. The
Flash-forward/FP32-backward hybrid passes the numerical gates but is slower in
the measured sequence-4096 cross-cache training operator. Its chunked backward
reduces memory. GPU PID coverage was incomplete during these shared-GPU timing
runs, so small timing differences do not support an implementation choice.

The compact experiment merges only frozen, byte-identical routed experts and
preserves the native 132-tensor B0 overlay. BF16 reduction order changes, so the
result is numerically approximate. Nine compact-model tiny tests passed, but
full 12B sequence-4096 parity failed. The measured forward+backward times were
**34.986 → 4.189 seconds (8.35×)**, with loss absolute error **5.72e-5** and
aggregate gradient relative L2 **2.95%**. The worst gradient tensor reached
**15.69%** and **65 of 132 tensors exceeded the fixed 5% limit**. No optimizer
updates were performed, and compact continuation was rejected. The timing is
not a validated continuation speedup.

A new `shared_storage_moe.py` candidate shares identical frozen weights while
keeping native dispatch, batched GEMM, and reduction order, using stride-zero
expert weight broadcasting. It remains unvalidated. Select it through
`model_bench.py --moe-layout shared-storage`; the default is `compact`. Both
layouts retain the 132-gradient/per-tensor/global 5% gates and must pass every
requested sequence before training (default 10 additional updates). Reports
record the selected layout, available Git revision, and source file hashes.

- `compact_model.py` implements the B0-only frozen-expert replacement. It changes frozen full-state keys; B1/B2 require rebuilding the native model.
- `shared_storage_moe.py` implements the separate B0 frozen-weight storage candidate; full-model parity and performance are pending.
- `model_bench.py` compares losses, selected outputs, and every trainable gradient against the native block-offload model, then permits bounded training only if all requested parity gates pass. `--parity-only` stops after checking.
- `run_compact_b0.py` performs separate bounded resident continuation after full-model parity has passed; the launcher assumes that validation was done.
- `gpu_wait.py` supplies read-only KFD PID observation through `--wait-gpu-idle` in both launchers. Two idle polls are required; `--gpu-idle-max-wait` sets an optional timeout. It neither stops other jobs nor reserves the GPU.

Resident compact training retains CPU Adam but uses global clipping instead of
the baseline's per-block clipping. Source Adam moments restart. Save only the
native trainable overlay to a separate directory; rebuild the native graph
before entering B1/B2. Reproduction commands and declared parity tolerances are
in [the operator documentation](../operators/rocm/README.md).
