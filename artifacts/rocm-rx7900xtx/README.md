# RX 7900 XTX B0 continuation (2026-10-02)

Ubuntu 24.04, Radeon RX 7900 XTX with 24 GiB VRAM, 125 GiB CPU RAM,
PyTorch `2.9.1+rocm6.4` / HIP `6.4.43484`.
Remote workspace: `/home/donz/cat-yoko-rocm-20261002/`.

B0 resumed from Hub `checkpoints/b0-3090-bf16/trainable.pt`, source step
**33800** and **158,140,416 tokens in phase**. The input SHA256 was
`2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690`;
the MiniCPM5 base SHA256 was
`d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`.
Both checks passed and the original input files were preserved.

The native model was built/upcycled in CPU BF16, with per-block GPU
forward/backward and CPU Adam. Configuration: sequence 4096, micro-batch 1,
accumulation 1, and the 8e9-token phase budget. `--run-steps` bounds additional
updates without replacing the gate/LR schedule. Saves contain only the native
132-tensor B0 trainable overlay; no full-model checkpoint is written.

## Verified initial continuation

| Check | Result |
| --- | --- |
| Step | 33800 → 33802 |
| Tokens in phase | 158,140,416 → 158,148,608 |
| NLL | 11.8067 / 11.8076 |
| Throughput | 90 / 113 tok/s, including per-block offload |
| Peak allocated | 8526.8 MiB, approximately 8.33 GiB |
| Parameter updates | 54 of the 132 tensors changed stored BF16 values |
| Checkpoint | Remote `runs/b0-rocm-verify/trainable.pt` |

Evidence: [result](verify/result.json), [metrics](verify/metrics.jsonl), and
[complete two-step log](experiments/b0-rocm-verify-retry.log). Required
continuation, attention, MoE, and offload regressions passed; their logs are in
`experiments/`.

The background baseline started from step **33802** in `runs/b0-rocm/`,
requesting **1000 additional updates**, saving every 10 steps, and keeping the
last two numbered overlays. Its initial PID was `1018872`, its remote log is
`experiments/b0-rocm.log`, and its target is step **34802**.

The latest recorded baseline save is **step 33850**, with **158,345,216 tokens
in phase**. The baseline process was stopped after this completed save to release
host memory. Its checkpoint is also retained in `runs/b0-layout-source/`. The
compact full-model experiment initially waited while other GPU jobs ran; one
job occupied approximately 22 GiB. [observed_status.json](observed_status.json) and the bundled
[metrics](metrics.jsonl) record the baseline through step 33850.

## Operator evidence and rejected full-model compaction

The experimental gfx11 AOTriton flag enables Flash/Efficient SDPA, but native
causal-GQA backward failed the initial comparison:
[failure log](experiments/flash-validation.log). The baseline keeps the verified
FP32 math attention path with the experimental flag disabled.

The isolated [attention layout ledger](../../operators/rocm/results/rx7900xtx-20261002/attention_layout.jsonl)
finished with **72 rows: 45 passed, 18 numerical failures, 9 unsupported, and
no execution errors**. Efficient native GQA is unsupported. Repeated-KV
Efficient attention runs but its dQ relative-L2 errors are 31%–56%; making the
inputs or upstream gradients contiguous does not repair it.

At sequence 4096 in the actual cross-cache layout, FP32 MATH measured
**19.29 ms forward / 39.40 ms forward+backward**, with **4208 MiB incremental
peak**. Flash forward plus full FP32 backward recomputation measured
**5.78 / 54.19 ms** and **4224 MiB**. Query-chunked hybrid backward measured
**5.33 / 71.60 ms** and **684 MiB**. Flash forward passed with approximately
0.21% output relative-L2 error; full FP32 recomputation gave reference-identical
gradients. The hybrid uses a reference derivative and remains experimental.

The runner stopped/resumed this session's baseline around operator runs, but
the GPU is shared and complete other-PID history was not recorded during the
measurements. Exclusive compute is not established. The approximately 8%
hybrid difference in another layout and small projection differences cannot
support a training recommendation. Initial MoE and projection measurements,
their exact compute/transfer scope, and failed attempts are retained in
[the operator documentation and ledgers](../../operators/rocm/README.md).

Nine compact-model tiny tests passed. The full native-versus-compact 12B
sequence-4096 comparison measured **34.986 → 4.189 seconds (8.35×)** for
forward+backward, loss absolute error **5.72e-5**, and aggregate gradient
relative L2 **2.95%**. However, the worst per-tensor gradient relative L2 was
**15.69%** and **65 of 132 tensors exceeded the fixed 5% gate**. Parity failed;
the experiment performed **zero optimizer updates**, and compact continuation
was rejected. This timing does not establish a validated training speedup.

The separate entry points are
`operators/rocm/compact_model.py`, `model_bench.py`, `run_compact_b0.py`, and
`gpu_wait.py`. Compaction is restricted to frozen, byte-identical B0 experts,
is numerically approximate in BF16, and retains native 132 trainable overlay
keys. Native reconstruction is required for B1/B2. The launchers can wait using
read-only KFD PID checks without changing another user's job.

## Validated shared storage and resumed training

`operators/rocm/shared_storage_moe.py` retains native MoE dispatch, batched GEMMs,
gate multiplication, and reduction order, while aliasing identical frozen expert
weights and broadcasting packed weights with zero expert stride. All native
full-state and 132-tensor overlay keys remain present. The first ordinary-mode
comparison failed; native BF16 atomic `index_add_` itself varies on identical
inputs, so that comparison could not isolate layout error from reference
variation. Both the [failed report](../../operators/rocm/results/rx7900xtx-20261002/shared_storage_parity.json)
and [standalone native probe](../../operators/rocm/results/rx7900xtx-20261002/index_add_probe.json)
are retained. The latter ran alongside this session's continuation; its timings
are not standalone operator benchmarks.

The strict deterministic full-model comparison passed **sequences 64, 256,
and 4096**. Loss, sampled hidden states, selected logits, and **all 132 trainable
gradients were exactly equal**; the native 4096-token reference repeated exactly.
The fixed gates were unchanged. At sequence 4096:

| Measurement | Native block offload → resident shared storage |
| --- | --- |
| Forward + backward, without optimizer | 31.310048 → 6.721015 s, **4.66×** |
| Unique parameter storage | 24,500,762,624 → 4,418,435,072 bytes, 22.82 → 4.11 GiB |
| Actual training throughput, 20-step medians | 131.283 → 584.001 tokens/s, **4.45×** |
| Actual peak allocated GPU memory | 8526.83 → 12006.71 MiB |

Storage sharing allows the complete model to remain on GPU, so the full-model
speedup includes removing block transfers. It is not an isolated kernel speedup.
Throughput windows are baseline steps 33831–33850 and shared steps 33888–33907,
with sequence 4096, micro-batch/accumulation 1, and CPU FP32 Adam; the shared
window predates the concurrent native-reduction probe.

After parity, the runner restored the original deterministic setting (**false**)
and started **952 additional updates from step 33850 toward 34802**, saving every
10 and keeping two numbered overlays. The recorded checkpoint **33950** has
**158,754,816 tokens in phase**, all 132 expected keys, and **54 tensors whose
stored BF16 values changed** from step 33850. This is a running job, not a
completed target. Its PID is `1032727`, its remote output directory is
`runs/b0-shared-storage-deterministic/train/`, and its log is
`experiments/shared-storage-deterministic.log`.

Evidence: [full deterministic comparison](../../operators/rocm/results/rx7900xtx-20261002/shared_storage_deterministic.json),
[log snapshot](experiments/shared-storage-deterministic.log),
[training metrics](shared-storage/metrics.jsonl),
[checkpoint update proof](shared-storage/update_proof.json), and
[observed status](observed_status.json). The parity report's `updates: 0` and
`status: parity_pass` describe its last write before training; metrics and
checkpoint evidence record subsequent updates.

The recorded benchmark source hash predates the published capture cleanup.
That cleanup clears pending MoE statistics after parity: the measured run's
first utilization row included parity loads, while later rows use only training
loads. B0's frozen-router guard prevents bias updates, so this affects logging,
not loss or optimizer updates. Two CPU regressions verify policy restoration
and complete tiny-model cleanup on success and exceptions, with unchanged
parameters and router bias: [log](experiments/parity-isolation.log).

Use `model_bench.py --moe-layout shared-storage --deterministic-parity
--reference-repeat` to reproduce validation and bounded training. The separate
`run_compact_b0.py` still installs compute collapse, which has not passed
whole-model deterministic validation.

These remain DummyStream experiments, with no language-quality improvement
claim. Source overlays lack Adam state, so moments restart. The native offload
path clips per block; the resident shared-storage path uses global clipping. CPU
router initialization and numerical limitations are documented in
[ROCM_TRAIN.md](../../docs/ROCM_TRAIN.md). The Hub B0-full pointer is preserved.

## Upstream integration validation

PRs #13 and #14 were integrated with the ROCm changes. The remote CPU suite
ran 580 tests, skipping 24 GPU tests; all code tests passed. Its sole error
was the Git-index check because the copied remote source has no `.git`. All
five repository checks passed separately against the actual local integration
index. The new shared-storage layout passed six CPU/GPU regressions, including
whole-model checkpointing and overlay round trips. The 63 parameter and 14
architecture ledger checks passed. See [check summary](integration_checks.json),
[CPU log](experiments/integration-cpu.log), and
[shared-storage log](experiments/shared-storage-regression.log).

The shared-storage run used the integrated source, waited until other GPU
tasks exited, and passed all requested checks before training. The published
runner restores both deterministic-policy flags even on exceptions and clears
pending parity statistics. It preserves the native overlay format and the
8e9-token schedule. The recorded maximum gradient norm is below 1, so clipping
was inactive throughout the captured continuation. This does not validate
compute collapse or the rejected ROCm fused-attention backward paths.
