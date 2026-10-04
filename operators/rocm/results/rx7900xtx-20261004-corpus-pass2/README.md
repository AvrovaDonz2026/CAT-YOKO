# B0 second-pass recovery receipts

These receipts record the completed first pilot pass and the new bounded
continuation from **54333** to **73864**. The current supervisor uses
`runs/pass2-54333-20261004T1045` and `source-pass2-20261004T1045` on the remote
RX 7900 XTX. It preserves the complete source checkpoint, Adam/RNG/absolute
cursor and original B0 clocks; it repeats the same corpus in the same order.

| Directory | Evidence |
| --- | --- |
| `first-pass-completion/` | Normal completion at 54333, full final CPU state audit, final fixed held-out NLL and all 1190 updates of the preceding operator continuation |
| `failed-native-attempt/` | Unchanged dense-native gate failure on two decoder-0 cross norm gradients, with no training updates; its original launch receipts are retained separately |
| `launch/` | Successful detached supervisor command, exact source inventory, source/window preflight, RNG checks, 32 remote audit tests, 11 remote continuation/supervisor tests, and the read-only checkpoint audit script |
| `live-snapshot/` | Running-state observations, full 132-gradient GPU comparison with zero error against the packed predecessor, consecutive real updates, and first periodic checkpoint/native recovery audits |

At **2026-10-04T11:11:22.228406+00:00**, the worker had completed **67** updates
through **54400**. The first periodic recovery point is **54388** / cursor and
Adam **19586**, with 132 BF16 weights and 264 FP32 moments. Native Adam
load/export is byte-exact, and CPU Python/Torch RNG restoration passes. The
audit checks serialized GPU RNG tensors without GPU allocation. Live files
were collected sequentially while updates continued, so their last-step
observations may differ. They do not certify second-pass completion.

The successful continuation reference uses the previous packed FP32 production
attention with native MoE offload. The candidate retains shared frozen MoE
storage, split attention and cached CPU Adam. The reference contexts exit
before candidate installation. All original 132-gradient, output and loss
thresholds remain. Passing this comparison does not turn the retained
dense-native failure into a pass.

All eight newly deployed scripts/tests match their recorded launch hashes.
The fixed 54333 checkpoint SHA256 is
`4966dcddbeb322b3f4e1b75e86cce66d3c5c95402346b853d4dee449a8fd796f`.
No checkpoint binaries or authentication credentials are stored here. The
published Hugging Face snapshot remains the separate immutable step 53307.
See [the continuation report](../../CORPUS_PASS2_20261004.md) for scope and
recovery policy.
