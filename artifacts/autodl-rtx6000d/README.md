# AutoDL RTX 6000D (Blackwell sm_120)

Instance `autodl-container-8x4c4zmh8d-e96e943a`, SSH was `connect.weste.seetacloud.com:34864` (**weste**, not the already-released westc). **This machine has been released**; do not connect again. The subsequent B200 has also been recycled. Current Hub nail and continuation of training: [`docs/STATUS.md`](../../docs/STATUS.md) and [`checkpoints/b0-full/README.md`](../../checkpoints/b0-full/README.md) (step **26940**).

Published B0 **has already switched to** `/root/autodl-tmp/venv-nightly`: torch `2.15.0.dev20260918+cu130`, driver 595.71.05, CUDA 13.2. miniconda still has 2.8.0+cu128.

| Item | Measured |
| --- | --- |
| GPU | NVIDIA RTX 6000D |
| compute | 12.0 (sm_120) |
| VRAM | 85651 MiB (≈83.6 GiB) |
| overlay `/` | 30G — do not put weights there |
| `/root/autodl-tmp` | 50G xfs |
| Hub | `HF_ENDPOINT=https://hf-mirror.com` `HF_HUB_DISABLE_XET=1` |

This is the target card for the NVFP4 recipe. After freeze, the trainer replaces allowed ``nn.Linear`` with ``Nvfp4Linear`` (E2M1/16 emulation; with TE it uses ``NVFP4BlockScaling``). Attention is still causal YOCO window + fp32 SDPA / qk_norm; topology is unchanged. fused TE WGRAD/RHT is still not a hard dependency of this repo.

## HuggingFace via hf-mirror

```bash
source scripts/autodl_env.sh   # HF_ENDPOINT=https://hf-mirror.com, cache /root/autodl-tmp/hf
python3 scripts/download_minicpm5.py
python3 -m cat_yoko.b0 --try --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base \
  --save-dir /root/autodl-tmp/runs/b0
```

`openbmb/MiniCPM5-2B-Base` `model.safetensors` is **5,033,557,128** bytes, sha256 `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a`.
hf-mirror `/resolve/` 302s to `cas-bridge.xethub.hf.co`; this machine returns **403** for that host (that is why the previous 4080 stuck at 20MiB). After `download_minicpm5.py` fails it falls back to ModelScope `OpenBMB/MiniCPM5-2B-Base` (same size, same sha). The tokenizer can still be pulled from hf-mirror git (`tokenizer.json` ≈9.8MiB, not Xet).

Do not cache the Hub on the overlay. Do not write SSH passwords or HuggingFace deploy keys into this directory. Published weights land on HuggingFace [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO), not GitHub.

## B2 `--try` (wait for a free GPU + B1 overlay)

Published envelope 15e9 tokens; 6000D first runs [`scripts/run_b2_try_autodl.sh`](../../scripts/run_b2_try_autodl.sh): `--try` 32 steps, resume `/root/autodl-tmp/runs/b1`, MiniCPM5 upcycle of encoder+embed, NVFP4 wrap of all allowed GEMMs, `--offload-blocks --optim-cpu --accum 1`. Do not `git fetch`, do not use 50B corpus, do not 23GiB `--save-full`. Pointers: [`b2/`](b2/), Hub [`checkpoints/b2/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b2).

## B0 `--try` (real MiniCPM5 upcycle)

After MiniCPM5-2B-Base sha256 `d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a` verifies, `python3 -m cat_yoko.b0 --try --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base` 32/32 steps exit 0.

| Item | Value |
| --- | --- |
| gate | 0.301 |
| last nll | 15.91 (DummyStream random tokens, not an evaluation) |
| peak | 24244 MiB |
| trainable | 219.21M / overlay 419MiB |
| overlay sha256 | `9012e5ac55c2f59ef7cacc34d5769444413d070116dbff0696c7b258b9aa0636` |

Logs: `runs/gpu_and_b0.log`, `runs/b0/metrics.jsonl`, `gpu_smoke/tiny.json`. Overlay goes to HuggingFace only, not GitHub.

## WESTE prep (2026-09-18 16:30Z)

hostname `autodl-container-8x4c4zmh8d-e96e943a`. GPU idle, 0 MiB / 85651 MiB, cap 12.0. torch `2.8.0+cu128`.

Remote `/root/autodl-tmp/CAT-YOKO` HEAD was `39b6e2dc740bf1693beffb58ab2c5321140bdb06` (`cursor/nvfp4-c1-theory-02c6`) with a dirty NVFP4 working tree (files copied on top; not the train branch checkout). MiniCPM5-2B-Base is at `/root/autodl-tmp/hf/MiniCPM5-2B-Base` (`model.safetensors` 4.7G). Existing `/root/autodl-tmp/runs/b0` still has the 32-step overlay.

Disk at prep end: overlay `/` 1.8G/30G used (29G free); `/root/autodl-tmp` 7.4G/50G used (43G free).

`HF_ENDPOINT` and `HF_HUB_DISABLE_XET=1` are in `/root/autodl-tmp/cat-yoko-env.sh`, `/etc/profile.d/cat-yoko-hf.sh`, and `/etc/environment` (non-interactive SSH picks them up).

Transformer Engine: `transformer-engine==2.19.0` + `transformer_engine_cu12==2.19.0` installed. `import transformer_engine` works and `NVFP4BlockScaling` exists, but `import transformer_engine.pytorch` fails (`libtorch_cuda.so: undefined symbol: ncclCommWindowRegister`). Isolated `transformer-engine[pytorch]` also failed: pip tried to download torch 2.14, then `--no-build-isolation` compile died on missing `nccl_dev_cap.hpp` (not in torch 2.8). GPU test path is E2M1/16 emulation. Logs: `te_install.log`, `te_error_extract.txt`, `prep_status.txt`.

After `cursor/nvfp4-train-6000d-02c6` is on origin, overlay the tree with tar/scp (GitHub `git fetch` hangs). Published B0 is [`scripts/run_b0_full_autodl.sh`](../../scripts/run_b0_full_autodl.sh): 8e9 tokens, seq=4096, `--no-offload-encoder`, resume `/root/autodl-tmp/runs/b0` if present. tmux `b0-full`. Overlay → Hub `checkpoints/b0-full/`; logs → [`b0-full/`](b0-full/). Do not overwrite the 32-step `checkpoints/b0/` on the Hub.

After published B0 finishes and the GPU is idle, run B1 `--try`: [`scripts/run_b1_try_autodl.sh`](../../scripts/run_b1_try_autodl.sh) (resume `runs/b0-full` else `b0`, MiniCPM5 upcycle, seq=64, 32 steps). Logs → [`b1/`](b1/). overlay → Hub `checkpoints/b1/`.

## PyTorch / TE nightly (NVFP4 GEMM)

torch **2.8.0+cu128** exposes `float4_e2m1fn_x2`, but `copy_` is `NotImplemented`, so cuBLAS NVFP4 GEMM does not go through. TE 2.19 `NVFP4BlockScaling` is present, but `transformer_engine.pytorch` is missing the `.so` (`ncclCommWindowRegister`).

Upgrade via a **new venv**; do not touch the running B0 miniconda process:

```bash
bash scripts/upgrade_torch_te_nightly_autodl.sh   # tmux nightly-upgrade
# probe: /root/autodl-tmp/venv-nightly/NVFP4_PROBE.json
# same-phase resume (first ln the latest trainable_step → trainable.pt):
PY=/root/autodl-tmp/venv-nightly/bin/python bash scripts/run_b0_full_autodl.sh
```

Nightly target: the day's torch 2.15.dev + `transformer_engine[pytorch,core-cu13]` from `https://download.pytorch.org/whl/nightly/cu130`. TE docs write **NVFP4 training kernels as SM100/103**; sm_120 is best-effort probing (`disable_rht` / `disable_2d_quantization`). If the probe fails, continue E2M1/16 emulation; do not change the C1 recipe.

**Switched (2026-09-18T18:06Z):** B0 same-phase resume from the step 1400 overlay; the process is nightly Python. `transformer_engine.pytorch` 2.19 can import; `float4_e2m1fn_x2` `copy_` still fails, so GEMM is still emulation. Probe JSON: [`nvfp4/NVFP4_PROBE_nightly.json`](nvfp4/NVFP4_PROBE_nightly.json). Throughput about 1350 → 1670 tok/s.

**Switched (2026-09-19T00:27Z):** same-phase resume from step **10560**, new operators in the in-memory graph. `grouped_mm=True`, `te=True`, `te_nvfp4=False` (TE NVFP4 Linear failed on sm_120 then disabled for the whole process), `return_logits=False`. Throughput about **1670 → 2820 tok/s**, mem **42092 → 56090 MiB**.

**Snapshot before release (2026-09-19T02:42Z):** Hub overlay step **16020**, `tokens_in_phase=65,488,896` (≈0.82%), sha256 `b5763b98…`. B0 did not finish 8e9. B200 has continued training from that overlay; the current Hub nail is [`checkpoints/b0-full/README.md`](../../checkpoints/b0-full/README.md) (step **26940**).

## NVFP4 wrap smoke test (2026-09-18)

tiny CUDA wrap + causal window passed; 12B B0 `--try` 2 steps `nvfp4=True`, wrap 2815 Linears, peak 34442 MiB. Logs in [`nvfp4/`](nvfp4/). **The 419MiB overlay goes to HuggingFace only** [`checkpoints/b0-nvfp4-try/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-nvfp4-try); it does not overwrite the 32-step `checkpoints/b0/trainable.pt`. Transformer Engine 2.19 cu12 installed, but `transformer_engine.pytorch` cannot import because of `ncclCommWindowRegister`; this run uses E2M1/16 emulation.
