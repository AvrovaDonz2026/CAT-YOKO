# CAT-YOKO Published Spec (Locked)

> This table is the input to the training code. The knobs are closed. If a knob must change, bump the version number; do not reopen the debate in the implementation.
> The middle-tier ledger, causality, C1, and C1+NVFP4 still follow the theory documents. This file pins **implementation defaults** only.

The machine-readable copy is `CATYokoConfig.middle_12b()` in `cat_yoko/config.py`.

---

## 0. One line

**CAT-YOKO-12B** is a causal YOCO MoE upcycled from MiniCPM5-2B. Phase B trains under **C1+NVFP4**. The published B0 run was measured on **NVIDIA B200 / SM 10.0** (frozen-encoder hardware NVFP4 FPROP); sm_120 (6000D) uses emulation. Phase B is sliding-window GQA + gated cross-attn, not a CSA CUDA kernel. M2 is off. KDA, mHC, MTP, Muon, and PDSA are off by default.

**Base license.** `MiniCPM-2B-sft-bf16` is OpenBMB GML and needs a commercial license. `MiniCPM5-2B` is **Apache-2.0**, so the published base is MiniCPM5, not MiniCPM-2B GML. Upcycle and teacher use `openbmb/MiniCPM5-2B-Base` (`LlamaForCausalLM` GQA). The tokenizer is `openbmb/MiniCPM5-2B`.

**Repository license.** CAT-YOKO code and derived weights are **Apache-2.0** ([`LICENSE`](../LICENSE)).

The training code in this repository is a **PyTorch reference implementation** of this 12B graph. When 12B does not fit, optional **DeepSpeed ZeRO** is available (`cat_yoko.deepspeed_zero`; CI does not hard-require it). Scale-out parallelism is reserved as a [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) adapter surface (`cat_yoko.megatron`, dual-stack TransformerConfig plus `model_provider` hooks). Do not implement YOCO as an off-the-shelf `GPTModel`. The tiny config is for unit tests only.

---

## 1. Architecture

| Item | Locked |
| --- | --- |
| Encoder | **Causal** self-decoder, not bidirectional |
| Split | MiniCPM5 **first 16 layers → Encoder, last 26 layers → Decoder** (42 total; the extra 2 layers go to the decoder). This is **16 encoder + 26 decoder**, not 20/22. |
| Hidden / vocab / heads | \(d=2048\), \(V=130560\), **16 Q / 2 KV** GQA, \(d_h=128\), \(d_{\mathrm{kv}}=256\), **untied** \(E\) / `lm_head` |
| μP | **No MiniCPM-2B μP** (MiniCPM5 is Llama identity scale): with `use_mup=False`, `embed_scale=1`, `residual_scale=1`, `logit_scale=1`; even if `dim_model_base` is filled by mistake, do not divide the logits. `rms_eps=1e-6`, `rope_theta=5e6` |
| MoE | DeepSeek fine-grained, `moe_intermediate_size=2048`, dense SwiGLU **6144** (\(6144/2048=3\) divides evenly, cleaner than MiniCPM-2B's 5760) |
| Enc experts | 1 shared + **20** routed, top-\(k=7\) |
| Dec experts | 1 shared + **20** routed, top-\(k=10\) |
| First-layer dense | **Off** (C1: token 0 is MoE on both stacks) |
| Routing | softmax-then-topK; score `sqrt(softplus(·))`; aux-loss-free bias + light seq-balance \(10^{-3}\) |
| Hash-MoE | Decoder **first 2 layers**. Encoder is only considered at **B2**, and still learns routing by default. |
| Residual | Ordinary residual. **mHC off** |
| MTP / KDA / PDSA | **Off** (KDA reference lives in `cat_yoko.kda`, `use_kda=False`; when on, it is a 3:1 layer swap and does not change the YOCO global cache) |

Bands change top-\(k\) only (the same 1+20 / **882** expert slots):

| Band | Enc top-\(k\) | Dec top-\(k\) |
| --- | ---: | ---: |
| low | 4 | 6 |
| **middle (published)** | **7** | **10** |
| near_dense | 12 | 16 |

Attention (implementation, **true GQA**, not the 1.25×MHA placeholder ledger):

| Item | Locked |
| --- | --- |
| Phase B | **Sliding-window causal GQA** (\(2d^2+2\cdot d\cdot d_{\mathrm{kv}}\approx 9.44\mathrm{M}\)/layer) + **gated cross-attn** (\(2d^2\), \(W_Q,W_O\) only). No top-k. No HCA compression. Phase B is sliding-window GQA + gated cross-attn, not a CSA CUDA kernel. |
| \(n_{\mathrm{win}}\) | **8192** (equals full causal on the 4K main train) |
| Encoder layer-type labels | **2 sliding + 7 CSA + 7 HCA** (still pinned on 16 layers; Phase B compute is still treated as sliding-window GQA, **not a CSA kernel**; CSA/HCA only light in Phase C). With `use_kda=True` this becomes **2 sliding + 11 KDA + 2 CSA + 1 HCA**; lighting order **KDA → indexer/CSA → HCA** (fewest HCA layers, write-first, placed last). |
| Decoder self-attn | **All sliding-window GQA**. Do not lay CSA. The extra 2 MiniCPM5 layers go to the decoder (26). With `use_kda`, 3:1 KDA:sliding (this is the source of decode-KV VRAM savings). |
| CSA \(m\) / HCA \(m'\) / `index_topk` | 4 / 128 / **256** (Phase C only) |
| M2 global cache pooling | **Off**. \(\hat K,\hat V = X^{16}W_K,X^{16}W_V\), \(W_K/W_V=\mathrm{Linear}(d,d_{\mathrm{kv}}=256)\), slot count = sequence length; **project once at the Encoder top** |
| M3 | After Phase C. Phase B decoder cross-attn is **dense causal** (position \(t\) reads cache \(0..t\)) |
| QK-Norm | **On** |
| Dual RMSNorm | **Off** (keep MiniCPM5 pre-norm) |

The published ledger counts true GQA, not a 1.25×MHA placeholder: storage ≈ **12.25B**; Encoder active ≈ **2.03B** / input token; Decoder active ≈ **4.33B** / output token; expert slots **882**. YOCO-cache \(W_K/W_V\) is not double-counted as per-layer \(4d^2\).

---

## 2. Phase B curriculum (C1+NVFP4)

| Stage | tokens | Trainable | detach | gate | dtype |
| --- | ---: | --- | --- | --- | --- |
| B0 | **8B** | cross-attn, \(W_K/W_V\), gate, new LN | yes | 0→0.3 | student **bf16**; frozen Encoder forward **nvfp4** |
| B1 | **27B** | entire Decoder + **untied `lm_head`** | yes | →1 | **nvfp4** (allowed linear GEMMs, including lm_head and attn QKV/O) |
| B2 | **15B** | full model (including input embed + `lm_head`) | no | 1 | **nvfp4** |

- MiniCPM5 is **untied**. B0/B1 **freeze Encoder + input embed**. B0 additionally freezes `lm_head` **and the final RMSNorm** (train only cross-attn / \(W_K/W_V\) / `ln_cross`). B1 trains `lm_head` and the final RMSNorm. B2 unlocks everything. Do not train the input table while the Encoder is frozen.
- The total envelope is **50B**. If quality is unstable, **lengthen B2**. Do not revert to joint bf16. Do not train two LMs.
- Wall-clock is **571 H100-h** (**43%** of joint-bf16 1,325; ≈ RTX PRO 6000-h). Joint bf16 1,325 is the control. C1 bf16 1,046 is the operand ledger. C1+FP8 729 is the Hopper/Ada fallback. **C1+NVFP4 = 571 H100-h**.
- The Phase C indexer is **bf16** and stacks after B2. L0 and unit tests are **bf16**; do not turn on NVFP4 there.
- **Must stay high precision:** input embed, RMSNorm / QK-Norm, router, gate, indexer, attn softmax / SDPA score. **`lm_head` and attn QKV/O projections are not required to be bf16; they run NVFP4.** In the implementation, RMSNorm, router logits/softmax, and attn softmax run fp32 under autocast.
- Published NVFP4 speedup is **2.0× vs bf16** (the older FP8 1.5×, times a further 1.33). B200 / SM 10.0 / 10.3 use ``TeNvfp4Linear`` (TE default ``NVFP4BlockScaling``: 2D + RHT + SR). Without Blackwell, or on sm_120, use ``Nvfp4Linear`` E2M1/16 emulation, then fall back to an FP8 placeholder / bf16 autocast. Attention stays a causal YOCO window; softmax / SDPA stay high precision.
- `use_muon=True` raises `NotImplementedError` from `build_optimizer` (published default is off).

---

## 3. Optimizer / hyperparameters

| Item | Locked |
| --- | --- |
| Optimizer | **AdamW** (\(\beta=(0.9,0.95)\), wd=0.1). **Muon off** (the switch exists; default false) |
| LR | WSD. Warmup **0.5B** tok. Stable **1e-4**. B2 **3e-5**. Decay is reserved for Phase E. |
| Grad clip | 1.0 |
| Sequence | Phase B **4096** |
| Global batch | **4M tok/step** (\(1024\times 4096\)). Micro-batch is split per GPU. |
| z-loss | router-z **1e-4** |
| KD | **On** if a MiniCPM5 teacher path is present (logit KL, T=2, weight 0.5→0); otherwise off |
| Document mask | **On** when packing |

---

## 4. Upcycling

- Weights: Enc ← MiniCPM5 0..15, Dec ← 16..41. **Untied** copies of `embed_tokens` and `lm_head` are separate.
- New modules (cross-attn, \(W_K/W_V\), router): small-scale random (`init_new_modules`, std=0.02; skip on the meta device).
- MoE: dense SwiGLU `6144` **divides** `2048` (exactly 3 groups). Each expert still copies the dense first 2048 rows and scales by \((E G^2 / T)^{1/3}\); cleaner than MiniCPM-2B's `5760`. Recovery is Phase B's job, not the surgical instant.
- Gate init **0** (Theorem A: bypass cross-attn).

---

## 5. Code scope (this repository)

Implement:

- The 12B YOCO MoE graph, the C1 freeze API, the C1+NVFP4 policy object, WSD, upcycling, and single-GPU / DDP / FSDP entry points.
- **C1 training loop.** Packing plus document mask (`labels_with_doc_boundaries` applies to the **last dimension**; `PackedBinStream`'s `[B,S]` must not misuse the batch dim). **DDP data sharding** (C1 re-wraps after `apply_freeze` at each stage; accumulation steps use `no_sync`). Gradient accumulation. Activation checkpointing `--grad-ckpt`. AdamW grouping (router has no decay; SwiGLU `gate_proj` still decays). Checkpoint / resume, including the packed cursor and RNG; skip if stream `kind` / DDP stride do not match. DummyStream offsets the seed by rank (the trainer no longer folds rank into the seed). Train/eval nll is weighted by `n_valid` (DDP sums `(nll·n_valid, n_valid)`; empty ranks contribute 0, not nan). jsonl logs (nll/ppl/aux/moe_cv/grad_norm/tok/s/mem/eval/n_valid/kd_w; non-finite floats are written as JSON null). `--eval-batches`. Optional MiniCPM5 logit KD (teacher is bf16 on CUDA; KL is a per-token mean; ignore `-100`). Frozen MoE does not enter aux and does not update router bias. DDP allreduces expert load before opt.step. Aux-loss-free bias updates once per optimizer step. When there is no window truncation and no cross-document span inside a row, attention uses causal SDPA and does not materialize an S×S mask. `init_process_group` passes `device_id` on NCCL. FSDP world=1 uses `NO_SHARD`. Tiny unit tests cover gate=0, detach with no Encoder gradients, frozen embed / B0 frozen `lm_head` and final RMSNorm, checkpoints, and 2-rank gloo DDP / C1 chain. **Cross-stage resume:** `--phase B1 --resume b0/latest.pt` takes over weights and the data cursor only; it does not restore B0's optimizer or step. The CLI phase is authoritative. `--resume` may be a file or a directory (a directory prefers `latest.pt`, else the newest `step_*.pt`). `--c1` / `--c1-smoke` run B0→B1→B2 on the same graph with a continuous packed cursor at `--save-dir/{B0,B1,B2}/latest.pt`. Checkpoints land on CPU first (non-FSDP also `.cpu()`-copies, then saves) so a 12B graph does not occupy the GPU twice. 12B defaults to weights-only (`--no-save-optim`). `--keep-last N` prunes `step_*.pt` (it does not delete a step file that shares an inode with `latest.pt`). **When `step_{last}.pt` already exists, `latest.pt` is a hardlink (or a same-volume copy). Do not `torch.save` a second 23GiB copy** — on an RTX 4080 SUPER, writing `step_1.pt` and then `latest.pt` under `/tmp` fills the overlay (`PytorchStreamWriter` / leftover `latest.pt.tmp`). Put 12B checkpoints on a large disk (for example `/root/autodl-tmp`), not `/tmp`. Clear `*.pt.tmp` before training. If volume space is short (payload+1GiB), fail immediately. B2 `--offload-blocks` only allows `--accum 1` (`--tokens` will not auto-expand to a 4M global batch either). DDP/FSDP and CPU offload are mutually exclusive.
- **B0 / B1 / B2 entry points:** `python3 -m cat_yoko.b0|b1|b2` (or `scripts/train_b{0,1,2}.py` / `cat-yoko-b0`). Without `--try`, inject the published token envelope (8e9 / 27e9 / 15e9) plus `--no-save-full --save-trainable --no-save-optim`; B0 `--offload-encoder`; B1 also `--optim-cpu`; B2 `--offload-blocks --optim-cpu --accum 1`. Cards `<40GiB` refuse the default envelope (32GB cannot finish 8B tokens) and require `--try` or an explicit `--steps` / `--tokens`. `--try`: 32 steps, seq=64, `--dummy-upcycle` when there is no `--upcycle*`, artifact `trainable.pt` (B0 ≈0.44GiB bf16). Weights do not live on GitHub (`.gitignore` drops `*.pt`); overlays, full graphs, and shards go to [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO). `--resume DIR` prefers `latest.pt`, then `trainable.pt`, then the newest `step_*.pt` / `trainable_step_*.pt`. B1 takes a B0 overlay = MiniCPM5 upcycle + `load_trainable_state`. B2 takes a B1 overlay = MiniCPM5 upcycle (encoder + input embed) + `load_trainable_state` (decoder + `lm_head` + final RMSNorm). `--offload-blocks` refuses `accum>1`.
- **B0 recipe for an unknown card:** `python3 -m cat_yoko.hw_recipe` (`--json` / `--argv` / `--shell`) picks a profile from SM family + VRAM. It does not attach a Megatron training loop. `scripts/run_b0_next.sh` pulls MiniCPM5 plus the Hub `b0-full` overlay, then hands flags to `cat_yoko.b0`. SM100/103 ≥160GiB → mb=2, no offload, no grad-ckpt, TE NVFP4. sm_120 ≥90GiB → mb=1, emulated NVFP4. Hopper ≥40GiB → published envelope + emulation (FP8 is a documentation fallback only). `<40GiB` → `--try`. CPU emits JSON only. The named scripts `run_b0_full_b200.sh` / `run_b0_full_autodl.sh` remain shortcuts for known SKUs.
- **C–G entry points:** `python3 -m cat_yoko.c|d|e|f|g` (`scripts/train_{c-g}.py` / `cat-yoko-c`). They reuse the same trainer and overlay. `--try` is still 32 steps, seq=64. **C** `--stage indexer|topk|hca|win` (default indexer) or `--chain`: freeze the backbone; a bf16 Lightning Indexer uses **intra-layer KL** to match dense sliding-window attention (hook Encoder CSA only; do not reuse on decoder queries). Then CSA is **sliding window ∪ compressed-block top-k** (own block excluded, Theorem B), and HCA is **sliding-window concat mean-pooled slots**. Neither is a CSA CUDA kernel. **KDA** (`cat_yoko.kda`) is off by default. **Wire the graph first, then enable it:** B `--use-kda` inserts the 3:1 graph and `KDAGates` into the graph (compute is still sliding window; KDA parameters stay frozen). C inherits from the overlay, then **`C-kda` → indexer → topk → hca → win**. Do not weld KDA onto a `use_kda=False` B overlay only at C. Do not shrink the YOCO global cache. This is not a DPLR CUDA kernel. Default `use_kda=False` still runs indexer→topk→hca→win. The published B0 overlay stays 132 tensors. **D** `--stage 8k|32k|128k` or `--chain`: 4K packed `.bin` is **rewindowed to the target seq** (concatenate rows; sidecar equality is not required). DummyStream plants a needle in the middle. D/E/F **pin `sparse=hca`** (even without KDA, do not fall back to window). `--use-kda` is inherited from resume extra. prepare `--mix phase-d` (web+code). **E** uses WSD 1-sqrt decay down to ~1/100 of peak. prepare `--mix phase-e` (HQ web + math + code + UltraChat as body text). **F** SFT seq=8192, prompt-span `labels=-100`. prepare `--mix phase-f` writes jsonl (UltraChat `data` / `messages` / alpaca, multi-span packed to seq). The Trainer consumes `tokens`+`labels` or `prompt_ids`/`response_ids`. **G** `--algo grpo|dpo` (step envelope, default 10000; `--try` uses dummy RLVR). C–G do not download Ultra-FineWeb / UltraChat by default.
- **OpenBMB data path.** `cat_yoko.prepare` packs the thinking mix into seq_len-aligned int32 mmap `.bin`: Ultra-FineWeb en/zh + UltraData-Math + **5% StarCoder** (default 0.55/0.30/0.10/0.05). Optional `phase-b-code` is 10% code. Do not download corpora in CI or on this VM. DummyStream uses the same `THINK_CODE_FRAC` to mix in-repo short snippets (utf-8 hashed to pad rows). This is not a 50B download. The tokenizer default is `openbmb/MiniCPM5-2B` (\(V=130560\)). The Trainer uses `PackedBinStream` on `.bin` and reads `eos_id` from `*.bin.meta.json`. `--upcycle-hf` / `--teacher-hf` pull `openbmb/MiniCPM5-2B-Base`.
- **Base loading.** Accept **MiniCPM5-2B** only (`openbmb/MiniCPM5-2B-Base` / `openbmb/MiniCPM5-2B`). MiniCPM-2B / MiniCPM3 / MiniCPM4 Hub ids are rejected in the CLI and in `load_minicpm_state`. HF load uses `LlamaForCausalLM`, **no** `trust_remote_code`, **`device_map="cpu"`** (call `empty_cache` immediately after copying the state_dict) so MiniCPM5 does not occupy 4–5GiB VRAM on top of the 12B graph. Upcycling does exact shape checks on embed / GQA K·V / dense SwiGLU. MiniCPM-2B MHA (\(d=2304\)) or `5760` FFN will not be truncated into the 12B graph. With `use_mup=False`, embed/residual/logit scales are identity 1; even a mistaken MiniCPM-2B `dim_model_base` does not divide the logits.
- **CUDA smoke.** Entry points `setdefault PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (B1 ≈ 28.4/32GiB; encoder offload fragments the caching allocator). On `<40GiB` cards, `--config 12b` requires `--seq-len` (default 4096 OOMs) and refuses `--teacher-hf` (student 23–28GiB + MiniCPM5 ~5GiB). 12B save checks the cgroup before the CPU copy (~23GiB host tensors; do not stack this with B1's resident CPU Adam). `python3 -m cat_yoko.gpu_smoke` runs tiny (including encoder offload / CPU Adam / B0→B1→B2 chain / KD / dummy upcycle / eval / cross-stage resume / 2-rank gloo DDP + **CUDA DDP** / **CUDA C1 DDP** / **1-rank FSDP**). `--middle` runs C1 steps on the **MiniCPM5 GQA 12.25B graph** on ≥28GiB GPUs (default 1; `--steps N` takes effect; build bf16 directly on CUDA; do not build CPU fp32 then cast). `--middle --phase B1` freezes the Encoder, offloads it to CPU, and uses CPU Adam. `--middle --save-dir DIR` writes `step_N.pt` and hardlinks `latest.pt`. `--c1` runs B0→B1→B2 on the same 12B graph (default 1 step each). B2 offloads layer by layer; after a layer's backward, clip+Adam the grads away. Skip if there is no GPU. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO. FSDP wrap passes `device_id` and `use_orig_params=True`. ckpt extra includes `cfg`/`seed`. Measured on RTX 4080 SUPER 32GB + 62GiB cgroup for the **MiniCPM5-2B GQA 12.25B graph** (params=12,250,381,312, `--seq-len 64`; peak counted after freeze/offload, excluding the `build_model` high-water mark): standalone `--middle` B0 one-step peak **22.83GiB** (23378MiB), B1 **28.6GiB** (29283MiB, Encoder offload + ephemeral CPU Adam + `expandable_segments:True`; previously 28.43/29115 on the same path), B2 **2.6GiB** (2662MiB, layer-wise offload working set); `--c1` same graph B0 **22.83GiB**, B1 **28.34GiB** (29019MiB), B2 **2.6GiB**. CPU `step_1.pt` resume to step 2 succeeded (nll=11.9425, mem=23378MiB). `/root/autodl-tmp/b0ckpt` weights-only **hardlink**: `latest.pt` and `step_1.pt` share an inode, nlink=2, 24,501,855,029 bytes, **no** second 23GiB copy; `--resume` from that directory for step 2 still peaked at 22.83GiB. The earlier second `torch.save` under `/tmp` that filled the overlay (`latest.pt.tmp` 7G leftover) is fixed. The old MiniCPM-2B placeholder-graph figures 21.65 / 26.3 / 13.38 / 22.73 GiB are **void**. Earlier B2 numbers 22.82 / 14.68 were leftover from graph build or the B1 decoder, not the B2 working set. Each 12B unit-test case runs in a **separate subprocess** (`python -m cat_yoko.gpu_smoke`). Do not stack two 12.25B copies in one process. tiny CUDA overall ok (peak_mib 17.5; includes 2-rank CUDA DDP / C1 DDP / 1-rank FSDP). GPU idle 1 MiB after the run. `expandable_segments` recap: after import, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `--config 12b --device cuda` without `--seq-len`, and `--teacher-hf`, both argparse-exit 2 (12B graph not built); standalone B0 nll=11.9544 peak 22.83GiB; standalone B1 nll=11.9345 peak **28.6GiB**, idle 1 MiB.
- Megatron-LM adapter surface: `ParallelPlan` (TP/PP/EP/CP), dual-stack `TransformerConfig` mapping, `model_provider` / `forward_step` hooks. Do not vendor Megatron.

Do not implement (this step):

- YOCO as `GPTModel` or a bidirectional T5 encoder. Do not implement a Megatron EP/TP loop (fill the provider after [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is installed).
- A CSA CUDA kernel / V4-style FP4 expert storage / a 50B download / checking 50B of corpus into git / an eval suite. The data interface consumes mmap `.bin` from prepare, or jsonl `tokens` / SFT `labels` / `prompt_ids`+`response_ids`. Phase C uses `cat_yoko.indexer.LightningIndexer` plus intra-layer KL, and Theorem B's sliding-window∪compressed SDPA mask / HCA concat; it is **not** a CSA kernel.
- Allocating 12B weights on local CPU (≈49GB fp32 / 24.5GB bf16), or downloading Ultra-FineWeb / MiniCPM5 weights in CI or on a small VM. `--config 12b --meta` builds a meta graph only. 12B training requires `--device cuda --dtype bf16` and an explicit `--steps` or `--tokens`. Single 32GB card: B0 one step can run directly; B1 uses frozen Encoder offload + CPU Adam (automatically switch to fp16 momentum if the host cgroup cannot hold fp32 momentum); B2 uses layer-wise offload, and one-step smoke uses ephemeral AdamW if the host cannot hold momentum. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO.

Megatron constraints (already in `cat_yoko.parallel`):

- TP must divide 16 heads and \(d=2048\): `{1,2,4,8,16}`.
- Routed experts 20; EP must divide 20: `{1,2,4,5,10,20}`.
- Encoder CSA ratios `{0,4,128}` match Megatron Core `csa_compress_ratios` legal values; light in Phase C.

---

## 6. Closed questions

| Was open | Locked |
| --- | --- |
| Encoder bidirectional? | **No. Causal YOCO** |
| 16/24 vs 20/20 | **16/26** (MiniCPM5 42 layers; the extra 2 layers go to the decoder). Not 20/22. |
| Train two stacks independently then weld | **Forbidden** |
| M2 default | **Off** |
| KDA / mHC / MTP / Muon | **Off** (KDA code is opt-in `use_kda`; default `use_kda=False` is not counted in the published graph; published B0 overlay is 132 tensors) |
| First-layer dense | **Off** (C1 full MoE) |
| Phase B CSA-sparse? | **No** (sliding-window GQA + gated cross-attn, not a CSA kernel) |
| Attention ledger | **True GQA 16/2**, not a 1.25×MHA placeholder |
| Unfreeze curriculum | **C1** (B0 8B / B1 27B / B2 15B) |
| Base | **MiniCPM5-2B Apache-2.0** (not MiniCPM-2B-sft-bf16 GML) |
| Wall-clock | **C1+NVFP4 = 571 H100-h** (C1+FP8 729 is the Hopper/Ada fallback) |
