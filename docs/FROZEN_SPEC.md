# CAT-YOKO published spec (frozen)

> This table is the input to the training code. Knobs are no longer open; change a version number instead of re-arguing them in the implementation.
> Middle-tier ledger / causality / C1 / C1+NVFP4 still follow the theory docs. This page only nails **implementation defaults**.

Machine-readable copy: `CATYokoConfig.middle_12b()` in `cat_yoko/config.py`.

---

## 0. One sentence

**CAT-YOKO-12B**: causal YOCO MoE upcycled from MiniCPM5-2B; Phase B trains under **C1+NVFP4**. Published B0 was measured on **NVIDIA B200 / SM 10.0** (frozen encoder hardware NVFP4 FPROP); sm_120 (6000D) uses emulation. Phase B attention is sliding-window GQA + gated cross-attn only; M2 off; KDA / mHC / MTP / Muon / PDSA off by default.

**Base license**: `MiniCPM-2B-sft-bf16` is OpenBMB GML / needs a commercial grant; `MiniCPM5-2B` is **Apache-2.0**, so the published base is MiniCPM5. Upcycle / teacher use `openbmb/MiniCPM5-2B-Base` (`LlamaForCausalLM` GQA); tokenizer is `openbmb/MiniCPM5-2B`.

**Repo license**: CAT-YOKO code and derived weights are **Apache-2.0** ([`LICENSE`](../LICENSE)).

Training code in this repo is the **PyTorch reference** of this 12B graph. When 12B does not fit, optional **DeepSpeed ZeRO** (`cat_yoko.deepspeed_zero`; CI does not require DeepSpeed). Scale-out parallelism is reserved as a [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) adapter (`cat_yoko.megatron`, dual-stack TransformerConfig + model_provider hooks). **Do not** stuff YOCO into a stock `GPTModel`. The tiny config is for unit tests only.

---

## 1. Architecture

| Item | Frozen |
| --- | --- |
| Encoder | **causal** self-decoder, not bidirectional |
| Split | MiniCPM5 **layers 0–15 → Encoder, 16–41 → Decoder** (42 total; the extra 2 layers go to the decoder) |
| Hidden / vocab / heads | \(d=2048\), \(V=130560\), **16 Q / 2 KV** GQA, \(d_h=128\), \(d_{\mathrm{kv}}=256\), **untied** \(E\) / `lm_head` |
| μP | **no MiniCPM-2B μP** (MiniCPM5 is Llama identity scale): with `use_mup=False`, `embed_scale=1`, `residual_scale=1`, `logit_scale=1`; even a mistaken `dim_model_base` does not divide logits. `rms_eps=1e-6`, `rope_theta=5e6` |
| MoE | DeepSeek fine-grained, `moe_intermediate_size=2048`, dense SwiGLU **6144** (\(6144/2048=3\) divides cleanly, cleaner than MiniCPM-2B's 5760) |
| Enc experts | 1 shared + **20** routed, top-\(k=7\) |
| Dec experts | 1 shared + **20** routed, top-\(k=10\) |
| First-layer dense | **off** (C1: both stacks are MoE from token 0) |
| Routing | softmax-then-topK; scores `sqrt(softplus(·))`; aux-loss-free bias + light seq-balance \(10^{-3}\) |
| Hash-MoE | Decoder **first 2 layers**; Encoder only considered at **B2**, still learned routing by default |
| Residual | ordinary residual. **mHC off** |
| MTP / KDA / PDSA | **off** (KDA reference lives in `cat_yoko.kda`, `use_kda=False`; lighting it is a 3:1 layer swap and does not change the YOCO global cache) |

Tiers only change top-\(k\) (same 1+20 / **882** expert slots):

| Tier | Enc top-\(k\) | Dec top-\(k\) |
| --- | ---: | ---: |
| low | 4 | 6 |
| **middle (published)** | **7** | **10** |
| near_dense | 12 | 16 |

Attention (implementation, **true GQA**, not a 1.25×MHA placeholder ledger):

| Item | Frozen |
| --- | --- |
| Phase B | **sliding-window causal GQA** (\(2d^2+2\cdot d\cdot d_{\mathrm{kv}}\approx 9.44\mathrm{M}\)/layer) + **gated cross-attn** (\(2d^2\), only \(W_Q,W_O\)). No top-k, no HCA compression. |
| \(n_{\mathrm{win}}\) | **8192** (equals full causal on 4K main training) |
| Encoder layer-type labels | **2 sliding + 7 CSA + 7 HCA** (still nailed to 16 layers; Phase B compute is still sliding GQA, **not a CSA kernel**; CSA/HCA light only in Phase C). With `use_kda=True` this becomes **2 sliding + 11 KDA + 2 CSA + 1 HCA**; lighting order **KDA → indexer/CSA → HCA** (fewest HCA layers, write-first, last) |
| Decoder self-attn | **all sliding GQA**. No CSA. The extra 2 MiniCPM5 layers go to the decoder (26). With `use_kda`, 3:1 KDA:sliding (the decode-KV memory save) |
| CSA \(m\) / HCA \(m'\) / `index_topk` | 4 / 128 / **256** (Phase C only) |
| M2 global-cache pooling | **off**. \(\hat K,\hat V = X^{16}W_K,X^{16}W_V\), \(W_K/W_V=\mathrm{Linear}(d,d_{\mathrm{kv}}=256)\), slot count = sequence length; **project once at encoder top** |
| M3 | after Phase C. Phase B decoder cross-attn is **dense causal** (position \(t\) reads cache \(0..t\)) |
| QK-Norm | **on** |
| Dual RMSNorm | **off** (keep MiniCPM5 pre-norm) |

The published ledger uses true GQA, not 1.25×MHA placeholder: stored ≈ **12.25B**; encoder active ≈ **2.03B** / input token; decoder active ≈ **4.33B** / output token; expert slots **882**. YOCO cache \(W_K/W_V\) are not counted as per-layer \(4d^2\).

---

## 2. Phase B curriculum (C1+NVFP4)

| Sub-phase | tokens | Trainable | detach | gate | dtype |
| --- | ---: | --- | --- | --- | --- |
| B0 | **8B** | cross-attn, \(W_K/W_V\), gate, new LN | yes | 0→0.3 | student **bf16**; frozen encoder forward **nvfp4** |
| B1 | **27B** | whole decoder + **untied `lm_head`** | yes | →1 | **nvfp4** (allowed linear GEMMs, including lm_head and attn QKV/O) |
| B2 | **15B** | full model (including input embed + `lm_head`) | no | 1 | **nvfp4** |

- MiniCPM5 is **untied**: B0/B1 **freeze encoder + input embed**; B0 also freezes `lm_head` **and the final RMSNorm** (train only cross-attn / \(W_K/W_V\) / `ln_cross`); B1 trains `lm_head` and the final RMSNorm; B2 opens everything. Never train the input table while the encoder is frozen.
- Total envelope **50B**. If quality is unstable, **lengthen B2**; do not revert to joint bf16, and do not train two LMs.
- Wall-clock **571 H100-h** (**43%** of joint bf16 1,325; ≈ RTX PRO 6000-h). Joint bf16 1,325 is the baseline; C1 bf16 1,046 is the operation-count ledger; C1+FP8 729 is the Hopper/Ada fallback.
- Phase C indexer is **bf16**, stacked after B2. L0/unit tests are **bf16**, no NVFP4.
- **Must stay high precision**: input embed, RMSNorm / QK-Norm, router, gate, indexer, attn softmax / SDPA scores. **lm_head and attn QKV/O projections are not required bf16; they go NVFP4.** Implementation: RMSNorm, router logits/softmax, and attn softmax run fp32 under autocast.
- Published NVFP4 speedup **2.0× vs bf16** (old FP8 1.5× times 1.33). B200 / SM 10.0 / 10.3 use `TeNvfp4Linear` (TE default `NVFP4BlockScaling`: 2D + RHT + SR). Without Blackwell or on sm_120, `Nvfp4Linear` E2M1/16 emulation, then FP8 placeholder / bf16 autocast. Attention stays causal YOCO window; softmax / SDPA stay high precision.
- `use_muon=True` raises `NotImplementedError` in `build_optimizer` (published default off).

---

## 3. Optimizer / hyperparameters

| Item | Frozen |
| --- | --- |
| Optimizer | **AdamW** (\(\beta=(0.9,0.95)\), wd=0.1). **Muon off** (switch exists, default false) |
| LR | WSD. warmup **0.5B** tok. stable **1e-4**. B2 **3e-5**. Decay is left to Phase E |
| Grad clip | 1.0 |
| Sequence | Phase B **4096** |
| Global batch | **4M tok/step** (\(1024\times 4096\)). micro-batch splits per GPU |
| z-loss | router-z **1e-4** |
| KD | **on** if a MiniCPM5 teacher path exists (logit KL, T=2, weight 0.5→0); else off |
| Document mask | **on** when packing |

---

## 4. Upcycling

- Weights: Enc ← MiniCPM5 0..15, Dec ← 16..41, **untied** copies of `embed_tokens` and `lm_head`.
- New modules (cross-attn, \(W_K/W_V\), router): small-scale random (`init_new_modules`, std=0.02; skip on meta device).
- MoE: dense SwiGLU `6144` **divides** `2048` (exactly 3 groups). Each expert still copies the first 2048 rows of dense and scales by \((E G^2 / T)^{1/3}\); cleaner than MiniCPM-2B `5760`. Recovery is Phase B, not the surgical instant.
- gate init **0** (Theorem A: bypass cross-attn).

---

## 5. Code scope (this repo)

Does:

- 12B YOCO MoE graph, C1 freeze API, C1+NVFP4 policy object, WSD, upcycling, single-GPU / DDP / FSDP entry points.
- **C1 training loop**: packing + document mask (`labels_with_doc_boundaries` hits the **last dim**, so `PackedBinStream`'s `[B,S]` does not misuse the batch dim), **DDP data sharding** (C1 re-wraps after each phase `apply_freeze`, accum steps `no_sync`), gradient accumulation, activation checkpointing `--grad-ckpt`, AdamW groups (router no decay; SwiGLU `gate_proj` still decays), checkpoint / resume (including packed cursor and RNG; skip if stream `kind` / DDP stride mismatch), DummyStream seed offset by rank (trainer no longer adds rank into the seed), train/eval nll weighted by `n_valid` (DDP sums `(nll·n_valid, n_valid)`; empty ranks contribute 0 not nan), jsonl logs (nll/ppl/aux/moe_cv/grad_norm/tok/s/mem/eval/n_valid/kd_w; non-finite floats become JSON null), `--eval-batches`, optional MiniCPM5 logit KD (teacher bf16 on CUDA; KL is token-mean, ignore `-100`). Frozen MoE is not in aux and does not update router bias. DDP allreduces expert load before opt.step; aux-loss-free bias updates once per optimizer step. With no window truncation and no cross-document rows, attention uses causal SDPA and does not materialize an S×S mask. `init_process_group` passes `device_id` on NCCL; FSDP world=1 uses `NO_SHARD`. tiny unit tests (gate=0, detach with no encoder grads, frozen embed / B0 frozen lm_head and final RMSNorm, ckpt, 2-rank gloo DDP / C1 chain). **Cross-phase resume**: `--phase B1 --resume b0/latest.pt` takes weights and data cursor only, not B0 optimizer / step; phase follows the CLI. `--resume` may be a file or a directory (directory prefers `latest.pt`, else newest `step_*.pt`). `--c1` / `--c1-smoke` run B0→B1→B2 on the same graph with a continuous packed cursor, `--save-dir/{B0,B1,B2}/latest.pt`. ckpt lands on CPU first (non-FSDP also `.cpu()` then save) so 12B does not occupy GPU twice. 12B defaults to weights-only (`--no-save-optim`); `--keep-last N` prunes `step_*.pt` (does not delete the step file that shares an inode with `latest.pt`). **`latest.pt` hardlinks (or same-volume copies) when `step_{last}.pt` already exists; do not `torch.save` a second 23GiB** — on RTX 4080 SUPER, writing `step_1.pt` then `latest.pt` into `/tmp` fills the overlay (`PytorchStreamWriter` / leftover `latest.pt.tmp`). Put 12B ckpts on a large disk (e.g. `/root/autodl-tmp`), not `/tmp`. Clear `*.pt.tmp` before training; fail immediately if the volume lacks payload+1GiB. B2 `--offload-blocks` requires `--accum 1` (`--tokens` will not auto-expand to a 4M global batch). DDP/FSDP is mutually exclusive with CPU offload.
- **B0 / B1 / B2 entry points**: `python3 -m cat_yoko.b0|b1|b2` (or `scripts/train_b{0,1,2}.py` / `cat-yoko-b0`). Without `--try`, injects the published token envelopes (8e9 / 27e9 / 15e9) plus `--no-save-full --save-trainable --no-save-optim`; B0 `--offload-encoder`; B1 also `--optim-cpu`; B2 `--offload-blocks --optim-cpu --accum 1`. `<40GiB` GPUs refuse the default envelope (32GB cannot finish 8B tokens); require `--try` or explicit `--steps` / `--tokens`. `--try`: 32 steps, seq=64, `--dummy-upcycle` when no `--upcycle*`, artifact `trainable.pt` (B0 ≈0.44GiB bf16). Weights **do not enter GitHub** (`.gitignore` `*.pt`); overlay / full graph / shards go to [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO). `--resume DIR` prefers `latest.pt`, then `trainable.pt`, then newest `step_*.pt` / `trainable_step_*.pt`. B1 takes B0 overlay = MiniCPM5 upcycle + `load_trainable_state`. B2 takes B1 overlay = MiniCPM5 upcycle (encoder + input embed) + `load_trainable_state` (decoder + `lm_head` + final RMSNorm); `--offload-blocks` refuses `accum>1`.
- **Unknown-GPU B0 recipe**: `python3 -m cat_yoko.hw_recipe` (`--json` / `--argv` / `--shell`) picks a profile from SM family + VRAM; does not attach a Megatron training loop. `scripts/run_b0_next.sh` pulls MiniCPM5 + Hub `b0-full` overlay then hands flags to `cat_yoko.b0`. SM100/103 ≥160GiB → mb=2, no offload, no grad-ckpt, TE NVFP4; sm_120 ≥90GiB → mb=1, emulate NVFP4; Hopper ≥40GiB → published envelope + emulation (FP8 is only the documented fallback); `<40GiB` → `--try`; CPU JSON only. Named scripts `run_b0_full_b200.sh` / `run_b0_full_autodl.sh` remain known-SKU shortcuts.
- **C–G entry points**: `python3 -m cat_yoko.c|d|e|f|g` (`scripts/train_{c-g}.py` / `cat-yoko-c`). Same trainer + overlay. `--try` is still 32 steps, seq=64. **C** `--stage indexer|topk|hca|win` (default indexer) or `--chain`: freeze backbone, bf16 Lightning Indexer **layer-internal KL** vs dense sliding attention (encoder CSA only, not reused on decoder queries); then CSA is **window ∪ compressed-block top-k** (own block excluded, Theorem B), HCA is **window concat mean-pooled slots**, neither is a CSA CUDA kernel. **KDA** (`cat_yoko.kda`) off by default. **Implement-then-light**: B `--use-kda` builds the 3:1 graph and `KDAGates` (compute still sliding, KDA params frozen); C inherits the overlay then **`C-kda` → indexer → topk → hca → win**. Do not weld KDA onto a `use_kda=False` B overlay at C. Does not shrink the YOCO global cache; not a DPLR CUDA kernel. Default `use_kda=False` is still indexer→topk→hca→win; published B0 overlay 132 tensors unchanged. **D** `--stage 8k|32k|128k` or `--chain`: 4K packed `.bin` **re-windowed to the target seq** (concatenate rows; sidecars need not match); DummyStream plants a mid-sequence needle. D/E/F **nail `sparse=hca`** (no KDA and also no return to window); `--use-kda` inherits from resume extra. prepare `--mix phase-d` (web+code). **E** WSD 1-sqrt decay to ~1/100 of peak; prepare `--mix phase-e` (HQ web + math + code + UltraChat as body text). **F** SFT seq=8192, prompt span `labels=-100`; prepare `--mix phase-f` writes jsonl (UltraChat `data` / `messages` / alpaca, multi-span packed to seq); Trainer accepts `tokens`+`labels` or `prompt_ids`/`response_ids`. **G** `--algo grpo|dpo` (step envelope, default 10000; `--try` uses dummy RLVR). C–G do not download Ultra-FineWeb / UltraChat by default.
- **OpenBMB data path**: `cat_yoko.prepare` packs Ultra-FineWeb en/zh + UltraData-Math (default 0.60/0.30/0.10) into seq_len-aligned int32 mmap `.bin`; tokenizer defaults to `openbmb/MiniCPM5-2B` (\(V=130560\)). Trainer uses `PackedBinStream` on `.bin` and reads `eos_id` from `*.bin.meta.json`. `--upcycle-hf` / `--teacher-hf` pull `openbmb/MiniCPM5-2B-Base`.
- **Base loading**: only **MiniCPM5-2B** (`openbmb/MiniCPM5-2B-Base` / `openbmb/MiniCPM5-2B`). MiniCPM-2B / MiniCPM3 / MiniCPM4 Hub ids are refused in the CLI and `load_minicpm_state`; HF load uses `LlamaForCausalLM`, **no** `trust_remote_code`, **`device_map="cpu"`** (`empty_cache` immediately after copying the state_dict) so MiniCPM5 does not occupy 4–5GiB VRAM on top of the 12B graph. Upcycling checks exact shapes for embed / GQA K·V / dense SwiGLU; MiniCPM-2B MHA (\(d=2304\)) or `5760` FFN will not be truncated into the 12B graph. With `use_mup=False`, embed/residual/logit scales are identity 1; a mistaken MiniCPM-2B `dim_model_base` still does not divide logits.
- **CUDA smoke**: entry `setdefault PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (B1 ~28.4/32GiB; encoder offload fragments the caching allocator). On `<40GiB` GPUs, `--config 12b` requires `--seq-len` (default 4096 OOMs) and refuses `--teacher-hf` (student 23–28GiB + MiniCPM5 ~5GiB). 12B save checks the cgroup before the CPU copy (~23GiB host tensors; do not stack with B1 resident CPU Adam). `python3 -m cat_yoko.gpu_smoke` runs tiny (encoder offload / CPU Adam / B0→B1→B2 chain / KD / dummy upcycle / eval / cross-phase resume / 2-rank gloo DDP + **CUDA DDP** / **CUDA C1 DDP** / **1-rank FSDP**). `--middle` takes C1 steps on the **MiniCPM5 GQA 12.25B graph** on ≥28GiB GPUs (default 1; `--steps N` honored; bf16 built directly on CUDA, no CPU fp32 then cast). `--middle --phase B1` freezes encoder, offloads to CPU + CPU Adam. `--middle --save-dir DIR` writes `step_N.pt` and hardlinks `latest.pt`. `--c1` runs B0→B1→B2 on the same 12B graph (default 1 step each). B2 offloads layer by layer; after each layer backward, clip+Adam the grads away. Skip if no GPU. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO. FSDP wrap passes `device_id` and `use_orig_params=True`. ckpt extra includes `cfg`/`seed`. Measured on RTX 4080 SUPER 32GB + 62GiB cgroup for the **MiniCPM5-2B GQA 12.25B graph** (params=12,250,381,312, `--seq-len 64`; peak counted after freeze/offload, excluding `build_model` high-water): isolated `--middle` B0 one-step peak **22.83GiB** (23378MiB), B1 **28.6GiB** (29283MiB, encoder offload + ephemeral CPU Adam + `expandable_segments:True`; previously 28.43/29115 on the same path), B2 **2.6GiB** (2662MiB, per-layer offload working set); `--c1` same graph B0 **22.83GiB**, B1 **28.34GiB** (29019MiB), B2 **2.6GiB**. CPU `step_1.pt` resume to step 2 succeeded (nll=11.9425, mem=23378MiB). `/root/autodl-tmp/b0ckpt` weights-only **hardlink**: `latest.pt` and `step_1.pt` same inode, nlink=2, 24,501,855,029 bytes, **no** second 23GiB; `--resume` from that directory for step 2 still peaks 22.83GiB. The earlier `/tmp` second `torch.save` that filled the overlay (7G leftover `latest.pt.tmp`) is fixed. Old MiniCPM-2B placeholder graph 21.65 / 26.3 / 13.38 / 22.73 GiB is **void**; earlier B2 numbers 22.82 / 14.68 were graph-build or leftover B1 decoder, not the B2 working set. 12B unit tests each run in a **separate subprocess** (`python -m cat_yoko.gpu_smoke`); do not stack two 12.25B graphs in one process. tiny CUDA overall ok (peak_mib 17.5; includes 2-rank CUDA DDP / C1 DDP / 1-rank FSDP). GPU idle 1 MiB after the run. `expandable_segments` check: after import `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `--config 12b --device cuda` without `--seq-len` and with `--teacher-hf` both argparse exit 2 (12B graph not built); isolated B0 nll=11.9544 peak 22.83GiB; isolated B1 nll=11.9345 peak **28.6GiB**, idle 1 MiB.
- Megatron-LM adapter: `ParallelPlan` (TP/PP/EP/CP), dual-stack `TransformerConfig` mapping, `model_provider` / `forward_step` hooks. Does not vendor Megatron.

Does not (this step):

- Treat `GPTModel` / bidirectional T5 encoder as YOCO; implement an EP/TP training loop (fill the provider after [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is installed).
- Homegrown CSA CUDA kernel / V4-style FP4 expert storage / **checking 50B corpus into git** / an eval suite. The data interface eats prepare mmap `.bin`, or jsonl `tokens` / SFT `labels` / `prompt_ids`+`response_ids`. Phase C uses `cat_yoko.indexer.LightningIndexer` + layer-internal KL, plus Theorem B window∪compressed SDPA mask / HCA concat; **not** a CSA kernel.
- Allocate 12B weights on this machine's CPU (~49GB fp32 / 24.5GB bf16), or download Ultra-FineWeb / MiniCPM5 weights in CI / a small VM. `--config 12b --meta` builds a meta graph only. 12B training requires `--device cuda --dtype bf16` and explicit `--steps` or `--tokens`. Single 32GB GPU: B0 one step can run directly; B1 uses frozen-encoder offload + CPU Adam (host cgroup too small for fp32 momentum falls back to fp16 momentum); B2 uses per-layer offload, and one-step smoke uses ephemeral AdamW when the host cannot hold momentum. 4M global batch / full-param GPU Adam still needs multi-GPU or ZeRO.

Megatron constraints (already in `cat_yoko.parallel`):

- TP must divide 16 heads and \(d=2048\): `{1,2,4,8,16}`.
- 20 routed experts, EP must divide 20: `{1,2,4,5,10,20}`.
- Encoder CSA ratios `{0,4,128}` match legal Megatron Core `csa_compress_ratios`; light in Phase C.

---

## 6. Closed open questions

| Once open | Frozen |
| --- | --- |
| Bidirectional encoder? | **No. Causal YOCO** |
| 16/24 vs 20/20 | **16/26** (MiniCPM5 42 layers; extra 2 go to decoder) |
| Train two stacks independently then weld | **forbidden** |
| M2 default | **off** |
| KDA / mHC / MTP / Muon | **off** (KDA code is opt-in `use_kda`; not in the published graph by default) |
| First-layer dense | **off** (C1 all MoE) |
| Phase B CSA sparse? | **No** (sliding GQA, not a CSA kernel) |
| Attention ledger | **true GQA 16/2**, not 1.25×MHA placeholder |
| Unfreeze curriculum | **C1** (B0 8B / B1 27B / B2 15B) |
| Base | **MiniCPM5-2B Apache-2.0** (not MiniCPM-2B-sft-bf16 GML) |
| Wall-clock | **C1+NVFP4 = 571 H100-h** (C1+FP8 729 is Hopper/Ada fallback) |
