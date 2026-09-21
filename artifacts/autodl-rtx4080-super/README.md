# AutoDL RTX 4080 SUPER pre-destroy migration

**The instance was released (2026-09-18).** Do not SSH to `connect.westc.seetacloud.com` again. The next card is RTX PRO 6000 / 6000D.

Instance `autodl-container-6d164c9f44-4dccdd73` (torch 2.8.0+cu128, driver 595.71.05).
Inventory timestamp UTC `2026-09-18T14:39:15Z`. The dummy-upcycle overlay was removed from GitHub; the real MiniCPM5 upcycle overlay is on [HuggingFace AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO).


## Copied into this directory

| Path | Contents |
| --- | --- |
| `minicpm5_peak/*.json` | 12B C1 peaks: B0 22.83GiB / B1 28.6GiB / B2 2.6GiB |
| `minicpm5_cuda/` | same-card unittest + standalone B0/B1/B2/C1 smokes |
| `p0_gpu/` | earlier MiniCPM5 graph smoke (including tiny) |
| `gpu_run*.log` | remote suite stdout |
| `runs/b0_train.log` | full B0 `--try` 32-step log |
| `runs/b0/metrics.jsonl` | same run jsonl (identical to `checkpoints/b0/metrics.jsonl`) |
| `MIGRATE_MANIFEST.txt` | pre-destroy file tree |

## Not migrated (intentionally dropped)

- Two 23GiB full-graph `b0ckpt` / `old_b0ckpt` copies: deleted before migration; weights do not go on GitHub.
- `trainable_step_24.pt`: an intermediate step with the same structure as the final overlay; the curve is in `metrics.jsonl`, and the final weights are step 32.
- MiniCPM5-2B-Base `model.safetensors`: hf-mirror stuck at a 20MiB incomplete download; full weights go back to HuggingFace.
- `miniconda3`, AutoDL `autopanel*.db`, gzip shards.
