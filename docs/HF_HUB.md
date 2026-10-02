# Hugging Face Hub (weights)

Published model repo: [AvrovaDonz/CAT-YOKO](https://huggingface.co/AvrovaDonz/CAT-YOKO)

GitHub **does not use LFS**. Code, docs, and logs stay on GitHub. `trainable.pt` / full graphs / shards go to Hub only. Code and derived weights are **Apache-2.0** ([`LICENSE`](../LICENSE)).

Progress pin: [`STATUS.md`](STATUS.md). **Do not** push the entire GitHub tree to Hub.

```bash
ssh -T git@hf.co
./scripts/push_to_hf.sh --dry-run
./scripts/push_to_hf.sh checkpoints/b0-full/trainable.pt
```

Each `push_to_hf.sh` run also ships:

- root card `huggingface/README.md` → Hub `README.md`
- `checkpoints/b0-full/README.md`
- `LICENSE`

Clone: `git clone git@hf.co:AvrovaDonz/CAT-YOKO`

## Current B0 overlay

The Vast B200 **has been recycled**. The table is the 2026-09-19T06:44Z pre-release snapshot, not a finished run.

| | |
| --- | --- |
| Hub file | [`checkpoints/b0-full/trainable.pt`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/trainable.pt) |
| Hub card | [`checkpoints/b0-full/README.md`](https://huggingface.co/AvrovaDonz/CAT-YOKO/blob/main/checkpoints/b0-full/README.md) |
| step | **26940** |
| tokens_in_phase | 130,041,856 (≈1.63% of 8e9) |
| sha256 | `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955` |
| next GPU | unknown SKU: `python3 -m cat_yoko.hw_recipe` + `bash scripts/run_b0_next.sh`. Known SM100: [`B200_TRAIN.md`](B200_TRAIN.md). `download_hub_overlay.py --name b0-full` |

[`scripts/pull_vast_b0_overlay.sh`](../scripts/pull_vast_b0_overlay.sh) uses SSH Host `vast-b200`. That machine is gone; do not assume it still answers.

Other Hub paths:

- `checkpoints/b0/` — 6000D `--try` 32 steps, **not** the 8e9 envelope
- `checkpoints/b0-nvfp4-try/` — 6000D NVFP4 wrap, 2 steps
- `checkpoints/b0-3090-bf16/` — RTX 3090 BF16 sibling snapshot (step **33800** / sha256 `2dc31406…`; machine pending release). **Not** the published pin; do not overwrite `b0-full`
- `checkpoints/b1/`, `checkpoints/b2/` — not uploaded yet (waiting on GPU)

Logs on GitHub: [`artifacts/vast-b200/`](../artifacts/vast-b200/README.md), [`artifacts/autodl-rtx6000d/`](../artifacts/autodl-rtx6000d/README.md), [`artifacts/autodl-rtx3090/`](../artifacts/autodl-rtx3090/b0-3090-bf16/README.md).

The deploy SSH key stays in `~/.ssh` on the machine (`HF_SSH_KEY` can override the path). Add the public key at https://huggingface.co/settings/keys. **Do not commit it.**

Downloads in China may use `HF_ENDPOINT=https://hf-mirror.com` (`scripts/autodl_env.sh`). **Uploads** go to `huggingface.co` / `hf.co`. Do not reconnect retired AutoDL westc / weste / westd hosts.
