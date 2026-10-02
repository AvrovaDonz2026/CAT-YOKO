# B1 overlay (Hub pointer)

Weights do not live on GitHub. Files are on Hugging Face:

https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b1

This is the C1 B1 **`trainable.pt` overlay** (decoder + untied `lm_head` + final RMSNorm), not the 27e9-token published envelope.

Resume (wait until B0 no longer holds the GPU; published overlay is Hub `b0-full`, not the 32-step `--try`):

```bash
# MiniCPM5 upcycle (frozen encoder + embed) + B0 overlay + this directory's B1 overlay
python3 -m cat_yoko.b1 --try \
  --resume /root/autodl-tmp/runs/b0-full \
  --upcycle-hf /root/autodl-tmp/hf/MiniCPM5-2B-Base \
  --save-dir /root/autodl-tmp/runs/b1
```

The trainer MiniCPM5-upcycles first, then `load_trainable_state` stacks B0 cache / cross-attn; this stage trains the decoder stack. Do not overwrite Hub `checkpoints/b0/` or `checkpoints/b0-full/`. The 12B full graph `latest.pt` (≈23GiB) does not go on GitHub.

Launch: [`scripts/run_b1_try_autodl.sh`](../../scripts/run_b1_try_autodl.sh) (`--try`: 32 steps, seq=64). Logs: [`artifacts/autodl-rtx6000d/b1/`](../../artifacts/autodl-rtx6000d/b1/).
