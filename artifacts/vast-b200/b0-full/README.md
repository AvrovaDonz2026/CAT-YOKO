# Vast B200 published B0 logs

Remote `/workspace/runs/b0-full`. tmux `b0-full`. Weights live only on Hub
[`checkpoints/b0-full/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-full).

**Pre-release snapshot (2026-09-19T06:44Z)** step **26940**, `tokens_in_phase=130,041,856` (≈1.63%), sha256 `7eebc9a4…`. B0 **had not finished** the 8e9 envelope. The machine then refused SSH; treat it as recycled. The next machine should resume from [`checkpoints/b0-full/README.md`](../../../checkpoints/b0-full/README.md). That Hub `b0-full` overlay is the published pin. The RTX 3090 BF16 run is a sibling, not a replacement.

| File | Contents |
| --- | --- |
| `metrics.jsonl` | per-step nll / tok/s / mem |
| `tmux-tail.txt` | tmux window + nvidia-smi just before release |
| `source.txt` | Hub overlay source at the time |
