# RTX 3090 B0 BF16 same-phase resume (does not overwrite the published pin)

The published pin remains Hub `checkpoints/b0-full` step **26940** / sha256 `7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955`.

The 3090 writes only the sibling `/root/autodl-tmp/b0-3090-bf16`. The 2026-09-20T13:53Z **pre-release copy** is a finished `trainable_step_33800.pt` (mtime age 669s, nlink=2, **no** SIGINT / kill). That file was copied here and uploaded to Hub [`checkpoints/b0-3090-bf16/`](https://huggingface.co/AvrovaDonz/CAT-YOKO/tree/main/checkpoints/b0-3090-bf16) (sha256 `2dc31406…`). It did **not** overwrite `b0-full` and did **not** use `--save-full`. The previous Hub sibling pin was step 28800 / `17a2c495…`.

Left on the machine (redownloadable, or already on Hub):

- `/root/autodl-tmp/hf/MiniCPM5-2B-Base` (~4.7GiB, `openbmb/MiniCPM5-2B-Base`)
- `/root/autodl-tmp/hub-b0-full/trainable.pt` (mode 444, sha256 `7eebc9a4…`, published pin already on Hub)
- miniconda / system packages

Copied off: logs in this directory, [`occupancy/`](../occupancy/README.md) GPU samples, and the Hub sibling overlay.

| | Published pin | 3090 sibling / Hub `b0-3090-bf16` |
| --- | --- | --- |
| Path | `checkpoints/b0-full/trainable.pt` | `checkpoints/b0-3090-bf16/trainable.pt` |
| step | **26940** | **33800** |
| tokens_in_phase | 130,041,856 | 158,140,416 (≈1.98% of 8e9) |
| n_tensors | 132 | 132 |
| sha256 | `7eebc9a4…` | `2dc31406c240ee8631eb49c22908e41734c6558325b4c19270dd7ab95679e690` |
| dtype | published C1+NVFP4 (B200) | BF16 `--no-nvfp4` |
| Throughput | ~15.7k tok/s (mb=2) | ~760 tok/s (`adam=ds-cpuadam`, mb=1) |

The live trainer had reached ~step 33960 at copy time; the pin is the aged **33800** file (`SAVE_EVERY=200`). DummyStream thinking mix is **5%** in-repo short code snippets (no StarCoder / 50B download). The overlay does not live on GitHub. Logs: [`metrics.jsonl`](metrics.jsonl), [`b0_3090.log`](b0_3090.log).
