#!/usr/bin/env python3
"""Download a CAT-YOKO Hub overlay (weights only). Does not fetch 50B tokens.

Default: ``AvrovaDonz/CAT-YOKO`` ``checkpoints/b0-full/trainable.pt``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

HUB_REPO = "AvrovaDonz/CAT-YOKO"
OVERLAYS = {
    "b0-full": {
        "filename": "checkpoints/b0-full/trainable.pt",
        "sha256": "7eebc9a4da78d79be71bbe52881f2a0eaffd899f58ada3a3325f410eca181955",
    },
    "b0-3090-bf16": {
        "filename": "checkpoints/b0-3090-bf16/trainable.pt",
        "sha256": "17a2c495b31033d149ed2b95a42ecaa49dcd3d81acb25b0b19b64b33386d3393",
    },
    "b0": {"filename": "checkpoints/b0/trainable.pt", "sha256": None},
    "b1": {"filename": "checkpoints/b1/trainable.pt", "sha256": None},
    "b2": {"filename": "checkpoints/b2/trainable.pt", "sha256": None},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="b0-full", choices=sorted(OVERLAYS))
    p.add_argument("--repo", default=HUB_REPO)
    p.add_argument("--out-dir", type=Path, default=None, help="directory that will hold trainable.pt")
    p.add_argument("--filename", default=None, help="override Hub path")
    args = p.parse_args(argv)
    spec = OVERLAYS[args.name]
    filename = args.filename or spec["filename"]
    out_dir = args.out_dir or Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "trainable.pt"
    want = spec.get("sha256")
    if dest.is_file() and dest.stat().st_size > 0:
        if want:
            got = _sha256(dest)
            if got == want:
                print(f"already ok {dest} sha256={got}", flush=True)
                return 0
            print(f"sha mismatch {dest}: {got} want {want}; re-download", flush=True)
        else:
            print(f"already present {dest} ({dest.stat().st_size} bytes)", flush=True)
            return 0
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub missing", file=sys.stderr)
        return 2
    endpoint = os.environ.get("HF_ENDPOINT") or None
    print(f"download {args.repo}/{filename} -> {dest} endpoint={endpoint}", flush=True)
    hub_dir = out_dir / "_hub"
    kwargs = {
        "repo_id": args.repo,
        "filename": filename,
        "local_dir": str(hub_dir),
    }
    try:
        path = hf_hub_download(endpoint=endpoint, **kwargs)
    except TypeError:
        path = hf_hub_download(**kwargs)
    src = Path(path)
    if not src.is_file():
        print(f"download missing {src}", file=sys.stderr)
        return 1
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    if want:
        got = _sha256(dest)
        if got != want:
            print(f"sha256 mismatch {got} want {want}", file=sys.stderr)
            return 3
        print(f"ok {dest} sha256={got}", flush=True)
    else:
        print(f"ok {dest} bytes={dest.stat().st_size}", flush=True)
    (out_dir / "source.txt").write_text(f"{args.repo}/{filename}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
