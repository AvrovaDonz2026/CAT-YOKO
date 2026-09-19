#!/usr/bin/env python3
"""Fetch MiniCPM5-2B-Base.

Order:
1. HuggingFace Hub via ``HF_ENDPOINT``. AutoDL (``/root/autodl-tmp`` present)
   defaults to ``https://hf-mirror.com``; otherwise ``https://huggingface.co``.
2. ModelScope ``OpenBMB/MiniCPM5-2B-Base`` if Hub fails (China AutoDL Xet 403).

Tokenizer json is in the same snapshot. Instruct ``openbmb/MiniCPM5-2B`` is
not required for upcycling; Base already ships ``tokenizer.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

MINICPM5_BASE_HF = "openbmb/MiniCPM5-2B-Base"
MINICPM5_BASE_MS = "OpenBMB/MiniCPM5-2B-Base"
SAFETENSORS = "model.safetensors"
SAFETENSORS_BYTES = 5_033_557_128
SAFETENSORS_SHA256 = "d80717e7b8eb21ef43070244ecebd85d6694e4a33602fdb817f366bdb04e1e5a"
DEFAULT_LOCAL = "/root/autodl-tmp/hf/MiniCPM5-2B-Base"
AUTODL_TMP = Path("/root/autodl-tmp")


def _autodl_root() -> Path | None:
    return AUTODL_TMP if AUTODL_TMP.is_dir() else None


def _env() -> None:
    autodl = _autodl_root()
    if autodl is not None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HOME", str(autodl / "hf"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(autodl / "ms"))
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
        work = Path("/workspace") if Path("/workspace").is_dir() else Path.home()
        os.environ.setdefault("HF_HOME", str(work / ".hf_home"))
        os.environ.setdefault("MODELSCOPE_CACHE", str(work / ".ms_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def safetensors_ok(root: Path, *, check_hash: bool) -> bool:
    blob = root / SAFETENSORS
    if not blob.is_file():
        return False
    size = blob.stat().st_size
    if size != SAFETENSORS_BYTES:
        print(f"incomplete {blob}: {size} bytes (want {SAFETENSORS_BYTES})", flush=True)
        return False
    if not check_hash:
        return True
    h = hashlib.sha256()
    with blob.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != SAFETENSORS_SHA256:
        print(f"sha256 mismatch {blob}: {digest} want {SAFETENSORS_SHA256}", flush=True)
        return False
    return True


def download_hf_mirror(local: Path) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub missing; skip hf-mirror", flush=True)
        return None
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"hf-mirror snapshot {MINICPM5_BASE_HF} endpoint={endpoint}", flush=True)
    try:
        path = snapshot_download(
            repo_id=MINICPM5_BASE_HF,
            local_dir=str(local),
            endpoint=endpoint,
            max_workers=4,
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        print(f"hf-mirror failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def download_modelscope(local: Path) -> Path | None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("modelscope missing; pip install modelscope", flush=True)
        return None
    cache = os.environ.get("MODELSCOPE_CACHE", "/root/autodl-tmp/ms")
    print(f"modelscope snapshot {MINICPM5_BASE_MS} cache={cache}", flush=True)
    try:
        path = snapshot_download(
            MINICPM5_BASE_MS,
            cache_dir=cache,
            local_dir=str(local),
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        print(f"modelscope failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-dir", default=DEFAULT_LOCAL)
    p.add_argument("--check-hash", action="store_true")
    p.add_argument("--source", choices=("auto", "hf-mirror", "modelscope"), default="auto")
    args = p.parse_args(argv)
    _env()
    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    if safetensors_ok(local, check_hash=args.check_hash):
        print(f"already complete {local / SAFETENSORS}", flush=True)
        return 0

    sources: list[str]
    if args.source == "auto":
        sources = ["hf-mirror", "modelscope"]
    else:
        sources = [args.source]

    for src in sources:
        got = download_hf_mirror(local) if src == "hf-mirror" else download_modelscope(local)
        if got is None:
            continue
        if safetensors_ok(local, check_hash=args.check_hash):
            print(f"ok via {src}: {local / SAFETENSORS}", flush=True)
            return 0
        print(f"{src} left incomplete snapshot", flush=True)

    print("MiniCPM5-2B-Base download failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
