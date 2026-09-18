#!/usr/bin/env bash
# China AutoDL Hub env. Source before any MiniCPM5 / tokenizer download.
#
# HuggingFace Hub API + tokenizer/json go through hf-mirror.
# MiniCPM5-2B-Base `model.safetensors` is Xet-backed (~5.03GiB). On this
# network `cas-bridge.xethub.hf.co` 403s, so `scripts/download_minicpm5.py`
# falls back to ModelScope `OpenBMB/MiniCPM5-2B-Base` (same sha256).
#
# Cache on the data disk, never the 30G overlay.

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/root/autodl-tmp/ms}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_HOME/hub" "$TRANSFORMERS_CACHE" "$MODELSCOPE_CACHE"
