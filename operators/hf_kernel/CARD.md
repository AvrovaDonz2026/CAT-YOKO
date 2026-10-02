---
license: apache-2.0
tags:
  - kernels
  - rocm
  - swiglu
  - moe
---

# CAT-YOKO frozen SwiGLU primitives

Pure PyTorch weight-packing and padded SwiGLU APIs for frozen, identical experts.
Weights use zero expert stride; the compute preserves native batched GEMM and
SiLU backward. This package does not route tokens or install a model layout.
The caller must establish expert identity and freezing before packing.

See the source README for build instructions, validation results, and the
current account-permission block. Existing full-model CAT-YOKO storage-sharing
measurements belong to its separate model installer, not this extracted API.
