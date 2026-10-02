#!/usr/bin/env python3
"""Validate ROCm AOTriton Flash causal GQA against fp32 math and at seq4096."""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def main() -> None:
    torch.manual_seed(0)
    rows = []
    for seq in (128, 4096):
        q = torch.randn(1, 16, seq, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, seq, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        grad = torch.randn_like(out)
        out.backward(grad)
        torch.cuda.synchronize()
        row = {"seq": seq, "elapsed_s": time.perf_counter() - start,
               "finite_output": bool(torch.isfinite(out).all()),
               "finite_gradients": all(bool(torch.isfinite(x.grad).all()) for x in (q, k, v))}
        if seq == 128:
            refs = [x.detach().float().requires_grad_(True) for x in (q, k, v)]
            with sdpa_kernel([SDPBackend.MATH]):
                expected = F.scaled_dot_product_attention(*refs, is_causal=True, enable_gqa=True)
            expected.backward(grad.float())
            torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.02)
            for x, ref in zip((q, k, v), refs):
                torch.testing.assert_close(x.grad.float(), ref.grad, atol=0.06, rtol=0.06)
            row["math_output_and_gradient_parity"] = True
        if not row["finite_output"] or not row["finite_gradients"]:
            raise FloatingPointError(row)
        rows.append(row)
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip,
                      "gpu": torch.cuda.get_device_name(0), "probes": rows}))


if __name__ == "__main__":
    main()
