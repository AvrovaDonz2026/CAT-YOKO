#!/usr/bin/env python3
"""Probe NVFP4 hardware paths: float4 cast, scaled_mm, Transformer Engine.

Prints one JSON object. Exit 0 even when a path is missing so the launcher
can record the result. Does not train. Does not download 50B tokens.
"""

from __future__ import annotations

import json
import sys
import traceback


def _exc(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def probe() -> dict:
    out: dict = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda": None,
        "cap": None,
        "float4_dtype": False,
        "float4_cast": False,
        "scaled_mm": False,
        "functional_scaled_mm": False,
        "grouped_mm": False,
        "te": None,
        "te_pytorch": False,
        "te_nvfp4_recipe": False,
        "te_nvfp4_linear": False,
        "hw_nvfp4_gemm": False,
        "errors": {},
    }
    try:
        import torch
    except Exception as err:
        out["errors"]["torch"] = _exc(err)
        return out
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["float4_dtype"] = bool(hasattr(torch, "float4_e2m1fn_x2"))
    out["scaled_mm"] = bool(hasattr(torch, "_scaled_mm"))
    out["grouped_mm"] = bool(hasattr(torch, "_grouped_mm"))
    try:
        import torch.nn.functional as F

        out["functional_scaled_mm"] = bool(hasattr(F, "scaled_mm"))
    except Exception as err:
        out["errors"]["functional"] = _exc(err)

    if not torch.cuda.is_available():
        out["errors"]["cuda"] = "cuda not available"
        return out
    out["cap"] = list(torch.cuda.get_device_capability(0))
    device = torch.device("cuda")

    if out["float4_dtype"]:
        try:
            x = torch.zeros(32, device=device, dtype=torch.bfloat16)
            y = x.to(torch.float4_e2m1fn_x2)
            out["float4_cast"] = True
            out["float4_cast_shape"] = list(y.shape)
        except Exception as err:
            out["errors"]["float4_cast"] = _exc(err)

    if out["float4_cast"] and out["scaled_mm"]:
        try:
            a = torch.zeros(32, device=device, dtype=torch.bfloat16).to(torch.float4_e2m1fn_x2)
            b = torch.zeros(32, device=device, dtype=torch.bfloat16).to(torch.float4_e2m1fn_x2)
            scale = torch.ones(1, device=device, dtype=torch.float32)
            y = torch._scaled_mm(a.reshape(16, 2), b.reshape(16, 2).T, scale, scale)
            out["hw_nvfp4_gemm"] = True
            out["scaled_mm_out"] = str(getattr(y, "dtype", type(y)))
        except Exception as err:
            out["errors"]["scaled_mm"] = _exc(err)

    if out["float4_cast"] and out["functional_scaled_mm"] and not out["hw_nvfp4_gemm"]:
        try:
            import torch.nn.functional as F

            a = torch.zeros(16, 32, device=device, dtype=torch.bfloat16)
            b = torch.zeros(32, 16, device=device, dtype=torch.bfloat16)
            # Best-effort: API moved across nightlies. Probe, do not require.
            y = F.scaled_mm(a, b, None, None, None, None)  # type: ignore[arg-type]
            out["hw_nvfp4_gemm"] = True
            out["functional_scaled_mm_out"] = str(getattr(y, "dtype", type(y)))
        except Exception as err:
            out["errors"]["functional_scaled_mm"] = _exc(err)

    try:
        import transformer_engine as te

        out["te"] = getattr(te, "__version__", "unknown")
    except Exception as err:
        out["errors"]["te"] = _exc(err)

    try:
        import transformer_engine.pytorch as tep  # noqa: F401

        out["te_pytorch"] = True
    except Exception as err:
        out["errors"]["te_pytorch"] = _exc(err)

    try:
        from transformer_engine.common.recipe import NVFP4BlockScaling

        out["te_nvfp4_recipe"] = NVFP4BlockScaling() is not None
    except Exception as err:
        out["errors"]["te_nvfp4_recipe"] = _exc(err)

    if out["te_pytorch"] and out["te_nvfp4_recipe"]:
        try:
            import transformer_engine.pytorch as tep
            from transformer_engine.common.recipe import NVFP4BlockScaling

            # sm_120 often needs RHT / 2D weight quant off.
            kwargs = {}
            try:
                recipe = NVFP4BlockScaling(disable_rht=True, disable_2d_quantization=True)
            except TypeError:
                recipe = NVFP4BlockScaling()
            # TE FP8: product of leading dims % 8 == 0, last dim % 16 == 0.
            # TE NVFP4: flattened leading dim must also be % 16 == 0 (block=16).
            # A 4×64 / 8×128 probe trips those checks and is not a kernel miss.
            layer = tep.Linear(128, 128, bias=False, params_dtype=torch.bfloat16)
            layer = layer.to(device)
            x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
            with tep.autocast(enabled=True, recipe=recipe):
                y = layer(x)
                loss = y.float().sum()
            loss.backward()
            out["te_nvfp4_linear"] = True
            out["te_nvfp4_linear_finite"] = bool(torch.isfinite(y).all().item())
        except Exception as err:
            out["errors"]["te_nvfp4_linear"] = _exc(err)
            out["errors"]["te_nvfp4_linear_tb"] = traceback.format_exc()[-1500:]

    return out


def main() -> int:
    result = probe()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
