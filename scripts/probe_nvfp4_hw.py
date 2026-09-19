#!/usr/bin/env python3
"""Probe NVFP4 hardware paths: float4 cast, scaled_mm, Transformer Engine.

Prints one JSON object. Exit 0 even when a path is missing so the launcher
can record the result. Does not train. Does not download 50B tokens.

SM 10.0 / 10.3 (B200): default ``NVFP4BlockScaling()`` (RHT + 2D + SR).
sm_120 (6000D): reduced recipe. Probe uses legal 16×128 and 64×2048,
copy-once ``te.Linear`` (not Parameter share). A single illegal shape is
not a kernel miss.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _exc(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def probe() -> dict:
    out: dict = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda": None,
        "cap": None,
        "compute_family": None,
        "nvfp4_training_capable": False,
        "te_recipe_kwargs": {},
        "float4_dtype": False,
        "float4_cast": False,
        "scaled_mm": False,
        "functional_scaled_mm": False,
        "grouped_mm": False,
        "te": None,
        "te_pytorch": False,
        "te_nvfp4_recipe": False,
        "te_nvfp4_linear": False,
        "te_nvfp4_linear_copy_once": False,
        "te_nvfp4_linear_64x2048": False,
        "te_grouped_linear": False,
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
        out["compute_family"] = "cpu"
        return out
    out["cap"] = list(torch.cuda.get_device_capability(0))
    device = torch.device("cuda")
    try:
        from cat_yoko.nvfp4_hw import (
            compute_family,
            nvfp4_training_capable,
            te_nvfp4_recipe_kwargs,
        )

        fam = compute_family(tuple(out["cap"]))
        out["compute_family"] = fam
        out["nvfp4_training_capable"] = nvfp4_training_capable(fam)
        out["te_recipe_kwargs"] = te_nvfp4_recipe_kwargs(fam)
    except Exception as err:
        out["errors"]["nvfp4_hw"] = _exc(err)
        major, minor = out["cap"]
        if major == 10 and minor == 3:
            out["compute_family"] = "sm103"
        elif major == 10:
            out["compute_family"] = "sm100"
        elif major == 12:
            out["compute_family"] = "sm120"
        else:
            out["compute_family"] = "other"
        out["nvfp4_training_capable"] = out["compute_family"] in {"sm100", "sm103"}
        if out["compute_family"] in {"sm100", "sm103"}:
            out["te_recipe_kwargs"] = {}
        elif out["compute_family"] == "sm120":
            out["te_recipe_kwargs"] = {
                "disable_rht": True,
                "disable_stochastic_rounding": True,
                "disable_2d_quantization": True,
            }

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

        kwargs = dict(out.get("te_recipe_kwargs") or {})
        try:
            recipe = NVFP4BlockScaling(**kwargs)
        except TypeError:
            recipe = NVFP4BlockScaling()
        out["te_nvfp4_recipe"] = recipe is not None
        out["te_recipe_used"] = sorted(kwargs.keys())
    except Exception as err:
        out["errors"]["te_nvfp4_recipe"] = _exc(err)
        recipe = None

    if out["te_pytorch"] and out["te_nvfp4_recipe"] and recipe is not None:
        try:
            import transformer_engine.pytorch as tep

            # Copy-once, legal 16×128. Do not require Parameter identity.
            layer = tep.Linear(128, 128, bias=False, params_dtype=torch.bfloat16)
            layer = layer.to(device)
            src = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                layer.weight.copy_(src)
            x = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
            with tep.autocast(enabled=True, recipe=recipe):
                y = layer(x)
                loss = y.float().sum()
            loss.backward()
            out["te_nvfp4_linear"] = True
            out["te_nvfp4_linear_copy_once"] = True
            out["te_nvfp4_linear_finite"] = bool(torch.isfinite(y).all().item())
        except Exception as err:
            out["errors"]["te_nvfp4_linear"] = _exc(err)
            out["errors"]["te_nvfp4_linear_tb"] = traceback.format_exc()[-1500:]

        if out["te_nvfp4_linear"]:
            try:
                import transformer_engine.pytorch as tep

                layer = tep.Linear(2048, 2048, bias=False, params_dtype=torch.bfloat16)
                layer = layer.to(device)
                x = torch.randn(64, 2048, device=device, dtype=torch.bfloat16)
                with tep.autocast(enabled=True, recipe=recipe):
                    y = layer(x)
                out["te_nvfp4_linear_64x2048"] = bool(
                    tuple(y.shape) == (64, 2048) and torch.isfinite(y.float()).all()
                )
            except Exception as err:
                out["errors"]["te_nvfp4_linear_64x2048"] = _exc(err)

        try:
            import transformer_engine.pytorch as tep

            grouped_cls = getattr(tep, "GroupedLinear", None)
            out["te_grouped_linear_cls"] = grouped_cls is not None
            if grouped_cls is not None:
                gl = grouped_cls(2, 128, 128, bias=False, params_dtype=torch.bfloat16, device=device)
                x = torch.randn(32, 128, device=device, dtype=torch.bfloat16)
                splits = torch.tensor([16, 16], dtype=torch.int64)
                with tep.autocast(enabled=True, recipe=recipe):
                    y = gl(x, splits)
                    loss = y.float().sum()
                loss.backward()
                out["te_grouped_linear"] = True
                out["te_grouped_linear_finite"] = bool(torch.isfinite(y).all().item())
        except Exception as err:
            out["errors"]["te_grouped_linear"] = _exc(err)
            out["errors"]["te_grouped_linear_tb"] = traceback.format_exc()[-1500:]

    return out


def main() -> int:
    result = probe()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
