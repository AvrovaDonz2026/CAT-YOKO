"""RTX 3090 Ampere BF16 operator roofline: theoretical MFU vs measured.

Peak is the **dense** BF16 tensor-core rate (no 2:4 sparsity). Roofline
MFU is ``min(1, I / ridge)`` with ``I = FLOPs / bytes`` and
``ridge = peak / HBM``. A kernel can beat the planning 40% Kaplan MFU
and still sit at 1% of peak if the shape is launch-bound (bf16-probe
seq=128). Approaching theoretical MFU means approaching this roofline,
not 100% of 71 TFLOPS on a tiny graph.

``torch._grouped_mm`` is SM90+; Ampere uses padded bmm.
FlexAttention sliding-window on this torch/Ampere pair is a speed trap
(~1.7 ms vs 0.02 ms SDPA) — do not swap CSA/HCA onto it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# GA102 RTX 3090 datasheet (dense, no sparsity). Measured large GEMM ~72.3.
RTX3090_BF16_PEAK = 71.16e12
RTX3090_FP32_PEAK = 35.58e12
RTX3090_HBM_BWS = 936.2e9  # 19.5 Gbps × 384-bit
RIDGE_BF16 = RTX3090_BF16_PEAK / RTX3090_HBM_BWS  # ~76 FLOP/byte


def gemm_flops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def gemm_bytes_bf16(m: int, n: int, k: int) -> float:
    return 2.0 * (m * k + k * n + m * n)


def gemm_bytes_fp32(m: int, n: int, k: int) -> float:
    return 4.0 * (m * k + k * n + m * n)


def sdpa_flops(b: int, h: int, q_len: int, k_len: int, hd: int, *, causal: bool) -> float:
    """QK^T + PV. Causal is ~half the dense S×S pair (2 instead of 4 BHS²D)."""
    full = 4.0 * b * h * q_len * k_len * hd
    return full * 0.5 if causal else full


def flash_bytes_bf16(b: int, n_q: int, n_kv: int, s: int, hd: int) -> float:
    """IO-aware Flash: Q, K, V, O once. Not the S² score matrix."""
    return 2.0 * b * (n_q * s * hd + 2 * n_kv * s * hd + n_q * s * hd)


def mask_bytes_bf16(q_len: int, k_len: int) -> float:
    return 2.0 * q_len * k_len


def roofline_mfu(flops: float, nbytes: float, *, peak: float = RTX3090_BF16_PEAK, bw: float = RTX3090_HBM_BWS) -> float:
    """Theoretical fraction of tensor-core peak. Memory-bound ops sit well below 1."""
    if flops <= 0 or peak <= 0:
        return 0.0
    if nbytes <= 0:
        return 1.0
    intensity = flops / nbytes
    ridge = peak / bw
    return min(1.0, intensity / ridge)


def achieved_mfu(flops: float, seconds: float, *, peak: float = RTX3090_BF16_PEAK) -> float:
    if seconds <= 0 or peak <= 0:
        return 0.0
    return (flops / seconds) / peak


@dataclass(frozen=True)
class OpTheory:
    name: str
    phase: str
    flops: float
    nbytes: float
    theo_mfu: float
    note: str


def probe_shapes() -> dict[str, int]:
    from cat_yoko.config import CATYokoConfig

    cfg = CATYokoConfig.bf16_probe()
    return {
        "hidden": cfg.hidden_size,
        "heads": cfg.num_heads,
        "kv": cfg.num_kv_heads,
        "hd": cfg.head_dim,
        "seq": cfg.seq_len,
        "n_win": cfg.n_win,
        "batch": 2,
        "kv_dim": cfg.kv_dim,
    }


def published_shapes() -> dict[str, int]:
    from cat_yoko.config import CATYokoConfig

    cfg = CATYokoConfig.middle_12b()
    return {
        "hidden": cfg.hidden_size,
        "heads": cfg.num_heads,
        "kv": cfg.num_kv_heads,
        "hd": cfg.head_dim,
        "seq": cfg.seq_len,
        "n_win": cfg.n_win,
        "batch": 2,
        "kv_dim": cfg.kv_dim,
    }


def theory_for_shape(shape: dict[str, int], *, tag: str) -> list[OpTheory]:
    b, s, h, kv, hd = shape["batch"], shape["seq"], shape["heads"], shape["kv"], shape["hd"]
    d, kv_dim, n_win = shape["hidden"], shape["kv_dim"], shape["n_win"]
    m = b * s
    qkv_n = d + 2 * kv_dim
    rows: list[OpTheory] = []

    fl = gemm_flops(m, qkv_n, d)
    nb = gemm_bytes_bf16(m, qkv_n, d)
    rows.append(
        OpTheory(
            "fused_qkv",
            "A–E",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: one GEMM [{m},{d}]×[{d},{qkv_n}]",
        )
    )

    fl = sdpa_flops(b, h, s, s, hd, causal=True)
    nb = flash_bytes_bf16(b, h, kv, s, hd)
    rows.append(
        OpTheory(
            "dense_cross_flash",
            "A/B YOCO cross",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: Flash GQA causal; IO-aware bytes",
        )
    )

    fl = sdpa_flops(b, h, s, s, hd, causal=True)
    nb = flash_bytes_bf16(b, h, h, s, hd) + mask_bytes_bf16(s, s)
    rows.append(
        OpTheory(
            "masked_window",
            "A/B encoder window (n_win<seq)",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: n_win={n_win}"
            + (
                " covers seq so B0 window is Flash; this row is C/probe hole"
                if n_win >= s
                else " still materializes S×S mask; not Flash"
            ),
        )
    )

    fl = sdpa_flops(b, h, s, s, hd, causal=False)
    nb = flash_bytes_bf16(b, h, h, s, hd) + mask_bytes_bf16(s, s)
    rows.append(
        OpTheory(
            "masked_csa_union",
            "C-topk",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: union mask, equal heads, cuDNN/Efficient bf16",
        )
    )

    n_slots = max(s // 8, 1)
    k_len = s + n_slots
    fl = sdpa_flops(b, h, s, k_len, hd, causal=False)
    nb = 2.0 * b * (h * s * hd + 2 * h * k_len * hd + h * s * hd) + mask_bytes_bf16(s, k_len)
    rows.append(
        OpTheory(
            "masked_hca_concat",
            "C-hca+",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: concat window+slots k_len={k_len}",
        )
    )

    # MoE: 20 experts, even split, SwiGLU 3 GEMMs with fused gate+up → 2 GEMMs
    e, inter = 20, 2048 if d >= 2048 else 64
    tok_e = max(m // e, 1)
    fl = e * (gemm_flops(tok_e, 2 * inter, d) + gemm_flops(tok_e, d, inter))
    nb = e * (gemm_bytes_bf16(tok_e, 2 * inter, d) + gemm_bytes_bf16(tok_e, d, inter))
    rows.append(
        OpTheory(
            "moe_bmm",
            "B2+ (Ampere; grouped_mm is SM90+)",
            fl,
            nb,
            roofline_mfu(fl, nb),
            f"{tag}: padded bmm, E={e} tokens/expert={tok_e}",
        )
    )

    idx_d = 32 if d <= 128 else 64
    fl = 2 * gemm_flops(m, idx_d, d) + b * gemm_flops(s, s, idx_d)
    nb = 2 * gemm_bytes_fp32(m, idx_d, d) + b * gemm_bytes_fp32(s, s, idx_d)
    rows.append(
        OpTheory(
            "indexer_fp32",
            "C-index",
            fl,
            nb,
            roofline_mfu(fl, nb, peak=RTX3090_FP32_PEAK),
            f"{tag}: fp32 scores, d_idx={idx_d}; peak is FP32/TF32 not BF16 TC",
        )
    )
    return rows


def theory_table() -> dict[str, list[dict[str, Any]]]:
    return {
        "peak": {
            "gpu": "RTX 3090 GA102",
            "bf16_dense_tflops": RTX3090_BF16_PEAK / 1e12,
            "fp32_tflops": RTX3090_FP32_PEAK / 1e12,
            "hbm_gb_s": RTX3090_HBM_BWS / 1e9,
            "ridge_flop_per_byte": RIDGE_BF16,
        },
        "bf16_probe": [asdict(r) for r in theory_for_shape(probe_shapes(), tag="probe")],
        "published_12b": [asdict(r) for r in theory_for_shape(published_shapes(), tag="12b")],
    }


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn: Callable[[], Any], *, warmup: int = 10, runs: int = 30) -> float:
    for _ in range(max(int(warmup), 1)):
        fn()
    _sync()
    t0 = time.perf_counter()
    n = max(int(runs), 1)
    for _ in range(n):
        fn()
    _sync()
    return (time.perf_counter() - t0) / n


def _row(name: str, flops: float, dt: float, nbytes: float, note: str, *, peak: float = RTX3090_BF16_PEAK) -> dict[str, Any]:
    theo = roofline_mfu(flops, nbytes, peak=peak)
    ach = achieved_mfu(flops, dt, peak=peak)
    return {
        "name": name,
        "tflops": flops / dt / 1e12 if dt > 0 else 0.0,
        "ms": dt * 1e3,
        "theo_mfu": theo,
        "achieved_mfu": ach,
        "frac_of_roofline": (ach / theo) if theo > 0 else 0.0,
        "note": note,
    }


def measure_cuda() -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from cat_yoko.attention import MASKED_SDPA_SWITCH_SEQ, _sdpa
    from cat_yoko.moe import grouped_mm_available
    from cat_yoko.nvfp4_linear import fused_cat_linear
    from cat_yoko.trainer import configure_cuda

    if not torch.cuda.is_available():
        return {"ok": False, "reason": "no cuda"}
    configure_cuda()
    device = torch.device("cuda")
    dt_peak = torch.bfloat16
    out: dict[str, Any] = {
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability()),
        "grouped_mm_available": grouped_mm_available(),
        "masked_switch_seq": MASKED_SDPA_SWITCH_SEQ,
        "ops": [],
    }
    cap = torch.cuda.get_device_capability()[0]

    # Calibrate peak
    m = n = k = 8192
    a = torch.randn(m, k, device=device, dtype=dt_peak)
    b = torch.randn(k, n, device=device, dtype=dt_peak)
    dt = _bench(lambda: torch.mm(a, b), warmup=15, runs=40)
    fl = gemm_flops(m, n, k)
    nb = gemm_bytes_bf16(m, n, k)
    out["ops"].append(_row("gemm_8192", fl, dt, nb, "cuBLAS bf16 peak probe"))
    out["measured_bf16_peak_tflops"] = fl / dt / 1e12

    # fused QKV 12b-ish and probe
    for tag, B, S, D, KV in (("probe", 2, 128, 128, 64), ("12b", 2, 4096, 2048, 256)):
        q = torch.nn.Linear(D, D, bias=False).to(device=device, dtype=dt_peak)
        k_lin = torch.nn.Linear(D, KV, bias=False).to(device=device, dtype=dt_peak)
        v = torch.nn.Linear(D, KV, bias=False).to(device=device, dtype=dt_peak)
        for p in list(q.parameters()) + list(k_lin.parameters()) + list(v.parameters()):
            p.requires_grad_(False)
        x = torch.randn(B, S, D, device=device, dtype=dt_peak)
        fused_cat_linear([q, k_lin, v], x)  # fill freeze cache
        dt = _bench(lambda: fused_cat_linear([q, k_lin, v], x), warmup=10, runs=30)
        fl = gemm_flops(B * S, D + 2 * KV, D)
        nb = gemm_bytes_bf16(B * S, D + 2 * KV, D)
        out["ops"].append(_row(f"fused_qkv_{tag}", fl, dt, nb, "frozen cached cat"))

    # Flash GQA
    for tag, B, H, KVH, S, HD in (("probe", 2, 4, 2, 128, 32), ("12b", 1, 16, 2, 4096, 128)):
        q = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        k = torch.randn(B, KVH, S, HD, device=device, dtype=dt_peak)
        v = torch.randn(B, KVH, S, HD, device=device, dtype=dt_peak)
        dt = _bench(lambda: _sdpa(q, k, v, causal=True), warmup=10, runs=25)
        fl = sdpa_flops(B, H, S, S, HD, causal=True)
        nb = flash_bytes_bf16(B, H, KVH, S, HD)
        out["ops"].append(_row(f"dense_flash_gqa_{tag}", fl, dt, nb, "YOCO cross / covering window"))

    # Masked window (equal heads)
    for tag, B, H, S, HD in (("probe", 2, 4, 128, 32), ("mid", 1, 16, 512, 128)):
        q = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        k = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        v = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        from cat_yoko.attention import _window_causal_bias

        bias = _window_causal_bias(S, S, 32, device, dt_peak, None)
        dt = _bench(lambda: _sdpa(q, k, v, bias), warmup=8, runs=20)
        fl = sdpa_flops(B, H, S, S, HD, causal=True)
        nb = flash_bytes_bf16(B, H, H, S, HD) + mask_bytes_bf16(S, S)
        out["ops"].append(_row(f"masked_window_{tag}", fl, dt, nb, f"seq={S} switch={MASKED_SDPA_SWITCH_SEQ}"))

    # CSA-style full extra_bias (non-covering union still S×S)
    for tag, B, H, S, HD in (("probe", 2, 4, 128, 32), ("mid", 1, 16, 512, 128)):
        q = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        k = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        v = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        from cat_yoko.attention import _window_causal_bias

        bias = _window_causal_bias(S, S, S, device, dt_peak, None)
        dt = _bench(lambda: _sdpa(q, k, v, bias), warmup=8, runs=20)
        fl = sdpa_flops(B, H, S, S, HD, causal=False)
        nb = flash_bytes_bf16(B, H, H, S, HD) + mask_bytes_bf16(S, S)
        out["ops"].append(_row(f"masked_csa_{tag}", fl, dt, nb, f"seq={S} equal-head union"))

    # HCA concat k_len = S + S//8
    for tag, B, H, S, HD in (("probe", 2, 4, 128, 32), ("mid", 1, 16, 512, 128)):
        extra = max(S // 8, 1)
        q = torch.randn(B, H, S, HD, device=device, dtype=dt_peak)
        k = torch.randn(B, H, S + extra, HD, device=device, dtype=dt_peak)
        v = torch.randn(B, H, S + extra, HD, device=device, dtype=dt_peak)
        bias = q.new_zeros(S, S + extra)
        dt = _bench(lambda: _sdpa(q, k, v, bias), warmup=8, runs=20)
        fl = sdpa_flops(B, H, S, S + extra, HD, causal=False)
        nb = 2.0 * B * (H * S * HD + 2 * H * (S + extra) * HD + H * S * HD) + mask_bytes_bf16(S, S + extra)
        out["ops"].append(_row(f"masked_hca_{tag}", fl, dt, nb, f"k_len={S + extra}"))

    # Masked GQA: fused kernels need equal heads; _sdpa repeats KV (stays bf16).
    from cat_yoko.attention import last_sdpa, reset_sdpa_counts

    reset_sdpa_counts()
    q = torch.randn(1, 16, 128, 32, device=device, dtype=dt_peak)
    k = torch.randn(1, 2, 128, 32, device=device, dtype=dt_peak)
    v = torch.randn(1, 2, 128, 32, device=device, dtype=dt_peak)
    from cat_yoko.attention import _window_causal_bias as _wcb

    bias = _wcb(128, 128, 32, device, dt_peak, None)
    dt = _bench(lambda: _sdpa(q, k, v, bias), warmup=8, runs=20)
    fl = sdpa_flops(1, 16, 128, 128, 32, causal=True)
    nb = flash_bytes_bf16(1, 16, 16, 128, 32) + mask_bytes_bf16(128, 128)
    out["ops"].append(
        _row("masked_gqa_probe", fl, dt, nb, f"repeat KV then fused; {last_sdpa()}")
    )

    # Indexer fp32 scores (KEEP_HIGH_PREC); peak is FP32/TF32
    for tag, B, S, D, DI in (("probe", 2, 128, 128, 32), ("12b", 1, 4096, 2048, 64)):
        x = torch.randn(B, S, D, device=device, dtype=torch.float32)
        wq = torch.randn(DI, D, device=device, dtype=torch.float32)
        wk = torch.randn(DI, D, device=device, dtype=torch.float32)

        def _idx(x=x, wq=wq, wk=wk):
            qq = F.relu(F.linear(x, wq))
            kk = F.linear(x, wk)
            return torch.matmul(qq, kk.transpose(-1, -2))

        dt = _bench(_idx, warmup=8, runs=20)
        fl = 2 * gemm_flops(B * S, DI, D) + B * gemm_flops(S, S, DI)
        nb = 2 * gemm_bytes_fp32(B * S, DI, D) + B * gemm_bytes_fp32(S, S, DI)
        out["ops"].append(
            _row(f"indexer_fp32_{tag}", fl, dt, nb, f"d_idx={DI}", peak=RTX3090_FP32_PEAK)
        )

    # MoE padded bmm vs grouped_mm trap
    e, tok, kdim, n_out = 20, 409, 2048, 2048
    xpad = torch.randn(e, tok, kdim, device=device, dtype=dt_peak)
    w = torch.randn(e, kdim, n_out, device=device, dtype=dt_peak)
    dt = _bench(lambda: torch.bmm(xpad, w), warmup=10, runs=25)
    fl = e * gemm_flops(tok, n_out, kdim)
    nb = e * gemm_bytes_bf16(tok, n_out, kdim)
    out["ops"].append(_row("moe_bmm_even", fl, dt, nb, "Ampere path after SM90 gate"))
    out["ops"].append(
        {
            "name": "grouped_mm",
            "tflops": 0.0,
            "ms": 0.0,
            "theo_mfu": 0.0 if cap < 9 else roofline_mfu(fl, nb),
            "achieved_mfu": 0.0,
            "frac_of_roofline": 0.0,
            "note": "SM90+ only; Ampere capability gate skips the RuntimeError tax",
            "available": grouped_mm_available(),
        }
    )
    return out


def run(*, out: Path | None = None) -> dict[str, Any]:
    blob: dict[str, Any] = {"theory": theory_table()}
    try:
        import torch

        if torch.cuda.is_available():
            blob["measure"] = measure_cuda()
        else:
            blob["measure"] = {"ok": False, "reason": "cpu"}
    except Exception as exc:
        blob["measure"] = {"ok": False, "reason": type(exc).__name__, "detail": str(exc)[:400]}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(blob, indent=2) + "\n")
    return blob


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ampere BF16 operator roofline / MFU")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    blob = run(out=args.out)
    if args.json:
        print(json.dumps(blob if "measure" in blob else blob, indent=2)[:8000])
        return 0
    peak = blob["theory"]["peak"]
    print(
        f"RTX 3090 peak BF16 {peak['bf16_dense_tflops']:.2f} TFLOPS  "
        f"HBM {peak['hbm_gb_s']:.0f} GB/s  ridge {peak['ridge_flop_per_byte']:.1f} F/B"
    )
    for tag in ("bf16_probe", "published_12b"):
        print(f"-- {tag} theoretical MFU --")
        for row in blob["theory"][tag]:
            print(
                f"  {row['name']:22s} theo={100*row['theo_mfu']:5.1f}%  "
                f"{row['note']}"
            )
    meas = blob.get("measure") or {}
    if meas.get("ok"):
        print(
            f"-- measured {meas.get('device')} cap={meas.get('capability')} "
            f"grouped_mm={meas.get('grouped_mm_available')} --"
        )
        if meas.get("measured_bf16_peak_tflops"):
            print(f"  calibrated GEMM {meas['measured_bf16_peak_tflops']:.2f} TFLOPS")
        for row in meas.get("ops") or []:
            print(
                f"  {row['name']:24s} {row.get('tflops', 0):6.2f} T  "
                f"ach={100*row.get('achieved_mfu', 0):5.1f}%  "
                f"theo={100*row.get('theo_mfu', 0):5.1f}%  "
                f"roof={100*row.get('frac_of_roofline', 0):5.1f}%  {row.get('note','')}"
            )
    else:
        print(f"measure skipped ({meas.get('reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
