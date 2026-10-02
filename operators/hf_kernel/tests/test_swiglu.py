"""Zero-stride storage, native output/input gradients, and input guards."""

import pytest
import torch
import torch.nn.functional as F
from kernels import get_kernel


def reference(x, gate, up, down, experts):
    gu = torch.cat((gate, up), dim=0).unsqueeze(0).repeat(experts, 1, 1)
    dw = down.unsqueeze(0).repeat(experts, 1, 1)
    g, u = torch.bmm(x, gu.transpose(1, 2)).chunk(2, dim=-1)
    return torch.bmm(F.silu(g) * u, dw.transpose(1, 2))


@pytest.mark.kernels_ci
def test_shared_storage(kernel):
    gate, up, down = torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
    gu, dw = kernel.pack_frozen_swiglu(gate, up, down, 20)
    assert gu.shape == (20, 64, 16)
    assert dw.shape == (20, 16, 32)
    assert gu.stride(0) == dw.stride(0) == 0
    assert gu.untyped_storage().nbytes() == 2 * gate.numel() * gate.element_size()
    assert dw.untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert set(kernel.__all__) == {"pack_frozen_swiglu", "frozen_swiglu"}


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_output_and_input_gradient(kernel, dtype):
    torch.manual_seed(2026)
    experts, rows, hidden, intermediate = 4, 8, 16, 32
    gate, up, down = (torch.randn(intermediate, hidden, dtype=dtype),
                      torch.randn(intermediate, hidden, dtype=dtype),
                      torch.randn(hidden, intermediate, dtype=dtype))
    gu, dw = kernel.pack_frozen_swiglu(gate, up, down, experts)
    x = torch.randn(experts, rows, hidden, dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y, yr = kernel.frozen_swiglu(x, gu, dw), reference(xr, gate, up, down, experts)
    probe = torch.randn_like(y)
    (y * probe).sum().backward()
    (yr * probe).sum().backward()
    torch.testing.assert_close(y, yr, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, xr.grad, atol=0, rtol=0)


def test_reused_pack_with_two_live_graphs(kernel):
    g, u, d = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    gu, dw = kernel.pack_frozen_swiglu(g, u, d, 4)
    x = torch.randn(4, 5, 8, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    actual = kernel.frozen_swiglu(x, gu, dw) + kernel.frozen_swiglu(x * 0.5, gu, dw)
    expected = reference(xr, g, u, d, 4) + reference(xr * 0.5, g, u, d, 4)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, xr.grad, atol=0, rtol=0)


def test_bf16_silu_derivative_near_zero(kernel):
    # A BF16 sigmoid expansion rounds this derivative to zero; native ATen
    # SiLU backward retains FP32 opmath before writing the BF16 result.
    gate = torch.full((1, 1), -1.28125, dtype=torch.bfloat16)
    up = down = torch.ones(1, 1, dtype=torch.bfloat16)
    gu, dw = kernel.pack_frozen_swiglu(gate, up, down, 4)
    x = torch.ones(4, 8, 1, dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    kernel.frozen_swiglu(x, gu, dw).sum().backward()
    reference(xr, gate, up, down, 4).sum().backward()
    assert torch.count_nonzero(xr.grad) == xr.numel()
    torch.testing.assert_close(x.grad, xr.grad, atol=0, rtol=0)


@pytest.mark.parametrize("experts", [0, -1, True, 1.5])
def test_invalid_expert_count(kernel, experts):
    with pytest.raises(ValueError, match="positive integer"):
        kernel.pack_frozen_swiglu(torch.randn(16, 8), torch.randn(16, 8),
                                  torch.randn(8, 16), experts)


def test_shape_frozen_dtype_and_autocast_guards(kernel):
    g, u, d = torch.randn(16, 8), torch.randn(16, 8), torch.randn(8, 16)
    with pytest.raises(ValueError, match="frozen"):
        kernel.pack_frozen_swiglu(g.requires_grad_(True), u, d, 4)
    g.requires_grad_(False)
    with pytest.raises(ValueError, match="shapes"):
        kernel.pack_frozen_swiglu(g, u, d.T, 4)
    with pytest.raises(ValueError, match="same device and dtype"):
        kernel.pack_frozen_swiglu(g, u.double(), d, 4)
    gu, dw = kernel.pack_frozen_swiglu(g, u, d, 4)
    with pytest.raises(ValueError, match="shapes"):
        kernel.frozen_swiglu(torch.randn(3, 5, 8), gu, dw)
    with pytest.raises(ValueError, match="same device and dtype"):
        kernel.frozen_swiglu(torch.randn(4, 5, 8).double(), gu, dw)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast dtype"):
            kernel.frozen_swiglu(torch.randn(4, 5, 8), gu, dw)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm")
@pytest.mark.kernels_ci
def test_rocm_loader_bf16_forward_and_gradient():
    kernel = get_kernel(
        "AvrovaDonz/CAT-YOKO-KERNEL", version=0, backend="rocm",
        trust_remote_code=["AvrovaDonz/CAT-YOKO-KERNEL"],
    )
    torch.manual_seed(2026)
    e, n, h, i = 20, 32, 64, 64
    g, u, d = (torch.randn(i, h, device="cuda", dtype=torch.bfloat16),
               torch.randn(i, h, device="cuda", dtype=torch.bfloat16),
               torch.randn(h, i, device="cuda", dtype=torch.bfloat16))
    gu, dw = kernel.pack_frozen_swiglu(g, u, d, e)
    x = torch.randn(e, n, h, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, yr = kernel.frozen_swiglu(x, gu, dw), reference(xr, g, u, d, e)
    dy = torch.randn_like(y)
    (y * dy).sum().backward()
    (yr * dy).sum().backward()
    for actual, expected in ((y, yr), (x.grad, xr.grad)):
        relative = ((actual.float() - expected.float()).norm() /
                    expected.float().norm().clamp_min(1e-8)).item()
        assert relative <= 0.015
