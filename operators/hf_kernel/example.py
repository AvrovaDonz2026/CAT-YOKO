"""Local development: LOCAL_KERNELS=AvrovaDonz/CAT-YOKO-KERNEL=. python example.py"""

import torch
from kernels import get_kernel

kernel = get_kernel(
    "AvrovaDonz/CAT-YOKO-KERNEL", version=0, backend="cpu",
    trust_remote_code=["AvrovaDonz/CAT-YOKO-KERNEL"],
)
g, u, d = torch.randn(32, 16), torch.randn(32, 16), torch.randn(16, 32)
gu, dw = kernel.pack_frozen_swiglu(g, u, d, experts=20)
x = torch.randn(20, 8, 16, requires_grad=True)
y = kernel.frozen_swiglu(x, gu, dw)
y.sum().backward()
print({"output_shape": tuple(y.shape), "input_grad_shape": tuple(x.grad.shape),
       "expert_stride": gu.stride(0)})
