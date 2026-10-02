"""Exercise the real kernel loader, with LOCAL_KERNELS for development."""

import pytest
import torch
from kernels import get_kernel


@pytest.fixture(scope="session")
def kernel():
    return get_kernel(
        "AvrovaDonz/CAT-YOKO-KERNEL", version=0,
        backend="rocm" if torch.version.hip else "cpu",
        trust_remote_code=["AvrovaDonz/CAT-YOKO-KERNEL"],
    )
