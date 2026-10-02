# CAT-YOKO Hugging Face Kernels source

This project prepares the frozen SwiGLU storage primitives for Hugging Face
Kernels. It uses the official edition-5 `torch-noarch` framework, exports only
PyTorch APIs, and builds `torch-cpu` and `torch-rocm` variants. The public API
is version 0 while integration is developed.

`pack_frozen_swiglu(gate, up, down, experts)` creates a single fused gate/up
copy and zero-stride expert views. `frozen_swiglu(x, gu, down)` keeps the native
padded batched GEMM, SiLU multiplication, and input-gradient arithmetic. It
does not perform routing or model installation. Callers must verify that all
represented experts are frozen and byte-identical before selecting a master
expert. Cast/move masters before packing, and repack after changing them.

The existing [ROCm model installer](../rocm/README.md) separately passed full
B0 parity and measured 4.45× continuation throughput by sharing storage and
removing block offload. Those timings do not validate this newly extracted
API's full-model integration or establish an isolated kernel speedup.

The [source validation record](source_validation.json) reports **11 CPU cases
and one real ROCm BF16 case passed** on the RX 7900 XTX. The tested GPU output
and input-gradient relative L2 errors were both zero. These are direct source
checks; official builder and `get_kernel()` loading checks are still pending.

## Repository access

On 2026-10-02, authenticated `hf repos create
AvrovaDonz/CAT-YOKO-KERNEL --repo-type kernel --public` returned **403**:
`Kernel repository creation is restricted. Request access in your user or
organizations settings.` Formal publication remains blocked on that account
permission. Request **Kernels Creation** in
[HF account settings](https://huggingface.co/settings/account).

The existing [CAT-YOKO-KERNEL source archive](https://huggingface.co/AvrovaDonz/CAT-YOKO-KERNEL)
is a `model`-type repository. It is not a published Kernels build and cannot be
loaded by current `get_kernel()` merely by changing tags. See
[migration requirements](https://huggingface.co/docs/kernels/migration).

## Build and verify

Use the pinned official builder from `flake.nix`. Lock and commit the source
before building so provenance describes the actual source tree. The builder
generates metadata, unique package IDs, and digests; do not create those by hand.

The initial remote Nix attempt was stopped while fetching the builder's
dependencies and did not produce a checked build. No metadata or build output
is published as an official Kernel release. Complete the build and loader
checks above before publishing.

```bash
cd operators/hf_kernel
nix flake lock
kernel-builder check-config
kernel-builder list-variants
kernel-builder build-and-copy --cores 4
kernel-builder check-builds

LOCAL_KERNELS=AvrovaDonz/CAT-YOKO-KERNEL=. python -m pytest tests -q
LOCAL_KERNELS=AvrovaDonz/CAT-YOKO-KERNEL=. python example.py
```

The tests use `get_kernel()` with `LOCAL_KERNELS` so the actual variant loader
resolves the package. They check expert-stride/storage size, native output and
input gradients, BF16 SiLU backward, live-graph reuse, and input guards. The
ROCm test also loads `torch-rocm`; CPU-only runs skip it. Runtime tests depend
on `kernels==0.17.1`, `pytest`, and a compatible PyTorch installation.

For local development only, `kernel-builder create-pyproject` followed by
`python setup.py build_kernel` can prepare an unpackaged build. Run
`kernel-builder hash` before validating it with the loader. Official
[local-development instructions](https://huggingface.co/docs/kernels/builder/local-dev)
require the Nix build for publishable artifacts; development builds must not
be presented as the final Hub release.

Once account access and the official build checks pass:

```bash
kernel-builder upload --repo-id AvrovaDonz/CAT-YOKO-KERNEL --repo-type kernel
```

The upload defaults to branch `v0`. Consumers must specify a version or
revision. The publisher is not assumed to be trusted:

```python
from kernels import get_kernel

kernel = get_kernel(
    "AvrovaDonz/CAT-YOKO-KERNEL", version=0, backend="rocm",
    trust_remote_code=["AvrovaDonz/CAT-YOKO-KERNEL"],
)
```

This Hub-loading command becomes usable only after formal kernel publication.

## Official references

- [Write kernels and torch-noarch](https://huggingface.co/docs/kernels/builder/writing-kernels#torch-noarch)
- [Pure Python build variants](https://huggingface.co/docs/kernels/builder/build-variants#python-only-kernels)
- [Kernel layout, metadata, and publisher trust](https://huggingface.co/docs/kernels/kernel-requirements)
- [Builder commands](https://huggingface.co/docs/kernels/builder-cli)
- [Pinned official builder source](https://github.com/huggingface/kernels/tree/bd5cc502105b741d4f13930d89e5fb5ac3c6f39d)

License: [Apache-2.0](LICENSE).
