{
  description = "CAT-YOKO frozen identical-expert PyTorch operators";

  inputs.kernel-builder.url = "github:huggingface/kernels/bd5cc502105b741d4f13930d89e5fb5ac3c6f39d";

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}

