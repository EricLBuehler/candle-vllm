from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

vllm_extension = CUDAExtension(
    name="vllm._C",
    sources=[
        "cache_kernels.cu",
        "attention/attention_kernels.cu",
        "pos_encoding_kernels.cu",
        "activation_kernels.cu",
        "layernorm_kernels.cu",
        "quantization/awq/gemm_kernels.cu",
        "quantization/squeezellm/quant_cuda_kernel.cu",
        "cuda_utils_kernels.cu",
        "ops.h",
    ],
)

for attr in dir(vllm_extension):
    if "__" in attr:
        continue
    print(attr, getattr(vllm_extension, attr))