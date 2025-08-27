from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cuda_src = "prpa/kernel/taichi/cuda"
cuda_sources = ["query.cu", "erosion.cu", "ext.cpp"]

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="prpa",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="prpa.kernel.taichi._C",
            sources=[os.path.join(cuda_src, s) for s in cuda_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
