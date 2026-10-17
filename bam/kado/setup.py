"""Build kado_bam (kado's BaM GPU-direct I/O engine, used by kado's kado/bam.py) against the
BaM sources one directory up (bam/): its libnvm user-space library (src/) is compiled into the
extension, its headers (device-side queue code) into the I/O kernel.

    pip install --no-build-isolation --no-deps .          # BAM_DIR overrides the BaM directory
"""
import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
bam = os.environ.get("BAM_DIR") or os.path.join(here, "..")
bam = os.path.abspath(bam)
srcs = [os.path.join(here, "kado_bam.cu")] + sorted(glob.glob(os.path.join(bam, "src", "*.cpp"))) \
    + sorted(glob.glob(os.path.join(bam, "src", "linux", "*.cpp")))
# shim/ first: maps BaM's simt:: (its bundled libcu++, which gcc >= 13 rejects) onto CUDA's libcu++
incs = [os.path.join(here, "shim"), os.path.join(bam, "include"),
        os.path.join(bam, "src"), os.path.join(bam, "src", "linux")]
# CUDA >= 13 keeps libcu++ (CCCL) under include/cccl, which the host compiler does not search
if CUDA_HOME and os.path.isdir(os.path.join(CUDA_HOME, "include", "cccl")):
    incs.append(os.path.join(CUDA_HOME, "include", "cccl"))

setup(
    name="kado_bam",
    version="0.1.0",
    ext_modules=[CUDAExtension(
        "kado_bam", srcs, include_dirs=incs,
        define_macros=[("_CUDA", None), ("__CUDA__", None)],
        extra_compile_args={"cxx": ["-O2", "-Wno-unused-variable", "-Wno-unused-function"],
                            "nvcc": ["-O3", "-lineinfo", "-Xcompiler", "-fno-forward-propagate"]})],
    cmdclass={"build_ext": BuildExtension},
)
