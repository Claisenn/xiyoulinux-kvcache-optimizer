import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


def _gds_flags():
  """Detect NVIDIA cuFile (GDS) and return (sources, defines, libs, lib_dirs).

  cuFile ships with the CUDA toolkit (>= 11.8) under
  ``$CUDA_HOME/lib64/libcufile.so`` with headers in ``$CUDA_HOME/include``.
  When not found, the GDS context compiles as a stub that reports
  ``is_available() == False`` and the runtime falls back to the CPU two-hop
  path, so the build never hard-depends on GDS.
  """
  cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
  include_dir = os.path.join(cuda_home, "include")
  lib_dir = os.path.join(cuda_home, "lib64")
  if os.path.exists(os.path.join(include_dir, "cufile.h")) and os.path.exists(
      os.path.join(lib_dir, "libcufile.so")):
    return (["MINIFLEX_WITH_CUFILE"], ["cufile"], [lib_dir])
  return ([], [], [])


_gds_defines, _gds_libs, _gds_lib_dirs = _gds_flags()

setup(
    ext_modules=[
        CUDAExtension(
            name="miniflex._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/ssd_io_uring.cpp",
                "csrc/gds_io.cpp",
                "csrc/transfer.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
            define_macros=[(d, None) for d in _gds_defines],
            libraries=["uring"] + _gds_libs,
            library_dirs=_gds_lib_dirs,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
