from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Path to the precompiled shared object
solver_so_path = "/workspace/build/lib/libsolver.so"  # <-- Change this to the correct path

# Ensure the .so file exists
if not os.path.exists(solver_so_path):
    raise FileNotFoundError(f"libsolver.so not found at {solver_so_path}")

ext_modules = [
    Extension(
        "gpusolver",
        sources=["gpusolver.pyx"],  # Only compile the Cython wrapper
        include_dirs=[np.get_include(), "/usr/local/cuda/include"],  # Include NumPy headers
        library_dirs=["/workspace/build/lib", "/usr/local/cuda/lib64"],  # Path to solver.so
        libraries=["solver"],  # Link against solver.so
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=[solver_so_path, "-lcudart"],  # Ensure solver.so is linked
        runtime_library_dirs=["/workspace/build/lib", "/usr/local/cuda/lib64"]
    )
]

setup(
    name="gpusolver",
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
)
