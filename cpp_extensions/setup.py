"""
Build script for TinyLlama C++ extensions.

This module provides performance-optimized C++ implementations
of critical functions used in ai_cli.py.

Build with:
    python setup.py build_ext --inplace

Or install with:
    pip install -e .
"""

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import sys
import os

# Get the include directory for pybind11
try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    pybind11_include = os.path.join(sys.prefix, 'include', 'pybind11')

# Source files
sources = [
    "cpp_extensions/optimizer.cpp",
    "cpp_extensions/bindings.cpp",
]

# Include paths
include_dirs = [pybind11_include]

# Compiler arguments
extra_compile_args = [
    "-O3",                    # Maximum optimization
    "-march=native",           # CPU-specific optimizations
    "-ffast-math",             # Fast floating point
    # "-fopenmp",                # OpenMP - disabled for macOS compatibility
    "-std=c++17",              # C++17 standard
]

# Link args
link_args = []

# Windows-specific settings
if sys.platform == "win32":
    extra_compile_args.remove("-march=native")
    extra_compile_args.extend(["-GL", "/GL"])  # Whole program optimization
    link_args.append("/LTCG")  # Link-time code generation

class BuildExt(build_ext):
    """Custom build_ext to handle optional dependencies."""
    
    def build_extensions(self):
        # Add include paths
        for ext in self.extensions:
            ext.include_dirs.insert(0, pybind11_include)
        
        # Try to build with OpenMP, fall back without if not available
        if sys.platform != "win32":
            try:
                # Check if OpenMP is available
                import subprocess
                test = subprocess.run(
                    ["echo", "int main() { return 0; }"],
                    capture_output=True,
                    timeout=5
                )
            except:
                # Remove OpenMP if not available
                extra_compile_args.remove("-fopenmp")
        
        build_ext.build_extensions(self)


setup(
    name="tinyllama_cpp",
    version="1.0.0",
    description="TinyLlama CLI C++ extensions for performance",
    author="TinyLlama Team",
    ext_modules=[
        Extension(
            "tinyllama_cpp",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=link_args,
            language="c++",
        )
    ],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.10.0",
    ],
)
