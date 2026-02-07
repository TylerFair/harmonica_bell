import pathlib
import os
import sys

from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ModuleNotFoundError:
    # Fall back to vendored pybind11 for offline/source builds.
    this_dir = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(this_dir / "vendor" / "pybind11"))
    from pybind11.setup_helpers import Pybind11Extension, build_ext


def _find_cuda_home():
    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = pathlib.Path(candidate)
        if path.exists():
            return path
    return None


enable_cuda = os.environ.get("HARMONICA_ENABLE_CUDA", "0") == "1"

sources = [
    "harmonica/orbit/kepler.cpp",
    "harmonica/orbit/trajectories.cpp",
    "harmonica/orbit/gradients.cpp",
    "harmonica/light_curve/fluxes.cpp",
    "harmonica/light_curve/gradients.cpp",
    "harmonica/core/bindings.cpp",
]

include_dirs = ["vendor/eigen", "vendor/pybind11", "harmonica"]
define_macros = []
libraries = []
library_dirs = []
extra_link_args = []

if enable_cuda:
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        raise RuntimeError(
            "HARMONICA_ENABLE_CUDA=1 but CUDA toolkit was not found. "
            "Set CUDA_HOME or CUDA_PATH."
        )

    cuda_include = cuda_home / "include"
    cuda_header = cuda_include / "cuda_runtime_api.h"
    if not cuda_header.exists():
        raise RuntimeError(
            "HARMONICA_ENABLE_CUDA=1 but CUDA headers were not found at "
            f"{cuda_header}."
        )

    cuda_lib_dirs = [cuda_home / "lib64", cuda_home / "lib"]
    existing_cuda_lib_dirs = [str(p) for p in cuda_lib_dirs if p.exists()]
    if not existing_cuda_lib_dirs:
        raise RuntimeError(
            "HARMONICA_ENABLE_CUDA=1 but CUDA libraries were not found under "
            f"{cuda_home}/lib64 or {cuda_home}/lib."
        )

    sources.append("harmonica/core/bindings_cuda.cpp")
    include_dirs.append(str(cuda_include))
    define_macros.append(("HARMONICA_ENABLE_CUDA", "1"))
    libraries.append("cudart")
    library_dirs.extend(existing_cuda_lib_dirs)

    # Keep CUDA runtime resolvable without requiring a global linker config.
    if sys.platform.startswith("linux"):
        for lib_dir in existing_cuda_lib_dirs:
            extra_link_args.append(f"-Wl,-rpath,{lib_dir}")


ext_modules = [
    Pybind11Extension(
        "harmonica.core.bindings",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=["-O2", "-ffast-math"],
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="planet-harmonica",
    version="0.2.1",
    author="David Grant",
    author_email="david.grant@bristol.ac.uk",
    url="https://github.com/DavoGrant/harmonica",
    license="MIT",
    packages=find_packages(where="."),
    include_package_data=True,
    description="Light curves for exoplanet transmission mapping.",
    long_description="Light curves for exoplanet transmission mapping.",
    python_requires=">=3.6",
    install_requires=["numpy", "jax>=0.5.3", "jaxlib>=0.5.3"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
