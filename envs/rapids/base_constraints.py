"""Print exact constraints for packages owned by the RAPIDS base image."""

import importlib.metadata
import re

EXACT_NAMES = {
    "aiohttp",
    "distributed",
    "fsspec",
    "llvmlite",
    "markdown-it-py",
    "mdit-py-plugins",
    "networkx",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
}
PREFIXES = (
    "cuda-",
    "cudf",
    "cugraph",
    "cuml",
    "cupy",
    "cuspatial",
    "cuxfilter",
    "dask",
    "libcudf",
    "libcugraph",
    "libcuml",
    "libcuspatial",
    "libraft",
    "librmm",
    "numba",
    "nx-cugraph",
    "polars",
    "pylib",
    "raft",
    "rmm",
    "ucx",
    "xarray",
)


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.lower())


def main() -> None:
    versions = {
        normalize_name(name): distribution.version
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata["Name"])
    }
    for name, version in sorted(versions.items()):
        if name in EXACT_NAMES or name.startswith(PREFIXES):
            print(f"{name}=={version}")


if __name__ == "__main__":
    main()
