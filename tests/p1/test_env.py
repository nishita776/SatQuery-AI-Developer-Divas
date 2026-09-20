"""ML-00: environment check for P1.

Covers both halves of the ML-00 acceptance criteria in one module, so the
same command proves the environment on a laptop and on a Kaggle T4.

Locally (no GPU):
    pytest tests/p1/test_env.py -v

On Kaggle (GPU T4, internet on):
    SATQUERY_EXPECT_GPU=1 pytest tests/p1/test_env.py -v -s

GPU checks are skipped unless SATQUERY_EXPECT_GPU is set, so this module
stays green on a laptop with no torch.  Checks that depend on P2's Day 0
skeleton (satquery.contracts) skip with an explanatory reason rather than
failing for a reason P1 cannot fix.

Version policy comes from master F3.9: requires-python = ">=3.10,<3.13".
Kaggle's image ships Python 3.12, so 3.12 gives laptop/Kaggle parity.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

# master F3.9: requires-python = ">=3.10,<3.13"
MIN_PY = (3, 10)
MAX_PY_EXCLUSIVE = (3, 13)
# Verified 2026-09-20 from Kaggle/docker-python Dockerfile.tmpl
KAGGLE_PY = (3, 12)

CORE_DEPS = ["pydantic", "numpy", "yaml"]
GEO_DEPS = ["rasterio", "scipy", "skimage", "shapely", "pyproj"]
MODEL_DEPS = ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]

EXPECT_GPU = os.environ.get("SATQUERY_EXPECT_GPU") == "1"
requires_gpu = pytest.mark.skipif(
    not EXPECT_GPU,
    reason="GPU checks run only where SATQUERY_EXPECT_GPU=1 (the Kaggle T4 notebook)",
)


def _import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.fail(f"could not import {name!r}: {exc}")


def test_python_version_is_supported() -> None:
    """The interpreter satisfies the pyproject pin in master F3.9."""
    got = sys.version_info[:2]
    assert MIN_PY <= got < MAX_PY_EXCLUSIVE, (
        f"Python {got[0]}.{got[1]} is outside the supported range "
        f">={MIN_PY[0]}.{MIN_PY[1]},<{MAX_PY_EXCLUSIVE[0]}.{MAX_PY_EXCLUSIVE[1]}. "
        "Create the venv with a supported interpreter."
    )


def test_python_version_matches_kaggle() -> None:
    """Warn if the laptop and Kaggle would run different minors."""
    got = sys.version_info[:2]
    if got != KAGGLE_PY:
        pytest.skip(
            f"running Python {got[0]}.{got[1]}, Kaggle ships "
            f"{KAGGLE_PY[0]}.{KAGGLE_PY[1]}; wheels may differ between the two"
        )


@pytest.mark.skipif(
    EXPECT_GPU, reason="Kaggle runs the image's system interpreter, not a venv"
)
def test_running_inside_a_virtualenv() -> None:
    """master F3.2: everyone works in a virtual environment."""
    assert sys.prefix != sys.base_prefix, (
        "not running inside a virtual environment; activate the venv first"
    )


@pytest.mark.parametrize("name", CORE_DEPS)
def test_core_dependencies_import(name: str) -> None:
    _import(name)


@pytest.mark.parametrize("name", GEO_DEPS)
def test_geo_dependencies_import(name: str) -> None:
    """The [geo] extra from master F3.9."""
    _import(name)


def test_rasterio_round_trips_a_uint8_mask(tmp_path) -> None:
    """GDAL works end to end for the mask convention in master F3.2.

    Masks are uint8 with values 0/1 and LZW compression, so write exactly
    that and read it back.  This is the check that catches a broken GDAL.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    arr = np.zeros((32, 32), dtype="uint8")
    arr[8:16, 8:16] = 1

    path = tmp_path / "mask.tif"
    profile = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 2500000, 2.0, 2.0),
        "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)

    with rasterio.open(path) as src:
        back = src.read(1)
        assert src.dtypes[0] == "uint8"
        assert src.crs.to_epsg() == 32643
        assert src.compression is not None
        assert src.compression.value.lower() == "lzw"

    assert back.dtype == np.uint8
    assert set(np.unique(back)) <= {0, 1}
    assert back.sum() == 64
    assert (back == arr).all()


def test_satquery_contracts_importable() -> None:
    """ML-00 done-when, local half.

    Skips until P2's Day 0 skeleton exists; P1 must not create
    satquery/contracts/ (session rule 2).
    """
    try:
        importlib.import_module("satquery.contracts")
    except ImportError as exc:
        pytest.skip(
            f"satquery.contracts not importable ({exc}). "
            "Blocked on P2's Day 0 skeleton (pyproject.toml + satquery/contracts/); "
            'then run: pip install -e ".[geo,server,dev]"'
        )


@requires_gpu
@pytest.mark.parametrize("name", MODEL_DEPS)
def test_model_dependencies_import(name: str) -> None:
    """The [models] extra, only where real inference runs."""
    _import(name)


@requires_gpu
def test_cuda_is_available() -> None:
    """ML-00 done-when, Kaggle half."""
    import torch

    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False. "
        "Set Accelerator = GPU T4 in the Kaggle notebook settings."
    )


@requires_gpu
def test_gpu_memory_is_recorded(capsys) -> None:
    """Record the GPU memory ML-00 step 4 asks for."""
    import torch

    assert torch.cuda.device_count() >= 1
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / (1024**3)

    with capsys.disabled():
        print(f"\n  GPU           : {props.name}")
        print(f"  device count  : {torch.cuda.device_count()}")
        print(f"  total memory  : {total_gib:.2f} GiB")
        print(f"  capability    : {props.major}.{props.minor}")
        print(f"  torch / cuda  : {torch.__version__} / {torch.version.cuda}")

    assert total_gib > 10, f"only {total_gib:.2f} GiB of GPU memory visible"


def test_environment_record(capsys) -> None:
    """Print the summary that goes into notebooks/ENVIRONMENT.md (use -s)."""
    import platform

    rows = [("python", platform.python_version()), ("platform", platform.platform())]
    for name in CORE_DEPS + GEO_DEPS:
        try:
            mod = importlib.import_module(name)
            rows.append((name, getattr(mod, "__version__", "?")))
        except ImportError:
            rows.append((name, "MISSING"))
    try:
        import rasterio

        rows.append(("GDAL", rasterio.__gdal_version__))
    except ImportError:
        pass
    if EXPECT_GPU:
        for name in MODEL_DEPS:
            try:
                mod = importlib.import_module(name)
                rows.append((name, getattr(mod, "__version__", "?")))
            except ImportError:
                rows.append((name, "MISSING"))

    with capsys.disabled():
        print("\n  --- environment record ---")
        for key, value in rows:
            print(f"  {key:14s} {value}")
