"""Shared test helpers. Default run mode for the whole suite is fake."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("SATQUERY_MODE", "fake")

from satquery.common.geo import sha1_file  # noqa: E402
from satquery.contracts import (  # noqa: E402
    ImageInfo,
    JobContext,
    Modality,
    PairInfo,
    SensorProfile,
)

REPO = Path(__file__).resolve().parents[1]
SCENES = REPO / "fixtures" / "scenes"
BANDS = ["blue", "green", "red", "nir"]


def image(i, role, fname, date, **kw) -> ImageInfo:
    return ImageInfo(
        image_id=f"img_{i}",
        role=role,
        file=fname,
        sha1=sha1_file(SCENES / fname),
        prepared_path=SCENES / fname,
        modality=Modality.OPTICAL_MX,
        bands=BANDS,
        pixel_m=2.0,
        width=512,
        height=512,
        date=date,
        crs="EPSG:32643",
        sensor="Cartosat-2S",
        cloud_pct=0.0,
        **kw,
    )


@pytest.fixture
def pair_profile() -> SensorProfile:
    return SensorProfile(
        input_mode="geotiff",
        images=[
            image(0, "T1", "pair_T1.tif", "2024-02-11"),
            image(1, "T2", "pair_T2.tif", "2025-02-08"),
        ],
        pair=PairInfo(kind="bi-temporal", overlap_pct=100.0, same_crs=True, ref_image="img_0"),
        can_compute={"NDVI": True, "NDWI": True, "NDBI": False},
    )


@pytest.fixture
def make_ctx(tmp_path, pair_profile):
    def _make(mode="fake") -> JobContext:
        return JobContext(job_id="t", workdir=tmp_path / "job", profile=pair_profile, mode=mode)

    return _make


@pytest.fixture
def scene_files():
    return {
        k: SCENES / v
        for k, v in dict(
            t1="pair_T1.tif",
            t2="pair_T2.tif",
            single="single.tif",
            cloudy="optical_cloudy.tif",
            sar="sar.tif",
            hand="hand.tif",
            png="scene.png",
        ).items()
    }
