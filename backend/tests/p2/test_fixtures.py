"""The fixtures match the geometry in master F3.7, and regenerating them is deterministic."""

import numpy as np
import pytest
import rasterio
from scipy import ndimage as ndi

import fixtures.make_fixtures as F
from satquery.common.geo import sha1_file
from tests.conftest import SCENES


def rd(name):
    """(array, meta) where meta has tags/crs/transform read while the file is still open."""
    with rasterio.open(SCENES / name) as s:
        return s.read(), dict(tags=s.tags(), crs=s.crs.to_string(), transform=s.transform)


def test_optical_pair_grid_and_tags():
    for name, date in (
        ("pair_T1.tif", "2024-02-11"),
        ("pair_T2.tif", "2025-02-08"),
        ("single.tif", "2025-02-08"),
        ("optical_cloudy.tif", "2025-02-06"),
    ):
        arr, m = rd(name)
        assert arr.shape == (4, 512, 512) and arr.dtype == np.uint16
        tr = m["transform"]
        assert m["crs"] == "EPSG:32643" and (tr.a, tr.c, tr.f) == (2.0, 500000, 2500000)
        assert (
            m["tags"]["SATQUERY_SENSOR"] == "Cartosat-2S" and m["tags"]["ACQUISITION_DATE"] == date
        )
    assert sha1_file(SCENES / "single.tif") == sha1_file(SCENES / "pair_T2.tif")


def test_sar_and_hand_and_png():
    sar, m = rd("sar.tif")
    assert sar.shape == (1, 512, 512) and sar.dtype == np.float32
    t = m["tags"]
    assert (t["SATQUERY_SENSOR"], t["ACQUISITION_DATE"], t["SAR_BAND"], t["POLARISATION"]) == (
        "RISAT-2B",
        "2025-02-06",
        "X",
        "HH",
    )
    sar = sar[0]
    assert (
        -23.5 < np.median(sar[F.lake_t2()]) < -20.5
        and -9
        < np.median(
            sar[~F.lake_t2() & ~F.boxes_mask(F.OLD_BUILDINGS + F.NEW_BUILDINGS) & ~F.shadow_truth()]
        )
        < -7
    )
    assert 1 < np.median(sar[F.boxes_mask(F.OLD_BUILDINGS + F.NEW_BUILDINGS)]) < 3
    hand = rd("hand.tif")[0][0]
    assert (
        hand[F.lake_t1()].max() == 0
        and hand[F.lake_t1() | (ndi.distance_transform_edt(~F.lake_t1()) < 100)].max() <= 5
    )
    assert (hand[F.shadow_truth()] >= 25).all() and hand[F.shadow_truth()].max() <= 40
    assert hand[~(np.hypot(F.XX - 435, F.YY - 75) < 40)].max() <= 15
    with rasterio.open(SCENES / "scene.png") as p:
        assert p.count == 3 and (p.width, p.height) == (512, 512) and p.crs is None


def test_lake_geometry_and_buildings():
    assert F.lake_t1().sum() == pytest.approx(np.pi * 120 * 70, rel=0.005)
    assert F.lake_t2().sum() == pytest.approx(np.pi * 100 * 55, rel=0.005)
    assert (F.lake_t2() & ~F.lake_t1()).sum() == 0
    for x, y in F.NEW_BUILDINGS:
        b = F.boxes_mask([(x, y)])
        assert (
            F.lake_t1()[y + 6, x + 6] and not F.lake_t2()[y + 6, x + 6]
        )  # centre: inside 2024, outside 2025
        assert (
            (b & F.lake_t1()).sum() >= 0.9 * 144
        )  # corners of the 12x12 square may stick out by 1-2 px and (b & F.lake_t2()).sum() == 0
    assert len(F.OLD_BUILDINGS) + len(F.NEW_BUILDINGS) == 7


def test_spectral_separation():
    t1, t2 = rd("pair_T1.tif")[0], rd("pair_T2.tif")[0]
    for arr, lake in ((t1, F.lake_t1()), (t2, F.lake_t2())):
        nd = F.ndwi(arr)
        assert nd[lake].min() > 0.3 and nd[~lake].max() < 0.3  # water is cleanly separable
    assert F.ndvi(t2)[~F.lake_t2() & ~F.boxes_mask(F.NEW_BUILDINGS + F.OLD_BUILDINGS)].mean() > 0.3


def test_cloud_is_about_35_percent_and_overlaps_the_lake():
    assert 0.33 < F.cloud_truth().mean() < 0.38
    cloudy = rd("optical_cloudy.tif")[0]
    assert (
        cloudy[:, F.cloud_truth()].mean() > 8000 > cloudy[:, ~F.cloud_truth()].mean() * 2
    )  # bright in all bands
    assert 0.3 < (F.cloud_truth() & F.lake_t2()).sum() / F.lake_t2().sum() < 0.7


def test_regeneration_is_deterministic(tmp_path):
    F.build_all(tmp_path / "a")
    F.build_all(tmp_path / "b")
    files = sorted(
        p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file()
    )
    assert len(files) > 30
    for f in files:
        assert sha1_file(tmp_path / "a" / f) == sha1_file(tmp_path / "b" / f), f


def test_committed_fixtures_are_up_to_date(tmp_path):
    """The committed rasters must equal a fresh build (compared by content, not bytes, so different
    zlib versions on different laptops do not cause false alarms)."""
    F.build_all(tmp_path)
    for f in sorted((tmp_path / "scenes").glob("*.tif")) + sorted(
        (tmp_path / "fake_results").rglob("*.tif")
    ):
        committed = SCENES.parent / f.relative_to(tmp_path)
        assert committed.exists(), f"{committed} missing: run python fixtures/make_fixtures.py"
        with rasterio.open(f) as a, rasterio.open(committed) as b:
            assert np.array_equal(a.read(), b.read()), (
                f"{committed} is stale: rerun make_fixtures.py"
            )
