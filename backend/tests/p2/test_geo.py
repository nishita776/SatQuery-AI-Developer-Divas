import numpy as np
import pytest
import rasterio

from satquery.common import geo
from satquery.contracts import ErrorCode, ToolError
from tests.conftest import SCENES, image


@pytest.fixture
def ref():
    return image(0, "T2", "pair_T2.tif", "2025-02-08")


def test_read_raster():
    arr, prof = geo.read_raster(SCENES / "pair_T1.tif")
    assert (
        arr.shape == (4, 512, 512)
        and arr.dtype == np.uint16
        and prof["crs"].to_string() == "EPSG:32643"
    )


def test_write_and_read_mask(tmp_path, ref):
    m = np.zeros((512, 512), bool)
    m[10:20, 30:50] = True
    ref_mask = geo.write_mask(m, ref, tmp_path / "m.tif", "region")
    assert ref_mask.pixels == 200 and ref_mask.area_m2 == 800.0 and ref_mask.meaning == "region"
    assert np.array_equal(geo.read_mask(ref_mask), m)
    with rasterio.open(ref_mask.path) as src:  # uint8 0/1, LZW, same grid as the reference image
        assert src.dtypes[0] == "uint8" and src.compression.value == "LZW"
        with rasterio.open(ref.prepared_path) as r:
            assert src.transform == r.transform and src.crs == r.crs


def test_write_mask_wrong_shape(tmp_path, ref):
    with pytest.raises(ToolError) as e:
        geo.write_mask(np.zeros((10, 10)), ref, tmp_path / "x.tif", "x")
    assert e.value.code is ErrorCode.INTERNAL


def test_area_m2():
    assert geo.area_m2(250_000, 2.0) == 1_000_000.0  # 1 km2 at 2 m
    assert geo.area_m2(1, 0.65) == pytest.approx(0.4225)
    assert geo.area_m2(100, None) is None
    assert geo.area_m2(np.ones((3, 3)), 2.0) == 36.0  # arrays are accepted too


def test_iou_and_diff():
    a = np.zeros((10, 10), bool)
    a[:5] = True
    b = np.zeros((10, 10), bool)
    b[:, :5] = True
    assert geo.mask_iou(a, a) == 1.0 and geo.mask_iou(a, ~a) == 0.0
    assert geo.mask_iou(a, b) == pytest.approx(25 / 75)
    assert geo.mask_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 0.0  # both empty
    d = geo.mask_diff(a, b)
    assert d.sum() == 25 and not (d & b).any()


def test_buffer_mask():
    m = np.zeros((101, 101), bool)
    m[50, 50] = True
    b = geo.buffer_mask(m, 10, 2.0)  # 10 m at 2 m/px = 5 px radius disk
    assert b[50, 55] and not b[50, 56] and b[55, 50] and not b[56, 50]
    assert not b[54, 54] or np.hypot(4, 4) <= 5  # disk, not a square
    assert not b[55, 55]  # sqrt(50) = 7.07 > 5
    assert geo.buffer_mask(m, 0, 2.0).sum() == 1
    assert geo.buffer_mask(m, 3, None)[50, 53]  # no pixel size: metres are treated as pixels
    assert not geo.buffer_mask(np.zeros((5, 5), bool), 10, 2.0).any()


def test_bbox_to_mask():
    m = geo.bbox_to_mask((2, 3, 6, 5), (10, 20))
    assert m.shape == (10, 20) and m.sum() == 8 and m[3:5, 2:6].all()
    assert geo.bbox_to_mask((-5, -5, 3, 3), (10, 10)).sum() == 9  # clipped
    assert geo.bbox_to_mask((8, 8, 30, 30), (10, 10)).sum() == 4
    assert geo.bbox_to_mask((5, 5, 5, 9), (10, 10)).sum() == 0  # degenerate


def test_px_to_lonlat_utm43(ref):
    lon, lat = geo.px_to_lonlat(ref, 0, 0)  # easting 500000 = central meridian of zone 43 (75 E)
    assert lon == pytest.approx(75.0, abs=1e-4) and 22.5 < lat < 22.7
    lon2, lat2 = geo.px_to_lonlat(ref, 512, 512)  # 1024 m east and south
    assert lon2 > lon and lat2 < lat
    assert lat - lat2 == pytest.approx(1024 / 110_574, rel=0.02)  # ~1 deg of latitude = 110.6 km


def test_px_to_lonlat_benchmark(ref):
    bench = ref.model_copy(update={"crs": None, "pixel_m": None})
    assert geo.px_to_lonlat(bench, 100, 200) == (pytest.approx(0.001), pytest.approx(-0.002))


def test_corners_order(ref):
    tl, tr, br, bl = geo.image_corners_lonlat(ref)
    assert tr[0] > tl[0] and br[0] > bl[0] and tl[1] > bl[1] and tr[1] > br[1]


def test_sha1_file(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("abc")
    assert geo.sha1_file(p) == "a9993e364706816aba3e25717850c26c9cd0d89d"
