"""ML-03: Scale Bridge (master F4.2).

The fixture scene matches master F3.7: 512x512 at 2 m on EPSG:32643. The
detector's native range is 0.3-1.0 m, so a 2 m scene is coarser than native and
gets upsampled by 2 -- which makes every scale conversion in here non-trivial,
which is the point.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.common.geo import write_mask
from satquery.contracts import Detection, ImageInfo, ToolError
from satquery.scale import (
    boxes_to_scene,
    merge_detections,
    plan_tiles,
    read_tile,
    resize_factor,
    scene_box_to_tile,
    stitch_mask,
)

H = W = 512
PIXEL_M = 2.0
DETECTOR_NATIVE = (0.3, 1.0)  # registry/detector.yaml
RSVLM_NATIVE = (0.3, 10.0)  # registry/rs_vlm.yaml


@pytest.fixture
def scene(tmp_path) -> ImageInfo:
    """A 3-band uint8 rgb8 GeoTIFF on the fixture grid."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(3, H, W), dtype=np.uint8)
    path = tmp_path / "img_0_rgb8.tif"
    profile = {
        "driver": "GTiff", "height": H, "width": W, "count": 3, "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 2500000, PIXEL_M, PIXEL_M),
        "compress": "LZW",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return ImageInfo(
        image_id="img_0", role="single", file="img_0.tif", sha1="0" * 40,
        prepared_path=path, rgb8_path=path, pixel_m=PIXEL_M,
        width=W, height=H, crs="EPSG:32643",
    )


def test_tiling_covers_the_whole_image(scene):
    covered = np.zeros((H, W), bool)
    for t in plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=32):
        covered[t.y0 : t.y1, t.x0 : t.x1] = True
    assert covered.all(), f"{(~covered).sum()} pixels never reach a model"


@pytest.mark.parametrize("tile_px", [640, 800, 1024])
def test_no_tile_exceeds_tile_px(scene, tile_px):
    tiles = plan_tiles(scene, DETECTOR_NATIVE, tile_px=tile_px)
    assert tiles
    for t in tiles:
        arr = read_tile(t)
        assert arr.dtype == np.uint8
        assert arr.shape == (t.model_height, t.model_width, 3)
        assert arr.shape[0] <= tile_px and arr.shape[1] <= tile_px


def test_boxes_survive_a_round_trip(scene):
    tiles = plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=32)
    tile = tiles[len(tiles) // 2]  # an interior tile, so nothing is clipped
    originals = [
        (tile.x0 + 4, tile.y0 + 6, tile.x0 + 40, tile.y0 + 38),
        (tile.x0 + 12, tile.y0 + 12, tile.x0 + 24, tile.y0 + 24),
    ]
    back = boxes_to_scene(tile, [scene_box_to_tile(tile, b) for b in originals])
    assert len(back) == len(originals)
    for want, got in zip(originals, back):
        assert all(abs(a - b) <= 1 for a, b in zip(want, got)), f"{want} -> {got}"


def test_boxes_land_on_the_reference_grid_not_the_tile(scene):
    """The bug this catches: returning tile-local coordinates (master F3.2)."""
    tiles = plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=0)
    far = [t for t in tiles if t.x0 > 0 and t.y0 > 0][-1]
    (x0, y0, _, _), = boxes_to_scene(far, [(0.0, 0.0, 20.0, 20.0)])
    assert (x0, y0) == (far.x0, far.y0)


def test_nms_removes_duplicates_that_straddle_a_tile_edge():
    same_a = Detection(label="building", score=0.80, bbox_px=(100, 100, 140, 140))
    same_b = Detection(label="building", score=0.72, bbox_px=(102, 101, 141, 139))
    elsewhere = Detection(label="building", score=0.65, bbox_px=(300, 300, 340, 340))
    other_class = Detection(label="road", score=0.60, bbox_px=(101, 100, 139, 141))

    kept = merge_detections([same_a, same_b, elsewhere, other_class], iou=0.5)

    assert len(kept) == 3, "the duplicate across the tile edge should collapse"
    assert kept[0].score == 0.80, "the higher-scoring copy survives"
    assert sorted(k.label for k in kept) == ["building", "building", "road"]
    assert merge_detections([], iou=0.5) == []


def test_stitched_mask_matches_the_reference_grid(scene, tmp_path):
    tiles = plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=0)
    masks = []
    for t in tiles:
        m = np.zeros((t.model_height, t.model_width), bool)
        m[: t.model_height // 2, : t.model_width // 2] = True
        masks.append(m)

    ref = stitch_mask(tiles, masks, scene, tmp_path / "change.tif", "change")

    assert ref.meaning == "change"
    with rasterio.open(ref.path) as src:
        assert (src.height, src.width) == (H, W)
        assert src.dtypes[0] == "uint8"
        arr = src.read(1)
    assert set(np.unique(arr)) <= {0, 1}
    assert int(arr.sum()) == ref.pixels > 0
    assert ref.area_m2 == pytest.approx(ref.pixels * PIXEL_M**2)


def test_resize_factor_rules(scene):
    assert resize_factor(scene, None) == 1.0, "change_diff has no native scale"
    assert resize_factor(scene, RSVLM_NATIVE) == 1.0, "2 m is inside 0.3-10"
    assert resize_factor(scene, DETECTOR_NATIVE) == pytest.approx(2.0)

    fine = scene.model_copy(update={"pixel_m": 0.1})
    assert resize_factor(fine, DETECTOR_NATIVE) == pytest.approx(1 / 3), "shrink to 0.3 m"

    benchmark = scene.model_copy(update={"pixel_m": None})
    assert resize_factor(benchmark, DETECTOR_NATIVE) == 1.0, "benchmark: tile only"


def test_benchmark_mode_tiles_without_resizing(scene):
    benchmark = scene.model_copy(update={"pixel_m": None})
    for t in plan_tiles(benchmark, DETECTOR_NATIVE, tile_px=256, overlap_px=0):
        assert t.scale == 1.0
        assert (t.model_height, t.model_width) == (t.height, t.width)


def test_region_skips_tiles_that_do_not_touch_it(scene, tmp_path):
    arr = np.zeros((H, W), np.uint8)
    arr[0:40, 0:40] = 1
    region = write_mask(arr, scene, tmp_path / "region.tif", "region")

    everything = plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=0)
    narrowed = plan_tiles(scene, DETECTOR_NATIVE, tile_px=256, overlap_px=0, region=region)

    assert 0 < len(narrowed) < len(everything)
    for t in narrowed:
        assert t.x0 < 40 and t.y0 < 40


def test_missing_rgb8_path_is_a_tool_error(scene):
    """Model tools read rgb8_path, never prepared_path (master F3.2)."""
    no_rgb = scene.model_copy(update={"rgb8_path": None})
    with pytest.raises(ToolError):
        plan_tiles(no_rgb, DETECTOR_NATIVE)
