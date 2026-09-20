"""ML-02: every P1 fake result is a valid ToolResult on the 512x512 fixture grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.common.paths import fake_results_dir
from satquery.contracts import ToolResult

TOOLS = ["rs_vlm", "detector", "change_map", "change_diff"]
GRID = (512, 512)  # master F3.7
PIXEL_M = 2.0

FAKES = [
    pytest.param(f, id=f"{tool}/{f.stem}")
    for tool in TOOLS
    for f in sorted((fake_results_dir() / tool).glob("*.json"))
]


def load(path: Path) -> ToolResult:
    data = json.loads(path.read_text())
    data["step_id"] = "s1"  # load_fake fills this in at run time
    return ToolResult.model_validate(data)


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_has_fake_results(tool):
    assert list((fake_results_dir() / tool).glob("*.json")), f"no fakes for {tool}"


@pytest.mark.parametrize("path", FAKES)
def test_fake_is_a_valid_tool_result(path):
    res = load(path)
    assert res.status == "ok"
    assert res.model_id
    assert res.source == "fake"
    if res.score is not None:
        assert 0.0 <= res.score <= 1.0, "scores are 0-1, never percentages (F3.2)"


@pytest.mark.parametrize("path", FAKES)
def test_boxes_are_on_the_reference_grid(path):
    h, w = GRID
    for d in load(path).detections:
        x0, y0, x1, y1 = d.bbox_px
        assert 0 <= x0 < x1 <= w, f"{d.bbox_px} outside the {w}px grid"
        assert 0 <= y0 < y1 <= h, f"{d.bbox_px} outside the {h}px grid"
        assert 0.0 <= d.score <= 1.0
        assert d.label


@pytest.mark.parametrize("path", FAKES)
def test_masks_are_uint8_on_the_reference_grid(path):
    for name, m in load(path).masks.items():
        tif = path.parent / Path(m.path).name
        assert tif.exists(), f"{path.name} references missing {tif.name}"
        with rasterio.open(tif) as src:
            assert (src.height, src.width) == GRID
            assert src.dtypes[0] == "uint8"
            arr = src.read(1)
        if name.endswith("_prob"):
            assert arr.max() <= 255
        else:
            assert set(np.unique(arr)) <= {0, 1}, f"{tif.name} is not a 0/1 mask"
            assert int(np.count_nonzero(arr)) == m.pixels
        if m.area_m2 is not None:
            assert m.area_m2 == pytest.approx(m.pixels * PIXEL_M**2)


@pytest.mark.parametrize("path", [p for p in FAKES if "detector/" in p.id])
def test_detector_count_matches_detections(path):
    res = load(path)
    assert res.metrics["count"] == len(res.detections)
    if res.detections:
        mean = sum(d.score for d in res.detections) / len(res.detections)
        assert res.score == pytest.approx(mean, abs=1e-3), "score is the mean box score"


def test_change_map_matches_the_fixture_geometry():
    """Six 12x12 buildings = 864 px = 3456 m2 (master F3.7)."""
    res = load(fake_results_dir() / "change_map" / "default.json")
    assert res.metrics["changed_pixels"] == 864
    assert res.metrics["changed_area_m2"] == 3456
    assert "change" in res.masks
    # change_prob pending the contract change request (see DECISION.md)


def test_change_diff_flags_more_than_change_map():
    """The blunt diff also catches the exposed shoreline -- that is the point."""
    cmap = load(fake_results_dir() / "change_map" / "default.json")
    cdiff = load(fake_results_dir() / "change_diff" / "default.json")
    assert cdiff.metrics["changed_pixels"] > cmap.metrics["changed_pixels"]
