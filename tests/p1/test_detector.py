"""ML-04: detector logic, with the model stubbed out (master F4.5).

The scene is the fixture grid (512x512 at 2 m, master F3.7) and the stubbed
detections are the seven fixture buildings, so these tests check the things
that actually go wrong: tile-to-scene mapping, region filtering, the cap, and
the score definition. The model itself is exercised separately on the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from satquery.common.geo import write_mask
from satquery.common.paths import REPO_ROOT
from satquery.contracts import (ImageInfo, JobContext, SensorProfile, ToolCall,
                                ToolCard, ToolError, ToolResult)
from satquery.models.detector import Detector, normalise_prompt

H = W = 512
PIXEL_M = 2.0

OLD_BUILDING = (100, 100, 112, 112)
NEW_BUILDINGS = [(140, 358, 152, 370), (140, 372, 152, 384), (140, 386, 152, 398),
                 (360, 358, 372, 370), (360, 372, 372, 384), (360, 386, 372, 398)]
SCORES = [0.66, 0.71, 0.68, 0.64, 0.69, 0.60, 0.57]


@pytest.fixture
def card():
    text = (REPO_ROOT / "registry" / "detector.yaml").read_text()
    return ToolCard.model_validate(yaml.safe_load(text))


@pytest.fixture
def scene(tmp_path) -> ImageInfo:
    arr = np.random.default_rng(0).integers(0, 256, size=(3, H, W), dtype=np.uint8)
    path = tmp_path / "img_0_rgb8.tif"
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=3,
                       dtype="uint8", crs="EPSG:32643",
                       transform=from_origin(500000, 2500000, PIXEL_M, PIXEL_M),
                       compress="LZW") as dst:
        dst.write(arr)
    return ImageInfo(image_id="img_0", role="single", file="img_0.tif", sha1="0" * 40,
                     prepared_path=path, rgb8_path=path, pixel_m=PIXEL_M,
                     width=W, height=H, crs="EPSG:32643")


@pytest.fixture
def ctx(scene, tmp_path) -> JobContext:
    return JobContext(job_id="job", workdir=tmp_path,
                      profile=SensorProfile(input_mode="geotiff", images=[scene]),
                      mode="real")


def stub_seven(monkeypatch):
    """Return the seven fixture buildings, in model-space tile coordinates."""
    def _fake(self, arr, prompt, box_threshold, text_threshold):
        boxes = [OLD_BUILDING] + NEW_BUILDINGS
        # ref -> model: multiply by the tile scale (2.0 for a 2 m scene at 0.3-1.0 m native)
        model = [tuple(v * 2.0 for v in b) for b in boxes]
        return model, list(SCORES), ["building"] * 7
    monkeypatch.setattr(Detector, "_detect_tile", _fake)


def make_call(**params):
    p = {"prompt": "building", "box_threshold": 0.35, "text_threshold": 0.25,
         "nms_iou": 0.50, "max_detections": 500, "tile_px": 1024}
    p.update(params)
    return ToolCall(step_id="s2", tool="detector", inputs={"image": "single"}, params=p)


# --------------------------------------------------------------- prompt


def test_prompt_becomes_lowercase_phrases_with_full_stops():
    assert normalise_prompt("building") == "building ."
    assert normalise_prompt("Buildings, water body") == "buildings . water body ."
    # the CAPTION multi-class prompt round-trips unchanged
    multi = "building . tree . road . water . vehicle ."
    assert normalise_prompt(multi) == multi


def test_empty_prompt_is_a_tool_error():
    with pytest.raises(ToolError):
        normalise_prompt("   ")


# --------------------------------------------------------------- run_real


def test_boxes_come_back_on_the_reference_grid(monkeypatch, card, ctx):
    """The count-off dies if boxes are tile-local (master F3.2)."""
    stub_seven(monkeypatch)
    res = Detector(card).run_real(make_call(), ctx)

    assert isinstance(res, ToolResult)
    assert res.metrics["count"] == 7
    got = {d.bbox_px for d in res.detections}
    assert got == {OLD_BUILDING, *NEW_BUILDINGS}


def test_score_is_the_mean_box_score(monkeypatch, card, ctx):
    stub_seven(monkeypatch)
    res = Detector(card).run_real(make_call(), ctx)
    assert res.score == pytest.approx(sum(SCORES) / len(SCORES), abs=1e-6)
    assert 0.0 <= res.score <= 1.0


def test_score_is_zero_when_nothing_is_found(monkeypatch, card, ctx):
    monkeypatch.setattr(Detector, "_detect_tile",
                        lambda self, a, p, b, t: ([], [], []))
    res = Detector(card).run_real(make_call(), ctx)
    assert res.detections == [] and res.metrics["count"] == 0
    assert res.score == 0.0, "master F3.5: 0 if nothing was found, not None"


def test_region_keeps_only_boxes_whose_centre_is_inside(monkeypatch, card, ctx, scene, tmp_path):
    """The fixture case: a buffered lake region returns the six new buildings,
    not the original one at (100, 100)."""
    stub_seven(monkeypatch)
    arr = np.zeros((H, W), np.uint8)
    arr[300:460, :] = 1
    region = write_mask(arr, scene, tmp_path / "region.tif", "region")
    ctx.results["s1"] = ToolResult(step_id="s1", tool="select_region", model_id="x",
                                   status="ok", masks={"region": region})

    call = make_call()
    call.inputs["region"] = "$s1.masks.region"
    res = Detector(card).run_real(call, ctx)

    assert res.metrics["count"] == 6
    assert {d.bbox_px for d in res.detections} == set(NEW_BUILDINGS)
    assert any("outside region" in w for w in res.warnings)


def test_max_detections_caps_and_keeps_the_best(monkeypatch, card, ctx):
    stub_seven(monkeypatch)
    res = Detector(card).run_real(make_call(max_detections=3), ctx)
    assert res.metrics["count"] == 3
    assert [d.score for d in res.detections] == sorted(SCORES, reverse=True)[:3]
    assert any("max_detections" in w for w in res.warnings)


def test_out_of_memory_retries_once_with_smaller_tiles(monkeypatch, card, ctx):
    """ML-12: OOM -> free -> retry smaller -> only then give up."""
    state = {"calls": 0}

    def _fake(self, arr, prompt, box_threshold, text_threshold):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [], [], []

    monkeypatch.setattr(Detector, "_detect_tile", _fake)
    res = Detector(card).run_real(make_call(), ctx)
    assert state["calls"] > 1, "it should have retried"
    assert any("out of memory" in w for w in res.warnings)


def test_persistent_out_of_memory_raises(monkeypatch, card, ctx):
    def _boom(self, arr, prompt, box_threshold, text_threshold):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(Detector, "_detect_tile", _boom)
    with pytest.raises(ToolError) as exc:
        Detector(card).run_real(make_call(), ctx)
    assert exc.value.code.value == "OUT_OF_MEMORY"
