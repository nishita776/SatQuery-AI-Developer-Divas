"""BaseTool mode dispatch, the cache round trip, and the fake-file lookup order."""

import json

import numpy as np
import pytest

from satquery.agent.registry import load_registry
from satquery.common import cache, fakes, geo
from satquery.common.toolbase import BaseTool
from satquery.contracts import Detection, MaskRef, ToolCall, ToolResult


class DummyTool(BaseTool):
    """A 'real' tool that writes a mask, to test the real -> cache round trip."""

    calls = 0

    def run_real(self, call, ctx):
        DummyTool.calls += 1
        ref = ctx.resolve(call.inputs["image"])
        m = np.zeros((ref.height, ref.width), bool)
        m[:10, :10] = True
        mask = geo.write_mask(m, ref, ctx.step_dir(call.step_id) / "water.tif", "water")
        return self.make_result(
            call, masks={"water": mask}, metrics={"water_area_m2": mask.area_m2}
        )


@pytest.fixture
def tools():
    return load_registry()


@pytest.fixture
def dummy(tools):
    DummyTool.calls = 0
    return DummyTool(tools["water_mask"].card)


def call(step="s1", tool="water_mask", inputs=None, params=None):
    return ToolCall(
        step_id=step,
        tool=tool,
        inputs=inputs or {"image": "T1"},
        params=params or {"index": "NDWI", "exclude_cloud": True},
    )


# ------------------------------------------------------------------ fake lookup
@pytest.mark.parametrize(
    "tool,inputs,params,expected",
    [
        ("water_mask", {"image": "T1"}, {"index": "NDWI"}, "default_T1"),
        ("water_mask", {"image": "T2"}, {"index": "NDWI"}, "default_T2"),
        ("index_mask", {"image": "T2"}, {"index": "NDVI"}, "ndvi"),
        ("select_region", {"detections": "$s1", "image": "T1"}, {"hint": "south"}, "south_T1"),
        ("sar_water", {"image": "T2"}, {"tile_px": 256}, "default"),
    ],
)
def test_candidate_order_reaches_the_handover_file_names(
    make_ctx, tools, tool, inputs, params, expected
):
    ctx = make_ctx()
    cands = fakes.candidates(call(tool=tool, inputs=inputs, params=params), ctx, "")
    fake_dir = fakes.fake_results_dir() / tool
    first_existing = next(c for c in cands if (fake_dir / f"{c}.json").exists())
    assert first_existing == expected


def test_variant_is_tried_first(make_ctx):
    cands = fakes.candidates(
        call(tool="rs_vlm", inputs={"t1": "T1", "t2": "T2"}, params={"mode": "describe_change"}),
        make_ctx(),
        "conflict",
    )
    assert cands[0] == "describe-change_T2_conflict" and cands.index(
        "describe-change_conflict"
    ) < cands.index("describe-change")
    assert fakes.slug("Water Body .") == "water-body"


def test_fake_mode_copies_masks_and_rewrites_paths(make_ctx, tools):
    ctx = make_ctx("fake")
    res = tools["select_region"].run(
        call(
            "s2",
            "select_region",
            {"detections": "$s1", "image": "T1"},
            {"hint": "south", "buffer_m": 50.0, "use_water": True},
        ),
        ctx,
    )
    assert res.source == "fake" and res.step_id == "s2" and res.detections
    for m in res.masks.values():
        assert m.path.parent == ctx.workdir / "steps" / "s2" and m.path.exists()
    assert res.masks["buffered"].pixels > res.masks["region"].pixels
    inner = geo.mask_iou(geo.read_mask(res.masks["region"]), geo.read_mask(res.masks["buffered"]))
    assert 0.5 < inner < 1.0


def test_water_loss_from_fakes_matches_the_fixture_story(make_ctx, tools):
    ctx = make_ctx("fake")
    r1 = tools["water_mask"].run(call("s3", inputs={"image": "T1"}), ctx)
    r2 = tools["water_mask"].run(call("s4", inputs={"image": "T2"}), ctx)
    a1, a2 = r1.metrics["water_area_m2"], r2.metrics["water_area_m2"]
    assert a1 == pytest.approx(np.pi * 120 * 70 * 4, rel=0.01) and a2 == pytest.approx(
        np.pi * 100 * 55 * 4, rel=0.01
    )
    lost = geo.mask_diff(geo.read_mask(r1.masks["water"]), geo.read_mask(r2.masks["water"]))
    assert lost.sum() * 4 == pytest.approx(
        36_400, rel=0.02
    )  # master 3.2: about 36,400 m2, about 34%
    assert lost.sum() * 4 / a1 == pytest.approx(0.34, abs=0.01)


def test_missing_fake_raises_clear_error(make_ctx, tools):
    from satquery.contracts import ErrorCode, ToolError

    with pytest.raises(ToolError) as e:
        tools["index_mask"].run(
            call("s1", "index_mask", params={"index": "NDWI"}), make_ctx("fake")
        )
        tools["water_mask"].run(call("s1", inputs={"image": "single"}), make_ctx("fake"))
    assert e.value.code is ErrorCode.INTERNAL


# ------------------------------------------------------------------ step_key
def test_step_key_is_stable_and_sensitive(make_ctx, dummy):
    ctx = make_ctx()
    k = cache.step_key(dummy, call(), ctx)
    assert k == cache.step_key(dummy, call(), ctx) and len(k) == 40
    assert k != cache.step_key(dummy, call(params={"index": "NDWI", "exclude_cloud": False}), ctx)
    assert k != cache.step_key(dummy, call(inputs={"image": "T2"}), ctx)
    assert cache.step_key(dummy, call(params={"b": 1, "a": 2}), ctx) == cache.step_key(
        dummy, call(params={"a": 2, "b": 1}), ctx
    )


def test_step_key_uses_detections_and_mask_content(make_ctx, dummy, tmp_path):
    ctx = make_ctx()

    def det(x):
        return ToolResult(
            step_id="s1",
            tool="detector",
            model_id="m",
            status="ok",
            detections=[Detection(label="a", score=0.5, bbox_px=(x, 0, 5, 5))],
        )

    ctx.results["s1"] = det(1)
    c = call("s2", inputs={"detections": "$s1", "image": "T1"})
    k1 = cache.step_key(dummy, c, ctx)
    ctx.results["s1"] = det(2)
    assert cache.step_key(dummy, c, ctx) != k1
    ref = ctx.resolve("T1")
    ma, mb = np.zeros((512, 512), bool), np.zeros((512, 512), bool)
    ma[:5, :5], mb[:6, :6] = True, True
    ctx.results["s3"] = ToolResult(
        step_id="s3",
        tool="x",
        model_id="m",
        status="ok",
        masks={"region": geo.write_mask(ma, ref, tmp_path / "a.tif", "region")},
    )
    kr = cache.step_key(
        dummy, call("s4", inputs={"image": "T1", "region": "$s3.masks.region"}), ctx
    )
    ctx.results["s3"].masks["region"] = geo.write_mask(mb, ref, tmp_path / "b.tif", "region")
    assert kr != cache.step_key(
        dummy, call("s4", inputs={"image": "T1", "region": "$s3.masks.region"}), ctx
    )


# ------------------------------------------------------------------ real -> cache -> cache mode
def test_real_run_saves_and_cache_mode_loads(make_ctx, dummy, tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SATQUERY_SCENE", "demo_a")
    real = dummy.run(call("s1"), make_ctx("real"))
    assert real.source == "real" and real.runtime_s >= 0 and DummyTool.calls == 1
    key = cache.step_key(dummy, call("s1"), make_ctx("real"))
    entry = tmp_path / "cache" / key
    saved = json.loads((entry / "result.json").read_text())
    assert (
        saved["masks"]["water"]["path"] == "water.tif" and (entry / "water.tif").exists()
    )  # relative path
    assert (
        json.loads((tmp_path / "cache" / "manifest.json").read_text())["demo_a"][key]["tool"]
        == "water_mask"
    )

    ctx2 = make_ctx("cache")
    hit = dummy.run(
        call("s7"), ctx2
    )  # a different step id must still hit: it is not part of the key
    assert DummyTool.calls == 1 and hit.source == "cache" and hit.step_id == "s7"
    assert (
        hit.masks["water"].path == ctx2.workdir / "steps" / "s7" / "water.tif"
        and hit.masks["water"].path.exists()
    )
    assert hit.metrics == real.metrics and isinstance(hit.masks["water"], MaskRef)


def test_cache_miss_falls_back_to_fake_with_warning(make_ctx, tools, tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_CACHE_DIR", str(tmp_path / "empty_cache"))
    res = tools["water_mask"].run(call("s1"), make_ctx("cache"))
    assert res.source == "fake" and "cache miss: fake result used" in res.warnings


def test_per_tool_mode_override(make_ctx, dummy, tmp_path, monkeypatch):
    monkeypatch.setenv("SATQUERY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SATQUERY_TOOL_water_mask", "real")
    assert (
        dummy.run(call(), make_ctx("fake")).source == "real"
    )  # overridden to real even in fake mode


def test_real_not_implemented_by_default(make_ctx, tools, monkeypatch):
    monkeypatch.setenv("SATQUERY_TOOL_water_mask", "real")
    with pytest.raises(NotImplementedError):
        tools["water_mask"].run(call(), make_ctx("fake"))
