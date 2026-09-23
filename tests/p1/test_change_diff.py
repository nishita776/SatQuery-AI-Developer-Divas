"""ML-05: ChangeDiff against the real fixture pair (master F3.7 / F4.6).

Acceptance criterion from the handover: on pair_T1 -> pair_T2 it flags the six
new buildings plus the ring of shoreline the shrinking lake exposed, and leaves
the building that was there all along alone.

The fixtures are 4-band uint16 (B, G, R, NIR); model tools read rgb8_path, so
the fixture is converted here the way ingest would -- bands R, G, B with a
2-98% stretch (master F3.2).
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
import yaml

from satquery.common.paths import REPO_ROOT
from satquery.contracts import (ImageInfo, JobContext, SensorProfile, ToolCall,
                                ToolCard, ToolError, ToolResult)
from satquery.models.change_map import ChangeDiff

SCENES = REPO_ROOT / "fixtures" / "scenes"
H = W = 512
PIXEL_M = 2.0

OLD_BUILDING = (100, 100, 112, 112)
NEW_BUILDINGS = [(140, 358, 152, 370), (140, 372, 152, 384), (140, 386, 152, 398),
                 (360, 358, 372, 370), (360, 372, 372, 384), (360, 386, 372, 398)]

pytestmark = pytest.mark.skipif(
    not (SCENES / "pair_T1.tif").exists(),
    reason="fixtures/scenes not generated; run fixtures/make_fixtures.py",
)


def stretch(band):
    lo, hi = np.percentile(band, [2, 98])
    if hi <= lo:
        return np.zeros(band.shape, np.uint8)
    return np.clip((band - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def build_rgb8_pair(tmp_path):
    """Both dates converted with ONE shared colour stretch.

    Master F4.6: ingest guarantees the rgb8 copies of T1 and T2 share a colour
    stretch. Stretching each date to its own histogram would re-normalise away
    part of the very difference change detection measures, and would give every
    unchanged pixel a small offset.
    """
    arrays, profiles, sources = [], [], []
    for name in ("pair_T1.tif", "pair_T2.tif"):
        src = SCENES / name
        with rasterio.open(src) as ds:
            arrays.append(ds.read())
            profiles.append(ds.profile)
        sources.append(src)

    # percentiles from T1 only, applied to both dates
    bounds = [np.percentile(arrays[0][i], [2, 98]) for i in (2, 1, 0)]

    out = []
    for src, bands, profile in zip(sources, arrays, profiles):
        channels = []
        for (lo, hi), idx in zip(bounds, (2, 1, 0)):
            b = bands[idx].astype(np.float32)
            channels.append(np.zeros(b.shape, np.uint8) if hi <= lo
                            else np.clip((b - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8))
        path = tmp_path / f"{src.stem}_rgb8.tif"
        profile.update(count=3, dtype="uint8", compress="LZW")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.stack(channels))
        out.append((path, src))
    return out


@pytest.fixture
def card():
    text = (REPO_ROOT / "registry" / "change_diff.yaml").read_text()
    return ToolCard.model_validate(yaml.safe_load(text))


@pytest.fixture
def images(tmp_path):
    out = []
    pair = build_rgb8_pair(tmp_path)
    for (rgb8, prepared), (name, role) in zip(pair, (("pair_T1.tif", "T1"), ("pair_T2.tif", "T2"))):
        out.append(ImageInfo(image_id=f"img_{role}", role=role, file=name,
                             sha1="0" * 40, prepared_path=prepared, rgb8_path=rgb8,
                             pixel_m=PIXEL_M, width=W, height=H, crs="EPSG:32643"))
    return out


@pytest.fixture
def ctx(images, tmp_path):
    return JobContext(job_id="job", workdir=tmp_path,
                      profile=SensorProfile(input_mode="geotiff", images=images),
                      mode="real")


def call(**params):
    p = {"k_sigma": 2.5}
    p.update(params)
    return ToolCall(step_id="s3", tool="change_diff",
                    inputs={"t1": "T1", "t2": "T2"}, params=p)


def coverage(mask, box):
    x0, y0, x1, y1 = box
    patch = mask[y0:y1, x0:x1]
    return float(patch.mean())


def read(res: ToolResult) -> np.ndarray:
    with rasterio.open(res.masks["change"].path) as src:
        return src.read(1)


def test_finds_the_six_new_buildings(card, ctx):
    res = ChangeDiff(card).run_real(call(), ctx)
    mask = read(res)
    for box in NEW_BUILDINGS:
        assert coverage(mask, box) > 0.4, f"missed the new building at {box}"


def test_leaves_the_unchanged_building_alone(card, ctx):
    res = ChangeDiff(card).run_real(call(), ctx)
    mask = read(res)
    assert coverage(mask, OLD_BUILDING) < 0.15, "flagged a building present in both dates"


def test_finds_the_exposed_shoreline(card, ctx):
    """The lake shrank from radii (120,70) to (100,55); that ring is real change."""
    res = ChangeDiff(card).run_real(call(), ctx)
    mask = read(res)
    yy, xx = np.mgrid[0:H, 0:W]
    ring = (((xx - 256) / 120) ** 2 + ((yy - 380) / 70) ** 2 <= 1.0) & \
           (((xx - 256) / 100) ** 2 + ((yy - 380) / 55) ** 2 > 1.0)
    assert mask[ring].mean() > 0.3, "did not flag the exposed shoreline"


def test_mask_and_metrics_follow_the_conventions(card, ctx):
    res = ChangeDiff(card).run_real(call(), ctx)
    ref = res.masks["change"]
    mask = read(res)

    assert mask.shape == (H, W)
    assert set(np.unique(mask)) <= {0, 1}, "master F3.2: masks are uint8 0/1"
    assert int(mask.sum()) == ref.pixels
    assert res.metrics["changed_pixels"] == ref.pixels
    assert res.metrics["changed_area_m2"] == pytest.approx(ref.pixels * PIXEL_M**2)
    assert res.model_id == card.model_id
    assert res.score is None, "change_diff reports no confidence (master F3.5)"
    assert str(ref.path).startswith(str(ctx.step_dir("s3")))


def test_higher_k_sigma_is_stricter(card, ctx):
    loose = ChangeDiff(card).run_real(call(k_sigma=1.5), ctx)
    tight = ChangeDiff(card).run_real(call(k_sigma=4.0), ctx)
    assert tight.metrics["changed_pixels"] < loose.metrics["changed_pixels"]


def test_region_restricts_what_is_reported(card, ctx, images, tmp_path):
    from satquery.common.geo import write_mask

    arr = np.zeros((H, W), np.uint8)
    arr[:, :256] = 1  # western half only
    region = write_mask(arr, images[0], tmp_path / "region.tif", "region")
    ctx.results["s2"] = ToolResult(step_id="s2", tool="select_region", model_id="x",
                                   status="ok", masks={"region": region})

    whole = ChangeDiff(card).run_real(call(), ctx)
    c = call()
    c.inputs["region"] = "$s2.masks.region"
    part = ChangeDiff(card).run_real(c, ctx)

    assert part.metrics["changed_pixels"] < whole.metrics["changed_pixels"]
    assert read(part)[:, 256:].sum() == 0, "reported change outside the region"
    # the threshold is global, so it must not move when a region is applied
    assert part.metrics["threshold"] == pytest.approx(whole.metrics["threshold"])


def test_mismatched_grids_are_rejected(card, ctx, images):
    bad = images[1].model_copy(update={"width": 256})
    ctx.profile.images[1] = bad
    with pytest.raises(ToolError):
        ChangeDiff(card).run_real(call(), ctx)


def test_report_the_actual_numbers(card, ctx, capsys):
    """Diagnostic, not a gate -- prints what the real pixels produce."""
    res = ChangeDiff(card).run_real(call(), ctx)
    mask = read(res)
    yy, xx = np.mgrid[0:H, 0:W]
    ring = (((xx - 256) / 120) ** 2 + ((yy - 380) / 70) ** 2 <= 1.0) & \
           (((xx - 256) / 100) ** 2 + ((yy - 380) / 55) ** 2 > 1.0)
    with capsys.disabled():
        print(f"\n  threshold      {res.metrics['threshold']:.2f}")
        print(f"  changed_pixels {int(res.metrics['changed_pixels'])} "
              f"({100 * res.metrics['changed_pixels'] / (H * W):.2f}% of the scene)")
        print(f"  old building   {coverage(mask, OLD_BUILDING):.3f}")
        for b in NEW_BUILDINGS:
            print(f"  new {str(b):<26} {coverage(mask, b):.3f}")
        print(f"  shoreline ring {mask[ring].mean():.3f}")
