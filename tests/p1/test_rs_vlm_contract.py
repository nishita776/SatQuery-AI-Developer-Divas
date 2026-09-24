"""ML-06: rs_vlm returns a valid ToolResult in all four modes (master F4.3).

GPU only -- set SATQUERY_EXPECT_GPU=1. Needs the EarthDial source, patched by
training/patch_earthdial.py.

This is also the first proof that the LoRA adapter loads: model_id must come
back with +lora-v1, because that is what Fusion copies into Trace.adapter_id
and how judges see the adaptation happened.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import rasterio
import yaml

from satquery.common.paths import REPO_ROOT
from satquery.contracts import (ImageInfo, JobContext, SensorProfile, ToolCall,
                                ToolCard)
from satquery.models.rs_vlm import RSVLM

SCENES = REPO_ROOT / "fixtures" / "scenes"
H = W = 512
PIXEL_M = 2.0

pytestmark = [
    pytest.mark.skipif(os.environ.get("SATQUERY_EXPECT_GPU") != "1",
                       reason="real model run; set SATQUERY_EXPECT_GPU=1"),
    pytest.mark.skipif(not (SCENES / "single.tif").exists(),
                       reason="fixtures/scenes not generated"),
]


def rgb8_pair(tmp_path, names):
    """One shared colour stretch across all dates, as ingest guarantees (F4.6)."""
    arrays, profiles, sources = [], [], []
    for name in names:
        src = SCENES / name
        with rasterio.open(src) as ds:
            arrays.append(ds.read())
            profiles.append(ds.profile)
        sources.append(src)
    bounds = [np.percentile(arrays[0][i], [2, 98]) for i in (2, 1, 0)]
    out = []
    for src, bands, profile in zip(sources, arrays, profiles):
        chans = []
        for (lo, hi), idx in zip(bounds, (2, 1, 0)):
            b = bands[idx].astype(np.float32)
            chans.append(np.zeros(b.shape, np.uint8) if hi <= lo else
                         np.clip((b - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8))
        path = tmp_path / f"{src.stem}_rgb8.tif"
        profile.update(count=3, dtype="uint8", compress="LZW")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.stack(chans))
        out.append((path, src))
    return out


def info(rgb8, prepared, role, name):
    return ImageInfo(image_id=f"img_{role}", role=role, file=name, sha1="0" * 40,
                     prepared_path=prepared, rgb8_path=rgb8, pixel_m=PIXEL_M,
                     width=W, height=H, crs="EPSG:32643")


@pytest.fixture(scope="module")
def card():
    return ToolCard.model_validate(
        yaml.safe_load((REPO_ROOT / "registry" / "rs_vlm.yaml").read_text()))


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rsvlm")
    single, t1, t2 = rgb8_pair(tmp, ["single.tif", "pair_T1.tif", "pair_T2.tif"])
    images = [info(*single, "single", "single.tif"),
              info(*t1, "T1", "pair_T1.tif"),
              info(*t2, "T2", "pair_T2.tif")]
    return JobContext(job_id="job", workdir=tmp,
                      profile=SensorProfile(input_mode="geotiff", images=images),
                      mode="real")


def check(res, card):
    assert res.status == "ok"
    assert isinstance(res.text, str) and res.text.strip(), "no text returned"
    assert res.score is None or 0.0 <= res.score <= 1.0
    assert res.detections == [] and res.masks == {}, "rs_vlm returns text, not boxes"
    assert res.model_id.startswith("earthdial-4b-rgb-4bit")
    print(f"\n  [{res.model_id}] score={res.score} :: {res.text[:160]}")


@pytest.mark.parametrize("mode", ["vqa", "caption", "describe_scene"])
def test_single_image_modes(card, ctx, mode, capsys):
    call = ToolCall(step_id="s1", tool="rs_vlm", inputs={"image": "single"},
                    params={"mode": mode, "max_new_tokens": 64,
                            "question": "What is the main land cover in this image?"})
    with capsys.disabled():
        check(RSVLM(card).run_real(call, ctx), card)


def test_describe_change_takes_both_dates(card, ctx, capsys):
    call = ToolCall(step_id="s2", tool="rs_vlm", inputs={"t1": "T1", "t2": "T2"},
                    params={"mode": "describe_change", "max_new_tokens": 64})
    with capsys.disabled():
        res = RSVLM(card).run_real(call, ctx)
        check(res, card)
        assert res.metrics["tiles"] >= 2, "describe_change must see both dates"


def test_adapter_is_in_the_model_id(card, ctx):
    """The mandatory adaptation has to be visible in the trace."""
    from satquery.models.rs_vlm import _adapter_dir
    if _adapter_dir() is None:
        pytest.skip("no adapter at training/adapters/lora-v1")
    call = ToolCall(step_id="s3", tool="rs_vlm", inputs={"image": "single"},
                    params={"mode": "caption", "max_new_tokens": 32})
    res = RSVLM(card).run_real(call, ctx)
    assert "+lora-v1" in res.model_id, f"adapter not reflected: {res.model_id}"
