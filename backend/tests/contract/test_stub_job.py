"""The stub run_job returns a valid, fully-backed JobResult for stories A, B and C."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.agent.job import run_job
from satquery.agent.stub import PATH_KEYS

STORIES = {
    "A": (["pair_T1.tif", "pair_T2.tif"], "CHANGE_GROUNDING"),
    "B": (["optical_cloudy.tif", "sar.tif"], "OPTICAL_SAR"),
    "C": (["single.tif"], "VQA"),
}


def all_paths(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in PATH_KEYS and isinstance(v, str):
                yield v
            else:
                yield from all_paths(v)
    elif isinstance(node, list):
        for v in node:
            yield from all_paths(v)


@pytest.mark.parametrize("story", STORIES)
def test_stub_story(story, tmp_path, scene_files):
    names, task = STORIES[story]
    files = [scene_files_path(scene_files, n) for n in names]
    seen = []
    workdir = tmp_path / "job42"
    res = run_job("my question", files, workdir, on_step=seen.append)

    assert res.status == "done" and res.job_id == "job42" and res.trace.job_id == "job42"
    assert res.trace.task.value == task and res.trace.question == "my question"
    assert 0.0 <= res.fusion.confidence <= 1.0 and res.fusion.confidence_label != "Conflict"
    assert (
        [s.step_id for s in seen]
        == [s.step_id for s in res.trace.steps]
        == [p.step_id for p in res.trace.plan]
    )
    assert res.trace.adapter_id == "lora-v1" and res.layers == [] and res.downloads == {}

    for p in all_paths(json.loads(res.model_dump_json())):
        assert Path(p).is_absolute() and str(workdir) in p and Path(p).exists(), p
    for name, m in res.fusion.display_masks.items():
        with rasterio.open(m.path) as src:
            assert (src.height, src.width) == (512, 512)
            assert int(np.count_nonzero(src.read(1))) == m.pixels, name
    for f in ("ingest/profile.json", "fusion.json", "trace.json"):
        assert (workdir / f).exists()


def scene_files_path(scene_files, name):
    return next(p for p in scene_files.values() if p.name == name)


def test_story_a_layers_and_facts(tmp_path, scene_files):
    res = run_job("q", [scene_files["t1"], scene_files["t2"]], tmp_path / "j")
    assert {"change", "water_loss", "water_T1", "water_T2", "region"} <= set(
        res.fusion.display_masks
    )
    assert {"buildings_new", "buildings_T2"} <= set(res.fusion.display_detections)
    assert res.fusion.facts["new_buildings"] == 6 and res.fusion.facts["water_loss_m2"] > 0
    assert len(res.fusion.display_detections["buildings_new"]) == 6


def test_story_b_layers(tmp_path, scene_files):
    res = run_job("q", [scene_files["cloudy"], scene_files["sar"]], tmp_path / "j")
    assert {"radar_only", "water_sar", "builtup", "cloud"} <= set(res.fusion.display_masks)
    assert any("radar" in r for r in res.fusion.reasons)
    imgs = {i.role: i for i in res.trace.sensor_profile.images}
    assert imgs["optical"].cloud_pct > 0 and imgs["sar"].hand_mask_path is not None


def test_conflict_variant(tmp_path, scene_files, monkeypatch):
    monkeypatch.setenv("SATQUERY_FAKE_VARIANT", "conflict")
    res = run_job("q", [scene_files["t1"], scene_files["t2"]], tmp_path / "j")
    assert res.fusion.confidence_label == "Conflict" and res.fusion.confidence <= 0.5
    assert res.fusion.conflicts and res.fusion.conflicts[0].between == ("s8", "s6")


def test_real_pipeline_not_wired_yet(tmp_path, scene_files, monkeypatch):
    monkeypatch.setenv("SATQUERY_STUB_JOB", "0")
    res = run_job("q", [scene_files["single"]], tmp_path / "j")
    assert res.status == "failed" and "BE-11" in res.message
