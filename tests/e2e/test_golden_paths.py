"""The seven golden-path tests of master F7. They describe the REAL pipeline, so they are marked
xfail until BE-11 (with the stub in place a few of them happen to pass; that is fine: not strict).
When BE-11 is done: delete the `pytestmark` line and make them all pass in fake mode."""

import pytest

from satquery.agent.job import run_job
from satquery.contracts import JobResult

pytestmark = pytest.mark.xfail(reason="real pipeline lands in BE-11", strict=False)


def _run(question, files, tmp_path, **kw) -> JobResult:
    return run_job(question, files, tmp_path / "job", **kw)


def test_story_a_encroachment(tmp_path, scene_files):
    r = _run(
        "Has anything been built on the southern lake shore since last year?",
        [scene_files["t1"], scene_files["t2"]],
        tmp_path,
    )
    assert r.status == "done" and r.trace.task.value == "CHANGE_GROUNDING"
    ran = [s.step_id for s in r.trace.steps]
    assert ran == [p.step_id for p in r.trace.plan if p.step_id in ran]  # steps run in plan order
    assert r.trace.adapter_id
    assert 0 <= r.fusion.confidence <= 1
    assert {"change"} <= set(
        r.fusion.display_masks
    ) and "buildings_new" in r.fusion.display_detections


def test_story_b_cloud_sar(tmp_path, scene_files):
    r = _run(
        "Use both images to identify built-up and water-covered areas.",
        [scene_files["cloudy"], scene_files["sar"]],
        tmp_path,
    )
    optical = next(i for i in r.trace.sensor_profile.images if i.role == "optical")
    assert optical.cloud_pct > 0
    assert "sar_water" in {s.tool for s in r.trace.steps}
    assert any("radar" in x.lower() for x in r.fusion.reasons)
    assert "radar_only" in r.fusion.display_masks


def test_story_c_vqa(tmp_path, scene_files):
    r = _run("What is the main land cover in this image?", [scene_files["single"]], tmp_path)
    tools = {s.tool for s in r.trace.steps}
    assert "rs_vlm" in tools and tools <= {"rs_vlm", "index_mask"}
    assert not tools & {"detector", "change_map", "change_diff"}


def test_needs_input(tmp_path, scene_files):
    r = _run("What changed?", [scene_files["single"]], tmp_path)
    assert r.status == "needs_input" and "second image" in r.message


def test_benchmark_png(tmp_path, scene_files):
    ok = _run("What is in this image?", [scene_files["png"]], tmp_path, input_mode="benchmark")
    assert ok.status == "done"
    bad = run_job(
        "What is in this image?", [scene_files["png"]], tmp_path / "job2", input_mode="geotiff"
    )
    assert bad.status == "failed" and "benchmark" in bad.message


def test_conflict_flag(tmp_path, scene_files, monkeypatch):
    monkeypatch.setenv("SATQUERY_MODE", "fake")
    monkeypatch.setenv("SATQUERY_FAKE_VARIANT", "conflict")
    r = _run(
        "Has anything been built on the southern lake shore since last year?",
        [scene_files["t1"], scene_files["t2"]],
        tmp_path,
    )
    assert r.fusion.confidence_label == "Conflict" and r.fusion.conflicts


def test_contract_all_tools_and_cards():
    from satquery.agent.registry import load_registry

    tools = load_registry()
    assert tools and all(t.card.tool == n for n, t in tools.items())
