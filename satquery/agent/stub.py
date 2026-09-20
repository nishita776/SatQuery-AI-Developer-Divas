"""Stub run_job (Day 0 -> BE-11): returns a realistic JobResult from fixtures/fake_jobs/story_{a,b,c}.

    2 optical images -> story A (encroachment)      optical + SAR -> story B (cloud -> SAR)
    anything else    -> story C (single-image VQA)

SATQUERY_FAKE_VARIANT=conflict on story A returns the amber "Conflict" version.
SATQUERY_STUB_DELAY=<seconds> sleeps between steps so the UI's progress timeline is visible.
Switch the stub off with SATQUERY_STUB_JOB=0 (BE-11 makes that the default).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import rasterio
from rasterio.errors import NotGeoreferencedWarning

from satquery.common.paths import fixtures_dir
from satquery.contracts import InputMode, JobResult, TraceStep

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
PATH_KEYS = {"path", "prepared_path", "rgb8_path", "cloud_mask_path", "hand_mask_path"}


def _is_sar(path: Path) -> bool:
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            sensor = tags.get("SATQUERY_SENSOR", "")
            return (
                sensor.upper().startswith(("RISAT", "SENTINEL-1"))
                or "SAR_BAND" in tags
                or (src.count == 1 and src.dtypes[0].startswith("float"))
            )
    except Exception:
        return False


def pick_story(files: list[Path]) -> str:
    n_sar = sum(_is_sar(f) for f in files)
    if len(files) == 2 and n_sar == 0:
        return "a"
    if len(files) == 2 and n_sar == 1:
        return "b"
    return "c"


def _rewrite_paths(node, base: Path):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in PATH_KEYS and isinstance(v, str) and not Path(v).is_absolute():
                node[k] = str(base / v)
            else:
                _rewrite_paths(v, base)
    elif isinstance(node, list):
        for v in node:
            _rewrite_paths(v, base)


def run_stub_job(
    question: str,
    files: list[Path],
    workdir: Path,
    input_mode: InputMode = "geotiff",
    on_step: Callable[[TraceStep], None] | None = None,
) -> JobResult:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    story = pick_story([Path(f) for f in files])
    src = fixtures_dir() / "fake_jobs" / f"story_{story}"

    variant = os.environ.get("SATQUERY_FAKE_VARIANT", "")
    result_file = src / "result.json"
    if variant == "conflict" and (src / "result_conflict.json").exists():
        result_file = src / "result_conflict.json"
    data = json.loads(result_file.read_text())

    # copy the story's files (rgb8 copies, masks) into the workdir, keeping relative layout
    for p in src.rglob("*"):
        if p.is_file() and not p.name.startswith("result"):
            dst = workdir / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    # the "prepared" images: copies of the fixture scenes the story was built from
    for im in data["trace"]["sensor_profile"]["images"]:
        dst = workdir / im["prepared_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixtures_dir() / "scenes" / im["file"], dst)

    _rewrite_paths(data, workdir)
    data["job_id"] = data["trace"]["job_id"] = workdir.name
    data["trace"]["question"] = question or data["trace"]["question"]
    data["trace"]["sensor_profile"]["input_mode"] = input_mode
    result = JobResult.model_validate(data)

    delay = float(os.environ.get("SATQUERY_STUB_DELAY", "0"))
    for step in result.trace.steps:
        if delay:
            time.sleep(delay)
        if on_step:
            on_step(step)

    # same files the real pipeline writes (master F3.1)
    (workdir / "ingest" / "profile.json").write_text(
        result.trace.sensor_profile.model_dump_json(indent=2)
    )
    (workdir / "fusion.json").write_text(result.fusion.model_dump_json(indent=2))
    (workdir / "trace.json").write_text(result.trace.model_dump_json(indent=2))
    return result
