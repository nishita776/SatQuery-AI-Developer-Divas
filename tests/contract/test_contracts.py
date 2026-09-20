"""Every contract round-trips through JSON; the JobContext resolver behaves as documented."""

from pathlib import Path

import pytest

from satquery.contracts import (
    CONTRACT_VERSION,
    Conflict,
    Detection,
    ErrorCode,
    FusionResult,
    ImageInfo,
    JobContext,
    JobResult,
    Layer,
    MaskRef,
    ParamSpec,
    SensorProfile,
    TaskType,
    ToolCall,
    ToolCard,
    ToolError,
    ToolResult,
    Trace,
    TraceStep,
)

FAKE_JOBS = Path(__file__).resolve().parents[2] / "fixtures" / "fake_jobs"


def roundtrip(obj):
    again = type(obj).model_validate_json(obj.model_dump_json())
    assert again == obj
    return again


def test_contract_version():
    assert CONTRACT_VERSION == "1.0"


def test_small_models_roundtrip():
    mask = MaskRef(path=Path("a.tif"), meaning="water", pixels=10, area_m2=40.0)
    det = Detection(label="building", score=0.9, bbox_px=(1, 2, 3, 4))
    for obj in (
        mask,
        det,
        ParamSpec(type="float", default=0.5, min=0.0, max=1.0),
        ToolCall(step_id="s1", tool="detector", inputs={"image": "T1"}, params={"prompt": "x"}),
        ToolResult(
            step_id="s1",
            tool="detector",
            model_id="m",
            status="ok",
            detections=[det],
            masks={"water": mask},
            metrics={"count": 1.0},
            score=0.5,
        ),
        Conflict(between=("s8", "s6"), description="d"),
        Layer(
            layer_id="water_T1",
            name="Water",
            kind="image",
            url="/x.png",
            style="water",
            corners=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        ),
        ToolError(ErrorCode.BAD_INPUT, "x").code,
    ):
        if hasattr(obj, "model_dump_json"):
            roundtrip(obj)


@pytest.mark.parametrize("story", ["story_a", "story_b", "story_c"])
def test_big_models_roundtrip(story):
    res = JobResult.model_validate_json((FAKE_JOBS / story / "result.json").read_text())
    roundtrip(res)
    roundtrip(res.fusion)
    roundtrip(res.trace)
    roundtrip(res.trace.sensor_profile)
    for im in res.trace.sensor_profile.images:
        roundtrip(im)
    for s in res.trace.steps:
        assert isinstance(s, TraceStep)
    assert isinstance(res.trace.task, TaskType) and isinstance(res.fusion, FusionResult)
    assert isinstance(res.trace, Trace) and isinstance(res.trace.sensor_profile, SensorProfile)
    assert isinstance(res.trace.sensor_profile.images[0], ImageInfo)


def test_tool_error_message():
    e = ToolError(ErrorCode.NEEDS_INPUT, "I need a second image")
    assert (
        e.code is ErrorCode.NEEDS_INPUT
        and "NEEDS_INPUT" in str(e)
        and e.message == "I need a second image"
    )


def test_context_resolve(make_ctx):
    ctx = make_ctx()
    ctx.variables["target"] = "water body"
    mask = MaskRef(path=Path("r.tif"), meaning="region", pixels=1)
    ctx.results["s2"] = ToolResult(
        step_id="s2", tool="select_region", model_id="m", status="ok", masks={"region": mask}
    )
    assert isinstance(ctx.resolve("T1"), ImageInfo) and ctx.resolve("T2").role == "T2"
    assert ctx.resolve("img_0").role == "T1"
    assert ctx.resolve("$target") == "water body"
    assert ctx.resolve("$s2").step_id == "s2"
    assert ctx.resolve("$s2.masks.region") is mask
    with pytest.raises(KeyError):
        ctx.resolve("$s9")
    assert ctx.step_dir("s3").is_dir() and ctx.step_dir("s3") == ctx.workdir / "steps" / "s3"
    assert isinstance(ctx, JobContext) and ToolCard is not None
