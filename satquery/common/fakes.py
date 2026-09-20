"""Fake tool results: instant, no GPU, no models (fake mode; also the cache-miss fallback).

Files live in fixtures/fake_results/<tool>/ . Lookup order (first hit wins):

    {key}_{role}_{v}, {key}_{v}, {key}_{role}, {key},
    default_{role}_{v}, default_{v}, default_{role}, default

key  = slug of params["mode"], else params["prompt"], else (extension, see below) params["index"]
       or params["hint"], else "".
role = role of the main input image (`image`, else `t2`, else the first image input).
v    = SATQUERY_FAKE_VARIANT (e.g. "conflict"); the variant entries are skipped when it is unset.

The JSON is a ToolResult without step_id. Mask files sit next to the JSON and are referenced by
relative paths; they are copied into the step folder and the paths are rewritten.

NOTE (extension of master F3.7): `index` and `hint` are used as key when there is no mode/prompt.
Without this the handover's own fake names (ndvi.json, south_T1.json) could never be found.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from satquery.common.paths import fake_results_dir
from satquery.contracts import ErrorCode, ImageInfo, JobContext, ToolCall, ToolError, ToolResult


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _image_or_none(ref: str, ctx: JobContext) -> ImageInfo | None:
    try:
        obj = ctx.resolve(ref)
    except (KeyError, AttributeError):
        return None
    return obj if isinstance(obj, ImageInfo) else None


def _main_role(call: ToolCall, ctx: JobContext) -> str:
    for key in ("image", "t2"):
        if key in call.inputs and (im := _image_or_none(call.inputs[key], ctx)):
            return im.role
    for ref in call.inputs.values():
        if im := _image_or_none(ref, ctx):
            return im.role
    return ""


def _key(call: ToolCall) -> str:
    for name in ("mode", "prompt", "index", "hint"):
        v = call.params.get(name)
        if v:
            return slug(str(v))
    return ""


def candidates(call: ToolCall, ctx: JobContext, variant: str) -> list[str]:
    role = _main_role(call, ctx)
    out: list[str] = []
    for key in (_key(call), "default"):
        if not key:
            continue
        for parts in ((key, role, variant), (key, variant), (key, role), (key,)):
            name = "_".join(p for p in parts if p)
            if name not in out:
                out.append(name)
    return out


def load_fake(tool, call: ToolCall, ctx: JobContext) -> ToolResult:
    import os

    name = tool.card.tool
    folder = fake_results_dir() / name
    variant = os.environ.get("SATQUERY_FAKE_VARIANT", "")
    tried = candidates(call, ctx, variant)
    for cand in tried:
        f = folder / f"{cand}.json"
        if f.exists():
            break
    else:
        raise ToolError(
            ErrorCode.INTERNAL, f"no fake result for '{name}' in {folder}; tried {tried}"
        )

    data = json.loads(f.read_text())
    data["step_id"] = call.step_id
    data.setdefault("tool", name)
    data.setdefault("model_id", tool.card.model_id)
    data["source"] = "fake"
    step_dir = ctx.step_dir(call.step_id)
    for m in data.get("masks", {}).values():
        dst = step_dir / Path(m["path"]).name
        shutil.copy2(folder / m["path"], dst)
        m["path"] = str(dst)
    return ToolResult.model_validate(data)
