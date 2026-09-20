"""Precomputed tool results (cache mode). Layout (master F3.7):

    cache/<step_key>/result.json     ToolResult with paths relative to this folder
    cache/<step_key>/<mask_name>.tif the mask files
    cache/manifest.json              {scene: {step_key: {tool, step_id}}}  (only for humans)

step_key = sha1(json.dumps({"tool", "params" (sorted), "inputs": {name: fingerprint}})).
The key is computed the same way by everyone, so P2's controller finds P1's cached results
without any coordination.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from satquery.common.geo import sha1_file
from satquery.common.paths import cache_dir
from satquery.contracts import ImageInfo, JobContext, MaskRef, ToolCall, ToolResult

log = logging.getLogger("satquery.common")


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _fingerprint(ref: str, ctx: JobContext) -> str:
    try:
        obj = ctx.resolve(ref)
    except (KeyError, AttributeError):
        return "unresolved:" + ref
    if isinstance(obj, ImageInfo):
        return obj.sha1  # hash of the ORIGINAL upload, stable between runs
    if isinstance(obj, MaskRef):
        return sha1_file(obj.path)
    if isinstance(obj, ToolResult):
        dets = [d.model_dump(mode="json") for d in obj.detections]
        dets.sort(key=lambda d: json.dumps(d, sort_keys=True))
        return _sha1_text(json.dumps(dets, sort_keys=True))
    return _sha1_text(json.dumps(obj, sort_keys=True, default=str))


def step_key(tool, call: ToolCall, ctx: JobContext) -> str:
    name = tool.card.tool if hasattr(tool, "card") else str(tool)
    payload = {
        "tool": name,
        "params": call.params,
        "inputs": {k: _fingerprint(v, ctx) for k, v in sorted(call.inputs.items())},
    }
    return _sha1_text(json.dumps(payload, sort_keys=True, default=str))


def save_cached(tool, call: ToolCall, ctx: JobContext, res: ToolResult) -> Path:
    key = step_key(tool, call, ctx)
    d = cache_dir() / key
    d.mkdir(parents=True, exist_ok=True)
    data = res.model_dump(mode="json")
    for name, m in res.masks.items():
        fname = f"{name}.tif"
        dst = d / fname
        if Path(m.path).resolve() != dst.resolve():
            shutil.copy2(m.path, dst)
        data["masks"][name]["path"] = fname
    (d / "result.json").write_text(json.dumps(data, indent=2))
    scene = os.environ.get("SATQUERY_SCENE")
    if scene:
        mf = cache_dir() / "manifest.json"
        manifest = json.loads(mf.read_text()) if mf.exists() else {}
        manifest.setdefault(scene, {})[key] = {"tool": res.tool, "step_id": call.step_id}
        mf.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return d


def load_cached(tool, call: ToolCall, ctx: JobContext) -> ToolResult | None:
    d = cache_dir() / step_key(tool, call, ctx)
    f = d / "result.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    data["step_id"] = call.step_id
    data["source"] = "cache"
    step_dir = ctx.step_dir(call.step_id)
    for m in data.get("masks", {}).values():
        dst = step_dir / Path(m["path"]).name
        shutil.copy2(d / m["path"], dst)
        m["path"] = str(dst)
    return ToolResult.model_validate(data)
