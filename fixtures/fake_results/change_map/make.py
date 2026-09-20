"""Generate the fake change masks and results for change_map and change_diff (ML-02).

    python fixtures/fake_results/change_map/make.py

Geometry is master F3.7 (512x512 at 2 m, EPSG:32643):
    T1 lake = ellipse centre (256, 380) radii (120, 70)
    T2 lake = ellipse centre (256, 380) radii (100, 55)
    six new 12x12 buildings on the shore the shrinking lake exposed

change_map fires on the six buildings only. change_diff, being a blunt pixel
difference, also flags the ring of shoreline the lake uncovered -- which is
exactly why it is the fallback and not the primary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from satquery.common.geo import write_raster

HERE = Path(__file__).resolve().parent
DIFF = HERE.parent / "change_diff"
REF = HERE.parents[1] / "scenes" / "pair_T1.tif"

H = W = 512
PIXEL_M = 2.0
BOX = 12
BUILDINGS = [(140, 358), (140, 372), (140, 386), (360, 358), (360, 372), (360, 386)]
PROB_INSIDE = 199  # 199/255 = 0.78, the score the handover specifies

yy, xx = np.mgrid[0:H, 0:W]


def ellipse(cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


buildings = np.zeros((H, W), bool)
for x0, y0 in BUILDINGS:
    buildings[y0 : y0 + BOX, x0 : x0 + BOX] = True

shoreline = ellipse(256, 380, 120, 70) & ~ellipse(256, 380, 100, 55)

cmap_mask = buildings
cdiff_mask = buildings | shoreline
prob = np.where(cmap_mask, PROB_INSIDE, 0).astype(np.uint8)

assert REF.exists(), f"missing {REF} -- run fixtures/make_fixtures.py first"
assert int(cmap_mask.sum()) == 864, f"expected 864 changed px, got {int(cmap_mask.sum())}"

DIFF.mkdir(parents=True, exist_ok=True)
write_raster(cmap_mask.astype(np.uint8), REF, HERE / "change.tif", dtype="uint8")
write_raster(prob, REF, HERE / "change_prob.tif", dtype="uint8")
write_raster(cdiff_mask.astype(np.uint8), REF, DIFF / "change.tif", dtype="uint8")


def mask_ref(filename, arr, meaning):
    px = int(np.count_nonzero(arr))
    return {"path": filename, "meaning": meaning, "pixels": px,
            "area_m2": px * PIXEL_M**2}


def result(tool, model_id, masks, pixels, score, runtime):
    return {"tool": tool, "model_id": model_id, "status": "ok", "text": None,
            "detections": [], "masks": masks,
            "metrics": {"changed_pixels": float(pixels),
                        "changed_area_m2": pixels * PIXEL_M**2},
            "score": score, "warnings": [], "runtime_s": runtime, "source": "fake"}


cmap_px = int(cmap_mask.sum())
cdiff_px = int(cdiff_mask.sum())
score = round(float(prob[cmap_mask].mean() / 255.0), 2)

(HERE / "default.json").write_text(json.dumps(result(
    "change_map", "ChangeFormer-LEVIR-CD",
    # change_prob.tif is still written to disk, but stays out of masks{}
    # until the contract change request lands: tests/contract asserts
    # every mask raster is 0/1, which a probability map cannot be.
    {"change": mask_ref("change.tif", cmap_mask, "change")},
    cmap_px, score, 2.6), indent=2) + "\n")

(DIFF / "default.json").write_text(json.dumps(result(
    "change_diff", "rgb-difference-ksigma",
    {"change": mask_ref("change.tif", cdiff_mask, "change")},
    cdiff_px, None, 0.2), indent=2) + "\n")

print(f"change_map : {cmap_px} px, {cmap_px * PIXEL_M**2:.0f} m2, score {score}")
print(f"change_diff: {cdiff_px} px, {cdiff_px * PIXEL_M**2:.0f} m2 "
      f"({cdiff_px - cmap_px} px of exposed shoreline)")
