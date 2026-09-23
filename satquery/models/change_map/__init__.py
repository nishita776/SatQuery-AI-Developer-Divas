"""change_map -- bi-temporal change detection, with change_diff as its fallback.

ChangeMap uses ChangeFormer (LEVIR-CD). ChangeDiff needs no learned model at
all: it is the contract-level fallback registered in registry/change_map.yaml,
so a change question never returns nothing (master F4.6).

Statistics are accumulated tile-by-tile so a large scene never has to be held
in memory as floats, but the threshold is global -- a per-tile threshold would
make the same building look "changed" in one tile and not its neighbour.
The morphological opening runs on the stitched mask for the same reason: doing
it per tile leaves seams along every tile edge.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

from satquery.common.toolbase import BaseTool
from satquery.contracts import (ErrorCode, JobContext, ToolCall, ToolError,
                                ToolResult)

__all__ = ["ChangeMap", "ChangeDiff"]

log = logging.getLogger("satquery.models")


def _pair(call: ToolCall, ctx: JobContext):
    """Resolve t1/t2 and check they share the reference grid."""
    try:
        t1 = ctx.resolve(call.inputs["t1"])
        t2 = ctx.resolve(call.inputs["t2"])
    except KeyError as exc:
        raise ToolError(ErrorCode.BAD_INPUT, f"change tools need t1 and t2: {exc}") from exc
    if (t1.width, t1.height) != (t2.width, t2.height):
        raise ToolError(
            ErrorCode.BAD_INPUT,
            f"t1 is {t1.width}x{t1.height} but t2 is {t2.width}x{t2.height}; "
            "ingest should have put both on one reference grid",
        )
    return t1, t2


class ChangeMap(BaseTool):
    """Inputs: t1, t2; optional region.

    Outputs masks.change, masks.change_prob, metrics.changed_pixels,
    metrics.changed_area_m2 and score (the mean probability inside the change
    mask). ChangeFormer with the LEVIR-CD checkpoint; lands later in ML-05.
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError("change_map.run_real: ChangeFormer still to land")


class ChangeDiff(BaseTool):
    """Per-pixel RGB difference, thresholded at mean + k_sigma * std.

    Blunt on purpose. It will flag illumination and seasonal differences as
    well as real change -- which is why it is the fallback and ChangeFormer is
    the primary -- but it never fails to produce an answer.
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        from scipy import ndimage as ndi

        from satquery.common.geo import area_m2, read_mask, write_mask
        from satquery.scale import plan_tiles, read_tile

        t1, t2 = _pair(call, ctx)
        region = ctx.resolve(call.inputs["region"]) if "region" in call.inputs else None
        k_sigma = float(call.params.get("k_sigma", 2.5))
        open_px = int(call.params.get("open_px", 2))

        region_arr = None
        if region is not None:
            region_arr = read_mask(region)
            if region_arr.shape != (t1.height, t1.width):
                raise ToolError(
                    ErrorCode.INTERNAL,
                    f"region {region_arr.shape} is not on the reference grid "
                    f"({t1.height}, {t1.width})",
                )

        # native_pixel_m is null for change_diff, so this tiles without resampling
        tiles = plan_tiles(t1, self.card.native_pixel_m, tile_px=1024, overlap_px=0)

        def difference(tile):
            a = read_tile(tile).astype(np.float32)
            b = read_tile(replace(tile, image=t2)).astype(np.float32)
            return np.linalg.norm(b - a, axis=2)  # Euclidean distance in RGB

        # pass 1: global mean and std, over the region when one is given
        # Statistics are global, never region-local. The same pixel must get the
        # same verdict whatever region a question happened to scope to, so two
        # answers about one scene stay comparable and the threshold stays
        # explainable. The region restricts what is *reported*, below.
        if region_arr is not None and not region_arr.any():
            raise ToolError(ErrorCode.BAD_INPUT, "region selects no pixels")

        n = 0.0
        total = 0.0
        total_sq = 0.0
        for tile in tiles:
            d = difference(tile)
            n += d.size
            total += float(d.sum())
            total_sq += float(np.square(d, dtype=np.float64).sum())
        if n == 0:
            raise ToolError(ErrorCode.INTERNAL, "no pixels to compare")

        mean = total / n
        var = max(total_sq / n - mean * mean, 0.0)
        threshold = mean + k_sigma * float(np.sqrt(var))

        # pass 2: threshold into a full-scene mask
        mask = np.zeros((t1.height, t1.width), dtype=bool)
        for tile in tiles:
            mask[tile.y0 : tile.y1, tile.x0 : tile.x1] = difference(tile) > threshold
        if region_arr is not None:
            mask &= region_arr

        if open_px > 0:
            mask = ndi.binary_opening(mask, np.ones((open_px, open_px), bool))

        out = ctx.step_dir(call.step_id) / "change.tif"
        change = write_mask(mask, t1, out, "change")

        pixels = int(change.pixels)
        return self.make_result(
            call,
            masks={"change": change},
            metrics={
                "changed_pixels": float(pixels),
                "changed_area_m2": float(area_m2(pixels, t1.pixel_m) or 0.0),
                "threshold": float(threshold),
                "k_sigma": k_sigma,
            },
            warnings=[] if pixels else ["no pixels exceeded the threshold"],
        )
