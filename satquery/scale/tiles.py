"""Scale Bridge (master F4.2): run any model on any scene at the right resolution.

A scene can be 20000 px across; a model wants a ~1024 px crop at roughly the
ground resolution it was trained on. This module plans those crops, reads them,
and maps whatever the model returns back onto the reference grid, because every
box and mask in a ToolResult is in reference-grid pixels (master F3.2).

Two coordinate spaces. Confusing them is the only real bug you can write here:

    ref    pixels on the reference grid -- what ToolResult must contain
    model  pixels in the resampled crop handed to the model

    model = ref * Tile.scale

Resizing rule (F4.2): factor = image.pixel_m / target, where target is the
nearest end of the tool's native_pixel_m range. An image already inside the
range is not resized. pixel_m is None in benchmark mode, and native_pixel_m is
None for tools with no native scale (change_diff) -- both mean "tile only".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from satquery.common.geo import read_mask, write_mask
from satquery.contracts import Detection, ErrorCode, ImageInfo, MaskRef, ToolError

__all__ = [
    "Tile",
    "resize_factor",
    "plan_tiles",
    "read_tile",
    "boxes_to_scene",
    "scene_box_to_tile",
    "merge_detections",
    "stitch_array",
    "stitch_mask",
]


@dataclass(frozen=True)
class Tile:
    """One crop. Internal to P1 (F4.2): where it sits on the ref grid, and its scale."""

    image: ImageInfo
    index: int
    x0: int
    y0: int
    x1: int  # half-open, ref grid
    y1: int
    scale: float  # ref px * scale = model px

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def model_width(self) -> int:
        return max(1, int(round(self.width * self.scale)))

    @property
    def model_height(self) -> int:
        return max(1, int(round(self.height * self.scale)))


def resize_factor(image: ImageInfo, native_pixel_m: tuple[float, float] | None) -> float:
    """ref px -> model px multiplier. 1.0 means "hand the model the raw pixels"."""
    if image.pixel_m is None or native_pixel_m is None:
        return 1.0  # benchmark mode, or a tool with no native scale
    lo, hi = sorted(float(v) for v in native_pixel_m)
    target = min(max(float(image.pixel_m), lo), hi)
    if target <= 0:
        raise ToolError(ErrorCode.INTERNAL, f"bad native_pixel_m {native_pixel_m!r}")
    return float(image.pixel_m) / target


def _starts(total: int, size: int, stride: int) -> list[int]:
    """Window origins that cover 0..total with no gaps."""
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] + size < total:
        starts.append(total - size)  # last tile snaps to the edge
    return starts


def plan_tiles(
    image: ImageInfo,
    native_pixel_m: tuple[float, float] | None,
    tile_px: int = 1024,
    overlap_px: int = 64,
    region: MaskRef | None = None,
) -> list[Tile]:
    """Cover `image` with crops that are at most tile_px on a side *in model space*.

    tile_px and overlap_px are model-space sizes, so the ref-grid window shrinks
    as the scale factor grows. That keeps memory bounded when a coarse scene has
    to be upsampled to reach a tool's native resolution.

    With a region, tiles that do not touch it are dropped -- this is what makes
    "the lake plus 50 m of shore" fast.
    """
    if image.rgb8_path is None:
        raise ToolError(
            ErrorCode.BAD_INPUT,
            f"{image.image_id} has no rgb8_path; model tools never read prepared_path",
        )
    if tile_px <= 0:
        raise ToolError(ErrorCode.BAD_INPUT, f"tile_px must be positive, got {tile_px}")

    scale = resize_factor(image, native_pixel_m)
    ref_tile = max(1, int(math.floor(tile_px / scale)))
    ref_overlap = max(0, min(int(math.floor(overlap_px / scale)), ref_tile - 1))
    stride = max(1, ref_tile - ref_overlap)

    region_arr = None
    if region is not None:
        region_arr = read_mask(region)
        if region_arr.shape != (image.height, image.width):
            raise ToolError(
                ErrorCode.INTERNAL,
                f"region {region_arr.shape} is not on the reference grid "
                f"({image.height}, {image.width})",
            )

    tiles: list[Tile] = []
    for y0 in _starts(image.height, ref_tile, stride):
        for x0 in _starts(image.width, ref_tile, stride):
            x1 = min(x0 + ref_tile, image.width)
            y1 = min(y0 + ref_tile, image.height)
            if region_arr is not None and not region_arr[y0:y1, x0:x1].any():
                continue
            tiles.append(Tile(image=image, index=len(tiles), x0=x0, y0=y0, x1=x1, y1=y1, scale=scale))
    return tiles


def read_tile(tile: Tile) -> np.ndarray:
    """(H, W, 3) uint8 RGB from rgb8_path, already resampled to model space.

    Area averaging when shrinking, bilinear when enlarging -- rasterio resamples
    during the windowed read, so the full scene is never held in memory.
    """
    path = tile.image.rgb8_path
    resampling = Resampling.average if tile.scale < 1.0 else Resampling.bilinear
    with rasterio.open(path) as src:
        indexes = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
        arr = src.read(
            indexes=indexes,
            window=Window(tile.x0, tile.y0, tile.width, tile.height),
            out_shape=(3, tile.model_height, tile.model_width),
            resampling=resampling,
        )
    return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)).astype(np.uint8))


def scene_box_to_tile(tile: Tile, box_ref_px) -> tuple[float, float, float, float]:
    """Inverse of boxes_to_scene. Mostly used to check the round trip."""
    x0, y0, x1, y1 = (float(v) for v in box_ref_px)
    return (
        (x0 - tile.x0) * tile.scale,
        (y0 - tile.y0) * tile.scale,
        (x1 - tile.x0) * tile.scale,
        (y1 - tile.y0) * tile.scale,
    )


def boxes_to_scene(tile: Tile, boxes_tile_px) -> list[tuple[int, int, int, int]]:
    """Model-space boxes -> [x0, y0, x1, y1] on the full reference grid.

    Clipped to the scene. Boxes that collapse to nothing after rounding are
    dropped rather than returned as zero-area rectangles.
    """
    w, h = tile.image.width, tile.image.height
    out: list[tuple[int, int, int, int]] = []
    for box in boxes_tile_px:
        bx0, by0, bx1, by1 = (float(v) for v in box)
        x0 = tile.x0 + bx0 / tile.scale
        y0 = tile.y0 + by0 / tile.scale
        x1 = tile.x0 + bx1 / tile.scale
        y1 = tile.y0 + by1 / tile.scale
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        ix0 = int(round(min(max(x0, 0), w)))
        iy0 = int(round(min(max(y0, 0), h)))
        ix1 = int(round(min(max(x1, 0), w)))
        iy1 = int(round(min(max(y1, 0), h)))
        if ix1 > ix0 and iy1 > iy0:
            out.append((ix0, iy0, ix1, iy1))
    return out


def merge_detections(dets: list[Detection], iou: float) -> list[Detection]:
    """Class-aware NMS across tiles: the same object seen in two overlapping
    tiles becomes one detection, but a building overlapping a road stays two."""
    if not dets:
        return []

    by_label: dict[str, list[Detection]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)

    kept: list[Detection] = []
    for group in by_label.values():
        group = sorted(group, key=lambda d: d.score, reverse=True)
        boxes = np.asarray([d.bbox_px for d in group], dtype=float)
        areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
            boxes[:, 3] - boxes[:, 1], 0, None
        )
        alive = np.ones(len(group), dtype=bool)
        for i in range(len(group)):
            if not alive[i]:
                continue
            kept.append(group[i])
            alive[i] = False
            if not alive.any():
                break
            xx0 = np.maximum(boxes[i, 0], boxes[:, 0])
            yy0 = np.maximum(boxes[i, 1], boxes[:, 1])
            xx1 = np.minimum(boxes[i, 2], boxes[:, 2])
            yy1 = np.minimum(boxes[i, 3], boxes[:, 3])
            inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
            union = areas[i] + areas - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            alive &= ious <= iou

    kept.sort(key=lambda d: d.score, reverse=True)
    return kept


def _resize_nearest(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour back to ref space. Nearest on purpose: it keeps a 0/1
    mask 0/1, where bilinear would invent values that are neither."""
    h, w = shape
    if a.shape == (h, w):
        return a
    yi = np.clip((np.arange(h) * a.shape[0] / h).astype(int), 0, a.shape[0] - 1)
    xi = np.clip((np.arange(w) * a.shape[1] / w).astype(int), 0, a.shape[1] - 1)
    return a[yi][:, xi]


def stitch_array(tiles: list[Tile], tile_arrays, ref: ImageInfo, dtype=np.uint8) -> np.ndarray:
    """Per-tile 2-D model-space arrays -> one (ref.height, ref.width) array.

    Overlaps take the maximum, so an object detected in one tile is not erased
    by the neighbour that missed it. Used for probability maps too (ML-05).
    """
    out = np.zeros((ref.height, ref.width), dtype=dtype)
    tiles = list(tiles)
    arrays = list(tile_arrays)
    if len(tiles) != len(arrays):
        raise ToolError(
            ErrorCode.INTERNAL, f"{len(tiles)} tiles but {len(arrays)} tile arrays"
        )
    for tile, arr in zip(tiles, arrays):
        a = np.asarray(arr)
        if a.ndim != 2:
            raise ToolError(ErrorCode.INTERNAL, f"tile array must be 2-D, got {a.shape}")
        block = _resize_nearest(a, (tile.height, tile.width)).astype(dtype, copy=False)
        view = out[tile.y0 : tile.y1, tile.x0 : tile.x1]
        np.maximum(view, block, out=view)
    return out


def stitch_mask(
    tiles: list[Tile],
    tile_masks,
    ref: ImageInfo,
    out: Path,
    meaning: str,
) -> MaskRef:
    """Per-tile boolean masks -> one uint8 0/1 GeoTIFF on the ref grid (F3.2)."""
    arr = stitch_array(tiles, [np.asarray(m) > 0 for m in tile_masks], ref, dtype=np.uint8)
    return write_mask(arr, ref, Path(out), meaning)
