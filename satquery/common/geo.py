"""Shared geo helpers. Everyone uses these; nobody re-implements them (master F7 boundary rules).

All rasters are GeoTIFFs on disk. Masks are uint8 0/1, LZW-compressed, on the reference grid.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy import ndimage as ndi

from satquery.contracts import ErrorCode, ImageInfo, MaskRef, ToolError

# PNG/JPEG in benchmark mode have no georeferencing; that is expected, not a problem.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

BENCHMARK_DEG_PER_PX = 1e-5  # benchmark mode: fake lon/lat so the map still works (master F6.3)


# ------------------------------------------------------------------ raster I/O
def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Return (bands, H, W) array and the rasterio profile."""
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def write_raster(array: np.ndarray, like_path: Path, path: Path, dtype=None) -> Path:
    """Write `array` ((H, W) or (B, H, W)) with the CRS + transform of `like_path`."""
    arr = np.asarray(array)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[None]
    out_dtype = np.dtype(dtype or arr.dtype).name
    with rasterio.open(like_path) as like:
        crs, transform = like.crs, like.transform
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(out_dtype))
    return path


def write_mask(mask: np.ndarray, ref: ImageInfo, path: Path, meaning: str) -> MaskRef:
    """uint8 0/1, LZW, on the ref grid (transform/crs from ref.prepared_path). Fills pixels + area_m2."""
    arr = (np.asarray(mask) > 0).astype(np.uint8)
    if arr.shape != (ref.height, ref.width):
        raise ToolError(
            ErrorCode.INTERNAL,
            f"mask shape {arr.shape} does not match the reference grid ({ref.height}, {ref.width})",
        )
    write_raster(arr, ref.prepared_path, path, dtype="uint8")
    pixels = int(arr.sum())
    return MaskRef(
        path=Path(path), meaning=meaning, pixels=pixels, area_m2=area_m2(pixels, ref.pixel_m)
    )


def read_mask(m: MaskRef) -> np.ndarray:
    """bool (H, W)."""
    with rasterio.open(m.path) as src:
        return src.read(1) > 0


# ------------------------------------------------------------------ measurements
def area_m2(pixels, pixel_m: float | None) -> float | None:
    """Area in m². `pixels` is a pixel count or a mask array. None if not georeferenced."""
    if pixel_m is None:
        return None
    if isinstance(pixels, np.ndarray):
        pixels = int(np.count_nonzero(pixels))
    return float(pixels) * float(pixel_m) ** 2


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union, 0 if both are empty."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def mask_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a AND NOT b, e.g. lost water = T1 water minus T2 water."""
    return np.asarray(a, bool) & ~np.asarray(b, bool)


def buffer_mask(mask: np.ndarray, dist_m: float, pixel_m: float | None) -> np.ndarray:
    """Grow a mask by `dist_m` (exact disk, via the distance transform). If pixel_m is None,
    dist_m is treated as pixels (benchmark mode)."""
    mask = np.asarray(mask, bool)
    radius_px = dist_m / pixel_m if pixel_m else dist_m
    if radius_px <= 0 or not mask.any():
        return mask.copy()
    return ndi.distance_transform_edt(~mask) <= radius_px


def bbox_to_mask(bbox_px: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    """Filled box [x0, y0, x1, y1) on an (H, W) grid, clipped to the image."""
    h, w = shape
    x0, y0, x1, y1 = (int(round(v)) for v in bbox_px)
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    out = np.zeros((h, w), bool)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = True
    return out


# ------------------------------------------------------------------ coordinates
@lru_cache(maxsize=64)
def _transform(path: str, mtime_ns: int):
    with rasterio.open(path) as src:
        return src.transform


@lru_cache(maxsize=16)
def _to_wgs84(crs: str):
    from pyproj import Transformer

    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def px_to_lonlat(ref: ImageInfo, x: float, y: float) -> tuple[float, float]:
    """Pixel (x = column, y = row) -> (lon, lat). Benchmark mode (crs None): (x*1e-5, -y*1e-5)."""
    if ref.crs is None:
        return (x * BENCHMARK_DEG_PER_PX, -y * BENCHMARK_DEG_PER_PX)
    p = str(ref.prepared_path)
    transform = _transform(p, os.stat(p).st_mtime_ns)
    easting = transform.a * x + transform.b * y + transform.c  # affine math, version-agnostic
    northing = transform.d * x + transform.e * y + transform.f
    lon, lat = _to_wgs84(ref.crs).transform(easting, northing)
    return (float(lon), float(lat))


def image_corners_lonlat(ref: ImageInfo) -> list[tuple[float, float]]:
    """TL, TR, BR, BL as (lon, lat), the order MapLibre image layers expect."""
    w, h = ref.width, ref.height
    return [px_to_lonlat(ref, x, y) for x, y in ((0, 0), (w, 0), (w, h), (0, h))]


# ------------------------------------------------------------------ misc
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
