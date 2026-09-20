"""Generate every fixture, identically on every run (seed 42).

    python fixtures/make_fixtures.py            # writes into fixtures/
    python fixtures/make_fixtures.py --out DIR  # writes into DIR (used by the determinism test)

Produces
    scenes/         the synthetic GeoTIFFs of master F3.7 (+ scene.png, + hand.tif)
    fake_results/   P2's fake tool results (water_mask, index_mask, sar_water, sar_builtup, select_region)
    fake_jobs/      story_a / story_b / story_c: full stub JobResults + the files they reference

All geometry (pixel coordinates are (x, y) on the 512x512 grid; boxes = top-left corner + size):
    lake T1  ellipse centre (256, 380), radii (120, 70)      lake T2  same centre, radii (100, 55)
    building 12x12 at (100, 100) in both dates; six new ones at
        (140,358) (140,372) (140,386) (360,358) (360,372) (360,386)  in T2
    shadow patch 30x30 at (420, 60) on a hill (SAR only)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from satquery.common.geo import area_m2, buffer_mask, mask_iou, sha1_file, write_raster
from satquery.contracts import (
    CONTRACT_VERSION,
    Conflict,
    Detection,
    FusionResult,
    ImageInfo,
    JobResult,
    MaskRef,
    Modality,
    PairInfo,
    SensorProfile,
    TaskType,
    ToolCall,
    ToolResult,
    Trace,
    TraceStep,
)
from satquery.fusion.format import fmt_area, fmt_int

SEED = 42
H = W = 512
PIXEL_M = 2.0
CRS = "EPSG:32643"
TRANSFORM = from_origin(500000, 2500000, PIXEL_M, PIXEL_M)

LAKE_C = (256, 380)
LAKE_T1 = (120, 70)
LAKE_T2 = (100, 55)
BUILD = 12
OLD_BUILDINGS = [(100, 100)]
NEW_BUILDINGS = [(140, 358), (140, 372), (140, 386), (360, 358), (360, 372), (360, 386)]
SHADOW = (420, 60, 30)  # x, y, size
HILL_C, HILL_R = (435, 75), 40
CLOUD_C, CLOUD_R = (150, 250), (200, 160)  # tuned so that about 35% of the scene is cloud

# band order B, G, R, NIR (digital numbers)
SPECTRA = {
    "veg": np.array([450, 800, 550, 2400], dtype=np.float32),
    "soil": np.array([800, 1050, 1400, 1900], dtype=np.float32),
    "water": np.array([900, 800, 450, 200], dtype=np.float32),
    "building": np.array([1500, 1520, 1540, 1500], dtype=np.float32),
}
BANDS = ["blue", "green", "red", "nir"]
QUESTION_A = "Has anything been built on the southern lake shore since last year?"
QUESTION_B = "Use both images to identify built-up and water-covered areas."
QUESTION_C = "What is the main land cover in this image?"

YY, XX = np.mgrid[0:H, 0:W]


# ============================================================ geometry (the "truth")
def ellipse(c, r) -> np.ndarray:
    return ((XX - c[0]) / r[0]) ** 2 + ((YY - c[1]) / r[1]) ** 2 <= 1.0


def boxes_mask(corners, size=BUILD) -> np.ndarray:
    m = np.zeros((H, W), bool)
    for x, y in corners:
        m[y : y + size, x : x + size] = True
    return m


def lake_t1() -> np.ndarray:
    return ellipse(LAKE_C, LAKE_T1)


def lake_t2() -> np.ndarray:
    return ellipse(LAKE_C, LAKE_T2)


def cloud_alpha() -> np.ndarray:
    return ndi.gaussian_filter(ellipse(CLOUD_C, CLOUD_R).astype(np.float32), 3)


def cloud_truth() -> np.ndarray:
    return cloud_alpha() > 0.5


def shadow_truth() -> np.ndarray:
    x, y, s = SHADOW
    m = np.zeros((H, W), bool)
    m[y : y + s, x : x + s] = True
    return m


# ============================================================ scene synthesis
def optical(lake, buildings, exposed, date_seed) -> np.ndarray:
    rng = np.random.default_rng(SEED)  # shared texture: unchanged pixels stay (almost) identical
    tex = ndi.gaussian_filter(rng.normal(size=(H, W)), 6)
    tex /= tex.std()
    pix = rng.normal(size=(4, H, W))
    rng_d = np.random.default_rng(SEED + date_seed)
    bands = np.zeros((4, H, W), np.float32)
    for cls, m in (
        ("veg", np.ones((H, W), bool)),
        ("soil", exposed),
        ("water", lake),
        ("building", buildings),
    ):
        bands[:, m] = SPECTRA[cls][:, None]
    bands *= 1 + 0.06 * tex + 0.012 * pix + 0.005 * rng_d.normal(size=(4, H, W))
    return bands


def add_cloud(bands: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED + 7)
    a = cloud_alpha()
    smooth = ndi.gaussian_filter(rng.normal(size=(H, W)), 25)
    smooth /= smooth.std()
    level = 9000 * (1 + 0.004 * smooth)  # clouds are smooth: very low texture
    white = np.array([9000, 9200, 9100, 8800], np.float32) / 9000
    return bands * (1 - a) + (level[None] * white[:, None, None]) * a


def sar_db(water, buildings, shadow) -> np.ndarray:
    rng = np.random.default_rng(SEED + 3)
    db = np.full((H, W), -8.0, np.float32)
    db[water] = -22.0
    db[shadow] = -21.0
    db[buildings] = 2.0
    looks = 8  # multiplicative gamma speckle
    speckle = rng.gamma(looks, 1.0 / looks, size=(H, W))
    return (db + 10 * np.log10(speckle)).astype(np.float32)


def hand_raster() -> np.ndarray:
    dist = ndi.distance_transform_edt(~lake_t1())
    hand = np.clip(dist * 0.03, 0, 10).astype(np.float32)  # 0-5 m near the shore, ramps to 10 m
    r = np.hypot(XX - HILL_C[0], YY - HILL_C[1])
    hill = r < HILL_R
    hand[hill] = (25 + 15 * (1 - r[hill] / HILL_R)).astype(np.float32)  # north-east hill: 25-40 m
    return hand


def rgb8_optical(arrs: list[np.ndarray]) -> list[np.ndarray]:
    """R,G,B stretched between the 2nd and 98th percentile, computed JOINTLY over all arrays."""
    rgbs = [a[[2, 1, 0]].astype(np.float32) for a in arrs]
    lohi = []
    for c in range(3):
        allv = np.concatenate([r[c].ravel() for r in rgbs])
        lohi.append(np.percentile(allv, [2, 98]))
    out = []
    for r in rgbs:
        o = np.stack(
            [
                np.clip((r[c] - lohi[c][0]) / (lohi[c][1] - lohi[c][0] + 1e-6), 0, 1)
                for c in range(3)
            ]
        )
        out.append((o * 255).round().astype(np.uint8))
    return out


def rgb8_sar(db: np.ndarray) -> np.ndarray:
    g = (np.clip((db + 25) / 30, 0, 1) * 255).round().astype(np.uint8)
    return np.stack([g, g, g])


# ============================================================ writers
def write_tif(path: Path, arr: np.ndarray, tags=None, descs=None) -> Path:
    arr = arr[None] if arr.ndim == 2 else arr
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=H,
        width=W,
        count=arr.shape[0],
        dtype=arr.dtype.name,
        crs=CRS,
        transform=TRANSFORM,
        compress="deflate",
        predictor=3 if arr.dtype.kind == "f" else 2,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        if tags:
            dst.update_tags(**tags)
        for i, d in enumerate(descs or [], start=1):
            dst.set_band_description(i, d)
    return path


def save_mask(folder: Path, fname: str, mask: np.ndarray, like: Path, meaning: str) -> MaskRef:
    """uint8 0/1 LZW GeoTIFF; returns a MaskRef whose path is relative to the tool folder
    (fake results) or to the story folder (when saving into story_x/fusion or story_x/ingest)."""
    write_raster(mask.astype(np.uint8), like, folder / fname, dtype="uint8")
    px = int(mask.sum())
    rel = Path(folder.name) / fname if folder.name in {"fusion", "ingest"} else Path(fname)
    return MaskRef(path=rel, meaning=meaning, pixels=px, area_m2=area_m2(px, PIXEL_M))


def dump_tool_result(folder: Path, fname: str, res: ToolResult) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / fname).write_text(res.model_dump_json(indent=2, exclude={"step_id"}) + "\n")


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ============================================================ 1. scenes
def build_scenes(out: Path) -> dict:
    sc = out / "scenes"
    sc.mkdir(parents=True, exist_ok=True)
    l1, l2 = lake_t1(), lake_t2()
    old_b = boxes_mask(OLD_BUILDINGS)
    new_b = boxes_mask(NEW_BUILDINGS)
    all_b = old_b | new_b

    t1 = optical(l1, old_b, np.zeros((H, W), bool), date_seed=1)
    t2 = optical(l2, all_b, l1 & ~l2, date_seed=2)
    cloudy = add_cloud(optical(l2, all_b, l1 & ~l2, date_seed=3))
    u16 = lambda a: np.clip(a, 0, 65535).round().astype(np.uint16)  # noqa: E731
    t1, t2, cloudy = u16(t1), u16(t2), u16(cloudy)

    def otags(date):
        return {"SATQUERY_SENSOR": "Cartosat-2S", "ACQUISITION_DATE": date}

    write_tif(sc / "pair_T1.tif", t1, otags("2024-02-11"), BANDS)
    write_tif(sc / "pair_T2.tif", t2, otags("2025-02-08"), BANDS)
    shutil.copyfile(sc / "pair_T2.tif", sc / "single.tif")
    write_tif(sc / "optical_cloudy.tif", cloudy, otags("2025-02-06"), BANDS)

    sar = sar_db(l2, all_b, shadow_truth())
    write_tif(
        sc / "sar.tif",
        sar,
        {
            "SATQUERY_SENSOR": "RISAT-2B",
            "ACQUISITION_DATE": "2025-02-06",
            "SAR_BAND": "X",
            "POLARISATION": "HH",
        },
        ["sigma0_hh"],
    )
    hand = hand_raster()
    write_tif(sc / "hand.tif", hand, {"DESCRIPTION": "synthetic HAND (m)"}, ["hand_m"])

    # PNG copy of single.tif for benchmark mode (3-band uint8)
    png = rgb8_optical([t2])[0]
    with rasterio.open(
        sc / "scene.png", "w", driver="PNG", width=W, height=H, count=3, dtype="uint8"
    ) as dst:
        dst.write(png)
    (sc / "scene.png.aux.xml").unlink(missing_ok=True)

    return dict(
        t1=t1,
        t2=t2,
        cloudy=cloudy,
        sar=sar,
        hand=hand,
        l1=l1,
        l2=l2,
        old_b=old_b,
        new_b=new_b,
        all_b=all_b,
    )


def ndwi(a):
    b = a.astype(np.float32)
    return (b[1] - b[3]) / (b[1] + b[3] + 1e-6)


def ndvi(a):
    b = a.astype(np.float32)
    return (b[3] - b[2]) / (b[3] + b[2] + 1e-6)


# ============================================================ 2. fake tool results (P2's five tools)
def build_fake_results(out: Path, S: dict) -> dict:
    fr = out / "fake_results"
    like = out / "scenes" / "pair_T2.tif"
    cloud = cloud_truth()
    info: dict = {}

    # ---- water_mask
    d = fr / "water_mask"
    for role, arr, lake, fname in (
        ("T1", S["t1"], S["l1"], "default_T1.json"),
        ("T2", S["t2"], S["l2"], "default_T2.json"),
    ):
        m = save_mask(d, f"water_{role}.tif", lake, like, "water")
        thr = round(float(threshold_otsu(ndwi(arr))), 3)
        dump_tool_result(
            d,
            fname,
            ToolResult(
                step_id="",
                tool="water_mask",
                model_id="ndwi-otsu",
                status="ok",
                masks={"water": m},
                metrics={"water_area_m2": m.area_m2, "threshold": thr, "excluded_cloud_m2": 0.0},
                runtime_s=0.2,
                source="fake",
            ),
        )
        info[f"water_{role}"] = m
    w_opt = S["l2"] & ~cloud
    m = save_mask(d, "water_optical.tif", w_opt, like, "water")
    valid = ~cloud
    thr = round(float(threshold_otsu(ndwi(S["cloudy"])[valid])), 3)
    cloud_m2 = area_m2(int(cloud.sum()), PIXEL_M)
    dump_tool_result(
        d,
        "default_optical.json",
        ToolResult(
            step_id="",
            tool="water_mask",
            model_id="ndwi-otsu",
            status="ok",
            masks={"water": m},
            metrics={"water_area_m2": m.area_m2, "threshold": thr, "excluded_cloud_m2": cloud_m2},
            runtime_s=0.2,
            source="fake",
        ),
    )
    info["water_optical"] = m

    # ---- index_mask (NDVI)
    d = fr / "index_mask"
    nd = ndvi(S["t2"])
    veg = nd > 0.3
    m = save_mask(d, "ndvi.tif", veg, like, "ndvi")
    dump_tool_result(
        d,
        "ndvi.json",
        ToolResult(
            step_id="",
            tool="index_mask",
            model_id="spectral-index-otsu",
            status="ok",
            masks={"ndvi": m},
            metrics={"mean_index": round(float(nd[veg].mean()), 3), "area_m2": m.area_m2},
            runtime_s=0.1,
            source="fake",
        ),
    )
    info["ndvi"] = (m, round(float(nd[veg].mean()), 3))

    # ---- sar_water: median filter stands in for the Lee filter; threshold -15 dB; HAND <= 15 m removes the shadow
    d = fr / "sar_water"
    dark = ndi.median_filter(S["sar"], size=5) < -15.0
    sar_w = dark & (S["hand"] <= 15.0)
    m = save_mask(d, "water.tif", sar_w, like, "water")
    dump_tool_result(
        d,
        "default.json",
        ToolResult(
            step_id="",
            tool="sar_water",
            model_id="nrsc-tile-otsu+hand",
            status="ok",
            masks={"water": m},
            metrics={"threshold_db": -15.0, "tiles_used": 4.0, "water_area_m2": m.area_m2},
            warnings=["band=X"],
            runtime_s=0.3,
            source="fake",
        ),
    )
    info["water_sar"] = m
    info["sar_water_px"] = sar_w

    # ---- sar_builtup: the 7 building squares
    d = fr / "sar_builtup"
    m = save_mask(d, "builtup.tif", S["all_b"], like, "builtup")
    dump_tool_result(
        d,
        "default.json",
        ToolResult(
            step_id="",
            tool="sar_builtup",
            model_id="sar-double-bounce-percentile",
            status="ok",
            masks={"builtup": m},
            metrics={"builtup_area_m2": m.area_m2},
            warnings=["band=X"],
            runtime_s=0.1,
            source="fake",
        ),
    )
    info["builtup"] = m

    # ---- select_region: hint=south on the T1 water-body detection
    d = fr / "select_region"
    reg = S["l1"]
    buf = buffer_mask(reg, 50, PIXEL_M)  # 25 px
    mr = save_mask(d, "region.tif", reg, like, "region")
    mb = save_mask(d, "buffered.tif", buf, like, "buffered")
    lake_det = Detection(label="water body", score=0.62, bbox_px=bbox_of(reg))
    dump_tool_result(
        d,
        "south_T1.json",
        ToolResult(
            step_id="",
            tool="select_region",
            model_id="rule-based-region-select",
            status="ok",
            masks={"region": mr, "buffered": mb},
            detections=[lake_det],
            metrics={"region_area_m2": mr.area_m2},
            runtime_s=0.05,
            source="fake",
        ),
    )
    info["region"], info["buffered"], info["lake_det"] = mr, mb, lake_det
    info["buf_px"] = buf
    return info


# ============================================================ 3. stub story folders
DET_MODEL = "IDEA-Research/grounding-dino-base"
CHG_MODEL = "changeformer-levir-cd"
VLM_MODEL = (
    "geochat-7b-4bit+lora-v1"  # P1 decides the real string; the "+lora-v1" part is what matters
)
DET_PARAMS = dict(
    box_threshold=0.35, text_threshold=0.25, nms_iou=0.5, max_detections=500, tile_px=1024
)


def confidence(model=None, agreement=None, spatial=None, quality=None) -> tuple[float, str]:
    """Master F5.5: weights 0.30/0.30/0.25/0.15, re-normalised over the terms that are present."""
    terms = [(0.30, model), (0.30, agreement), (0.25, spatial), (0.15, quality)]
    terms = [(w, v) for w, v in terms if v is not None]
    c = sum(w * v for w, v in terms) / sum(w for w, _ in terms)
    label = "High" if c >= 0.75 else "Medium" if c >= 0.5 else "Low"
    return round(c, 2), label


def _tc(step_id, tool, inputs, params=None) -> ToolCall:
    return ToolCall(step_id=step_id, tool=tool, inputs=inputs, params=params or {})


def _ts(step_id, tool, model_id, params, summary, runtime_s) -> TraceStep:
    return TraceStep(
        step_id=step_id,
        tool=tool,
        model_id=model_id,
        params=params,
        status="ok",
        summary=summary,
        runtime_s=runtime_s,
        source="fake",
    )


def _image(i, role, file, scenes, modality, bands, date, cloud_pct=0.0, **kw) -> ImageInfo:
    return ImageInfo(
        image_id=f"img_{i}",
        role=role,
        file=file,
        sha1=sha1_file(scenes / file),
        prepared_path=Path(f"ingest/img_{i}.tif"),
        rgb8_path=Path(f"ingest/img_{i}_rgb8.tif"),
        modality=modality,
        bands=bands,
        pixel_m=PIXEL_M,
        width=W,
        height=H,
        date=date,
        crs=CRS,
        cloud_pct=cloud_pct,
        **kw,
    )


def _write_story(
    folder: Path, name: str, result: JobResult, extra_result: JobResult | None = None
) -> None:
    (folder / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    if extra_result is not None:
        (folder / name).write_text(extra_result.model_dump_json(indent=2) + "\n")


def build_stories(out: Path, S: dict, fk: dict) -> None:
    sc = out / "scenes"
    like = sc / "pair_T2.tif"
    jobs = out / "fake_jobs"
    cloud = cloud_truth()
    cloud_pct = round(100 * float(cloud.mean()), 1)
    notes_ndbi = "NDBI skipped: no SWIR band on Cartosat-2S"

    # ------------------------------------------------------------ story A: encroachment
    a = jobs / "story_a"
    (a / "ingest").mkdir(parents=True, exist_ok=True)
    (a / "fusion").mkdir(parents=True, exist_ok=True)
    rgb_t1, rgb_t2 = rgb8_optical([S["t1"], S["t2"]])  # shared stretch for the pair
    write_tif(a / "ingest" / "img_0_rgb8.tif", rgb_t1)
    write_tif(a / "ingest" / "img_1_rgb8.tif", rgb_t2)

    l1, l2, new_b = S["l1"], S["l2"], S["new_b"]
    loss = l1 & ~l2
    change = ndi.binary_dilation(new_b, structure=np.ones((3, 3), bool), iterations=2)
    fm = {
        "water_T1": save_mask(a / "fusion", "water_T1.tif", l1, like, "water"),
        "water_T2": save_mask(a / "fusion", "water_T2.tif", l2, like, "water"),
        "water_loss": save_mask(a / "fusion", "water_loss.tif", loss, like, "water_loss"),
        "change": save_mask(a / "fusion", "change.tif", change, like, "change"),
        "region": save_mask(a / "fusion", "region.tif", l1, like, "region"),
    }
    w1, w2 = fm["water_T1"].area_m2, fm["water_T2"].area_m2
    lost, pct = fm["water_loss"].area_m2, round(100 * fm["water_loss"].area_m2 / w1)
    scores_t2 = [0.71, 0.68, 0.74, 0.69, 0.72, 0.66, 0.77]
    boxes_t2 = [(x, y, x + BUILD, y + BUILD) for x, y in OLD_BUILDINGS + NEW_BUILDINGS]
    det_t2 = [
        Detection(label="building", score=s, bbox_px=b)
        for s, b in zip(scores_t2, boxes_t2, strict=True)
    ]
    det_t1 = [Detection(label="building", score=0.73, bbox_px=boxes_t2[0])]
    det_new = det_t2[1:]
    facts_a = {
        "water_T1_m2": w1,
        "water_T2_m2": w2,
        "water_loss_m2": lost,
        "water_loss_pct": float(pct),
        "buildings_T1": 1.0,
        "buildings_T2": 7.0,
        "new_buildings": 6.0,
        "changed_area_m2": fm["change"].area_m2,
    }
    score_det = round(float(np.mean([d.score for d in det_t2 + det_t1 + [fk["lake_det"]]])), 2)
    model_term = round(float(np.mean([score_det, 0.83, 0.78])), 3)  # detector, change_map, rs_vlm
    conf_a, label_a = confidence(model_term, 1.0, 1.0, 1.0)
    answer_a = (
        f"Yes. 6 new structures have appeared within 50 m of the water body. "
        f"The water surface shrank by about {fmt_area(lost)} ({pct}%)."
    )
    vlm_a = "New structures are visible along the southern shore, where the water has receded."
    reasons_a = ["4 sources agree", "6 of 6 new buildings fall inside the changed area", "cloud 0%"]

    profile_a = SensorProfile(
        input_mode="geotiff",
        images=[
            _image(
                0,
                "T1",
                "pair_T1.tif",
                sc,
                Modality.OPTICAL_MX,
                BANDS,
                "2024-02-11",
                sensor="Cartosat-2S",
            ),
            _image(
                1,
                "T2",
                "pair_T2.tif",
                sc,
                Modality.OPTICAL_MX,
                BANDS,
                "2025-02-08",
                sensor="Cartosat-2S",
            ),
        ],
        pair=PairInfo(kind="bi-temporal", overlap_pct=100.0, same_crs=True, ref_image="img_0"),
        can_compute={"NDVI": True, "NDWI": True, "NDBI": False},
        notes=[notes_ndbi, "HAND not needed: no radar input"],
    )
    q = QUESTION_A
    plan_a = [
        _tc("s1", "detector", {"image": "T1"}, {"prompt": "water body", **DET_PARAMS}),
        _tc(
            "s2",
            "select_region",
            {"detections": "$s1", "image": "T1"},
            {"hint": "south", "buffer_m": 50.0, "use_water": True},
        ),
        _tc(
            "s3",
            "water_mask",
            {"image": "T1", "region": "$s2.masks.region"},
            {"index": "NDWI", "exclude_cloud": True},
        ),
        _tc(
            "s4",
            "water_mask",
            {"image": "T2", "region": "$s2.masks.region"},
            {"index": "NDWI", "exclude_cloud": True},
        ),
        _tc(
            "s5",
            "change_map",
            {"t1": "T1", "t2": "T2", "region": "$s2.masks.buffered"},
            {"threshold": 0.5},
        ),
        _tc(
            "s6",
            "detector",
            {"image": "T2", "region": "$s2.masks.buffered"},
            {"prompt": "building", **DET_PARAMS},
        ),
        _tc(
            "s7",
            "detector",
            {"image": "T1", "region": "$s2.masks.buffered"},
            {"prompt": "building", **DET_PARAMS},
        ),
        _tc(
            "s8",
            "rs_vlm",
            {"t1": "T1", "t2": "T2", "region": "$s2.masks.region"},
            {"mode": "describe_change", "question": q, "max_new_tokens": 64},
        ),
    ]
    steps_a = [
        _ts("s1", "detector", DET_MODEL, plan_a[0].params, "1 detection >= 0.35", 1.9),
        _ts(
            "s2",
            "select_region",
            "rule-based-region-select",
            plan_a[1].params,
            f"southern water body selected, {fmt_area(w1)}",
            0.1,
        ),
        _ts("s3", "water_mask", "ndwi-otsu", plan_a[2].params, f"water {fmt_area(w1)} (T1)", 0.3),
        _ts("s4", "water_mask", "ndwi-otsu", plan_a[3].params, f"water {fmt_area(w2)} (T2)", 0.3),
        _ts(
            "s5",
            "change_map",
            CHG_MODEL,
            plan_a[4].params,
            f"{fmt_int(fm['change'].pixels)} changed pixels in the shoreline zone",
            2.4,
        ),
        _ts("s6", "detector", DET_MODEL, plan_a[5].params, "7 detections >= 0.35 in region", 1.7),
        _ts("s7", "detector", DET_MODEL, plan_a[6].params, "1 detection >= 0.35 in region", 1.6),
        _ts("s8", "rs_vlm", VLM_MODEL, plan_a[7].params, "described the change", 3.1),
    ]

    def trace_a(vlm_text: str) -> Trace:
        return Trace(
            job_id="STORY",
            question=q,
            parsed_query={
                "task": "CHANGE_GROUNDING",
                "target": "water body",
                "place_hint": "south",
                "looking_for": "building",
                "question": q,
            },
            task=TaskType.CHANGE_GROUNDING,
            sensor_profile=profile_a,
            plan=plan_a,
            steps=steps_a,
            skipped=[notes_ndbi],
            adapter_id="lora-v1",
            versions={
                "git": "stub",
                "contract": CONTRACT_VERSION,
                "rs_vlm": VLM_MODEL,
                "detector": DET_MODEL,
                "change_map": CHG_MODEL,
            },
        )

    display_masks_a = dict(fm)
    display_dets_a = {"buildings_new": det_new, "buildings_T2": det_t2}
    fusion_a = FusionResult(
        answer=answer_a,
        model_description=vlm_a,
        facts=facts_a,
        confidence=conf_a,
        confidence_label=label_a,
        reasons=reasons_a,
        display_masks=display_masks_a,
        display_detections=display_dets_a,
    )
    res_a = JobResult(job_id="STORY", status="done", fusion=fusion_a, trace=trace_a(vlm_a))

    vlm_conf = "No change is visible along the southern shore."
    conf_c, _ = confidence(
        model_term, 0.5, 1.0, 1.0
    )  # agreement drops when the VLM text contradicts
    conf_c = min(conf_c, 0.5)
    fusion_conf = fusion_a.model_copy(
        update=dict(
            model_description=vlm_conf,
            confidence=conf_c,
            confidence_label="Conflict",
            reasons=[
                "language model disagrees with the detector",
                "6 new buildings measured",
                "human check recommended",
            ],
            conflicts=[
                Conflict(
                    between=("s8", "s6"),
                    description=(
                        f"The language model reports no change, but the detector finds 6 new buildings "
                        f"and the change map shows {fmt_area(fm['change'].area_m2)} changed."
                    ),
                )
            ],
        )
    )
    res_a_conflict = JobResult(
        job_id="STORY", status="done", fusion=fusion_conf, trace=trace_a(vlm_conf)
    )
    _write_story(a, "result_conflict.json", res_a, res_a_conflict)

    # ------------------------------------------------------------ story B: cloud -> SAR
    b = jobs / "story_b"
    (b / "ingest").mkdir(parents=True, exist_ok=True)
    (b / "fusion").mkdir(parents=True, exist_ok=True)
    write_tif(b / "ingest" / "img_0_rgb8.tif", rgb8_optical([S["cloudy"]])[0])
    write_tif(b / "ingest" / "img_1_rgb8.tif", rgb8_sar(S["sar"]))
    save_mask(b / "ingest", "cloud_img_0.tif", cloud, like, "cloud")
    shutil.copyfile(sc / "hand.tif", b / "ingest" / "hand_img_1.tif")

    water_sar, water_opt = fk["sar_water_px"], S["l2"] & ~cloud
    iou = round(mask_iou(water_sar & ~cloud, water_opt), 2)
    fb = {
        "water_sar": save_mask(b / "fusion", "water_sar.tif", water_sar, like, "water"),
        "water_optical": save_mask(b / "fusion", "water_optical.tif", water_opt, like, "water"),
        "builtup": save_mask(b / "fusion", "builtup.tif", S["all_b"], like, "builtup"),
        "radar_only": save_mask(b / "fusion", "radar_only.tif", cloud, like, "radar_only"),
        "cloud": save_mask(b / "fusion", "cloud.tif", cloud, like, "cloud"),
    }
    facts_b = {
        "water_sar_m2": fb["water_sar"].area_m2,
        "water_optical_m2": fb["water_optical"].area_m2,
        "water_iou": iou,
        "builtup_m2": fb["builtup"].area_m2,
        "radar_only_pct": cloud_pct,
    }
    conf_b, label_b = confidence(
        0.74, iou, 1.0, round(float(np.mean([1 - cloud_pct / 100, 1.0])), 3)
    )
    answer_b = (
        f"Water covers about {fmt_area(facts_b['water_sar_m2'])} and built-up areas about "
        f"{fmt_area(facts_b['builtup_m2'])}. Optical and SAR agree on water (IoU {iou:.2f}). "
        f"{cloud_pct:.0f}% of the area was answered from radar alone because of cloud."
    )
    profile_b = SensorProfile(
        input_mode="geotiff",
        images=[
            _image(
                0,
                "optical",
                "optical_cloudy.tif",
                sc,
                Modality.OPTICAL_MX,
                BANDS,
                "2025-02-06",
                cloud_pct=cloud_pct,
                sensor="Cartosat-2S",
                cloud_mask_path=Path("ingest/cloud_img_0.tif"),
            ),
            _image(
                1,
                "sar",
                "sar.tif",
                sc,
                Modality.SAR,
                ["sigma0_hh"],
                "2025-02-06",
                cloud_pct=None,
                sensor="RISAT-2B",
                sar_band="X",
                polarisation=["HH"],
                hand_mask_path=Path("ingest/hand_img_1.tif"),
            ),
        ],
        pair=PairInfo(kind="cross-modal", overlap_pct=100.0, same_crs=True, ref_image="img_0"),
        can_compute={"NDVI": True, "NDWI": True, "NDBI": False},
        notes=[notes_ndbi],
    )
    q = QUESTION_B
    plan_b = [
        _tc(
            "s1",
            "sar_water",
            {"image": "sar"},
            {"tile_px": 256, "hand_max_m": 15.0, "use_hand": True},
        ),
        _tc("s2", "sar_builtup", {"image": "sar"}, {"bright_pct": 98.0, "open_px": 2}),
        _tc("s3", "water_mask", {"image": "optical"}, {"index": "NDWI", "exclude_cloud": True}),
        _tc("s4", "index_mask", {"image": "optical"}, {"index": "NDVI", "exclude_cloud": True}),
        _tc(
            "s5",
            "rs_vlm",
            {"image": "optical"},
            {"mode": "describe_scene", "question": q, "max_new_tokens": 64},
        ),
    ]
    steps_b = [
        _ts(
            "s1",
            "sar_water",
            "nrsc-tile-otsu+hand",
            plan_b[0].params,
            f"water {fmt_area(facts_b['water_sar_m2'])}, threshold -15.0 dB, HAND <= 15 m",
            0.6,
        ),
        _ts(
            "s2",
            "sar_builtup",
            "sar-double-bounce-percentile",
            plan_b[1].params,
            f"built-up {fmt_area(facts_b['builtup_m2'])}",
            0.3,
        ),
        _ts(
            "s3",
            "water_mask",
            "ndwi-otsu",
            plan_b[2].params,
            f"water {fmt_area(facts_b['water_optical_m2'])} on clear-sky pixels",
            0.3,
        ),
        _ts("s4", "index_mask", "spectral-index-otsu", plan_b[3].params, "NDVI mean 0.63", 0.2),
        _ts("s5", "rs_vlm", VLM_MODEL, plan_b[4].params, "described the scene", 2.8),
    ]
    trace_b = Trace(
        job_id="STORY",
        question=q,
        parsed_query={
            "task": "OPTICAL_SAR",
            "target": "object",
            "place_hint": "none",
            "looking_for": "object",
            "question": q,
        },
        task=TaskType.OPTICAL_SAR,
        sensor_profile=profile_b,
        plan=plan_b,
        steps=steps_b,
        skipped=[notes_ndbi],
        adapter_id="lora-v1",
        versions={"git": "stub", "contract": CONTRACT_VERSION, "rs_vlm": VLM_MODEL},
    )
    fusion_b = FusionResult(
        answer=answer_b,
        model_description="A lake in the south, farmland and a few buildings; the north-west is covered by cloud.",
        facts=facts_b,
        confidence=conf_b,
        confidence_label=label_b,
        reasons=[
            f"optical and SAR water IoU {iou:.2f}",
            f"{cloud_pct:.0f}% answered from radar only",
            f"cloud {cloud_pct:.0f}%",
        ],
        display_masks=fb,
    )
    _write_story(b, "", JobResult(job_id="STORY", status="done", fusion=fusion_b, trace=trace_b))

    # ------------------------------------------------------------ story C: single-image VQA
    c = jobs / "story_c"
    (c / "ingest").mkdir(parents=True, exist_ok=True)
    (c / "fusion").mkdir(parents=True, exist_ok=True)
    write_tif(c / "ingest" / "img_0_rgb8.tif", rgb8_optical([S["t2"]])[0])
    ndvi_mask, ndvi_mean = fk["ndvi"]
    fc = {"ndvi": save_mask(c / "fusion", "ndvi.tif", ndvi(S["t2"]) > 0.3, like, "ndvi")}
    text_c = "The image is mostly agricultural land and vegetation, with a lake in the south."
    conf_cc, label_c = confidence(0.81, None, 1.0, 1.0)
    profile_c = SensorProfile(
        input_mode="geotiff",
        images=[
            _image(
                0,
                "single",
                "single.tif",
                sc,
                Modality.OPTICAL_MX,
                BANDS,
                "2025-02-08",
                sensor="Cartosat-2S",
            )
        ],
        pair=None,
        can_compute={"NDVI": True, "NDWI": True, "NDBI": False},
        notes=[notes_ndbi],
    )
    q = QUESTION_C
    plan_c = [
        _tc(
            "s1",
            "rs_vlm",
            {"image": "single"},
            {"mode": "vqa", "question": q, "max_new_tokens": 64},
        ),
        _tc("s2", "index_mask", {"image": "single"}, {"index": "NDVI", "exclude_cloud": True}),
    ]
    steps_c = [
        _ts("s1", "rs_vlm", VLM_MODEL, plan_c[0].params, "answered the question", 2.2),
        _ts(
            "s2",
            "index_mask",
            "spectral-index-otsu",
            plan_c[1].params,
            f"NDVI mean {ndvi_mean}",
            0.2,
        ),
    ]
    trace_c = Trace(
        job_id="STORY",
        question=q,
        parsed_query={
            "task": "VQA",
            "target": "object",
            "place_hint": "none",
            "looking_for": "object",
            "question": q,
        },
        task=TaskType.VQA,
        sensor_profile=profile_c,
        plan=plan_c,
        steps=steps_c,
        skipped=[notes_ndbi],
        adapter_id="lora-v1",
        versions={"git": "stub", "contract": CONTRACT_VERSION, "rs_vlm": VLM_MODEL},
    )
    fusion_c = FusionResult(
        answer=text_c,
        facts={"ndvi_mean": ndvi_mean},
        confidence=conf_cc,
        confidence_label=label_c,
        reasons=[
            "answer from the adapted language model",
            f"NDVI mean {ndvi_mean} supports vegetation",
            "cloud 0%",
        ],
        display_masks=fc,
    )
    _write_story(c, "", JobResult(job_id="STORY", status="done", fusion=fusion_c, trace=trace_c))


# ============================================================ entry point
def build_all(out: Path) -> None:
    out = Path(out)
    S = build_scenes(out)
    fk = build_fake_results(out, S)
    build_stories(out, S, fk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    build_all(args.out)
    cp = round(100 * float(cloud_truth().mean()), 1)
    print(f"fixtures written to {args.out} (cloud {cp}%)")
