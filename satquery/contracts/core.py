# satquery/contracts/core.py  (master F3.3, verbatim)
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.0"


class TaskType(str, Enum):
    VQA = "VQA"
    CAPTION = "CAPTION"
    GROUNDING = "GROUNDING"
    COUNT = "COUNT"
    CHANGE_DESCRIBE = "CHANGE_DESCRIBE"
    CHANGE_VQA = "CHANGE_VQA"
    CHANGE_GROUNDING = "CHANGE_GROUNDING"
    OPTICAL_SAR = "OPTICAL_SAR"


class Modality(str, Enum):
    OPTICAL_MX = "optical-MX"
    OPTICAL_PAN = "optical-PAN"
    OPTICAL_RGB = "optical-RGB"
    SAR = "sar"
    UNKNOWN = "unknown"


InputMode = Literal["geotiff", "benchmark"]
RunMode = Literal["real", "cache", "fake"]
Role = Literal["single", "T1", "T2", "optical", "sar"]


# ---------- written by Ingest (P2), read by everyone ----------
class ImageInfo(BaseModel):
    image_id: str  # "img_0", upload order
    role: Role
    file: str  # original filename
    sha1: str  # sha1 of the original upload (cache key)
    prepared_path: Path  # runs/<job>/ingest/img_0.tif (full bands, ref grid)
    rgb8_path: Path | None = None  # 3-band uint8 display/model copy, same grid
    sensor: str = "unknown"  # "Cartosat-2S" | "RISAT-2B" | "RISAT-1A" | "Sentinel-2" | ...
    modality: Modality = Modality.UNKNOWN
    bands: list[str] = Field(default_factory=list)  # ["blue","green","red","nir"] / ["sigma0_hh"]
    pixel_m: float | None = None  # None in benchmark mode (no georeferencing)
    width: int  # pixels on the ref grid
    height: int
    date: str | None = None
    crs: str | None = None  # "EPSG:32643"
    sar_band: Literal["X", "C", "L"] | None = None
    polarisation: list[str] = Field(default_factory=list)
    cloud_pct: float | None = None
    cloud_mask_path: Path | None = None
    hand_mask_path: Path | None = None


class PairInfo(BaseModel):
    kind: Literal["bi-temporal", "cross-modal"]
    overlap_pct: float
    same_crs: bool
    ref_image: str  # image_id whose grid everything is on


class SensorProfile(BaseModel):
    contract_version: str = CONTRACT_VERSION
    input_mode: InputMode
    images: list[ImageInfo]
    pair: PairInfo | None = None
    can_compute: dict[str, bool] = Field(
        default_factory=dict
    )  # {"NDVI": True, "NDWI": True, "NDBI": False}
    notes: list[str] = Field(default_factory=list)  # human-readable; copied into the trace


# ---------- the toolbox ----------
class ParamSpec(BaseModel):
    type: Literal["float", "int", "str", "bool", "enum"]
    default: Any = None
    min: float | None = None
    max: float | None = None
    choices: list[Any] | None = None


class ToolCard(BaseModel):  # = one file registry/<tool>.yaml
    tool: str  # "detector"
    owner: Literal["P1", "P2"]
    impl: str  # "satquery.models.detector:Detector"
    model_id: str  # "IDEA-Research/grounding-dino-base"
    does: list[TaskType]
    needs: list[Modality]
    gives: list[str]  # ["detections", "count"]
    native_pixel_m: tuple[float, float] | None = None
    params: dict[str, ParamSpec] = Field(default_factory=dict)
    fallback: str | None = None  # tool to use if this one fails


class MaskRef(BaseModel):
    path: Path  # uint8 0/1 GeoTIFF on the ref grid
    meaning: str  # "water", "change", "cloud", "region", ...
    pixels: int
    area_m2: float | None = None  # None when not georeferenced


class Detection(BaseModel):
    label: str
    score: float  # 0-1
    bbox_px: tuple[int, int, int, int]  # x0, y0, x1, y1 on the ref grid


class ToolCall(BaseModel):
    step_id: str  # "s3"
    tool: str
    inputs: dict[str, str]  # {"image": "T2", "region": "$s2.masks.region"}
    params: dict[str, Any] = Field(
        default_factory=dict
    )  # validated + defaults filled by the controller


class ToolResult(BaseModel):
    step_id: str
    tool: str
    model_id: str
    status: Literal["ok", "skipped", "failed"]
    text: str | None = None  # RS-VLM only
    detections: list[Detection] = Field(default_factory=list)
    masks: dict[str, MaskRef] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(
        default_factory=dict
    )  # {"count": 6, "changed_area_m2": 4944.0}
    score: float | None = None  # the tool's own confidence, 0-1
    warnings: list[str] = Field(default_factory=list)
    runtime_s: float = 0.0
    source: RunMode = "real"


# ---------- written by Fusion + Trace (P2), read by Render/UI (P3) ----------
class Conflict(BaseModel):
    between: tuple[str, str]  # ("s8", "s6")
    description: str


class FusionResult(BaseModel):
    answer: str  # composed from measured facts
    model_description: str | None = None  # the RS-VLM's own sentence, shown separately
    facts: dict[str, float | str] = Field(default_factory=dict)
    confidence: float  # 0-1
    confidence_label: Literal["High", "Medium", "Low", "Conflict"]
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    display_masks: dict[str, MaskRef] = Field(default_factory=dict)  # keys = layer ids
    display_detections: dict[str, list[Detection]] = Field(default_factory=dict)


class TraceStep(BaseModel):
    step_id: str
    tool: str
    model_id: str
    params: dict[str, Any]
    status: str
    summary: str  # one line, e.g. "7 detections above 0.35"
    runtime_s: float
    source: str


class Trace(BaseModel):
    job_id: str
    question: str
    parsed_query: dict[str, Any]  # task, target, place_hint, looking_for
    task: TaskType
    sensor_profile: SensorProfile
    plan: list[ToolCall]
    steps: list[TraceStep]
    skipped: list[str] = Field(default_factory=list)  # ["NDBI: no SWIR band"]
    adapter_id: str | None = None  # the LoRA adapter, proving adaptation
    versions: dict[str, str] = Field(
        default_factory=dict
    )  # git commit, contract version, model ids


# ---------- returned to the web (P3 fills layers/downloads) ----------
class Layer(BaseModel):
    layer_id: str
    name: str
    kind: Literal["image", "vector"]
    url: str
    corners: list[tuple[float, float]] | None = None  # image layers: TL, TR, BR, BL as (lon, lat)
    style: str  # key in render/styles.json
    visible: bool = True


class JobResult(BaseModel):
    job_id: str
    status: Literal["done", "failed", "needs_input"]
    message: str | None = None  # e.g. "I need a second image from a different date"
    fusion: FusionResult | None = None
    trace: Trace | None = None
    layers: list[Layer] = Field(default_factory=list)
    downloads: dict[str, str] = Field(default_factory=dict)  # {"report_pdf": url, "geojson": url}
