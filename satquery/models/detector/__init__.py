"""detector -- open-vocabulary detection with Grounding DINO (master F4.5).

The "detect, don't estimate" tool. Every count the system reports is backed by
boxes a judge can see, which is the difference between "around 15 buildings" and
"7 buildings, each boxed".

Scale Bridge does the geometry: plan_tiles crops at the model's native
resolution, boxes_to_scene maps detections back onto the reference grid, and
merge_detections collapses the same object seen in two overlapping tiles.
bbox_px is ALWAYS on the full reference grid, never tile-local (master F3.2).

torch/transformers are imported inside run_real so fake and cache modes need no
GPU and no model libraries (master F3.4).
"""

from __future__ import annotations

import logging
import re

from satquery.common.toolbase import BaseTool
from satquery.contracts import (Detection, ErrorCode, JobContext, ToolCall,
                                ToolError, ToolResult)

__all__ = ["Detector"]

log = logging.getLogger("satquery.models")

_CACHE: dict = {}  # model_id -> (processor, model); loaded once per process


def normalise_prompt(prompt: str) -> str:
    """Grounding DINO wants lower-case phrases each ending in a full stop.

    "Buildings, water body" -> "buildings. water body."
    """
    parts = [p.strip().lower() for p in re.split(r"[.,;]", prompt or "") if p.strip()]
    if not parts:
        raise ToolError(ErrorCode.BAD_INPUT, "detector needs a non-empty prompt")
    return " ".join(f"{p} ." for p in parts)


def _load(model_id: str):
    """Lazy, cached. Imports live here so fake mode never touches torch."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise ToolError(ErrorCode.MODEL_LOAD_FAILED,
                        f"model libraries not installed: {exc}") from exc
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    except Exception as exc:
        raise ToolError(ErrorCode.MODEL_LOAD_FAILED,
                        f"could not load {model_id}: {exc}") from exc
    log.info("detector loaded %s on %s", model_id, device)
    _CACHE[model_id] = (processor, model)
    return _CACHE[model_id]


class Detector(BaseTool):
    """Inputs: image; optional region. Outputs: detections, metrics.count, score."""

    def _detect_tile(self, arr, prompt, box_threshold, text_threshold):
        """-> (boxes_model_px, scores, labels) for one tile."""
        import torch
        from PIL import Image

        processor, model = _load(self.card.model_id)
        device = next(model.parameters()).device
        inputs = processor(images=Image.fromarray(arr), text=prompt,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # transformers 5.x: the box argument is `threshold`, not `box_threshold`,
        # and `labels` returns integer ids -- phrase strings are in `text_labels`.
        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(arr.shape[0], arr.shape[1])],
        )[0]

        boxes = [tuple(float(v) for v in b) for b in results["boxes"].tolist()]
        scores = [float(s) for s in results["scores"].tolist()]
        labels = [str(t) for t in results.get("text_labels", results.get("labels", []))]
        if len(labels) != len(boxes):  # be forgiving, keep boxes
            labels = [prompt.split(" .")[0].strip()] * len(boxes)
        return boxes, scores, labels

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        from satquery.common.geo import read_mask
        from satquery.scale import (boxes_to_scene, merge_detections, plan_tiles,
                                    read_tile)

        image = ctx.resolve(call.inputs["image"])
        region = ctx.resolve(call.inputs["region"]) if "region" in call.inputs else None
        p = call.params

        prompt = normalise_prompt(p.get("prompt", ""))
        box_threshold = float(p.get("box_threshold", 0.35))
        text_threshold = float(p.get("text_threshold", 0.25))
        nms_iou = float(p.get("nms_iou", 0.50))
        max_detections = int(p.get("max_detections", 500))
        tile_px = int(p.get("tile_px", 1024))

        warnings: list[str] = []
        dets: list[Detection] = []

        def sweep(tile_size: int):
            found: list[Detection] = []
            tiles = plan_tiles(image, self.card.native_pixel_m,
                               tile_px=tile_size, region=region)
            for tile in tiles:
                boxes, scores, labels = self._detect_tile(
                    read_tile(tile), prompt, box_threshold, text_threshold)
                for bbox, score, label in zip(boxes_to_scene(tile, boxes), scores, labels):
                    found.append(Detection(label=label or "object",
                                           score=max(0.0, min(1.0, score)),
                                           bbox_px=bbox))
            return found

        try:
            dets = sweep(tile_px)
        except ToolError:
            raise
        except Exception as exc:
            # ML-12: out of memory -> free, retry once smaller, then give up
            if "out of memory" not in str(exc).lower():
                raise ToolError(ErrorCode.INTERNAL, f"detector failed: {exc}") from exc
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            smaller = 640
            warnings.append(f"out of memory at tile_px={tile_px}, retried at {smaller}")
            log.warning("detector OOM at %d, retrying at %d", tile_px, smaller)
            try:
                dets = sweep(smaller)
            except Exception as exc2:
                raise ToolError(ErrorCode.OUT_OF_MEMORY,
                                f"detector out of memory even at {smaller} px: {exc2}") from exc2

        dets = merge_detections(dets, nms_iou)

        if region is not None:
            mask = read_mask(region)
            h, w = mask.shape
            kept = []
            for d in dets:
                x0, y0, x1, y1 = d.bbox_px
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                if 0 <= cy < h and 0 <= cx < w and mask[cy, cx]:
                    kept.append(d)
            if len(kept) != len(dets):
                warnings.append(f"{len(dets) - len(kept)} detections outside region")
            dets = kept

        dets.sort(key=lambda d: d.score, reverse=True)
        if len(dets) > max_detections:
            warnings.append(f"capped at max_detections={max_detections}")
            dets = dets[:max_detections]

        score = sum(d.score for d in dets) / len(dets) if dets else 0.0

        return self.make_result(
            call,
            detections=dets,
            metrics={"count": float(len(dets))},
            score=score,
            warnings=warnings,
        )
