"""detector — open-vocabulary detection with Grounding DINO. "Detect, don't estimate."

run_real() lands in ML-04. Model libraries are imported lazily inside it.
"""

from __future__ import annotations

from satquery.common.toolbase import BaseTool
from satquery.contracts import JobContext, ToolCall, ToolResult

__all__ = ["Detector"]


class Detector(BaseTool):
    """Inputs: image; optional region. Outputs: detections, metrics.count, score.

    bbox_px is [x0, y0, x1, y1] on the full reference grid, never tile
    coordinates -- the Scale Bridge maps tile boxes back. score is the mean
    box score, 0.0 when nothing was found.
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError("detector.run_real lands in ML-04")
