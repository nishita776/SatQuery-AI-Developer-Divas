"""change_map — learned bi-temporal change detection, with change_diff as its fallback.

ChangeMap.run_real lands in ML-05 (ChangeFormer, LEVIR-CD checkpoint).
ChangeDiff.run_real also lands in ML-05 and needs no learned model -- it is the
safety net the controller switches to when ChangeMap fails.
"""

from __future__ import annotations

from satquery.common.toolbase import BaseTool
from satquery.contracts import JobContext, ToolCall, ToolResult

__all__ = ["ChangeMap", "ChangeDiff"]


class ChangeMap(BaseTool):
    """Inputs: t1, t2; optional region.

    Outputs: masks.change, masks.change_prob, metrics.changed_pixels,
    metrics.changed_area_m2, score (mean probability inside the change mask).
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError("change_map.run_real lands in ML-05")


class ChangeDiff(BaseTool):
    """Inputs: t1, t2; optional region.

    Per-pixel RGB difference thresholded at mean + k_sigma * std, then a
    morphological opening. Outputs: masks.change, metrics.changed_pixels,
    metrics.changed_area_m2.
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError("change_diff.run_real lands in ML-05")
