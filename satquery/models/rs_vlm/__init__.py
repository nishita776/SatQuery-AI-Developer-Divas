"""rs_vlm — remote-sensing VLM. Backbone: EarthDial 4B RGB in 4-bit (see DECISION.md).

run_real() lands in ML-06. torch/transformers/peft are imported lazily *inside*
run_real so fake and cache modes need no GPU and no model libraries.
"""

from __future__ import annotations

from satquery.common.toolbase import BaseTool
from satquery.contracts import JobContext, ToolCall, ToolResult

__all__ = ["RSVLM"]


class RSVLM(BaseTool):
    """Inputs: image, or t1 + t2; optional region. Outputs: text, score.

    Modes: vqa, caption, describe_change, describe_scene.
    score is the mean softmax probability of the generated tokens.
    """

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError("rs_vlm.run_real lands in ML-06")
