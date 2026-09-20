"""BaseTool: every tool extends this and implements run_real() (master F3.4).

BaseTool.run() handles the run mode, so tools never think about fake/cache:
    fake  -> fixtures/fake_results/<tool>/
    cache -> cache/<step_key>/ ; on a miss it falls back to the fake and adds a warning
    real  -> run_real(), then always save to the cache
The mode can be overridden per tool: SATQUERY_TOOL_<tool>=real|cache|fake
"""

from __future__ import annotations

import logging
import os
import time

from satquery.common.cache import load_cached, save_cached
from satquery.common.fakes import load_fake
from satquery.contracts import JobContext, ToolCall, ToolCard, ToolResult

log = logging.getLogger("satquery.common")


class BaseTool:
    def __init__(self, card: ToolCard):
        self.card = card

    def run(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        mode = os.environ.get(f"SATQUERY_TOOL_{self.card.tool}", ctx.mode)
        if mode == "fake":
            return load_fake(self, call, ctx)
        if mode == "cache":
            res = load_cached(self, call, ctx)
            if res is not None:
                return res
            res = load_fake(self, call, ctx)
            res.warnings.append("cache miss: fake result used")
            return res
        t0 = time.time()
        res = self.run_real(call, ctx)
        res.runtime_s = time.time() - t0
        res.source = "real"
        res.step_id = call.step_id
        if res.status == "ok":
            try:
                save_cached(self, call, ctx, res)
            except Exception:  # a cache problem must never fail the job
                log.exception("could not write cache for %s", call.step_id)
        return res

    def run_real(self, call: ToolCall, ctx: JobContext) -> ToolResult:
        raise NotImplementedError  # model libraries are imported lazily inside here

    # convenience for run_real() implementations
    def make_result(self, call: ToolCall, **fields) -> ToolResult:
        fields.setdefault("status", "ok")
        return ToolResult(
            step_id=call.step_id, tool=self.card.tool, model_id=self.card.model_id, **fields
        )
