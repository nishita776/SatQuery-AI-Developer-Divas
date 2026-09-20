# satquery/contracts/tools.py  (master F3.3, verbatim)
from typing import Protocol, runtime_checkable

from .context import JobContext
from .core import ToolCall, ToolCard, ToolResult


@runtime_checkable
class Tool(Protocol):
    card: ToolCard

    def run(self, call: ToolCall, ctx: JobContext) -> ToolResult: ...
