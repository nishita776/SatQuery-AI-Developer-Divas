# satquery/contracts/errors.py  (master F3.3, verbatim)
from enum import Enum


class ErrorCode(str, Enum):
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    BAD_INPUT = "BAD_INPUT"
    MISSING_BAND = "MISSING_BAND"
    NO_OVERLAP = "NO_OVERLAP"
    NEEDS_INPUT = "NEEDS_INPUT"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INTERNAL = "INTERNAL"


class ToolError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message
