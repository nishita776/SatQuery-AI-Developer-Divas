"""run_job: the ONLY function P3's server calls (master F3.4).

Until BE-11 it returns the stub stories. BE-11 replaces the `else` branch with the real pipeline
(ingest -> parse -> plan -> execute -> fuse -> trace) and flips the default of SATQUERY_STUB_JOB to "0".
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from satquery.contracts import InputMode, JobResult, TraceStep


def run_job(
    question: str,
    files: list[Path],
    workdir: Path,
    input_mode: InputMode = "geotiff",
    on_step: Callable[[TraceStep], None] | None = None,
) -> JobResult:
    if os.environ.get("SATQUERY_STUB_JOB", "1") != "0":
        from satquery.agent.stub import run_stub_job

        return run_stub_job(question, files, workdir, input_mode, on_step)
    return JobResult(
        job_id=Path(workdir).name,
        status="failed",
        message="The real pipeline is not wired yet (BE-11). Set SATQUERY_STUB_JOB=1 to use the stub.",
    )
