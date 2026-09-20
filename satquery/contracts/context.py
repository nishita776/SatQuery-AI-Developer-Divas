# satquery/contracts/context.py  (master F3.3, verbatim)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import ImageInfo, RunMode, SensorProfile, ToolResult


@dataclass
class JobContext:
    job_id: str
    workdir: Path  # runs/<job_id>/
    profile: SensorProfile
    mode: RunMode  # fake | cache | real
    results: dict[str, ToolResult] = field(default_factory=dict)  # filled as steps finish
    variables: dict[str, Any] = field(default_factory=dict)  # question, target, place_hint, ...

    def step_dir(self, step_id: str) -> Path:
        d = self.workdir / "steps" / step_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def image(self, id_or_role: str) -> ImageInfo:
        for im in self.profile.images:
            if im.image_id == id_or_role or im.role == id_or_role:
                return im
        raise KeyError(id_or_role)

    def resolve(self, ref: str) -> Any:
        """'T1' -> ImageInfo · '$target' -> variable · '$s2' -> ToolResult · '$s2.masks.region' -> MaskRef"""
        if not ref.startswith("$"):
            return self.image(ref)
        name, *path = ref[1:].split(".")
        if name in self.variables:
            return self.variables[name]
        obj: Any = self.results[name]
        for part in path:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        return obj
