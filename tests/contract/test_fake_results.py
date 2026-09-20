"""Every fake tool result validates as a ToolResult and its masks are 512x512 uint8 0/1."""

import json

import numpy as np
import pytest
import rasterio

from satquery.common.paths import fake_results_dir
from satquery.contracts import ToolResult

FILES = sorted(fake_results_dir().glob("*/*.json"))


def test_there_are_fakes():
    assert FILES


@pytest.mark.parametrize("f", FILES, ids=lambda f: f"{f.parent.name}/{f.stem}")
def test_fake_validates(f):
    data = json.loads(f.read_text())
    assert "step_id" not in data or data["step_id"] == ""
    data["step_id"] = "s1"
    res = ToolResult.model_validate(data)
    assert res.tool == f.parent.name and res.status == "ok"
    assert all(0.0 <= d.score <= 1.0 for d in res.detections) and (
        res.score is None or 0 <= res.score <= 1
    )
    for name, m in res.masks.items():
        p = f.parent / m.path
        assert p.exists(), p
        with rasterio.open(p) as src:
            assert (src.height, src.width) == (512, 512) and src.dtypes[0] == "uint8"
            arr = src.read(1)
        assert set(np.unique(arr)) <= {0, 1}
        assert m.pixels == int(arr.sum()), f"{f.name}:{name} pixel count does not match the file"
