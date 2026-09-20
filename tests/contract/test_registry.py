"""Every registry/*.yaml validates as a ToolCard, its impl imports, and it has fake results."""

from importlib import import_module

import pytest
import yaml

from satquery.agent.registry import load_cards, load_registry
from satquery.common.paths import REPO_ROOT, fake_results_dir
from satquery.contracts import ToolCard

CARDS = sorted((REPO_ROOT / "registry").glob("*.yaml"))


def test_there_are_cards():
    assert CARDS


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_card_validates_and_imports(path):
    card = ToolCard.model_validate(yaml.safe_load(path.read_text()))
    assert card.tool == path.stem
    module, _, cls = card.impl.partition(":")
    assert hasattr(import_module(module), cls), f"{card.impl} does not import"
    for name, spec in card.params.items():
        if spec.type in ("int", "float") and spec.default is not None and spec.min is not None:
            assert spec.min <= spec.default <= spec.max, (
                f"{card.tool}.{name}: default outside range"
            )
        if spec.type == "enum":
            assert spec.choices and (spec.default is None or spec.default in spec.choices)
    if card.fallback:
        assert (REPO_ROOT / "registry" / f"{card.fallback}.yaml").exists()


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.stem)
def test_every_tool_has_fake_results(path):
    assert list((fake_results_dir() / path.stem).glob("*.json")), f"no fake results for {path.stem}"


def test_load_registry():
    tools = load_registry()
    assert set(tools) == set(load_cards())
    for name, tool in tools.items():
        assert tool.card.tool == name and hasattr(tool, "run")
