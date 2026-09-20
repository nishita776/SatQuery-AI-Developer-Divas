"""Registry loader: registry/*.yaml -> validated ToolCards -> instantiated tools."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml

from satquery.common.paths import REPO_ROOT
from satquery.common.toolbase import BaseTool
from satquery.contracts import ErrorCode, ToolCard, ToolError


def _dir(registry_dir: str | Path) -> Path:
    p = Path(registry_dir)
    return p if p.is_absolute() else REPO_ROOT / p


def load_cards(registry_dir: str | Path = "registry") -> dict[str, ToolCard]:
    """Only validate the YAML (no imports): what the server's /api/registry needs."""
    cards: dict[str, ToolCard] = {}
    for f in sorted(_dir(registry_dir).glob("*.yaml")):
        card = ToolCard.model_validate(yaml.safe_load(f.read_text()))
        if card.tool != f.stem:
            raise ToolError(
                ErrorCode.INTERNAL, f"{f.name}: card says tool='{card.tool}', file name must match"
            )
        cards[card.tool] = card
    return cards


def load_registry(registry_dir: str | Path = "registry") -> dict[str, BaseTool]:
    """Validate every card, import `impl` ("module:Class") and create the tool."""
    tools: dict[str, BaseTool] = {}
    for name, card in load_cards(registry_dir).items():
        module_name, _, cls_name = card.impl.partition(":")
        try:
            cls = getattr(import_module(module_name), cls_name)
        except (ImportError, AttributeError) as e:
            raise ToolError(ErrorCode.MODEL_LOAD_FAILED, f"cannot import {card.impl}: {e}") from e
        tools[name] = cls(card)
    return tools
