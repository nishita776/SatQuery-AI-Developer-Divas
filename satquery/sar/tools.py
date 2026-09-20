"""SAR expert tools. Real implementations arrive in BE-03; the classes exist now so that the
registry cards import and fake/cache mode work from Day 0."""

from satquery.common.toolbase import BaseTool


class SarWater(BaseTool):
    """NRSC tile-based Otsu + HAND <= hand_max_m. TODO BE-03."""


class SarBuiltup(BaseTool):
    """Very bright SAR returns (double-bounce) above a per-image percentile. TODO BE-03."""
