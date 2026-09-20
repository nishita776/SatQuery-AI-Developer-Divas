"""Physics tools. Real implementations arrive in BE-04 (water_mask, index_mask) and BE-05
(select_region); the classes exist now so that the registry cards import from Day 0."""

from satquery.common.toolbase import BaseTool


class WaterMask(BaseTool):
    """NDWI + Otsu. TODO BE-04."""


class IndexMask(BaseTool):
    """NDVI or NDWI + Otsu. TODO BE-04."""


class SelectRegion(BaseTool):
    """Pick one detection by hint, turn it into a region mask + buffer. TODO BE-05."""
