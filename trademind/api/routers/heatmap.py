"""Sector Heatmap API endpoints.

Provides visual heatmap data for sector performance analysis.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter

from trademind.engines.sector_heatmap.engine import (
    HeatmapResult,
    SectorHeatCell,
    SectorHeatmapEngine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Sector Heatmap"])

_engine: SectorHeatmapEngine | None = None


def _get_engine() -> SectorHeatmapEngine:
    global _engine  # noqa: PLW0603
    if _engine is None:
        _engine = SectorHeatmapEngine()
    return _engine


def _cell_to_dict(cell: SectorHeatCell) -> dict:
    return {
        "sector_index": cell.sector_index,
        "sector_name": cell.sector_name,
        "color": cell.color,
        "change_1d": cell.change_1d,
        "change_1w": cell.change_1w,
        "change_1m": cell.change_1m,
        "momentum_score": cell.momentum_score,
        "heat_value": cell.heat_value,
        "volume_trend": cell.volume_trend,
        "relative_strength": cell.relative_strength,
        "top_stocks": cell.top_stocks,
        "current_price": cell.current_price,
    }


def _result_to_dict(result: HeatmapResult) -> dict:
    return {
        "timestamp": result.timestamp,
        "generated_at": result.generated_at,
        "nifty50_change_1d": result.nifty50_change_1d,
        "nifty50_change_1w": result.nifty50_change_1w,
        "nifty50_change_1m": result.nifty50_change_1m,
        "sectors": [_cell_to_dict(s) for s in result.sectors],
    }


@router.get("/sector-heatmap")
async def get_sector_heatmap() -> dict:
    """Full sector heatmap — all sectors with performance metrics."""
    engine = _get_engine()
    result = await engine.generate_heatmap()
    return _result_to_dict(result)


@router.get("/sector-heatmap/top")
async def get_top_sectors() -> dict:
    """Top 3 performing sectors by momentum score."""
    engine = _get_engine()
    cells = await engine.get_top_sectors(n=3)
    return {
        "top_sectors": [_cell_to_dict(c) for c in cells],
        "generated_at": engine._cache.generated_at if engine._cache else "",
    }


@router.get("/sector-heatmap/worst")
async def get_worst_sectors() -> dict:
    """Bottom 3 performing sectors by momentum score."""
    engine = _get_engine()
    cells = await engine.get_worst_sectors(n=3)
    return {
        "worst_sectors": [_cell_to_dict(c) for c in cells],
        "generated_at": engine._cache.generated_at if engine._cache else "",
    }
