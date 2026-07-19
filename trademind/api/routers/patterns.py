"""Pattern Scanner API endpoints.

Detects common chart patterns from Yahoo Finance daily OHLCV data
for all tracked FNO symbols.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Query

from trademind.api.schemas import (
    PatternScanResponse,
    PatternScanResultResponse,
    PatternSummaryResponse,
)
from trademind.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["Pattern Scanner"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from trademind.engines.pattern_scanner.engine import PatternScannerEngine
        _engine = PatternScannerEngine()
    return _engine


def _result_to_response(r) -> PatternScanResultResponse:
    return PatternScanResultResponse(
        symbol=r.symbol,
        pattern_name=r.pattern_name,
        pattern_type=r.pattern_type,
        confidence=r.confidence,
        entry_price=r.entry_price,
        stop_loss=r.stop_loss,
        target_price=r.target_price,
        timeframe=r.timeframe,
        supporting_evidence=r.supporting_evidence,
        detected_at=r.detected_at,
    )


@router.get("/scan", response_model=PatternScanResponse)
async def scan_all_symbols(
    limit: Optional[int] = Query(None, ge=1, le=50, description="Max symbols to scan"),
) -> PatternScanResponse:
    """Scan all tracked FNO symbols for chart patterns."""
    engine = _get_engine()
    symbols = settings.fno_symbols[:limit] if limit else settings.fno_symbols

    start = time.time()
    results = await engine.scan_all(symbols)
    elapsed = round(time.time() - start, 2)

    logger.info("Pattern scan completed: %d patterns from %d symbols in %.2fs", len(results), len(symbols), elapsed)

    return PatternScanResponse(
        patterns=[_result_to_response(r) for r in results],
        total=len(results),
        scanned_symbols=len(symbols),
        scan_time_seconds=elapsed,
    )


@router.get("/scan/{symbol}", response_model=PatternScanResponse)
async def scan_single_symbol(
    symbol: str,
) -> PatternScanResponse:
    """Scan a single symbol for chart patterns."""
    engine = _get_engine()
    sym = symbol.upper()

    start = time.time()
    results = await engine.scan_symbol(sym)
    elapsed = round(time.time() - start, 2)

    logger.info("Pattern scan for %s: %d patterns in %.2fs", sym, len(results), elapsed)

    return PatternScanResponse(
        patterns=[_result_to_response(r) for r in results],
        total=len(results),
        scanned_symbols=1,
        scan_time_seconds=elapsed,
    )


@router.get("/summary", response_model=PatternSummaryResponse)
async def pattern_summary(
    limit: Optional[int] = Query(None, ge=1, le=50, description="Max symbols to scan"),
) -> PatternSummaryResponse:
    """Summary count of detected patterns by type and name."""
    engine = _get_engine()
    symbols = settings.fno_symbols[:limit] if limit else settings.fno_symbols

    results = await engine.scan_all(symbols)
    summary = engine.get_summary(results)

    return PatternSummaryResponse(
        total_patterns=summary["total_patterns"],
        by_type=summary["by_type"],
        by_pattern=summary["by_pattern"],
        avg_confidence=summary["avg_confidence"],
    )
