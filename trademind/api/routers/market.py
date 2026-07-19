"""Market data API endpoints."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from trademind.api.schemas import (
    DataCollectionResponse,
    IndexDataResponse,
    MarketBreadthResponse,
    MarketCollectRequest,
    MarketDataResponse,
    OptionChainResponse,
    OptionStrikeResponse,
    VIXDataResponse,
)
from trademind.config.settings import settings as cfg
from trademind.engines.market_data.collector import MarketDataCollector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market Data"])

# Populated at startup from live NSE data; used as fallback cache
_indices_cache: list = []

# BhavcopyEngine singleton
_bhavcopy = None


def _get_bhavcopy():
    global _bhavcopy  # noqa: PLW0603
    if _bhavcopy is None:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        _bhavcopy = BhavcopyEngine()
    return _bhavcopy


def _decimal_to_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


@router.get("/data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str) -> MarketDataResponse:
    """Get current market data (OHLCV + VWAP + delivery) for a symbol."""
    symbol = symbol.upper().strip()
    if symbol not in cfg.fno_symbols:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in F&O universe. Use a valid NSE F&O symbol.",
        )

    collector = MarketDataCollector()
    try:
        price_data = await collector.collect_price_data(symbol)
    finally:
        await collector.close()

    if price_data is None:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch market data for '{symbol}'. NSE API may be down.",
        )

    return MarketDataResponse(
        symbol=price_data.symbol,
        timestamp=price_data.timestamp,
        open=_decimal_to_float(price_data.open),
        high=_decimal_to_float(price_data.high),
        low=_decimal_to_float(price_data.low),
        close=_decimal_to_float(price_data.close),
        volume=price_data.volume,
        vwap=_decimal_to_float(price_data.vwap),
        delivery_volume=price_data.delivery_volume,
        delivery_percent=_decimal_to_float(price_data.delivery_percent),
        prev_close=_decimal_to_float(price_data.prev_close),
        change_percent=_decimal_to_float(price_data.change_percent),
        turnover=_decimal_to_float(price_data.turnover),
    )


@router.get("/option-chain/{symbol}", response_model=OptionChainResponse)
async def get_option_chain(symbol: str) -> OptionChainResponse:
    """Get full option chain for a symbol with OI, IV, Greeks."""
    symbol = symbol.upper().strip()
    if symbol not in cfg.fno_symbols:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in F&O universe.",
        )

    collector = MarketDataCollector()
    try:
        option_data = await collector.collect_option_chain(symbol)
    finally:
        await collector.close()

    if option_data is None:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to fetch option chain for '{symbol}'.",
        )

    strikes = []
    for s in option_data.strikes:
        if isinstance(s, dict):
            strikes.append(
                OptionStrikeResponse(
                    strike_price=_decimal_to_float(s.get("strike_price")) or 0.0,
                    expiry_date=s.get("expiry_date", ""),
                    option_type=s.get("option_type", ""),
                    open_interest=s.get("open_interest", 0),
                    change_in_oi=s.get("change_in_oi", 0),
                    volume=s.get("volume", 0),
                    iv=_decimal_to_float(s.get("iv")) or 0.0,
                    last_price=_decimal_to_float(s.get("last_price")) or 0.0,
                    delta=_decimal_to_float(s.get("delta")),
                    gamma=_decimal_to_float(s.get("gamma")),
                    theta=_decimal_to_float(s.get("theta")),
                    vega=_decimal_to_float(s.get("vega")),
                )
            )

    return OptionChainResponse(
        symbol=option_data.symbol,
        timestamp=option_data.timestamp,
        expiry_date=option_data.expiry_date,
        strikes=strikes,
        pcr=_decimal_to_float(option_data.pcr),
        total_ce_oi=option_data.total_ce_oi,
        total_pe_oi=option_data.total_pe_oi,
        max_pain=_decimal_to_float(option_data.max_pain),
        iv_rank=_decimal_to_float(option_data.iv_rank),
        iv_percentile=_decimal_to_float(option_data.iv_percentile),
    )


@router.get("/indices", response_model=list[IndexDataResponse])
async def get_indices() -> list[IndexDataResponse]:
    """Get all tracked NSE indices with live data."""
    if _indices_cache:
        return _indices_cache

    collector = MarketDataCollector()
    try:
        indices = await collector.collect_index_data()
    finally:
        await collector.close()

    result = []
    for name, idx in indices.items():
        result.append(
            IndexDataResponse(
                symbol=idx.symbol,
                timestamp=idx.timestamp,
                open=_decimal_to_float(idx.open),
                high=_decimal_to_float(idx.high),
                low=_decimal_to_float(idx.low),
                close=_decimal_to_float(idx.close),
                change_percent=_decimal_to_float(idx.change_percent),
                volume=idx.volume,
            )
        )
    return result


@router.get("/vix", response_model=VIXDataResponse)
async def get_vix() -> VIXDataResponse:
    """Get current India VIX level."""
    collector = MarketDataCollector()
    try:
        vix_data = await collector.collect_vix()
    finally:
        await collector.close()

    if vix_data is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch India VIX data.",
        )

    return VIXDataResponse(
        timestamp=vix_data.timestamp,
        value=_decimal_to_float(vix_data.value) or 0.0,
        change_percent=_decimal_to_float(vix_data.change_percent),
        prev_close=_decimal_to_float(vix_data.prev_close),
    )


@router.get("/breadth", response_model=MarketBreadthResponse)
async def get_market_breadth() -> MarketBreadthResponse:
    """Get market breadth (advances vs declines)."""
    collector = MarketDataCollector()
    try:
        breadth = await collector.collect_market_breadth()
    finally:
        await collector.close()

    if breadth is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch market breadth data.",
        )

    return MarketBreadthResponse(
        timestamp=breadth.timestamp,
        advances=breadth.advances,
        declines=breadth.declines,
        unchanged=breadth.unchanged,
        advance_decline_ratio=_decimal_to_float(breadth.advance_decline_ratio),
        total_traded=breadth.total_traded,
    )


@router.post("/collect", response_model=DataCollectionResponse)
async def trigger_data_collection(
    request: Optional[MarketCollectRequest] = None,
) -> DataCollectionResponse:
    """Trigger manual data collection for specified symbols or all F&O symbols."""
    symbols = request.symbols if request and request.symbols else cfg.fno_symbols
    symbols = [s.upper().strip() for s in symbols]

    invalid = [s for s in symbols if s not in cfg.fno_symbols]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid symbols: {', '.join(invalid)}",
        )

    collector = MarketDataCollector()
    collected = 0
    try:
        tasks = [collector.collect_all_for_symbol(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        collected = sum(
            1 for r in results
            if not isinstance(r, Exception) and r[0] is not None
        )
    finally:
        await collector.close()

    return DataCollectionResponse(
        status="completed",
        message=f"Collected data for {collected}/{len(symbols)} symbols",
        symbols_collected=collected,
        timestamp=datetime.now(),
    )


# ── Bhavcopy Endpoints ────────────────────────────────────

@router.get("/bhavcopy")
async def get_bhavcopy_summary() -> dict:
    """Get Bhavcopy summary: stock count, indices count, options count, advances/declines."""
    bhavcopy = _get_bhavcopy()
    try:
        data = await bhavcopy.get_bhavcopy()
        return {
            "source": data.source,
            "timestamp": data.timestamp,
            "stock_count": data.fo_count,
            "index_count": len(data.indices),
            "option_count": data.option_count,
            "advances": data.advances,
            "declines": data.declines,
            "total_stocks": len(data.stocks),
        }
    except Exception as exc:
        return {"source": "error", "error": str(exc)}


@router.get("/bhavcopy/stocks")
async def get_bhavcopy_stocks(
    limit: int = Query(default=50, ge=1, le=200),
    sort_by: str = Query(default="change_pct", regex="^(change_pct|volume|oi|close)$"),
) -> list[dict]:
    """Get individual stock data from Bhavcopy."""
    bhavcopy = _get_bhavcopy()
    try:
        data = await bhavcopy.get_bhavcopy()
        stocks = list(data.stocks.values())

        if sort_by == "change_pct":
            stocks.sort(key=lambda s: abs(s.change_pct), reverse=True)
        elif sort_by == "volume":
            stocks.sort(key=lambda s: s.volume, reverse=True)
        elif sort_by == "oi":
            stocks.sort(key=lambda s: s.oi, reverse=True)
        elif sort_by == "close":
            stocks.sort(key=lambda s: s.close, reverse=True)

        return [
            {
                "symbol": s.symbol,
                "close": s.close,
                "change_pct": s.change_pct,
                "volume": s.volume,
                "oi": s.oi,
                "open": s.open,
                "high": s.high,
                "low": s.low,
                "turnover": s.turnover,
            }
            for s in stocks[:limit]
        ]
    except Exception as exc:
        return []


@router.get("/bhavcopy/options/{symbol}")
async def get_bhavcopy_options(symbol: str) -> list[dict]:
    """Get options data for a specific symbol from Bhavcopy."""
    symbol = symbol.upper().strip()
    bhavcopy = _get_bhavcopy()
    try:
        data = await bhavcopy.get_bhavcopy()
        opts = data.options.get(symbol, [])
        return [
            {
                "contract": o.contract,
                "strike": o.strike,
                "expiry": o.expiry,
                "option_type": o.option_type,
                "close": o.close,
                "oi": o.oi,
                "volume": o.volume,
                "underlying": o.underlying,
                "premium_traded": o.premium_traded,
            }
            for o in opts
        ]
    except Exception as exc:
        return []
