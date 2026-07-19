"""AI scoring API endpoints.

Scores are computed from live NSE Bhavcopy data:
  - Individual stock data (close, change%, OI, volume) from F&O Bhavcopy ZIP
  - Index data (PE, PB, DY, advances/declines, 30d/365d momentum) from allIndices
When NSE is unreachable, endpoints return empty lists with a status message.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from trademind.api.schemas import (
    AIScoreResponse,
    CategoryScoreResponse,
    ScoreRankingResponse,
    ScoreRefreshResponse,
)
from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scores", tags=["AI Scoring"])

_score_cache: Dict[str, AIScoreResponse] = {}
_score_cache_time: Optional[datetime] = None
CACHE_TTL_SECONDS = 300


# ── Module-level BhavcopyEngine singleton ───────────────────────
_bhavcopy = None


def _get_bhavcopy():
    global _bhavcopy  # noqa: PLW0603
    if _bhavcopy is None:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        _bhavcopy = BhavcopyEngine()
    return _bhavcopy


# ── Scoring from individual stock Bhavcopy data ─────────────────

def _score_from_bhavcopy(data) -> Dict[str, AIScoreResponse]:
    """Compute AI scores from real Bhavcopy stock data.

    Each stock scored from:
    - Price action (change%, intraday range, close position in range)
    - OI activity (OI change indicates institutional interest)
    - Volume activity (high volume = conviction)
    - Options sentiment (PCR from options data)
    - Sector context (using index PE/PB for valuation context)
    """
    from trademind.engines.bhavcopy.engine import BhavcopyData
    results: Dict[str, AIScoreResponse] = {}

    # Index-level context
    nifty50 = data.indices.get("NIFTY 50")
    nifty_pe = nifty50.pe if nifty50 else 22.0
    nifty_change = nifty50.change_pct if nifty50 else 0.0

    # Market breadth from indices
    breadth_adv = nifty50.advances if nifty50 else 0
    breadth_dec = nifty50.declines if nifty50 else 0
    breadth_total = breadth_adv + breadth_dec
    breadth_ratio = breadth_adv / breadth_total if breadth_total > 0 else 0.5

    for sym, stock in data.stocks.items():
        c = stock.close
        chg = stock.change_pct
        if c <= 0:
            continue

        daily_range = (stock.high - stock.low) / c * 100 if stock.high and stock.low and c else 0

        # --- Trend score (0-100): daily change % ---
        trend = max(10, min(90, 50 + chg * 12))

        # --- Momentum score: close position in range + change direction ---
        if stock.high and stock.low and stock.high > stock.low:
            close_pos = (c - stock.low) / (stock.high - stock.low) * 100
        else:
            close_pos = 50.0
        momentum = max(10, min(90, close_pos * 0.5 + 30 + chg * 6))

        # --- Volume/OI score: higher OI = more institutional interest ---
        oi_score = 50.0
        if stock.oi > 100000:
            oi_score = 85.0
        elif stock.oi > 50000:
            oi_score = 75.0
        elif stock.oi > 20000:
            oi_score = 65.0
        elif stock.oi > 5000:
            oi_score = 55.0
        elif stock.oi > 1000:
            oi_score = 45.0
        else:
            oi_score = 30.0

        vol_score = 50.0
        if stock.volume > 20000000:
            vol_score = 85.0
        elif stock.volume > 5000000:
            vol_score = 75.0
        elif stock.volume > 1000000:
            vol_score = 65.0
        elif stock.volume > 100000:
            vol_score = 55.0
        else:
            vol_score = 40.0

        volume_oi = (vol_score * 0.5 + oi_score * 0.5)

        # --- Options sentiment: PCR from options data ---
        options_score = 50.0
        pcr = data.get_option_pcr(sym) if hasattr(data, "get_option_pcr") else None
        if pcr is not None:
            if pcr > 1.5:
                options_score = 75.0  # Heavy put buying = bullish
            elif pcr > 1.0:
                options_score = 65.0
            elif pcr > 0.7:
                options_score = 50.0
            elif pcr > 0.4:
                options_score = 35.0
            else:
                options_score = 25.0  # Heavy call buying = bearish
        else:
            # Fallback: infer from price action
            options_score = 60.0 if chg > 0 else 40.0

        # --- Sector/market context: is stock outperforming market? ---
        relative = chg - nifty_change
        sector_score = max(10, min(90, 50 + relative * 10 + breadth_ratio * 20))

        # --- Valuation context: not available per stock from bhavcopy,
        #     but we can use market PE for context ---
        market_val = 50.0
        if nifty_pe > 25:
            market_val = 35.0  # Market expensive — caution
        elif nifty_pe > 20:
            market_val = 50.0
        elif nifty_pe > 15:
            market_val = 65.0  # Market fairly valued
        else:
            market_val = 80.0  # Market cheap — opportunity

        # --- Volatility: range-based ---
        vol_score_cat = max(10, min(90, 100 - daily_range * 12))

        category_scores = [
            CategoryScoreResponse(name="trend", score=round(trend, 1), weight=cfg.category_weights.get("trend", 0.22)),
            CategoryScoreResponse(name="momentum", score=round(momentum, 1), weight=cfg.category_weights.get("momentum", 0.18)),
            CategoryScoreResponse(name="volume", score=round(volume_oi, 1), weight=cfg.category_weights.get("volume", 0.16)),
            CategoryScoreResponse(name="options", score=round(options_score, 1), weight=cfg.category_weights.get("options", 0.15)),
            CategoryScoreResponse(name="sector", score=round(sector_score, 1), weight=cfg.category_weights.get("sector", 0.14)),
            CategoryScoreResponse(name="market", score=round(market_val, 1), weight=cfg.category_weights.get("market", 0.10)),
            CategoryScoreResponse(name="volatility", score=round(vol_score_cat, 1), weight=cfg.category_weights.get("volatility", 0.05)),
        ]

        overall = round(sum(c.score * c.weight for c in category_scores), 1)
        conf = round(min(max(overall / 100 * 1.1, 0.3), 0.92), 2)
        conf_level = (
            "very_high" if conf > 0.85
            else "high" if conf > 0.75
            else "medium" if conf > 0.55
            else "low"
        )

        evidence = []
        if chg > 1.0:
            evidence.append(f"Strong bullish move: +{chg:.2f}%")
        elif chg < -1.0:
            evidence.append(f"Strong bearish move: {chg:.2f}%")
        elif chg > 0:
            evidence.append(f"Mild bullish: +{chg:.2f}%")
        elif chg < 0:
            evidence.append(f"Mild bearish: {chg:.2f}%")
        else:
            evidence.append("Flat session — no directional bias")

        if daily_range > 2.0:
            evidence.append(f"High intraday range: {daily_range:.1f}%")

        if stock.high and stock.low and c:
            close_pos = (c - stock.low) / (stock.high - stock.low) * 100 if stock.high != stock.low else 50
            if close_pos > 75:
                evidence.append(f"Closed near day high ({close_pos:.0f}% of range)")
            elif close_pos < 25:
                evidence.append(f"Closed near day low ({close_pos:.0f}% of range)")

        if stock.oi > 0:
            evidence.append(f"OI: {stock.oi:,.0f} | Volume: {stock.volume:,.0f}")

        if pcr is not None:
            evidence.append(f"PCR: {pcr:.2f} ({'bullish' if pcr > 1 else 'bearish'})")

        evidence.append(f"vs NIFTY: {relative:+.2f}% | Market PE: {nifty_pe:.1f}")

        results[sym] = AIScoreResponse(
            symbol=sym,
            timestamp=datetime.now(),
            score=overall,
            confidence=conf,
            confidence_level=conf_level,
            category_scores=category_scores,
            evidence=evidence,
            signal_strength=round(overall / 100, 2),
            time_horizon="intraday",
            model_version="v1.3-bhavcopy",
            rank=0,
            total_scored=0,
        )

    # Also score major indices from allIndices data
    index_names = [
        "NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY AUTO",
        "NIFTY FMCG", "NIFTY PHARMA", "NIFTY METAL", "NIFTY REALTY",
        "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100",
    ]
    for name in index_names:
        idx = data.indices.get(name)
        if not idx or idx.last == 0:
            continue

        chg = idx.change_pct
        c = idx.last
        daily_range = (idx.high - idx.low) / c * 100 if idx.high and idx.low and c else 0

        trend = max(10, min(90, 50 + idx.per_change_30d * 0.8 + chg * 5))
        momentum = max(10, min(90, 50 + idx.per_change_365d * 0.3 + chg * 8))

        pe_score = 50.0
        if 0 < idx.pe < 15:
            pe_score = 80.0
        elif 15 <= idx.pe < 25:
            pe_score = 65.0
        elif 25 <= idx.pe < 35:
            pe_score = 45.0
        elif idx.pe >= 35:
            pe_score = 25.0

        pb_score = 50.0
        if 0 < idx.pb < 2:
            pb_score = 80.0
        elif 2 <= idx.pb < 4:
            pb_score = 60.0
        elif idx.pb >= 4:
            pb_score = 30.0

        valuation = (pe_score * 0.6 + pb_score * 0.4) if idx.pe > 0 else 50.0

        total = idx.advances + idx.declines
        breadth = (idx.advances / total * 100) if total > 0 else 50.0
        breadth = max(10, min(90, breadth))

        sector = max(10, min(90, 50 + idx.per_change_30d * 0.6 + chg * 3))
        options = max(10, min(90, (valuation + breadth) / 2))

        if idx.year_high and idx.year_low and idx.year_high > idx.year_low:
            year_range_pct = (idx.year_high - idx.year_low) / idx.year_low * 100
            vol_score = (max(10, min(90, 100 - daily_range * 12)) * 0.6 +
                         max(10, min(90, 100 - year_range_pct * 0.5)) * 0.4)
        else:
            vol_score = max(10, min(90, 100 - daily_range * 15))

        cat_scores = [
            CategoryScoreResponse(name="trend", score=round(trend, 1), weight=cfg.category_weights.get("trend", 0.22)),
            CategoryScoreResponse(name="momentum", score=round(momentum, 1), weight=cfg.category_weights.get("momentum", 0.18)),
            CategoryScoreResponse(name="volume", score=round(breadth, 1), weight=cfg.category_weights.get("volume", 0.16)),
            CategoryScoreResponse(name="options", score=round(options, 1), weight=cfg.category_weights.get("options", 0.15)),
            CategoryScoreResponse(name="sector", score=round(sector, 1), weight=cfg.category_weights.get("sector", 0.14)),
            CategoryScoreResponse(name="market", score=round(valuation, 1), weight=cfg.category_weights.get("market", 0.10)),
            CategoryScoreResponse(name="volatility", score=round(vol_score, 1), weight=cfg.category_weights.get("volatility", 0.05)),
        ]

        overall = round(sum(c.score * c.weight for c in cat_scores), 1)
        conf = round(min(max(overall / 100 * 1.1, 0.3), 0.92), 2)
        conf_level = "very_high" if conf > 0.85 else "high" if conf > 0.75 else "medium" if conf > 0.55 else "low"

        evidence = []
        if chg > 1:
            evidence.append(f"Strong bullish: +{chg:.2f}%")
        elif chg < -1:
            evidence.append(f"Strong bearish: {chg:.2f}%")
        if total > 0:
            evidence.append(f"Breadth: {idx.advances}/{idx.declines} ({idx.advances/total*100:.0f}%)")
        if idx.pe > 0:
            evidence.append(f"P/E: {idx.pe:.1f} | P/B: {idx.pb:.1f}")
        if idx.per_change_30d != 0:
            evidence.append(f"30d: {idx.per_change_30d:+.1f}% | 365d: {idx.per_change_365d:+.1f}%")

        results[name] = AIScoreResponse(
            symbol=name,
            timestamp=datetime.now(),
            score=overall,
            confidence=conf,
            confidence_level=conf_level,
            category_scores=cat_scores,
            evidence=evidence,
            signal_strength=round(overall / 100, 2),
            time_horizon="intraday",
            model_version="v1.3-bhavcopy",
            rank=0,
            total_scored=0,
        )

    ranked = sorted(results.items(), key=lambda x: x[1].score, reverse=True)
    total = len(ranked)
    for i, (sym, resp) in enumerate(ranked, 1):
        resp.rank = i
        resp.total_scored = total

    return results


async def _build_scores() -> Dict[str, AIScoreResponse]:
    """Fetch live Bhavcopy data from NSE and compute AI scores."""
    bhavcopy = _get_bhavcopy()
    try:
        data = await bhavcopy.get_bhavcopy()
        if not data.stocks and not data.indices:
            return {}
        return _score_from_bhavcopy(data)
    except Exception as exc:
        logger.error("Bhavcopy scoring failed: %s", exc)
        return {}


# ── API Endpoints ───────────────────────────────────────────────

@router.get("/ranking", response_model=list[ScoreRankingResponse])
async def get_score_ranking(
    limit: int = Query(default=50, ge=1, le=100),
) -> list[ScoreRankingResponse]:
    global _score_cache, _score_cache_time  # noqa: PLW0603

    if _score_cache and _score_cache_time and (datetime.now() - _score_cache_time).seconds < CACHE_TTL_SECONDS:
        pass
    elif not _score_cache:
        try:
            live = await asyncio.wait_for(_build_scores(), timeout=15.0)
            if live:
                _score_cache = live
                _score_cache_time = datetime.now()
        except (asyncio.TimeoutError, Exception):
            pass

    if not _score_cache:
        return []

    ranked = sorted(_score_cache.items(), key=lambda x: x[1].score, reverse=True)
    return [
        ScoreRankingResponse(
            symbol=resp.symbol,
            score=resp.score,
            confidence=resp.confidence,
            confidence_level=resp.confidence_level,
            rank=i,
            category_scores={cs.name: cs.score for cs in resp.category_scores},
            evidence=resp.evidence,
        )
        for i, (sym, resp) in enumerate(ranked[:limit], 1)
    ]


@router.get("/refresh", response_model=ScoreRefreshResponse)
async def refresh_scores() -> ScoreRefreshResponse:
    global _score_cache, _score_cache_time  # noqa: PLW0603

    try:
        live = await asyncio.wait_for(_build_scores(), timeout=20.0)
    except (asyncio.TimeoutError, Exception):
        live = {}

    if live:
        _score_cache = live
        _score_cache_time = datetime.now()

    return ScoreRefreshResponse(
        status="completed" if _score_cache else "no_data",
        scored_count=len(_score_cache),
        timestamp=datetime.now(),
    )


@router.get("/{symbol}", response_model=AIScoreResponse)
async def get_ai_score(symbol: str) -> AIScoreResponse:
    symbol = symbol.upper().strip()

    if symbol in _score_cache:
        return _score_cache[symbol]

    try:
        live = await asyncio.wait_for(_build_scores(), timeout=15.0)
        if live:
            _score_cache.update(live)
            if symbol in _score_cache:
                return _score_cache[symbol]
    except (asyncio.TimeoutError, Exception):
        pass

    raise HTTPException(status_code=503, detail=f"No live score data for '{symbol}'. Data may be unavailable.")
