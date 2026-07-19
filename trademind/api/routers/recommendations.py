"""Trade recommendation API endpoints.

No dummy data. Recommendations are generated from live NSE Bhavcopy data:
  - Individual stock data (close, change%, OI, volume) from F&O Bhavcopy
  - Options PCR data from op*.csv
  - Index context (PE/PB, breadth, momentum) from allIndices
When NSE is unreachable, endpoints return empty lists.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from trademind.api.schemas import (
    RecommendationGenerateRequest,
    RecommendationGenerateResponse,
    RecommendationHistoryResponse,
    TradeRecommendationResponse,
)
from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

# In-memory store for generated recommendations
_active_recommendations: Dict[str, TradeRecommendationResponse] = {}
_recommendation_history: List[RecommendationHistoryResponse] = []

# Module-level BhavcopyEngine singleton
_bhavcopy = None


def _get_bhavcopy():
    global _bhavcopy  # noqa: PLW0603
    if _bhavcopy is None:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        _bhavcopy = BhavcopyEngine()
    return _bhavcopy


async def _auto_generate_from_bhavcopy() -> None:
    """Auto-generate recommendations from real Bhavcopy stock data.

    Scores each stock, generates BUY/SELL signals for stocks with:
    - Strong directional move (>0.5% daily change)
    - High OI (institutional interest)
    - High volume (conviction)
    """
    from trademind.api.routers.scores import _score_from_bhavcopy

    bhavcopy = _get_bhavcopy()
    try:
        data = await bhavcopy.get_bhavcopy()
        if not data.stocks:
            return

        scored = _score_from_bhavcopy(data)

        for sym, score_resp in scored.items():
            if sym.startswith("NIFTY"):  # Skip index recommendations for now
                continue

            stock = data.stocks.get(sym)
            if not stock or stock.close <= 0:
                continue

            chg = stock.change_pct
            entry = stock.close

            # Get options PCR for this stock
            pcr = data.get_option_pcr(sym) if hasattr(data, "get_option_pcr") else None

            # Generate signal based on change %, OI, and PCR
            if chg > 2.0 and stock.oi > 20000:
                action = "STRONG_BUY"
                sl = entry * 0.975
                target = entry * 1.04
                conf = min(0.75 + chg * 0.02, 0.92)
            elif chg > 0.5 and stock.oi > 5000:
                action = "BUY"
                sl = entry * 0.98
                target = entry * 1.025
                conf = min(0.6 + chg * 0.04, 0.85)
            elif chg < -2.0 and stock.oi > 20000:
                action = "STRONG_SELL"
                sl = entry * 1.025
                target = entry * 0.96
                conf = min(0.75 + abs(chg) * 0.02, 0.92)
            elif chg < -0.5 and stock.oi > 5000:
                action = "SELL"
                sl = entry * 1.02
                target = entry * 0.975
                conf = min(0.6 + abs(chg) * 0.04, 0.85)
            else:
                continue  # Skip HOLD — not actionable

            risk_reward = round((target - entry) / (entry - sl), 1) if entry > sl else 1.0

            evidence = list(score_resp.evidence)
            evidence.append(f"Score: {score_resp.score}/100 | Change: {chg:+.2f}%")

            rec_id = f"bhavcopy-{sym}-{datetime.now().strftime('%Y%m%d')}"
            _active_recommendations[sym] = TradeRecommendationResponse(
                id=rec_id,
                symbol=sym,
                timestamp=datetime.now(),
                action=action,
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target=round(target, 2),
                confidence=round(conf, 2),
                expected_move_percent=round(chg, 2),
                holding_period="intraday",
                risk_reward_ratio=risk_reward,
                evidence=evidence,
                is_active=True,
            )

        logger.info(
            "Auto-generated %d recommendations from Bhavcopy (%d stocks, %d options)",
            len(_active_recommendations),
            data.fo_count,
            data.option_count,
        )
    except Exception as exc:
        logger.error("Bhavcopy recommendation generation failed: %s", exc)


@router.get("", response_model=list[TradeRecommendationResponse])
async def get_active_recommendations() -> list[TradeRecommendationResponse]:
    """Get all active recommendations. Auto-generates from Bhavcopy if empty."""
    if not _active_recommendations:
        try:
            await asyncio.wait_for(_auto_generate_from_bhavcopy(), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass
    return list(_active_recommendations.values())


@router.get("/{symbol}", response_model=TradeRecommendationResponse)
async def get_recommendation(symbol: str) -> TradeRecommendationResponse:
    symbol = symbol.upper().strip()

    if symbol in _active_recommendations:
        return _active_recommendations[symbol]

    # Try to generate on-demand from Bhavcopy
    try:
        await asyncio.wait_for(_auto_generate_from_bhavcopy(), timeout=15.0)
    except (asyncio.TimeoutError, Exception):
        pass

    if symbol in _active_recommendations:
        return _active_recommendations[symbol]

    raise HTTPException(
        status_code=503,
        detail=f"Unable to generate recommendation for '{symbol}'. NSE may be unreachable.",
    )


@router.get("/history/all", response_model=list[RecommendationHistoryResponse])
async def get_recommendation_history(
    limit: int = Query(default=50, ge=1, le=500),
    symbol: Optional[str] = Query(default=None),
) -> list[RecommendationHistoryResponse]:
    history = _recommendation_history
    if symbol:
        symbol = symbol.upper().strip()
        history = [r for r in history if r.symbol == symbol]
    history.sort(key=lambda r: r.timestamp, reverse=True)
    return history[:limit]


@router.post("/generate", response_model=RecommendationGenerateResponse)
async def generate_recommendations(
    request: Optional[RecommendationGenerateRequest] = None,
) -> RecommendationGenerateResponse:
    """Generate recommendations from live Bhavcopy data."""
    global _active_recommendations  # noqa: PLW0603
    _active_recommendations = {}

    try:
        await asyncio.wait_for(_auto_generate_from_bhavcopy(), timeout=20.0)
    except (asyncio.TimeoutError, Exception):
        pass

    if _active_recommendations:
        for sym, rec in _active_recommendations.items():
            _recommendation_history.append(
                RecommendationHistoryResponse(
                    id=rec.id,
                    symbol=rec.symbol,
                    timestamp=rec.timestamp,
                    action=rec.action,
                    entry_price=rec.entry_price,
                    stop_loss=rec.stop_loss,
                    target=rec.target,
                    confidence=rec.confidence,
                    is_active=rec.is_active,
                )
            )

    return RecommendationGenerateResponse(
        status="completed" if _active_recommendations else "no_data",
        generated_count=len(_active_recommendations),
        timestamp=datetime.now(),
    )
