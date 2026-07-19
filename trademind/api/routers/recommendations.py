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
from pydantic import BaseModel

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

    SL / target thresholds respect the current trading mode from settings.
    """
    from trademind.api.routers.scores import _score_from_bhavcopy

    mode = cfg.trading_mode
    if mode == "intraday":
        sl_mult = cfg.intraday_sl_percent / 100.0
        tgt_mult = cfg.intraday_target_percent / 100.0
        holding_label = "intraday"
    else:
        sl_mult = cfg.swing_sl_percent / 100.0
        tgt_mult = cfg.swing_target_percent / 100.0
        holding_label = "swing"

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

            # Generate signal based on change %, OI, volume, and PCR
            has_oi = stock.oi > 0
            has_volume = stock.volume > 100000

            action = None
            conf = 0.0

            if has_oi:
                if chg > 2.0 and stock.oi > 20000:
                    action = "STRONG_BUY"
                    conf = min(0.75 + chg * 0.02, 0.92)
                elif chg > 0.5 and stock.oi > 5000:
                    action = "BUY"
                    conf = min(0.6 + chg * 0.04, 0.85)
                elif chg < -2.0 and stock.oi > 20000:
                    action = "STRONG_SELL"
                    conf = min(0.75 + abs(chg) * 0.02, 0.92)
                elif chg < -0.5 and stock.oi > 5000:
                    action = "SELL"
                    conf = min(0.6 + abs(chg) * 0.04, 0.85)
            else:
                if chg > 2.5 and has_volume:
                    action = "STRONG_BUY"
                    conf = min(0.70 + chg * 0.015, 0.88)
                elif chg > 0.8 and has_volume:
                    action = "BUY"
                    conf = min(0.55 + chg * 0.03, 0.80)
                elif chg < -2.5 and has_volume:
                    action = "STRONG_SELL"
                    conf = min(0.70 + abs(chg) * 0.015, 0.88)
                elif chg < -0.8 and has_volume:
                    action = "SELL"
                    conf = min(0.55 + abs(chg) * 0.03, 0.80)

            if action is None:
                continue

            # Compute SL / target using mode-specific thresholds
            if action in ("BUY", "STRONG_BUY"):
                sl = entry * (1.0 - sl_mult)
                target = entry * (1.0 + tgt_mult)
            else:
                sl = entry * (1.0 + sl_mult)
                target = entry * (1.0 - tgt_mult)

            risk_reward = round((target - entry) / (entry - sl), 1) if entry > sl else 1.0

            evidence = list(score_resp.evidence)
            evidence.append(f"Score: {score_resp.score}/100 | Change: {chg:+.2f}% | Mode: {mode}")

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
                holding_period=holding_label,
                risk_reward_ratio=risk_reward,
                evidence=evidence,
                is_active=True,
            )

        logger.info(
            "Auto-generated %d recommendations from Bhavcopy (%d stocks, %d options, mode=%s)",
            len(_active_recommendations),
            data.fo_count,
            data.option_count,
            mode,
        )
    except Exception as exc:
        logger.error("Bhavcopy recommendation generation failed: %s", exc)


@router.get("", response_model=list[TradeRecommendationResponse])
async def get_active_recommendations(
    mode: Optional[str] = Query(default=None, description="Filter by trading mode: intraday or swing"),
) -> list[TradeRecommendationResponse]:
    """Get all active recommendations. Auto-generates from Bhavcopy if empty.

    Pass ?mode=intraday or ?mode=swing to filter by holding_period.
    """
    if not _active_recommendations:
        try:
            await asyncio.wait_for(_auto_generate_from_bhavcopy(), timeout=15.0)
        except (asyncio.TimeoutError, Exception):
            pass

    recs = list(_active_recommendations.values())
    if mode:
        mode_lower = mode.lower().strip()
        recs = [r for r in recs if r.holding_period == mode_lower]

    return recs


# ── Trading Mode Endpoints ─────────────────────────────────
class TradingModeResponse(BaseModel):
    mode: str
    sl_percent: float
    target_percent: float
    max_holding: str


class TradingModeRequest(BaseModel):
    mode: str


@router.get("/mode", response_model=TradingModeResponse)
async def get_trading_mode() -> TradingModeResponse:
    """Get the current trading mode and its associated thresholds."""
    mode = cfg.trading_mode
    if mode == "intraday":
        return TradingModeResponse(
            mode=mode,
            sl_percent=cfg.intraday_sl_percent,
            target_percent=cfg.intraday_target_percent,
            max_holding=f"{cfg.intraday_max_holding_hours} hours",
        )
    return TradingModeResponse(
        mode=mode,
        sl_percent=cfg.swing_sl_percent,
        target_percent=cfg.swing_target_percent,
        max_holding=f"{cfg.swing_max_holding_days} days",
    )


@router.post("/mode", response_model=TradingModeResponse)
async def set_trading_mode(request: TradingModeRequest) -> TradingModeResponse:
    """Switch between intraday and swing trading mode."""
    new_mode = request.mode.lower().strip()
    if new_mode not in ("intraday", "swing"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Must be 'intraday' or 'swing'.",
        )

    cfg.trading_mode = new_mode
    logger.info("Trading mode switched to %s", new_mode)

    # Clear existing recommendations so they regenerate with new thresholds
    global _active_recommendations  # noqa: PLW0603
    _active_recommendations = {}

    if new_mode == "intraday":
        return TradingModeResponse(
            mode=new_mode,
            sl_percent=cfg.intraday_sl_percent,
            target_percent=cfg.intraday_target_percent,
            max_holding=f"{cfg.intraday_max_holding_hours} hours",
        )
    return TradingModeResponse(
        mode=new_mode,
        sl_percent=cfg.swing_sl_percent,
        target_percent=cfg.swing_target_percent,
        max_holding=f"{cfg.swing_max_holding_days} days",
    )


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
