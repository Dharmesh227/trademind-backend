"""Paper trading API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from trademind.api.schemas import (
    PaginatedResponse,
    PaperTradeResponse,
    PortfolioSummary,
    TradeCloseRequest,
    TradeCloseResponse,
    TradeExecuteResponse,
)
from trademind.config.settings import settings as cfg
from trademind.database.models import PaperTrade as PaperTradeModel
from trademind.engines.paper_trading.executor import PaperTradingEngine
from trademind.api.routers.recommendations import _active_recommendations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trades", tags=["Paper Trading"])

# Module-level singleton paper trading engine
_paper_engine: Optional[PaperTradingEngine] = None


def _get_paper_engine() -> PaperTradingEngine:
    """Get or create the singleton paper trading engine."""
    global _paper_engine  # noqa: PLW0603
    if _paper_engine is None:
        _paper_engine = PaperTradingEngine()
    return _paper_engine


def _decimal_to_float(val) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _trade_to_response(trade) -> PaperTradeResponse:
    """Convert a PaperTrade dataclass to API response."""
    return PaperTradeResponse(
        id=trade.trade_id if hasattr(trade, "trade_id") else trade.id,
        recommendation_id=getattr(trade, "recommendation_id", None),
        symbol=trade.symbol,
        action=getattr(trade, "action", None),
        entry_time=trade.entry_time,
        entry_price=_decimal_to_float(trade.entry_price) or 0.0,
        exit_time=trade.exit_time,
        exit_price=_decimal_to_float(trade.exit_price),
        stop_loss=_decimal_to_float(trade.stop_loss) or 0.0,
        target=_decimal_to_float(trade.target) or 0.0,
        quantity=trade.quantity,
        pnl=_decimal_to_float(trade.pnl),
        pnl_percent=trade.pnl_percent,
        holding_time_hours=getattr(trade, "holding_period_hours", None),
        max_drawdown=getattr(trade, "max_drawdown", None),
        max_profit=getattr(trade, "max_profit", None),
        exit_reason=getattr(trade, "exit_reason", None),
        status=trade.status,
        commission=getattr(trade, "commission", 0.0) or 0.0,
        slippage=getattr(trade, "slippage", 0.0) or 0.0,
    )


def _portfolio_to_response(summary) -> PortfolioSummary:
    """Convert a PortfolioSummary dataclass to API response."""
    return PortfolioSummary(
        total_trades=summary.total_trades,
        open_positions=summary.open_positions,
        closed_trades=summary.closed_trades,
        total_pnl=_decimal_to_float(summary.total_pnl) or 0.0,
        total_pnl_percent=summary.total_pnl_percent,
        win_rate=summary.win_rate,
        average_win=summary.average_win,
        average_loss=summary.average_loss,
        profit_factor=summary.profit_factor,
        max_drawdown=summary.max_drawdown,
        max_drawdown_percent=summary.max_drawdown_percent,
        current_capital=_decimal_to_float(summary.current_capital) or 0.0,
        initial_capital=_decimal_to_float(summary.initial_capital) or 0.0,
        winning_trades=summary.winning_trades,
        losing_trades=summary.losing_trades,
    )


@router.get("/portfolio", response_model=PortfolioSummary)
async def get_portfolio() -> PortfolioSummary:
    """Get full portfolio summary with P&L, win rate, drawdown."""
    engine = _get_paper_engine()
    summary = engine.get_portfolio_summary()
    return _portfolio_to_response(summary)


@router.get("/open", response_model=list[PaperTradeResponse])
async def get_open_positions() -> list[PaperTradeResponse]:
    """Get all currently open paper trade positions."""
    engine = _get_paper_engine()
    open_trades = engine.get_open_positions()
    return [_trade_to_response(t) for t in open_trades]


@router.get("/history", response_model=PaginatedResponse[PaperTradeResponse])
async def get_trade_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    symbol: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    outcome: Optional[str] = Query(default=None),
    min_score: Optional[float] = Query(default=None, ge=0, le=100),
) -> PaginatedResponse[PaperTradeResponse]:
    """Get paginated trade history with filters."""
    engine = _get_paper_engine()

    filters = {}
    if symbol:
        filters["symbol"] = symbol
    if status:
        filters["status"] = status
    if action:
        filters["action"] = action
    if outcome:
        filters["outcome"] = outcome
    if min_score is not None:
        filters["min_score"] = min_score

    result = engine.get_trade_history(
        filters=filters if filters else None,
        page=page,
        page_size=page_size,
    )

    items = [_trade_to_response(t) for t in result["trades"]]

    return PaginatedResponse(
        items=items,
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        total_pages=result["total_pages"],
    )


@router.get("/{trade_id}", response_model=PaperTradeResponse)
async def get_trade(trade_id: str) -> PaperTradeResponse:
    """Get details for a specific paper trade."""
    engine = _get_paper_engine()
    trade = engine.get_trade_by_id(trade_id)

    if trade is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trade '{trade_id}' not found.",
        )

    return _trade_to_response(trade)


@router.post("/execute/{recommendation_id}", response_model=TradeExecuteResponse)
async def execute_paper_trade(
    recommendation_id: str,
) -> TradeExecuteResponse:
    """Execute a paper trade based on an existing recommendation."""
    rec = None
    for r in _active_recommendations.values():
        if r.id == recommendation_id:
            rec = r
            break

    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation '{recommendation_id}' not found.",
        )

    if rec.action in ("HOLD",):
        return TradeExecuteResponse(
            status="skipped",
            message=f"Recommendation is {rec.action}. Cannot execute HOLD trades.",
        )

    engine = _get_paper_engine()

    from trademind.database.models import TradeRecommendation
    rec_data = TradeRecommendation(
        symbol=rec.symbol,
        timestamp=rec.timestamp,
        action=rec.action,
        entry_price=Decimal(str(rec.entry_price)),
        stop_loss=Decimal(str(rec.stop_loss)),
        target=Decimal(str(rec.target)),
        confidence=rec.confidence,
        expected_move_percent=rec.expected_move_percent,
        holding_period=rec.holding_period,
        evidence_json="[]",
    )

    trade = engine.execute_trade(rec_data)

    if trade is None:
        return TradeExecuteResponse(
            status="rejected",
            message="Trade execution rejected. Check position limits and risk parameters.",
        )

    return TradeExecuteResponse(
        status="executed",
        trade=_trade_to_response(trade),
        message=f"Paper trade executed: {trade.action} {trade.symbol} x {trade.quantity}",
    )


@router.post("/close/{trade_id}", response_model=TradeCloseResponse)
async def close_paper_trade(
    trade_id: str,
    request: Optional[TradeCloseRequest] = None,
) -> TradeCloseResponse:
    """Manually close a paper trade position."""
    engine = _get_paper_engine()
    trade = engine.get_trade_by_id(trade_id)

    if trade is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trade '{trade_id}' not found.",
        )

    if trade.status != "OPEN":
        raise HTTPException(
            status_code=400,
            detail=f"Trade '{trade_id}' is already {trade.status}.",
        )

    exit_price = Decimal(str(request.exit_price)) if request and request.exit_price else None
    exit_reason = request.exit_reason if request else "manual"

    closed = engine.close_trade(
        trade_id=trade_id,
        exit_reason=exit_reason,
        exit_price=exit_price,
    )

    if closed is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to close trade '{trade_id}'.",
        )

    try:
        from trademind.scheduler.tasks import learn_from_closed_trade
        import asyncio
        asyncio.create_task(learn_from_closed_trade(closed))
    except Exception as e:
        logger.warning("Learning trigger after manual close failed: %s", e)

    try:
        from trademind.engines.notifications.service import notification_service
        pnl_val = _decimal_to_float(closed.pnl)
        pnl_pct = closed.pnl_percent
        notif_task = notification_service.send_to_all(
            title="Trade Closed",
            body=f"{closed.symbol} — P&L: \u20b9{pnl_val} ({pnl_pct}%)",
            data={"type": "trade_closed", "symbol": closed.symbol, "pnl": str(pnl_val)},
        )
        import asyncio
        asyncio.create_task(notif_task)
    except Exception as e:
        logger.warning("Push notification after trade close failed: %s", e)

    return TradeCloseResponse(
        status="closed",
        trade=_trade_to_response(closed),
        message=f"Trade closed: {closed.symbol} | P&L: {_decimal_to_float(closed.pnl)}",
    )
