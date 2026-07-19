"""Portfolio P&L Dashboard API endpoints.

Computes P&L analytics from PaperTrade records stored in the database.
Uses BhavcopyEngine for live price lookups on open positions.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, and_

from trademind.config.settings import settings as cfg
from trademind.database.connection import get_session
from trademind.database.models import PaperTrade

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["Portfolio P&L"])

INITIAL_CAPITAL = cfg.default_capital


# ── Response Schemas ────────────────────────────────────────
class PortfolioSummaryResponse(BaseModel):
    total_invested: float = 0.0
    current_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    today_pnl: float = 0.0
    today_trade_count: int = 0


class PositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    holding_days: int
    status: str
    action: Optional[str] = None
    entry_time: datetime
    stop_loss: Optional[float] = None
    target: Optional[float] = None


class TradeHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_days: int
    status: str
    action: Optional[str] = None


class EquityCurvePoint(BaseModel):
    date: str
    equity_value: float
    daily_pnl: float
    cumulative_pnl: float


class MonthlyReturnItem(BaseModel):
    month: str
    pnl: float
    pnl_percent: float
    trades_count: int
    win_rate: float


# ── Helpers ─────────────────────────────────────────────────
_bhavcopy_engine = None


def _get_bhavcopy():
    global _bhavcopy_engine  # noqa: PLW0603
    if _bhavcopy_engine is None:
        from trademind.engines.bhavcopy.engine import BhavcopyEngine
        _bhavcopy_engine = BhavcopyEngine()
    return _bhavcopy_engine


def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _holding_days(entry_time: datetime, exit_time: Optional[datetime] = None) -> int:
    ref = exit_time or datetime.now(timezone.utc)
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    delta = ref - entry_time
    return max(0, delta.days)


# ── Endpoints ───────────────────────────────────────────────
@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary() -> PortfolioSummaryResponse:
    """Overall portfolio summary with P&L, win rate, drawdown, Sharpe."""
    async with get_session() as session:
        result = await session.execute(select(PaperTrade))
        trades = list(result.scalars().all())

    if not trades:
        return PortfolioSummaryResponse()

    closed = [t for t in trades if t.status in ("CLOSED", "completed", "closed")]
    open_trades = [t for t in trades if t.status in ("OPEN", "active", "open")]

    total_pnl = 0.0
    wins = 0
    losses = 0
    profit_sum = 0.0
    loss_sum = 0.0
    daily_pnls: Dict[str, float] = {}

    for t in trades:
        pnl = _safe_float(t.pnl)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
            profit_sum += pnl
        elif pnl < 0:
            losses += 1
            loss_sum += abs(pnl)

        entry_dt = t.entry_time
        if entry_dt:
            day_key = entry_dt.strftime("%Y-%m-%d")
            daily_pnls[day_key] = daily_pnls.get(day_key, 0.0) + pnl

    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    avg_profit = profit_sum / wins if wins > 0 else 0.0
    avg_loss = loss_sum / losses if losses > 0 else 0.0
    profit_factor = profit_sum / loss_sum if loss_sum > 0 else (float("inf") if profit_sum > 0 else 0.0)

    # Max drawdown from equity curve
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    equity_values: List[float] = []
    for day_key in sorted(daily_pnls.keys()):
        equity += daily_pnls[day_key]
        equity_values.append(equity)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized, using daily returns)
    if len(equity_values) >= 2:
        returns = []
        prev = equity_values[0]
        for ev in equity_values[1:]:
            if prev != 0:
                returns.append((ev - prev) / prev)
            prev = ev
        if returns:
            mean_ret = sum(returns) / len(returns)
            std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in returns) / len(returns))
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Today's P&L
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_pnl = 0.0
    today_count = 0
    for t in trades:
        if t.entry_time and t.entry_time.strftime("%Y-%m-%d") == today_str:
            today_pnl += _safe_float(t.pnl)
            today_count += 1

    total_invested = sum(
        _safe_float(t.entry_price) * t.quantity for t in trades
    )

    current_value = total_invested + total_pnl

    return PortfolioSummaryResponse(
        total_invested=round(total_invested, 2),
        current_value=round(current_value, 2),
        total_pnl=round(total_pnl, 2),
        total_pnl_percent=round((total_pnl / INITIAL_CAPITAL) * 100, 2) if INITIAL_CAPITAL else 0.0,
        win_count=wins,
        loss_count=losses,
        win_rate=round(win_rate, 2),
        avg_profit=round(avg_profit, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
        max_drawdown=round(max_dd * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        today_pnl=round(today_pnl, 2),
        today_trade_count=today_count,
    )


@router.get("/positions", response_model=list[PositionResponse])
async def get_open_positions() -> list[PositionResponse]:
    """Current open positions with live prices from Bhavcopy cache."""
    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).where(
                PaperTrade.status.in_(["OPEN", "active", "open"])
            )
        )
        trades = list(result.scalars().all())

    if not trades:
        return []

    # Get live prices from Bhavcopy cache
    current_prices: Dict[str, float] = {}
    try:
        bhavcopy = _get_bhavcopy()
        data = await bhavcopy.get_bhavcopy()
        for sym, stock in data.stocks.items():
            if stock.close > 0:
                current_prices[sym] = stock.close
    except Exception as exc:
        logger.warning("Could not fetch live prices: %s", exc)

    positions: list[PositionResponse] = []
    for t in trades:
        entry_p = _safe_float(t.entry_price)
        qty = t.quantity or 0

        # Try live price, fall back to entry price
        sym_upper = t.symbol.upper().strip() if t.symbol else ""
        cp = current_prices.get(sym_upper, entry_p)

        unrealized_pnl = (cp - entry_p) * qty
        pnl_pct = ((cp - entry_p) / entry_p * 100) if entry_p > 0 else 0.0

        positions.append(
            PositionResponse(
                symbol=t.symbol,
                entry_price=round(entry_p, 2),
                current_price=round(cp, 2),
                quantity=qty,
                pnl=round(unrealized_pnl, 2),
                pnl_percent=round(pnl_pct, 2),
                holding_days=_holding_days(t.entry_time),
                status=t.status,
                action=getattr(t, "action", None),
                entry_time=t.entry_time,
                stop_loss=_safe_float(t.stop_loss) or None,
                target=_safe_float(t.target) or None,
            )
        )

    return positions


@router.get("/history", response_model=dict)
async def get_trade_history(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict:
    """Paginated trade history with P&L for each trade."""
    async with get_session() as session:
        count_result = await session.execute(select(func.count(PaperTrade.id)))
        total = count_result.scalar() or 0

        total_pages = max(1, math.ceil(total / limit))
        offset = (page - 1) * limit

        result = await session.execute(
            select(PaperTrade).order_by(PaperTrade.entry_time.desc()).offset(offset).limit(limit)
        )
        trades = list(result.scalars().all())

    items: list[dict] = []
    for t in trades:
        entry_p = _safe_float(t.entry_price)
        exit_p = _safe_float(t.exit_price) if t.exit_time else None
        qty = t.quantity or 0

        if t.pnl is not None:
            pnl = _safe_float(t.pnl)
            pnl_pct = t.pnl_percent if t.pnl_percent is not None else (
                round((pnl / (entry_p * qty)) * 100, 2) if entry_p * qty > 0 else 0.0
            )
        elif exit_p is not None:
            pnl = (exit_p - entry_p) * qty
            pnl_pct = round(((exit_p - entry_p) / entry_p) * 100, 2) if entry_p > 0 else 0.0
        else:
            pnl = 0.0
            pnl_pct = 0.0

        status = t.status
        if status in ("OPEN", "active", "open"):
            status = "open"
        else:
            status = "closed"

        items.append({
            "symbol": t.symbol,
            "entry_date": t.entry_time.isoformat() if t.entry_time else None,
            "exit_date": t.exit_time.isoformat() if t.exit_time else None,
            "entry_price": round(entry_p, 2),
            "exit_price": round(exit_p, 2) if exit_p is not None else None,
            "quantity": qty,
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
            "holding_days": _holding_days(t.entry_time, t.exit_time),
            "status": status,
            "action": getattr(t, "action", None),
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": limit,
        "total_pages": total_pages,
    }


@router.get("/equity-curve", response_model=list[EquityCurvePoint])
async def get_equity_curve() -> list[EquityCurvePoint]:
    """Daily portfolio value over time."""
    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).order_by(PaperTrade.entry_time.asc())
        )
        trades = list(result.scalars().all())

    if not trades:
        return []

    # Aggregate P&L by day
    daily_pnl: Dict[str, float] = {}
    for t in trades:
        if t.entry_time:
            day_key = t.entry_time.strftime("%Y-%m-%d")
            daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + _safe_float(t.pnl)

    if not daily_pnl:
        return []

    curve: list[EquityCurvePoint] = []
    equity = INITIAL_CAPITAL
    cumulative = 0.0
    for day_key in sorted(daily_pnl.keys()):
        dp = round(daily_pnl[day_key], 2)
        equity += dp
        cumulative += dp
        curve.append(
            EquityCurvePoint(
                date=day_key,
                equity_value=round(equity, 2),
                daily_pnl=dp,
                cumulative_pnl=round(cumulative, 2),
            )
        )

    return curve


@router.get("/monthly-returns", response_model=list[MonthlyReturnItem])
async def get_monthly_returns() -> list[MonthlyReturnItem]:
    """Monthly P&L breakdown."""
    async with get_session() as session:
        result = await session.execute(
            select(PaperTrade).order_by(PaperTrade.entry_time.asc())
        )
        trades = list(result.scalars().all())

    if not trades:
        return []

    # Group by month
    monthly: Dict[str, Dict[str, float]] = {}
    for t in trades:
        if not t.entry_time:
            continue
        month_key = t.entry_time.strftime("%Y-%m")
        if month_key not in monthly:
            monthly[month_key] = {"pnl": 0.0, "wins": 0.0, "losses": 0.0, "count": 0.0}

        pnl = _safe_float(t.pnl)
        monthly[month_key]["pnl"] += pnl
        monthly[month_key]["count"] += 1
        if pnl > 0:
            monthly[month_key]["wins"] += 1
        elif pnl < 0:
            monthly[month_key]["losses"] += 1

    results: list[MonthlyReturnItem] = []
    for month_key in sorted(monthly.keys()):
        m = monthly[month_key]
        count = int(m["count"])
        wins = int(m["wins"])
        win_rate = (wins / count * 100) if count > 0 else 0.0
        pnl_pct = (m["pnl"] / INITIAL_CAPITAL * 100) if INITIAL_CAPITAL else 0.0
        results.append(
            MonthlyReturnItem(
                month=month_key,
                pnl=round(m["pnl"], 2),
                pnl_percent=round(pnl_pct, 2),
                trades_count=count,
                win_rate=round(win_rate, 2),
            )
        )

    return results
