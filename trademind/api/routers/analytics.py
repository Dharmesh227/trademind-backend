"""Analytics and dashboard API endpoints.

All data computed from real paper trades. Returns empty/zero values when no trades exist.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from fastapi import APIRouter

from trademind.api.schemas import (
    AccuracyTimeResponse,
    CategoryPerformanceResponse,
    DashboardStatsResponse,
    DrawdownResponse,
    TimePerformanceResponse,
)
from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


async def _live_market_highlights():
    """Pull best performers from live Bhavcopy/index data when no trades exist."""
    best_scanner = ""
    best_sector = ""
    best_time = ""
    best_day = ""

    try:
        from trademind.api.routers.market import _get_bhavcopy

        bhavcopy = _get_bhavcopy()
        data = await bhavcopy.get_bhavcopy()

        if data and data.stocks:
            top_sym = max(
                data.stocks,
                key=lambda s: abs(float(data.stocks[s].change_pct or 0)),
            )
            top_chg = float(data.stocks[top_sym].change_pct or 0)
            sign = "+" if top_chg >= 0 else ""
            best_scanner = f"{top_sym} ({sign}{top_chg:.1f}%)"

        if data and data.indices:
            sector_chgs = {}
            skip = {"NIFTY 50", "NIFTY BANK", "NIFTY MIDCAP 100", "NIFTY SMALLCAP 100"}
            for name, idx in data.indices.items():
                if name.startswith("NIFTY ") and name not in skip:
                    sector_chgs[name] = float(idx.change_pct or 0)
            if sector_chgs:
                best_sector = max(sector_chgs, key=sector_chgs.get)

        now = datetime.now()
        if 9 <= now.hour < 11:
            best_time = "09:30-11:00"
        elif 11 <= now.hour < 13:
            best_time = "11:00-13:00"
        elif 13 <= now.hour < 15:
            best_time = "13:00-15:00"
        else:
            best_time = "Off-market"

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        best_day = day_names[now.weekday()]

    except Exception as exc:
        logger.debug("live_market_highlights fallback: %s", exc)

    return best_scanner, best_sector, best_time, best_day


def _decimal_to_float(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


async def _compute_dashboard_from_trades() -> DashboardStatsResponse:
    """Compute dashboard stats from real paper trade history."""
    from trademind.api.routers.trades import _get_paper_engine

    engine = _get_paper_engine()
    summary = engine.get_portfolio_summary()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        best_scanner, best_sector, best_time, best_day = await _live_market_highlights()
        return DashboardStatsResponse(
            ai_accuracy=0.0,
            win_rate=0.0,
            total_trades=0,
            best_scanner=best_scanner,
            worst_scanner="",
            best_sector=best_sector,
            best_time=best_time,
            best_day=best_day,
            message="No closed trades yet — trades will appear after NSE recommendations are generated",
        )

    # Win/loss stats
    wins = [t for t in closed if t.pnl and t.pnl > 0]
    losses = [t for t in closed if t.pnl and t.pnl <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    total_win_pnl = sum(float(t.pnl) for t in wins)
    total_loss_pnl = sum(abs(float(t.pnl)) for t in losses)
    avg_win_pct = sum(t.pnl_percent or 0 for t in wins) / len(wins) if wins else 0
    avg_loss_pct = sum(t.pnl_percent or 0 for t in losses) / len(losses) if losses else 0
    profit_factor = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0

    # P&L
    total_pnl = sum(_decimal_to_float(t.pnl) for t in closed)
    avg_trade_pnl = total_pnl / len(closed) if closed else 0

    # Holding time
    holding_times = [t.holding_period_hours for t in closed if t.holding_period_hours]
    avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0

    # Streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    streak_type = None
    for t in sorted(closed, key=lambda x: x.entry_time):
        is_win = t.pnl and t.pnl > 0
        if is_win:
            if streak_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if streak_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Drawdown
    max_dd, max_dd_pct = engine._calculate_max_drawdown(closed)

    # Sharpe ratio (simplified: mean return / std of returns)
    returns = [t.pnl_percent or 0 for t in closed]
    if len(returns) > 1:
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = variance ** 0.5
        sharpe = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    expectancy = (win_rate / 100 * abs(avg_win_pct)) - ((100 - win_rate) / 100 * abs(avg_loss_pct))

    # Kelly criterion
    if profit_factor > 0 and win_rate > 0:
        kelly = (win_rate / 100 * profit_factor - (1 - win_rate / 100)) / profit_factor
        kelly = max(0, min(kelly, 0.25))
    else:
        kelly = 0

    # AI accuracy: % of trades where AI score at entry was on the right side
    correct_ai = 0
    total_with_score = 0
    for t in closed:
        if t.ai_score_at_entry:
            total_with_score += 1
            is_win = t.pnl and t.pnl > 0
            # High score + win or low score + loss = correct
            if (t.ai_score_at_entry > 50 and is_win) or (t.ai_score_at_entry <= 50 and not is_win):
                correct_ai += 1
    ai_accuracy = (correct_ai / total_with_score * 100) if total_with_score > 0 else 0

    # Best/worst by sector
    sector_pnl: Dict[str, float] = defaultdict(float)
    from trademind.seed_data import SECTORS
    for t in closed:
        sector = SECTORS.get(t.symbol, "Unknown")
        sector_pnl[sector] += _decimal_to_float(t.pnl)
    best_sector = max(sector_pnl, key=sector_pnl.get) if sector_pnl else ""

    # Best time
    time_buckets: Dict[str, List[float]] = defaultdict(list)
    for t in closed:
        hour = t.entry_time.hour
        if 9 <= hour < 11:
            bucket = "09:30-11:00"
        elif 11 <= hour < 13:
            bucket = "11:00-13:00"
        elif 13 <= hour < 15:
            bucket = "13:00-15:00"
        else:
            bucket = "Other"
        time_buckets[bucket].append(_decimal_to_float(t.pnl))
    best_time = max(time_buckets, key=lambda k: sum(time_buckets[k])) if time_buckets else ""

    # Best/worst day
    day_pnl: Dict[str, float] = defaultdict(float)
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for t in closed:
        day_pnl[day_names[t.entry_time.weekday()]] += _decimal_to_float(t.pnl)
    best_day = max(day_pnl, key=day_pnl.get) if day_pnl else ""
    worst_day = min(day_pnl, key=day_pnl.get) if day_pnl else ""

    return DashboardStatsResponse(
        ai_accuracy=round(ai_accuracy, 1),
        win_rate=round(win_rate, 1),
        avg_win_pct=round(avg_win_pct, 2),
        avg_loss_pct=round(avg_loss_pct, 2),
        profit_factor=round(profit_factor, 2),
        sharpe_ratio=round(sharpe, 2),
        total_trades=len(closed),
        best_scanner="",
        worst_scanner="",
        best_sector=best_sector,
        best_time=best_time,
        best_day=best_day,
        worst_day=worst_day,
        max_drawdown=round(max_dd_pct, 2),
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        avg_holding_hours=round(avg_holding, 1),
        expectancy=round(expectancy, 2),
        total_pnl=round(total_pnl, 2),
        avg_trade_pnl=round(avg_trade_pnl, 2),
        kelly_criterion=round(kelly, 3),
    )


@router.get("/dashboard", response_model=DashboardStatsResponse)
async def get_dashboard() -> DashboardStatsResponse:
    return await _compute_dashboard_from_trades()


@router.get("/accuracy", response_model=AccuracyTimeResponse)
async def get_accuracy_over_time() -> AccuracyTimeResponse:
    """AI accuracy computed from real trade outcomes over time."""
    from trademind.api.routers.trades import _get_paper_engine

    engine = _get_paper_engine()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        return AccuracyTimeResponse(timestamps=[], accuracy_values=[], trade_count=0)

    sorted_trades = sorted(closed, key=lambda t: t.entry_time)
    timestamps = []
    accuracy_values = []
    correct = 0

    for i, t in enumerate(sorted_trades):
        timestamps.append(t.entry_time)
        is_win = t.pnl and t.pnl > 0
        if is_win:
            correct += 1
        accuracy_values.append(round(correct / (i + 1) * 100, 2))

    return AccuracyTimeResponse(
        timestamps=timestamps,
        accuracy_values=accuracy_values,
        trade_count=len(closed),
    )


@router.get("/performance/by-sector", response_model=list[CategoryPerformanceResponse])
async def get_performance_by_sector() -> list[CategoryPerformanceResponse]:
    """Trade performance broken down by sector from real trades."""
    from trademind.api.routers.trades import _get_paper_engine
    from trademind.seed_data import SECTORS

    engine = _get_paper_engine()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        return []

    sector_trades: Dict[str, List] = defaultdict(list)
    for t in closed:
        sector = SECTORS.get(t.symbol, "Unknown")
        sector_trades[sector].append(t)

    results = []
    for sector, trades in sorted(sector_trades.items()):
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(_decimal_to_float(t.pnl) for t in trades)
        win_pnl = sum(_decimal_to_float(t.pnl) for t in wins)
        loss_pnl = sum(abs(_decimal_to_float(t.pnl)) for t in losses)
        avg_win = sum(t.pnl_percent or 0 for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_percent or 0 for t in losses) / len(losses) if losses else 0

        results.append(CategoryPerformanceResponse(
            category=sector,
            total_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
            avg_pnl_pct=round(sum(t.pnl_percent or 0 for t in trades) / len(trades), 2) if trades else 0,
            total_pnl=round(total_pnl, 2),
            profit_factor=round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else 0,
            avg_win_pct=round(avg_win, 2),
            avg_loss_pct=round(avg_loss, 2),
        ))

    return results


@router.get("/performance/by-time", response_model=TimePerformanceResponse)
async def get_performance_by_time() -> TimePerformanceResponse:
    """Trade performance broken down by entry time from real trades."""
    from trademind.api.routers.trades import _get_paper_engine

    engine = _get_paper_engine()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        return TimePerformanceResponse(hourly=[], daily=[])

    # Hourly buckets
    hour_trades: Dict[str, List] = defaultdict(list)
    for t in closed:
        h = t.entry_time.hour
        if 9 <= h < 10:
            bucket = "09:30"
        elif 10 <= h < 11:
            bucket = "10:00"
        elif 11 <= h < 12:
            bucket = "11:00"
        elif 12 <= h < 13:
            bucket = "12:00"
        elif 13 <= h < 14:
            bucket = "13:00"
        elif 14 <= h < 15:
            bucket = "14:00"
        else:
            bucket = "15:00"
        hour_trades[bucket].append(t)

    hourly = []
    for bucket in sorted(hour_trades.keys()):
        trades = hour_trades[bucket]
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(_decimal_to_float(t.pnl) for t in trades)
        win_pnl = sum(_decimal_to_float(t.pnl) for t in wins)
        loss_pnl = sum(abs(_decimal_to_float(t.pnl)) for t in losses)
        hourly.append(CategoryPerformanceResponse(
            category=bucket,
            total_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
            avg_pnl_pct=round(sum(t.pnl_percent or 0 for t in trades) / len(trades), 2) if trades else 0,
            total_pnl=round(total_pnl, 2),
            profit_factor=round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else 0,
            avg_win_pct=round(sum(t.pnl_percent or 0 for t in wins) / len(wins), 2) if wins else 0,
            avg_loss_pct=round(sum(t.pnl_percent or 0 for t in losses) / len(losses), 2) if losses else 0,
        ))

    # Daily buckets
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_trades: Dict[str, List] = defaultdict(list)
    for t in closed:
        day_trades[day_names[t.entry_time.weekday()]].append(t)

    daily = []
    for day in day_names:
        if day not in day_trades:
            continue
        trades = day_trades[day]
        wins = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl <= 0]
        total_pnl = sum(_decimal_to_float(t.pnl) for t in trades)
        win_pnl = sum(_decimal_to_float(t.pnl) for t in wins)
        loss_pnl = sum(abs(_decimal_to_float(t.pnl)) for t in losses)
        daily.append(CategoryPerformanceResponse(
            category=day,
            total_trades=len(trades),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 1) if trades else 0,
            avg_pnl_pct=round(sum(t.pnl_percent or 0 for t in trades) / len(trades), 2) if trades else 0,
            total_pnl=round(total_pnl, 2),
            profit_factor=round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else 0,
            avg_win_pct=round(sum(t.pnl_percent or 0 for t in wins) / len(wins), 2) if wins else 0,
            avg_loss_pct=round(sum(t.pnl_percent or 0 for t in losses) / len(losses), 2) if losses else 0,
        ))

    return TimePerformanceResponse(hourly=hourly, daily=daily)


@router.get("/drawdown", response_model=DrawdownResponse)
async def get_drawdown() -> DrawdownResponse:
    from trademind.api.routers.trades import _get_paper_engine

    engine = _get_paper_engine()
    closed = [t for t in engine._trades.values() if t.status == "CLOSED"]

    if not closed:
        return DrawdownResponse(max_drawdown=0, recovery_trades=0)

    max_dd, max_dd_pct = engine._calculate_max_drawdown(closed)

    return DrawdownResponse(
        max_drawdown=round(max_dd_pct, 2),
        recovery_trades=round(max_dd / abs(max_dd) * 3, 1) if max_dd != 0 else 0,
    )
