"""Backtest metrics — compute performance statistics from simulated trades."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BacktestMetricsResult:
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_holding_days: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    pnl_curve: List[Tuple[str, float]] = field(default_factory=list)
    win_count: int = 0
    loss_count: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0


class BacktestMetrics:
    """Compute performance metrics from simulated trades."""

    @staticmethod
    def compute(
        trades: list,
        equity_curve: List[Tuple[str, float]],
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.06,
    ) -> BacktestMetricsResult:
        result = BacktestMetricsResult()

        if not trades:
            return result

        result.total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        result.win_count = len(winning)
        result.loss_count = len(losing)
        result.win_rate = round(len(winning) / len(trades) * 100, 2) if trades else 0

        total_pnl = sum(t.pnl for t in trades)
        final_capital = equity_curve[-1][1] if equity_curve else initial_capital
        result.total_return_pct = round(
            (final_capital - initial_capital) / initial_capital * 100, 2
        )

        if trades:
            holding_days = [t.holding_days for t in trades if t.holding_days > 0]
            result.avg_holding_days = round(
                sum(holding_days) / len(holding_days), 1
            ) if holding_days else 0

        result.avg_trade_pnl = round(total_pnl / len(trades), 2) if trades else 0

        if winning:
            result.avg_win_pct = round(
                sum(t.pnl_pct for t in winning) / len(winning), 2
            )
        if losing:
            result.avg_loss_pct = round(
                sum(t.pnl_pct for t in losing) / len(losing), 2
            )

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        result.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

        avg_win = gross_profit / len(winning) if winning else 0
        avg_loss = gross_loss / len(losing) if losing else 1
        result.expectancy = round(
            (result.win_rate / 100 * avg_win) - ((1 - result.win_rate / 100) * avg_loss), 2
        )

        if result.win_rate > 0 and result.win_rate < 100:
            win_pct = result.win_rate / 100
            result.kelly_criterion = round(
                (win_pct * avg_win - (1 - win_pct) * avg_loss) / avg_win * 100, 2
            ) if avg_win > 0 else 0

        daily_returns = BacktestMetrics._compute_daily_returns(equity_curve)
        if daily_returns:
            avg_daily = sum(daily_returns) / len(daily_returns)
            std_daily = BacktestMetrics._std(daily_returns)
            annual_factor = math.sqrt(252)

            if std_daily > 0:
                daily_rf = risk_free_rate / 252
                result.sharpe_ratio = round(
                    (avg_daily - daily_rf) / std_daily * annual_factor, 2
                )

                neg_returns = [r for r in daily_returns if r < 0]
                neg_std = BacktestMetrics._std(neg_returns) if neg_returns else 0
                if neg_std > 0:
                    result.sortino_ratio = round(
                        (avg_daily - daily_rf) / neg_std * annual_factor, 2
                    )

        if equity_curve:
            result.max_drawdown, result.max_drawdown_pct = BacktestMetrics._max_drawdown(
                equity_curve, initial_capital
            )

        if equity_curve:
            dates = [e[0] for e in equity_curve]
            monthly: Dict[str, float] = {}
            for i, (date, equity) in enumerate(equity_curve):
                month_key = date[:7]
                monthly[month_key] = equity
            prev = initial_capital
            for month, equity in monthly.items():
                result.monthly_returns[month] = round(
                    (equity - prev) / prev * 100, 2
                )
                prev = equity

        result.pnl_curve = equity_curve

        max_win = max_loss = current_win = current_loss = 0
        for t in trades:
            if t.pnl > 0:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        result.max_win_streak = max_win
        result.max_loss_streak = max_loss

        return result

    @staticmethod
    def _compute_daily_returns(curve: List[Tuple[str, float]]) -> List[float]:
        returns = []
        for i in range(1, len(curve)):
            prev = curve[i - 1][1]
            curr = curve[i][1]
            if prev > 0:
                returns.append((curr - prev) / prev)
        return returns

    @staticmethod
    def _max_drawdown(
        curve: List[Tuple[str, float]], initial: float
    ) -> Tuple[float, float]:
        peak = initial
        max_dd = 0.0
        max_dd_pct = 0.0
        for _, equity in curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        return round(max_dd, 2), round(max_dd_pct, 2)

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
