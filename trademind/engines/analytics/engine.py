from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from trademind.database.models import (
    AdaptiveWeight,
    AIScore,
    LearningRecord,
    PaperTrade,
    TradeRecommendation,
    Pattern,
    KnowledgeBase,
    FeatureVector,
)
from trademind.config.settings import settings


@dataclass
class DashboardStats:
    ai_accuracy: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    best_scanner: str = ""
    worst_scanner: str = ""
    best_sector: str = ""
    best_time: str = ""
    best_day: str = ""
    worst_day: str = ""
    max_drawdown: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_holding_hours: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    avg_trade_pnl: float = 0.0
    kelly_criterion: float = 0.0


@dataclass
class CategoryPerformance:
    category: str
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0


def _profit_factor_from_arrays(wins: np.ndarray, losses: np.ndarray) -> float:
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(round(gross_profit / gross_loss, 4))


class AnalyticsEngine:
    """Engine 6: Comprehensive trade analytics and dashboard metrics."""

    def __init__(self) -> None:
        self._db_url: str = "sqlite+aiosqlite:///trademind.db"

    async def get_dashboard_stats(self) -> DashboardStats:
        trades = await self._get_closed_trades()
        if not trades:
            return DashboardStats()

        df = self._trades_to_dataframe(trades)

        stats = DashboardStats()
        stats.total_trades = len(df)
        stats.total_pnl = float(df["pnl"].sum())
        stats.avg_trade_pnl = float(df["pnl"].mean())

        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]

        stats.win_rate = float(len(wins) / len(df) * 100) if len(df) > 0 else 0.0
        stats.avg_win_pct = float(wins["pnl_percent"].mean()) if len(wins) > 0 else 0.0
        stats.avg_loss_pct = float(losses["pnl_percent"].mean()) if len(losses) > 0 else 0.0

        stats.profit_factor = self.calculate_profit_factor(trades)
        stats.sharpe_ratio = self.calculate_sharpe_ratio(trades)
        stats.max_drawdown, _ = self.get_drawdown_analysis(trades)
        stats.max_win_streak, stats.max_loss_streak = self.get_win_streak_analysis(trades)

        if "holding_time_hours" in df.columns:
            stats.avg_holding_hours = float(df["holding_time_hours"].mean())

        win_rate_dec = stats.win_rate / 100.0
        win_avg = abs(stats.avg_win_pct / 100.0) if stats.avg_win_pct else 0.0
        loss_avg = abs(stats.avg_loss_pct / 100.0) if stats.avg_loss_pct else 1.0
        stats.expectancy = (win_rate_dec * win_avg) - ((1.0 - win_rate_dec) * loss_avg)

        if loss_avg > 0 and win_avg > 0:
            stats.kelly_criterion = (
                (win_rate_dec * win_avg - (1.0 - win_rate_dec) * loss_avg) / win_avg
            )
        else:
            stats.kelly_criterion = 0.0

        stats.ai_accuracy = await self.calculate_accuracy(trades)

        stats.best_scanner, stats.worst_scanner = self._best_worst_by_column(df, "exit_reason")
        stats.best_sector, _ = self._best_worst_by_column(df, "symbol")

        if "entry_time" in df.columns and df["entry_time"].notna().any():
            entry_dt = pd.to_datetime(df["entry_time"])
            df["_hour"] = entry_dt.dt.hour
            hour_perf = df.groupby("_hour")["pnl"].mean()
            if not hour_perf.empty:
                stats.best_time = f"{int(hour_perf.idxmax()):02d}:00"

            df["_dow"] = entry_dt.dt.day_name()
            dow_perf = df.groupby("_dow")["pnl"].mean()
            if not dow_perf.empty:
                stats.best_day = str(dow_perf.idxmax())
                stats.worst_day = str(dow_perf.idxmin())

        return stats

    async def calculate_accuracy(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> float:
        if trades is None:
            trades = await self._get_closed_trades()
        if not trades:
            return 0.0

        correct = 0
        total = 0
        for t in trades:
            if t.pnl is None:
                continue
            total += 1
            if t.pnl > 0:
                correct += 1

        return float(correct / total * 100) if total > 0 else 0.0

    def calculate_sharpe_ratio(
        self,
        trades: Optional[List[PaperTrade]] = None,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
    ) -> float:
        returns = self._get_returns_array(trades)
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return == 0:
            return 0.0

        sharpe = (mean_return - risk_free_rate) / std_return
        if annualize:
            sharpe *= np.sqrt(252)
        return float(round(sharpe, 4))

    def calculate_profit_factor(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> float:
        if trades is None:
            return 0.0
        pnl_values = np.array([t.pnl for t in trades if t.pnl is not None])
        if len(pnl_values) == 0:
            return 0.0

        gross_profit = float(np.sum(pnl_values[pnl_values > 0]))
        gross_loss = float(abs(np.sum(pnl_values[pnl_values < 0])))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return float(round(gross_profit / gross_loss, 4))

    async def get_performance_by_category(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> List[CategoryPerformance]:
        if trades is None:
            trades = await self._get_closed_trades()
        if not trades:
            return []

        df = self._trades_to_dataframe(trades)
        if "exit_reason" not in df.columns:
            return []

        return self._build_category_performances(df, "exit_reason")

    async def get_performance_by_sector(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> List[CategoryPerformance]:
        if trades is None:
            trades = await self._get_closed_trades()
        if not trades:
            return []

        df = self._trades_to_dataframe(trades)
        if "symbol" not in df.columns:
            return []

        return self._build_category_performances(df, "symbol")

    async def get_performance_by_time(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> Dict[str, List[CategoryPerformance]]:
        if trades is None:
            trades = await self._get_closed_trades()
        if not trades:
            return {"hourly": [], "daily": []}

        df = self._trades_to_dataframe(trades)
        result: Dict[str, List[CategoryPerformance]] = {"hourly": [], "daily": []}

        if "entry_time" in df.columns and df["entry_time"].notna().any():
            entry_dt = pd.to_datetime(df["entry_time"])

            df["_hour"] = entry_dt.dt.hour
            result["hourly"] = self._build_category_performances(df, "_hour")

            df["_dow"] = entry_dt.dt.day_name()
            result["daily"] = self._build_category_performances(df, "_dow")

        return result

    def get_drawdown_analysis(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> Tuple[float, float]:
        if trades is None:
            return 0.0, 0.0

        pnl_values = np.array([t.pnl for t in trades if t.pnl is not None])
        if len(pnl_values) == 0:
            return 0.0, 0.0

        cumulative = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max

        max_drawdown = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0

        recovery_trades = 0.0
        if max_drawdown > 0:
            trough_idx = int(np.argmin(drawdowns))
            for i in range(trough_idx, len(drawdowns)):
                recovery_trades += 1.0
                if drawdowns[i] >= 0:
                    break
            else:
                recovery_trades = float(len(drawdowns) - trough_idx)

        return float(round(max_drawdown, 2)), float(round(recovery_trades, 1))

    def get_win_streak_analysis(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> Tuple[int, int]:
        if trades is None:
            return 0, 0

        pnl_values = [t.pnl for t in trades if t.pnl is not None]
        if not pnl_values:
            return 0, 0

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for pnl in pnl_values:
            if pnl > 0:
                current_win += 1
                current_loss = 0
            else:
                current_loss += 1
                current_win = 0
            max_win_streak = max(max_win_streak, current_win)
            max_loss_streak = max(max_loss_streak, current_loss)

        return max_win_streak, max_loss_streak

    async def _get_closed_trades(self) -> List[PaperTrade]:
        from trademind.database.models import Base
        from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        engine = create_async_engine(self._db_url, echo=False)
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            result = await session.execute(
                select(PaperTrade).where(PaperTrade.status.in_(["closed", "completed"]))
            )
            trades = list(result.scalars().all())
        await engine.dispose()
        return trades

    def _trades_to_dataframe(self, trades: List[PaperTrade]) -> pd.DataFrame:
        records = []
        for t in trades:
            records.append({
                "id": t.id,
                "symbol": t.symbol,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price or 0.0,
                "pnl": t.pnl or 0.0,
                "pnl_percent": t.pnl_percent or 0.0,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "holding_time_hours": t.holding_time_hours or 0.0,
                "exit_reason": t.exit_reason or "unknown",
                "status": t.status,
            })
        return pd.DataFrame(records)

    def _get_returns_array(
        self, trades: Optional[List[PaperTrade]] = None
    ) -> np.ndarray:
        if trades is None:
            return np.array([])
        returns = [t.pnl_percent / 100.0 for t in trades if t.pnl_percent is not None]
        return np.array(returns) if returns else np.array([])

    def _build_category_performances(
        self, df: pd.DataFrame, column: str
    ) -> List[CategoryPerformance]:
        results: List[CategoryPerformance] = []
        for cat_val, group in df.groupby(column):
            arr = group["pnl"].values.astype(float)
            wins = arr[arr > 0]
            losses = arr[arr <= 0]
            results.append(
                CategoryPerformance(
                    category=str(cat_val),
                    total_trades=len(arr),
                    win_count=int(len(wins)),
                    loss_count=int(len(losses)),
                    win_rate=float(len(wins) / len(arr) * 100) if len(arr) > 0 else 0.0,
                    avg_pnl_pct=float(np.mean(arr)),
                    total_pnl=float(np.sum(arr)),
                    profit_factor=_profit_factor_from_arrays(wins, losses),
                    avg_win_pct=float(np.mean(wins)) if len(wins) > 0 else 0.0,
                    avg_loss_pct=float(np.mean(losses)) if len(losses) > 0 else 0.0,
                )
            )
        results.sort(key=lambda x: x.total_pnl, reverse=True)
        return results

    async def close(self) -> None:
        pass

    @staticmethod
    def _best_worst_by_column(
        df: pd.DataFrame, column: str
    ) -> Tuple[str, str]:
        if column not in df.columns:
            return "", ""
        grouped = df.groupby(column)["pnl"].mean()
        if grouped.empty:
            return "", ""
        best = str(grouped.idxmax())
        worst = str(grouped.idxmin())
        return best, worst
