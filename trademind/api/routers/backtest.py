"""Backtest API endpoints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from trademind.config.settings import settings as cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["Backtesting"])


class BacktestRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: cfg.fno_symbols[:10])
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    capital: float = Field(default=100000.0)
    score_threshold: float = Field(default=55.0)
    max_positions: int = Field(default=10)
    slippage_pct: float = Field(default=0.05)
    max_holding_days: int = Field(default=10)


class BacktestResponse(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    avg_holding_days: float
    max_win_streak: int
    max_loss_streak: int
    expectancy: float
    kelly_criterion: float
    monthly_returns: Dict[str, float]
    equity_curve: List[List]
    trades: List[Dict]


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    """Run a walk-forward backtest over historical data."""
    from trademind.engines.backtesting.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(
        capital=request.capital,
        score_threshold=request.score_threshold,
        max_positions=request.max_positions,
        slippage_pct=request.slippage_pct,
        max_holding_days=request.max_holding_days,
    )

    try:
        result = await engine.run_backtest(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        return BacktestResponse(
            symbols=result.symbols,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_capital=result.final_capital,
            total_return_pct=result.total_return_pct,
            total_trades=result.total_trades,
            win_rate=result.win_rate,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.metrics.sortino_ratio if result.metrics else 0,
            max_drawdown_pct=result.max_drawdown_pct,
            profit_factor=result.profit_factor,
            avg_holding_days=result.avg_holding_days,
            max_win_streak=result.max_win_streak,
            max_loss_streak=result.max_loss_streak,
            expectancy=result.expectancy,
            kelly_criterion=result.kelly_criterion,
            monthly_returns=result.monthly_returns,
            equity_curve=[[date, val] for date, val in result.equity_curve],
            trades=result.trades,
        )
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/symbols")
async def list_backtest_symbols() -> List[str]:
    """Return available symbols for backtesting."""
    return cfg.fno_symbols
