"""Walk-forward backtesting engine — replay AI pipeline against historical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from trademind.config.settings import settings as cfg
from trademind.database.models import PriceData, AIScoreResult
from trademind.engines.backtesting.data_store import historical_data_store
from trademind.engines.backtesting.trade_simulator import (
    TradeSimulator,
    TradeSignal,
)
from trademind.engines.backtesting.metrics import BacktestMetrics, BacktestMetricsResult
from trademind.engines.feature_engine.extractor import FeatureExtractor
from trademind.engines.ai_scoring.scorer import AIScoreEngine
from trademind.engines.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0
    metrics: Optional[BacktestMetricsResult] = None
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)


class WalkForwardEngine:
    """Walks through historical data day-by-day, running the full AI pipeline."""

    def __init__(
        self,
        capital: float = 100000.0,
        score_threshold: float = 55.0,
        max_positions: int = 10,
        slippage_pct: float = 0.05,
        trailing_stop_activate_pct: float = 1.5,
        trailing_stop_gap_pct: float = 1.0,
        max_holding_days: int = 10,
    ) -> None:
        self._capital = capital
        self._score_threshold = score_threshold
        self._max_positions = max_positions
        self._slippage = slippage_pct
        self._trailing_activate = trailing_stop_activate_pct
        self._trailing_gap = trailing_stop_gap_pct
        self._max_holding = max_holding_days

    async def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """Run a full walk-forward backtest."""
        logger.info(
            "Backtest starting: %s symbols, %s to %s, capital=%.0f",
            len(symbols), start_date, end_date, self._capital,
        )

        all_data = await historical_data_store.download_history(
            symbols, start_date, end_date
        )

        if not all_data:
            logger.warning("No historical data available for backtest")
            return BacktestResult(
                symbols=symbols, start_date=start_date, end_date=end_date,
                initial_capital=self._capital, final_capital=self._capital,
            )

        trading_days = self._get_trading_days(all_data)
        if not trading_days:
            return BacktestResult(
                symbols=symbols, start_date=start_date, end_date=end_date,
                initial_capital=self._capital, final_capital=self._capital,
            )

        warmup_days = 60
        if len(trading_days) <= warmup_days:
            logger.warning("Not enough data for warmup (%d days, need %d)", len(trading_days), warmup_days)
            return BacktestResult(
                symbols=symbols, start_date=start_date, end_date=end_date,
                initial_capital=self._capital, final_capital=self._capital,
            )

        simulator = TradeSimulator(
            initial_capital=self._capital,
            slippage_pct=self._slippage,
            max_positions=self._max_positions,
            trailing_stop_activate_pct=self._trailing_activate,
            trailing_stop_gap_pct=self._trailing_gap,
            max_holding_days=self._max_holding,
        )

        extractor = FeatureExtractor()
        scorer = AIScoreEngine()
        rec_engine = RecommendationEngine()

        for day_idx, day in enumerate(trading_days):
            date_str = day.strftime("%Y-%m-%d")
            simulator.set_date(date_str)

            if day_idx < warmup_days:
                self._populate_history_up_to(extractor, all_data, symbols, day)
                simulator.record_equity(date_str, self._get_prices(all_data, symbols, day))
                continue

            self._populate_history_up_to(extractor, all_data, symbols, day)

            current_prices = self._get_prices(all_data, symbols, day)
            closed = simulator.update_positions(date_str, current_prices)

            all_scores: Dict[str, float] = {}
            for symbol in symbols:
                features = self._compute_features_at_date(
                    extractor, all_data, symbol, day
                )
                if not features:
                    continue

                current_price = self._get_bar_at_date(all_data, symbol, day)
                if not current_price or current_price.close <= 0:
                    continue

                try:
                    ai_score = scorer.score_symbol(symbol, features)
                    all_scores[symbol] = ai_score.overall_score

                    if ai_score.overall_score >= self._score_threshold:
                        rec = rec_engine.generate_recommendation(
                            symbol=symbol,
                            ai_score=ai_score,
                            features=features,
                            market_data=current_price,
                            all_scores=all_scores,
                        )
                        if rec.action in ("BUY", "STRONG_BUY"):
                            signal = TradeSignal(
                                symbol=symbol,
                                date=date_str,
                                action="BUY",
                                score=ai_score.overall_score,
                                entry_price=float(rec.entry_price),
                                stop_loss=float(rec.stop_loss),
                                target=float(rec.target),
                                confidence=rec.confidence,
                            )
                            simulator.try_open_position(signal)

                except Exception as e:
                    logger.debug("Scoring failed for %s on %s: %s", symbol, date_str, e)

            simulator.record_equity(date_str, current_prices)

        final_prices = {}
        for symbol in symbols:
            if trading_days:
                last_bar = self._get_bar_at_date(all_data, symbol, trading_days[-1])
                if last_bar:
                    final_prices[symbol] = float(last_bar.close)

        simulator.close_all_positions(final_prices)
        if trading_days:
            simulator.record_equity(
                trading_days[-1].strftime("%Y-%m-%d"), final_prices
            )

        metrics = BacktestMetrics.compute(
            simulator.closed_trades,
            simulator.equity_curve,
            self._capital,
        )

        trades_data = []
        for t in simulator.closed_trades:
            trades_data.append({
                "symbol": t.symbol,
                "action": t.action,
                "entry_date": t.entry_date,
                "entry_price": round(t.entry_price, 2),
                "exit_date": t.exit_date,
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "quantity": t.quantity,
                "pnl": round(t.pnl, 2),
                "pnl_pct": t.pnl_pct,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
                "ai_score_at_entry": round(t.ai_score_at_entry, 1),
            })

        result = BacktestResult(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self._capital,
            final_capital=round(simulator.equity_curve[-1][1], 2) if simulator.equity_curve else self._capital,
            total_trades=metrics.total_trades,
            win_rate=metrics.win_rate,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown_pct=metrics.max_drawdown_pct,
            total_return_pct=metrics.total_return_pct,
            profit_factor=metrics.profit_factor,
            avg_holding_days=metrics.avg_holding_days,
            max_win_streak=metrics.max_win_streak,
            max_loss_streak=metrics.max_loss_streak,
            expectancy=metrics.expectancy,
            kelly_criterion=metrics.kelly_criterion,
            metrics=metrics,
            equity_curve=simulator.equity_curve,
            trades=trades_data,
            monthly_returns=metrics.monthly_returns,
        )

        logger.info(
            "Backtest complete: %d trades, %.1f%% return, Sharpe=%.2f, MaxDD=%.1f%%",
            result.total_trades, result.total_return_pct,
            result.sharpe_ratio, result.max_drawdown_pct,
        )

        return result

    def _get_trading_days(self, all_data: Dict[str, List[PriceData]]) -> List[datetime]:
        """Get sorted unique trading days from all symbol data."""
        all_dates = set()
        for prices in all_data.values():
            for p in prices:
                all_dates.add(p.timestamp.replace(tzinfo=None))
        return sorted(all_dates)

    def _populate_history_up_to(
        self,
        extractor: FeatureExtractor,
        all_data: Dict[str, List[PriceData]],
        symbols: List[str],
        cutoff: datetime,
    ) -> None:
        """Populate extractor._price_history with bars up to cutoff date."""
        for symbol in symbols:
            prices = all_data.get(symbol, [])
            bars = [p for p in prices if p.timestamp.replace(tzinfo=None) <= cutoff]
            if bars:
                extractor._price_history[symbol] = bars

    def _get_prices(
        self, all_data: Dict[str, List[PriceData]], symbols: List[str], day: datetime
    ) -> Dict[str, float]:
        """Get closing prices for all symbols on a specific day."""
        prices = {}
        for symbol in symbols:
            bar = self._get_bar_at_date(all_data, symbol, day)
            if bar:
                prices[symbol] = float(bar.close)
        return prices

    def _get_bar_at_date(
        self, all_data: Dict[str, List[PriceData]], symbol: str, day: datetime
    ) -> Optional[PriceData]:
        """Get a single bar for a symbol on a specific day."""
        for p in all_data.get(symbol, []):
            if p.timestamp.replace(tzinfo=None).date() == day.date():
                return p
        return None

    def _compute_features_at_date(
        self,
        extractor: FeatureExtractor,
        all_data: Dict[str, List[PriceData]],
        symbol: str,
        day: datetime,
    ) -> Optional[Dict[str, float]]:
        """Compute features for a symbol at a specific date."""
        bars = all_data.get(symbol, [])
        bars_up_to = [p for p in bars if p.timestamp.replace(tzinfo=None) <= day]

        if len(bars_up_to) < 20:
            return None

        current = bars_up_to[-1]
        extractor._price_history[symbol] = bars_up_to

        try:
            return extractor.compute_all_features(
                symbol=symbol,
                market_data=current,
                option_data=None,
            )
        except Exception:
            return None
