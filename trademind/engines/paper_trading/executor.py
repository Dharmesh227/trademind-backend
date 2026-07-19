import logging
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from math import floor
from typing import Dict, List, Optional, Tuple

from trademind.config.settings import settings as cfg
from trademind.database.models import (
    PaperTrade,
    PortfolioSummary,
    PriceData,
    TradeRecommendation,
)

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Executes and monitors paper trades based on recommendations.

    Tracks entry/exit, P&L, holding time, max drawdown, max profit,
    trailing stops, and provides portfolio-level summaries.
    """

    def __init__(self, initial_capital: Decimal = None) -> None:
        self._trades: Dict[str, PaperTrade] = {}
        self._capital = initial_capital or cfg.default_capital
        self._initial_capital = self._capital
        self._price_cache: Dict[str, PriceData] = {}

    # ------------------------------------------------------------------
    # Trade Execution
    # ------------------------------------------------------------------

    def execute_trade(
        self, recommendation: TradeRecommendation
    ) -> Optional[PaperTrade]:
        trade_id = str(uuid.uuid4())[:12].upper()

        if recommendation.action in ("HOLD",):
            logger.info(f"Recommendation is HOLD for {recommendation.symbol}, skipping")
            return None

        open_count = self._count_open_positions()
        if open_count >= cfg.max_open_positions:
            logger.warning(
                f"Max open positions reached ({cfg.max_open_positions}), "
                f"cannot execute {recommendation.symbol}"
            )
            return None

        position_size = self._calculate_position_size(
            recommendation.entry_price, recommendation.stop_loss
        )
        if position_size <= 0:
            logger.warning(
                f"Invalid position size for {recommendation.symbol}: {position_size}"
            )
            return None

        if recommendation.entry_price * Decimal(str(position_size)) > self._capital:
            max_qty = int(float(self._capital) / float(recommendation.entry_price))
            position_size = max(1, max_qty)

        trade = PaperTrade(
            trade_id=trade_id,
            symbol=recommendation.symbol,
            action=recommendation.action,
            entry_price=recommendation.entry_price,
            quantity=position_size,
            entry_time=datetime.now(),
            stop_loss=recommendation.stop_loss,
            target=recommendation.target,
            initial_sl=recommendation.stop_loss,
            current_sl=recommendation.stop_loss,
            status="OPEN",
            ai_score_at_entry=recommendation.ai_score,
            recommendation_id=recommendation.recommendation_id,
            evidence_at_entry_json=recommendation.evidence_panel_json,
        )

        self._trades[trade_id] = trade
        cost = trade.entry_price * Decimal(str(trade.quantity))
        self._capital -= cost

        logger.info(
            f"Paper trade executed: {trade.action} {trade.symbol} "
            f"x {trade.quantity} @ {trade.entry_price} (ID: {trade_id})"
        )

        return trade

    def _calculate_position_size(
        self, entry_price: Decimal, stop_loss: Decimal
    ) -> int:
        if entry_price <= 0 or stop_loss < 0 or entry_price <= stop_loss:
            return 0

        risk_per_share = entry_price - stop_loss
        max_risk = self._capital * Decimal(
            str(cfg.max_position_size_percent / 100.0)
        )
        quantity = int(floor(float(max_risk) / float(risk_per_share)))
        return max(1, quantity)

    def _count_open_positions(self) -> int:
        return sum(
            1 for t in self._trades.values() if t.status == "OPEN"
        )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    async def monitor_open_trades(
        self, current_prices: Dict[str, PriceData]
    ) -> List[PaperTrade]:
        self._price_cache.update(current_prices)
        closed_trades: List[PaperTrade] = []

        for trade in list(self._trades.values()):
            if trade.status != "OPEN":
                continue

            current_data = current_prices.get(trade.symbol)
            if current_data is None:
                continue

            current_price = current_data.close

            exit_reason = self._check_exit_conditions(trade, current_data)
            if exit_reason:
                self.close_trade(trade.trade_id, exit_reason, current_price)
                closed_trades.append(self._trades.get(trade.trade_id))

            self._update_trade_metrics(trade, current_price)

        return closed_trades

    def _check_exit_conditions(
        self, trade: PaperTrade, current_data: PriceData
    ) -> Optional[str]:
        current_price = current_data.close
        holding_hours = (datetime.now() - trade.entry_time).total_seconds() / 3600

        # Stop loss hit
        stop = trade.current_sl or trade.stop_loss
        if trade.action in ("STRONG_BUY", "BUY") and current_price <= stop:
            return "STOP_LOSS_HIT"
        elif trade.action in ("STRONG_SELL", "SELL") and current_price >= stop:
            return "STOP_LOSS_HIT"

        # Target hit
        if trade.action in ("STRONG_BUY", "BUY") and current_price >= trade.target:
            return "TARGET_HIT"
        elif trade.action in ("STRONG_SELL", "SELL") and current_price <= trade.target:
            return "TARGET_HIT"

        # Time-based exit
        max_hours = cfg.max_holding_days * 24
        if holding_hours >= max_hours:
            return "TIME_BASED_EXIT"

        # Minimum holding period check
        if holding_hours < cfg.min_holding_hours:
            return None

        reentry_guard = self._check_reentry(trade, current_price)
        if reentry_guard:
            return reentry_guard

        return None

    def _check_reentry(
        self, trade: PaperTrade, current_price: Decimal
    ) -> Optional[str]:
        for existing in self._trades.values():
            if (
                existing.trade_id != trade.trade_id
                and existing.symbol == trade.symbol
                and existing.status == "CLOSED"
                and (datetime.now() - existing.exit_time).total_seconds() < 86400
            ):
                if existing.pnl and existing.pnl < 0:
                    return "REENTRY_BLOCKED_LOSS"
        return None

    # ------------------------------------------------------------------
    # Trade Closure
    # ------------------------------------------------------------------

    def close_trade(
        self,
        trade_id: str,
        exit_reason: str,
        exit_price: Optional[Decimal] = None,
    ) -> Optional[PaperTrade]:
        trade = self._trades.get(trade_id)
        if trade is None or trade.status != "OPEN":
            logger.warning(f"Trade {trade_id} not found or already closed")
            return None

        if exit_price is None:
            exit_price = self._get_last_price(trade.symbol)

        if exit_price is None or exit_price == 0:
            logger.error(f"Cannot close trade {trade_id}: no exit price")
            return None

        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = exit_reason

        if trade.action in ("STRONG_BUY", "BUY"):
            trade.pnl = (exit_price - trade.entry_price) * Decimal(str(trade.quantity))
        else:
            trade.pnl = (trade.entry_price - exit_price) * Decimal(str(trade.quantity))

        if trade.entry_price > 0:
            trade.pnl_percent = round(
                float(trade.pnl) / (float(trade.entry_price) * trade.quantity) * 100, 2
            )

        holding_seconds = (trade.exit_time - trade.entry_time).total_seconds()
        trade.holding_period_hours = round(holding_seconds / 3600, 2)
        trade.status = "CLOSED"

        self._capital += abs(float(exit_price) * trade.quantity) \
            if float(trade.pnl) >= 0 else \
            (abs(float(exit_price) * trade.quantity) - abs(float(trade.pnl)))

        logger.info(
            f"Trade closed: {trade.symbol} | {exit_reason} | "
            f"P&L: {trade.pnl:.2f} ({trade.pnl_percent:.2f}%) | "
            f"Held: {trade.holding_period_hours:.1f}h"
        )

        return trade

    def _get_last_price(self, symbol: str) -> Optional[Decimal]:
        cached = self._price_cache.get(symbol)
        if cached:
            return cached.close
        return None

    # ------------------------------------------------------------------
    # Trailing Stop
    # ------------------------------------------------------------------

    def _apply_trailing_stop(
        self, trade: PaperTrade, current_price: Decimal
    ) -> Optional[Decimal]:
        if trade.action not in ("STRONG_BUY", "BUY"):
            return trade.current_sl

        entry = trade.entry_price
        pct_profit = float((current_price - entry) / entry * 100)

        if pct_profit < cfg.trailing_stop_activate_percent:
            return trade.current_sl

        atr_val = float(entry) * 0.02
        new_sl = current_price - Decimal(str(
            atr_val * cfg.trailing_stop_gap_atr
        ))

        current_sl = trade.current_sl or trade.stop_loss
        if new_sl > current_sl:
            trade.trailing_stop_activated = True
            return new_sl

        return trade.current_sl

    def _update_trade_metrics(
        self, trade: PaperTrade, current_price: Decimal
    ) -> None:
        trade.current_sl = self._apply_trailing_stop(trade, current_price)

        if trade.action in ("STRONG_BUY", "BUY"):
            unrealized_pnl_pct = float(
                (current_price - trade.entry_price) / trade.entry_price * 100
            )
        else:
            unrealized_pnl_pct = float(
                (trade.entry_price - current_price) / trade.entry_price * 100
            )

        if trade.max_profit is None or unrealized_pnl_pct > trade.max_profit:
            trade.max_profit = round(unrealized_pnl_pct, 2)

        if trade.max_drawdown is None or unrealized_pnl_pct < trade.max_drawdown:
            trade.max_drawdown = round(unrealized_pnl_pct, 2)

    # ------------------------------------------------------------------
    # Portfolio Summary
    # ------------------------------------------------------------------

    def get_portfolio_summary(self) -> PortfolioSummary:
        all_trades = list(self._trades.values())
        open_trades = [t for t in all_trades if t.status == "OPEN"]
        closed_trades = [t for t in all_trades if t.status == "CLOSED"]

        total_pnl = Decimal("0.00")
        winning_trades = 0
        losing_trades = 0
        total_win_pnl = 0.0
        total_loss_pnl = 0.0

        for t in closed_trades:
            if t.pnl:
                total_pnl += t.pnl
                if t.pnl > 0:
                    winning_trades += 1
                    total_win_pnl += float(t.pnl)
                else:
                    losing_trades += 1
                    total_loss_pnl += abs(float(t.pnl))

        total_closed = len(closed_trades)
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
        avg_win = (total_win_pnl / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (total_loss_pnl / losing_trades) if losing_trades > 0 else 0.0
        profit_factor = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else 0.0

        max_drawdown_val, max_drawdown_pct = self._calculate_max_drawdown(closed_trades)

        summary = PortfolioSummary(
            total_trades=len(all_trades),
            open_positions=len(open_trades),
            closed_trades=total_closed,
            total_pnl=total_pnl,
            total_pnl_percent=round(
                float(total_pnl) / float(self._initial_capital) * 100, 2
            ) if self._initial_capital > 0 else 0.0,
            win_rate=round(win_rate, 1),
            average_win=round(avg_win, 2),
            average_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown=round(max_drawdown_val, 2),
            max_drawdown_percent=round(max_drawdown_pct, 2),
            current_capital=self._capital,
            initial_capital=self._initial_capital,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            open_positions_list=open_trades,
        )

        return summary

    def _calculate_max_drawdown(
        self, closed_trades: List[PaperTrade]
    ) -> Tuple[float, float]:
        if not closed_trades:
            return 0.0, 0.0

        peak = float(self._initial_capital)
        valley = peak
        max_dd = 0.0

        running_capital = float(self._initial_capital)
        for trade in sorted(closed_trades, key=lambda t: t.entry_time):
            if trade.pnl:
                running_capital += float(trade.pnl)
                if running_capital > peak:
                    peak = running_capital
                    valley = peak
                elif running_capital < valley:
                    valley = running_capital
                    dd = peak - valley
                    dd_pct = (dd / peak) * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd_pct)

        drawdown_amount = max_dd / 100.0 * peak if peak > 0 else 0.0
        return drawdown_amount, max_dd

    # ------------------------------------------------------------------
    # Trade History
    # ------------------------------------------------------------------

    def get_trade_history(
        self,
        filters: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict:
        trades = list(self._trades.values())

        if filters:
            if "symbol" in filters:
                trades = [
                    t for t in trades
                    if t.symbol.upper() == filters["symbol"].upper()
                ]
            if "status" in filters:
                trades = [
                    t for t in trades
                    if t.status == filters["status"].upper()
                ]
            if "action" in filters:
                trades = [
                    t for t in trades
                    if t.action == filters["action"].upper()
                ]
            if "date_from" in filters:
                date_from = filters["date_from"]
                trades = [t for t in trades if t.entry_time >= date_from]
            if "date_to" in filters:
                date_to = filters["date_to"]
                trades = [t for t in trades if t.entry_time <= date_to]
            if "min_score" in filters:
                min_score = filters["min_score"]
                trades = [
                    t for t in trades
                    if (t.ai_score_at_entry or 0) >= min_score
                ]
            if "outcome" in filters:
                outcome = filters["outcome"].lower()
                if outcome == "win":
                    trades = [t for t in trades if t.pnl and t.pnl > 0]
                elif outcome == "loss":
                    trades = [t for t in trades if t.pnl and t.pnl < 0]
                elif outcome == "open":
                    trades = [t for t in trades if t.status == "OPEN"]

        trades.sort(key=lambda t: t.entry_time, reverse=True)
        total = len(trades)
        start = (page - 1) * page_size
        end = start + page_size
        page_trades = trades[start:end]

        return {
            "trades": page_trades,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def get_trade_by_id(self, trade_id: str) -> Optional[PaperTrade]:
        return self._trades.get(trade_id)

    def get_open_positions(self) -> List[PaperTrade]:
        return [
            t for t in self._trades.values() if t.status == "OPEN"
        ]

    def get_recent_trades(self, n: int = 10) -> List[PaperTrade]:
        sorted_trades = sorted(
            self._trades.values(),
            key=lambda t: t.entry_time,
            reverse=True,
        )
        return sorted_trades[:n]

    def reset_portfolio(self, capital: Optional[Decimal] = None) -> None:
        self._trades.clear()
        self._capital = capital or cfg.default_capital
        self._initial_capital = self._capital
        self._price_cache.clear()
        logger.info("Portfolio reset completed")
