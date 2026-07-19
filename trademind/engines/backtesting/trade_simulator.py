"""Trade simulator — manages simulated positions during backtest."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    date: str
    action: str  # "BUY" / "SELL"
    score: float
    entry_price: float
    stop_loss: float
    target: float
    confidence: float


@dataclass
class SimulatedTrade:
    symbol: str
    entry_date: str
    entry_price: float
    action: str = "BUY"
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    stop_loss: float = 0.0
    current_sl: float = 0.0
    target: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""
    ai_score_at_entry: float = 0.0
    trailing_stop_active: bool = False
    max_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def unrealized_pnl(self) -> float:
        if not self.is_open:
            return self.pnl
        return 0.0


@dataclass
class TradeEvent:
    date: str
    symbol: str
    event_type: str  # "open", "close", "modify_sl"
    trade: Optional[SimulatedTrade] = None
    reason: str = ""


class TradeSimulator:
    """Manages simulated positions during backtest."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        slippage_pct: float = 0.05,
        max_positions: int = 10,
        trailing_stop_activate_pct: float = 1.5,
        trailing_stop_gap_pct: float = 1.0,
        max_holding_days: int = 10,
    ) -> None:
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._slippage_pct = slippage_pct
        self._max_positions = max_positions
        self._trailing_activate = trailing_stop_activate_pct
        self._trailing_gap = trailing_stop_gap_pct
        self._max_holding = max_holding_days

        self._open_trades: Dict[str, SimulatedTrade] = {}
        self._closed_trades: List[SimulatedTrade] = []
        self._equity_curve: List[Tuple[str, float]] = []
        self._current_date: str = ""

    @property
    def equity_curve(self) -> List[Tuple[str, float]]:
        return self._equity_curve

    @property
    def closed_trades(self) -> List[SimulatedTrade]:
        return self._closed_trades

    @property
    def open_trades(self) -> List[SimulatedTrade]:
        return list(self._open_trades.values())

    def set_date(self, date: str) -> None:
        self._current_date = date

    def try_open_position(self, signal: TradeSignal) -> Optional[TradeEvent]:
        """Open a position if conditions are met."""
        if signal.symbol in self._open_trades:
            return None
        if len(self._open_trades) >= self._max_positions:
            return None
        if signal.score < 55:
            return None

        entry_price = signal.entry_price * (1 + self._slippage_pct / 100)

        max_risk = self._capital * 0.10
        risk_per_share = abs(entry_price - signal.stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02

        quantity = max(1, int(max_risk / risk_per_share))
        cost = entry_price * quantity
        if cost > self._capital * 0.95:
            quantity = max(1, int(self._capital * 0.95 / entry_price))

        self._capital -= entry_price * quantity

        trade = SimulatedTrade(
            symbol=signal.symbol,
            entry_date=signal.date,
            entry_price=entry_price,
            action=signal.action,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            current_sl=signal.stop_loss,
            target=signal.target,
            ai_score_at_entry=signal.score,
            max_price=entry_price,
        )

        self._open_trades[signal.symbol] = trade
        logger.debug("OPEN %s %s x%d @ %.2f (SL=%.2f T=%.2f)",
                      signal.action, signal.symbol, quantity, entry_price,
                      signal.stop_loss, signal.target)

        return TradeEvent(
            date=signal.date,
            symbol=signal.symbol,
            event_type="open",
            trade=trade,
        )

    def update_positions(
        self, date: str, prices: Dict[str, float]
    ) -> List[TradeEvent]:
        """Check all open positions against current prices."""
        events: List[TradeEvent] = []

        for symbol in list(self._open_trades.keys()):
            trade = self._open_trades[symbol]
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            trade.holding_days += 1

            if trade.action == "BUY":
                trade.max_price = max(trade.max_price, current_price)
                pnl_per_share = current_price - trade.entry_price
            else:
                trade.max_price = min(trade.max_price, current_price) if trade.max_price > 0 else current_price
                pnl_per_share = trade.entry_price - current_price

            unrealized = pnl_per_share * trade.quantity

            exit_reason = self._check_exit(trade, current_price, unrealized)
            if exit_reason:
                self._close_trade(trade, date, current_price, exit_reason)
                events.append(TradeEvent(
                    date=date, symbol=symbol,
                    event_type="close", trade=trade, reason=exit_reason,
                ))
            else:
                self._update_trailing_stop(trade, current_price)

        return events

    def _check_exit(
        self, trade: SimulatedTrade, current_price: float, unrealized: float
    ) -> Optional[str]:
        """Check if a position should be closed."""
        if trade.action == "BUY":
            if current_price <= trade.current_sl:
                return "STOP_LOSS"
            if current_price >= trade.target:
                return "TARGET"
        else:
            if current_price >= trade.current_sl:
                return "STOP_LOSS"
            if current_price <= trade.target:
                return "TARGET"

        if trade.holding_days >= self._max_holding:
            return "MAX_HOLDING"

        return None

    def _update_trailing_stop(
        self, trade: SimulatedTrade, current_price: float
    ) -> None:
        """Update trailing stop if price has moved enough."""
        if trade.action == "BUY":
            gain_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
            if gain_pct >= self._trailing_activate:
                new_sl = current_price * (1 - self._trailing_gap / 100)
                if new_sl > trade.current_sl:
                    trade.current_sl = round(new_sl, 2)
                    trade.trailing_stop_active = True
        else:
            gain_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100
            if gain_pct >= self._trailing_activate:
                new_sl = current_price * (1 + self._trailing_gap / 100)
                if new_sl < trade.current_sl or trade.current_sl == trade.stop_loss:
                    trade.current_sl = round(new_sl, 2)
                    trade.trailing_stop_active = True

    def _close_trade(
        self, trade: SimulatedTrade, date: str, exit_price: float, reason: str
    ) -> None:
        """Close a position and update P&L."""
        exit_price_adjusted = exit_price * (1 - self._slippage_pct / 100)

        if trade.action == "BUY":
            trade.pnl = (exit_price_adjusted - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price_adjusted) * trade.quantity

        if trade.entry_price > 0:
            trade.pnl_pct = round(
                trade.pnl / (trade.entry_price * trade.quantity) * 100, 2
            )

        trade.exit_date = date
        trade.exit_price = exit_price_adjusted
        trade.exit_reason = reason

        self._capital += exit_price_adjusted * trade.quantity

        self._closed_trades.append(trade)
        del self._open_trades[trade.symbol]

    def close_all_positions(self, prices: Dict[str, float]) -> List[TradeEvent]:
        """Close all remaining positions at end of backtest."""
        events: List[TradeEvent] = []
        for symbol in list(self._open_trades.keys()):
            trade = self._open_trades[symbol]
            price = prices.get(symbol, trade.entry_price)
            self._close_trade(trade, self._current_date, price, "END_OF_PERIOD")
            events.append(TradeEvent(
                date=self._current_date, symbol=symbol,
                event_type="close", trade=trade, reason="END_OF_PERIOD",
            ))
        return events

    def record_equity(self, date: str, prices: Dict[str, float]) -> None:
        """Record current total equity (capital + unrealized)."""
        unrealized = 0.0
        for symbol, trade in self._open_trades.items():
            price = prices.get(symbol, trade.entry_price)
            if trade.action == "BUY":
                unrealized += (price - trade.entry_price) * trade.quantity
            else:
                unrealized += (trade.entry_price - price) * trade.quantity

        total = self._capital + unrealized
        self._equity_curve.append((date, round(total, 2)))

    def reset(self) -> None:
        """Reset for a new backtest run."""
        self._capital = self._initial_capital
        self._open_trades.clear()
        self._closed_trades.clear()
        self._equity_curve.clear()
        self._current_date = ""
