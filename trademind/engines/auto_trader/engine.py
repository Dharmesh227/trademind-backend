"""Auto-trader engine — automatically executes paper trades from high-confidence recommendations.

Runs as a background task. After recommendations are generated, it:
1. Filters by confidence threshold (>= 0.70 by default)
2. Checks position limits (max open positions)
3. Deduplicates (no double-trade same symbol within cooldown)
4. Executes paper trades via PaperTradingEngine
5. Logs all actions for transparency
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from trademind.config.settings import settings as cfg
from trademind.database.models import TradeRecommendation

logger = logging.getLogger(__name__)

AUTO_TRADE_CONFIDENCE_THRESHOLD: float = 0.70
SYMBOL_COOLDOWN_HOURS: int = 4


class AutoTrader:
    """Executes paper trades automatically from qualifying recommendations."""

    def __init__(self) -> None:
        self._trade_log: List[Dict] = []
        self._symbol_last_trade: Dict[str, datetime] = {}

    def process_recommendations(
        self,
        recommendations: Dict[str, Any],
        paper_engine: Any,
    ) -> List[Dict]:
        """Evaluate all active recommendations and auto-execute qualifying ones."""
        actions: List[Dict] = []

        for symbol, rec in recommendations.items():
            action_result = self._evaluate_and_execute(rec, paper_engine)
            actions.append(action_result)
            self._trade_log.append({
                **action_result,
                "timestamp": datetime.now().isoformat(),
                "confidence": getattr(rec, "confidence", 0),
                "rec_id": getattr(rec, "id", symbol),
            })

        executed = sum(1 for a in actions if a["status"] == "executed")
        skipped = sum(1 for a in actions if a["status"] == "skipped")
        logger.info(
            "Auto-trade scan: %d recommendations evaluated, %d executed, %d skipped",
            len(actions), executed, skipped,
        )
        return actions

    def _evaluate_and_execute(
        self,
        rec: Any,
        paper_engine: Any,
    ) -> Dict:
        symbol = getattr(rec, "symbol", "?")

        if getattr(rec, "action", "HOLD") == "HOLD":
            return {"symbol": symbol, "action": "HOLD", "status": "skipped", "reason": "HOLD action — no trade"}

        confidence = getattr(rec, "confidence", 0)
        if confidence < AUTO_TRADE_CONFIDENCE_THRESHOLD:
            return {"symbol": symbol, "action": getattr(rec, "action", "?"), "status": "skipped",
                    "reason": f"Confidence {confidence:.2f} below threshold {AUTO_TRADE_CONFIDENCE_THRESHOLD}"}

        last_trade_time = self._symbol_last_trade.get(symbol)
        if last_trade_time:
            hours_since = (datetime.now() - last_trade_time).total_seconds() / 3600
            if hours_since < SYMBOL_COOLDOWN_HOURS:
                return {"symbol": symbol, "action": getattr(rec, "action", "?"), "status": "skipped",
                        "reason": f"Cooldown — last trade {hours_since:.1f}h ago (need {SYMBOL_COOLDOWN_HOURS}h)"}

        open_count = paper_engine._count_open_positions()
        if open_count >= cfg.max_open_positions:
            return {"symbol": symbol, "action": getattr(rec, "action", "?"), "status": "skipped",
                    "reason": f"Max open positions ({cfg.max_open_positions}) reached"}

        for t in paper_engine._trades.values():
            if t.symbol == symbol and t.status == "OPEN":
                return {"symbol": symbol, "action": getattr(rec, "action", "?"), "status": "skipped",
                        "reason": "Already have open position in this symbol"}

        rec_data = TradeRecommendation(
            symbol=symbol,
            timestamp=getattr(rec, "timestamp", datetime.now()),
            action=getattr(rec, "action", "BUY"),
            entry_price=float(getattr(rec, "entry_price", 0)),
            stop_loss=float(getattr(rec, "stop_loss", 0)),
            target=float(getattr(rec, "target", 0)),
            confidence=confidence,
            expected_move_percent=getattr(rec, "expected_move_percent", None),
            holding_period=getattr(rec, "holding_period", "intraday"),
            evidence_json=json.dumps(getattr(rec, "evidence", [])),
            ai_score=confidence * 100,
            recommendation_id=getattr(rec, "id", symbol),
            evidence_panel_json=json.dumps(getattr(rec, "evidence", [])),
        )

        trade = paper_engine.execute_trade(rec_data)

        if trade is None:
            return {"symbol": symbol, "action": getattr(rec, "action", "?"), "status": "rejected",
                    "reason": "Execution rejected by paper engine (risk limits)"}

        self._symbol_last_trade[symbol] = datetime.now()
        return {
            "symbol": symbol,
            "action": getattr(rec, "action", "?"),
            "status": "executed",
            "reason": f"Confidence {confidence:.2f} >= {AUTO_TRADE_CONFIDENCE_THRESHOLD}",
            "trade_id": getattr(trade, "trade_id", trade.id),
            "entry_price": float(trade.entry_price),
            "quantity": trade.quantity,
            "stop_loss": float(trade.stop_loss),
            "target": float(trade.target),
        }

    def get_trade_log(self) -> List[Dict]:
        return list(self._trade_log)

    def get_stats(self) -> Dict:
        executed = [a for a in self._trade_log if a["status"] == "executed"]
        skipped = [a for a in self._trade_log if a["status"] == "skipped"]
        rejected = [a for a in self._trade_log if a["status"] == "rejected"]
        return {
            "total_evaluated": len(self._trade_log),
            "executed": len(executed),
            "skipped": len(skipped),
            "rejected": len(rejected),
            "symbols_traded": list({a["symbol"] for a in executed}),
            "last_scan": self._trade_log[-1]["timestamp"] if self._trade_log else None,
        }


auto_trader = AutoTrader()
