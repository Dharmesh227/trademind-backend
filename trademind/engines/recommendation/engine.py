import logging
import uuid
from datetime import datetime
from decimal import Decimal
from math import floor
from typing import Dict, List, Optional, Tuple

from trademind.config.settings import settings as cfg
from trademind.database.models import (
    AIScoreResult,
    MarketBreadthData,
    PriceData,
    TradeRecommendation,
)

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates trade recommendations with evidence panels.

    Converts AI scores and features into actionable trade signals with
    entry price, stop loss (ATR-based), target (risk:reward), holding
    period, and a comprehensive evidence panel explaining the rationale.
    """

    ACTION_THRESHOLDS: List[Tuple[float, str]] = [
        (70.0, "STRONG_BUY"),
        (55.0, "BUY"),
        (38.0, "HOLD"),
        (25.0, "SELL"),
        (0.0, "STRONG_SELL"),
    ]

    CONFIDENCE_TO_PROBABILITY: Dict[str, float] = {
        "High": 0.91,
        "Medium": 0.72,
        "Low": 0.55,
    }

    def generate_recommendation(
        self,
        symbol: str,
        ai_score: AIScoreResult,
        features: Dict[str, float],
        market_data: Optional[PriceData] = None,
        all_scores: Optional[Dict[str, float]] = None,
        sector_scores: Optional[Dict[str, float]] = None,
        historical_patterns: Optional[Dict] = None,
    ) -> TradeRecommendation:
        action = self._determine_action(ai_score.overall_score, features)

        entry_price = self._get_entry_price(market_data)
        stop_loss = self._calculate_stop_loss(features, market_data)
        target = self._calculate_target(features, market_data, entry_price, stop_loss)

        expected_move = self._calculate_expected_move(
            entry_price, target
        )

        holding_period = self._suggest_holding_period(features, action)
        risk = self._assess_risk(features, market_data, ai_score)

        evidence = self._build_evidence_panel(
            symbol=symbol,
            ai_score=ai_score,
            features=features,
            action=action,
            all_scores=all_scores,
            sector_scores=sector_scores,
            historical_patterns=historical_patterns,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
        )

        probability = self.CONFIDENCE_TO_PROBABILITY.get(
            ai_score.confidence_level, 0.75
        )

        return TradeRecommendation(
            symbol=symbol,
            timestamp=datetime.now(),
            action=action,
            ai_score=ai_score.overall_score,
            entry_price=float(entry_price),
            stop_loss=float(stop_loss),
            target=float(target),
            expected_move_percent=expected_move,
            holding_period=holding_period,
            confidence=probability,
            evidence_json="[]",
            recommendation_id=str(uuid.uuid4())[:8].upper(),
            evidence_panel_json=str(evidence),
        )

    def _determine_action(
        self, score: float, features: Dict[str, float]
    ) -> str:
        for threshold, action in self.ACTION_THRESHOLDS:
            if score >= threshold:
                return action
        return "HOLD"

    def _get_entry_price(
        self, market_data: Optional[PriceData]
    ) -> Decimal:
        if market_data and market_data.close:
            return market_data.close
        return Decimal("0")

    def _calculate_stop_loss(
        self,
        features: Dict[str, float],
        market_data: Optional[PriceData],
    ) -> Decimal:
        if market_data is None or market_data.close == 0:
            return Decimal("0")

        entry = market_data.close
        atr_val = self._get_atr_value(features, market_data)

        if atr_val <= 0:
            atr_val = Decimal(str(float(entry) * 0.02))

        sl_distance = atr_val * Decimal(str(cfg.default_atr_multiplier_sl))
        return max(Decimal("0"), entry - sl_distance)

    def _get_atr_value(
        self, features: Dict[str, float], market_data: PriceData
    ) -> Decimal:
        close_float = float(market_data.close)
        atr_pct = features.get("atr_value", 50)
        if atr_pct != 50:
            atr_approx = close_float * (atr_pct / 100.0) * 0.1
        else:
            atr_approx = close_float * 0.02
        return Decimal(str(max(atr_approx, close_float * 0.005)))

    def _calculate_target(
        self,
        features: Dict[str, float],
        market_data: Optional[PriceData],
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
    ) -> Decimal:
        if entry_price is None or stop_loss is None or stop_loss == 0:
            if market_data:
                entry_price = market_data.close
                atr_val = self._get_atr_value(features, market_data)
                sl = entry_price - atr_val * Decimal(str(cfg.default_atr_multiplier_sl))
                stop_loss = sl
            else:
                return Decimal("0")

        risk = entry_price - stop_loss
        if risk <= 0:
            return entry_price * Decimal("1.05")

        reward_ratio = Decimal(str(cfg.default_risk_reward_ratio))
        target = entry_price + risk * reward_ratio
        return target

    def _calculate_expected_move(
        self, entry: Decimal, target: Decimal
    ) -> float:
        if entry and entry > 0 and target > 0:
            return round(float((target - entry) / entry * 100), 2)
        return 0.0

    def _suggest_holding_period(
        self, features: Dict[str, float], action: str
    ) -> str:
        if action in ("STRONG_BUY", "STRONG_SELL"):
            return "1-3 Days"
        elif action in ("BUY", "SELL"):
            return "3-7 Days"
        else:
            return "Monitor"

    def _assess_risk(
        self,
        features: Dict[str, float],
        market_data: Optional[PriceData],
        ai_score: AIScoreResult,
    ) -> str:
        risk_factors = 0

        vix = features.get("vix_level", 50)
        if vix >= 70:
            risk_factors += 2
        elif vix >= 55:
            risk_factors += 1

        vol_score = features.get("historical_volatility", 50)
        if vol_score >= 70:
            risk_factors += 1

        delivery = features.get("delivery_percent", 50)
        if delivery <= 30:
            risk_factors += 1

        oi_build = features.get("oi_build_type", 50)
        if oi_build <= 25:
            risk_factors += 1

        adx = features.get("adx", 50)
        if adx <= 25:
            risk_factors += 1

        if market_data is None or market_data.close == 0:
            risk_factors += 2

        if risk_factors >= 4:
            return "High"
        elif risk_factors >= 2:
            return "Medium"
        return "Low"

    def _build_evidence_panel(
        self,
        symbol: str,
        ai_score: AIScoreResult,
        features: Dict[str, float],
        action: str,
        all_scores: Optional[Dict[str, float]] = None,
        sector_scores: Optional[Dict[str, float]] = None,
        historical_patterns: Optional[Dict] = None,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        target: Optional[Decimal] = None,
    ) -> List[str]:
        evidence: List[str] = []

        # Relative strength vs market
        if all_scores and len(all_scores) > 1:
            sorted_scores = sorted(all_scores.values(), reverse=True)
            rank = sum(1 for s in sorted_scores if s > ai_score.overall_score) + 1
            total = len(sorted_scores)
            pct = round((total - rank) / total * 100, 1)
            evidence.append(
                f"Stronger than {pct}% of F&O stocks"
            )

        # Sector rank
        sector_rank = features.get("sector_rank", 50)
        if sector_rank >= 80:
            evidence.append("Sector ranked in top quintile today")
        elif sector_rank >= 60:
            evidence.append("Sector performing above average today")

        # Breakout on volume
        vol_ratio = features.get("volume_expansion_ratio", 50)
        if vol_ratio >= 70:
            vol_mult = 1 + (vol_ratio - 50) / 25
            evidence.append(
                f"Breakout on {vol_mult:.1f}x average volume"
            )

        # Fresh OI build-up
        oi_build = features.get("oi_build_type", 50)
        if oi_build >= 80:
            evidence.append("Fresh long build-up in derivatives")
        elif oi_build <= 25:
            evidence.append("Bearish derivative positioning")

        # Trend alignment
        ema_bull = features.get("ema_bull_count", 0)
        if ema_bull >= 75:
            evidence.append("Above VWAP and all key EMAs (5/10/20/50/200)")

        # Historical pattern match
        if historical_patterns:
            pattern = self._match_historical_patterns(features, historical_patterns)
            if pattern:
                evidence.append(
                    f"Similar setups ({pattern.get('count', 0)} historical paper trades)\n"
                    f"   Win Rate: {pattern.get('win_rate', 0):.0f}%\n"
                    f"   Average Gain: +{pattern.get('avg_gain', 0):.1f}%\n"
                    f"   Average Loss: -{pattern.get('avg_loss', 0):.1f}%"
                )

        # MACD confirmation
        macd_cross = features.get("macd_crossover", 50)
        if macd_cross >= 75:
            evidence.append("Fresh MACD bullish crossover")
        elif macd_cross <= 25:
            evidence.append("MACD bearish crossover")

        # RSI condition
        rsi = features.get("rsi_14", 50)
        if rsi >= 75:
            evidence.append(f"RSI at {rsi:.0f} - strong momentum")
        elif rsi <= 35:
            evidence.append(f"RSI at {rsi:.0f} - potential reversal zone")

        # VWAP
        vwap_pos = features.get("vwap_position", 50)
        if vwap_pos >= 70:
            evidence.append("Trading above VWAP - intraday bias positive")
        elif vwap_pos <= 30:
            evidence.append("Trading below VWAP - intraday bias negative")

        # Delivery quality
        delivery = features.get("delivery_percent", 50)
        if delivery >= 70:
            evidence.append(f"High quality delivery at {delivery:.0f}%")

        # Entry/SL/Target
        if entry_price and stop_loss and target:
            risk_reward = float((target - entry_price) / (entry_price - stop_loss)) \
                if (entry_price - stop_loss) > 0 else 0
            evidence.append(
                f"Entry: {float(entry_price):.2f} | SL: {float(stop_loss):.2f} | "
                f"Target: {float(target):.2f} (R:R {risk_reward:.1f})"
            )

        return evidence[:12]

    def _match_historical_patterns(
        self,
        features: Dict[str, float],
        historical_patterns: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if historical_patterns is None:
            return None

        best_match = None
        best_score = 0.0

        for pattern_name, pattern_data in historical_patterns.items():
            conditions = pattern_data.get("conditions", {})
            match_count = 0
            total_conditions = len(conditions)

            for feat_name, (min_val, max_val) in conditions.items():
                feat_val = features.get(feat_name, 50)
                if min_val <= feat_val <= max_val:
                    match_count += 1

            if total_conditions > 0:
                match_score = match_count / total_conditions
                if match_score > best_score:
                    best_score = match_score
                    best_match = pattern_data

        if best_score >= 0.7:
            return best_match
        return None

    def _calculate_historical_percentile(
        self, score: float, all_scores: Dict[str, float]
    ) -> float:
        if not all_scores:
            return 0.0
        sorted_scores = sorted(all_scores.values(), reverse=True)
        rank = sum(1 for s in sorted_scores if s > score) + 1
        return round((len(sorted_scores) - rank) / len(sorted_scores) * 100, 1)
