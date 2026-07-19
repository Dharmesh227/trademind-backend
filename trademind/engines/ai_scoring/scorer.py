import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from trademind.config.settings import settings as cfg
from trademind.database.models import AIScoreResult

logger = logging.getLogger(__name__)


class AIScoreEngine:
    """Dynamic weighted scoring system for ranking trading opportunities.

    Computes an overall AI score (0-100) from 100+ normalized features,
    grouped into categories: Trend, Momentum, Volume, Options, Sector,
    Market, and Volatility.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(cfg.category_weights)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            for k in self.weights:
                self.weights[k] /= total

    # ------------------------------------------------------------------
    # Category Feature Mappings
    # ------------------------------------------------------------------

    TREND_FEATURES: List[str] = [
        "ema_alignment_score", "ema_alignment", "ema_bull_count",
        "ema_price_vs_ema5", "ema_price_vs_ema20", "ema_price_vs_ema50",
        "ema5_slope", "ema20_slope",
        "vwap_distance", "vwap_position",
        "supertrend", "supertrend_direction",
        "adx", "plus_di", "minus_di", "di_cross",
        "bb_position",
    ]

    MOMENTUM_FEATURES: List[str] = [
        "rsi_14", "rsi_slope", "rsi_strength",
        "macd_value", "macd_histogram", "macd_signal", "macd_crossover",
        "momentum_score", "momentum_strength",
        "roc", "return_1d", "return_5d", "return_20d",
        "cci", "williams_r", "stoch_k", "stoch_d", "mfi",
    ]

    VOLUME_FEATURES: List[str] = [
        "volume_expansion_ratio", "delivery_percent", "volume_profile",
        "accumulation_distribution", "liquidity_score",
    ]

    OPTIONS_FEATURES: List[str] = [
        "pcr", "iv_rank", "iv_percentile", "oi_build_type",
        "max_pain_distance", "oi_ratio", "pcr_change",
    ]

    SECTOR_FEATURES: List[str] = [
        "sector_change", "sector_relative_strength", "sector_rank",
        "sector_momentum",
    ]

    MARKET_FEATURES: List[str] = [
        "market_breadth", "fii_dii_net", "vix_level", "vix_change",
        "gap_percent", "opening_range_percent",
    ]

    VOLATILITY_FEATURES: List[str] = [
        "historical_volatility", "hv_20", "implied_volatility",
        "iv_hv_spread", "atr_expansion_ratio",
    ]

    CATEGORY_MAP: Dict[str, List[str]] = {
        "trend": TREND_FEATURES,
        "momentum": MOMENTUM_FEATURES,
        "volume": VOLUME_FEATURES,
        "options": OPTIONS_FEATURES,
        "sector": SECTOR_FEATURES,
        "market": MARKET_FEATURES,
        "volatility": VOLATILITY_FEATURES,
    }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_symbol(
        self, symbol: str, features: Dict[str, float]
    ) -> AIScoreResult:
        if not features:
            return AIScoreResult(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_score=0.0,
                confidence_level="Low",
                evidence=["Insufficient features to score"],
            )

        scores: Dict[str, float] = {}
        for category, weight in self.weights.items():
            cat_score = self._calculate_category_score(
                category, features, self.weights
            )
            scores[category] = cat_score

        overall = sum(
            scores.get(cat, 0.0) * weight
            for cat, weight in self.weights.items()
        )
        overall = round(min(100.0, max(0.0, overall)), 2)

        completeness = features.get("feature_completeness", 50.0)
        feature_quality = self._calculate_feature_quality(features)
        confidence = self._calculate_confidence(overall, completeness, feature_quality)

        evidence = self._generate_evidence(features, scores, overall)

        return AIScoreResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall,
            trend_score=round(scores.get("trend", 0.0), 2),
            momentum_score=round(scores.get("momentum", 0.0), 2),
            volume_score=round(scores.get("volume", 0.0), 2),
            options_score=round(scores.get("options", 0.0), 2),
            sector_score=round(scores.get("sector", 0.0), 2),
            market_score=round(scores.get("market", 0.0), 2),
            volatility_score=round(scores.get("volatility", 0.0), 2),
            confidence_level=confidence[0],
            confidence_score=round(confidence[1], 2),
            evidence=evidence,
        )

    def _calculate_category_score(
        self, category: str, features: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        cat_features = self.CATEGORY_MAP.get(category, [])
        if not cat_features:
            return 0.0

        matched = []
        for fname in cat_features:
            if fname in features:
                matched.append(features[fname])

        if not matched:
            return 50.0

        avg = sum(matched) / len(matched)

        # Weighted toward extreme values (>70 or <30)
        extreme_count = sum(1 for v in matched if v >= 70 or v <= 30)
        extreme_boost = (extreme_count / len(matched)) * 10.0

        score = avg + (extreme_boost if avg >= 50 else -extreme_boost)
        return min(100.0, max(0.0, score))

    def _calculate_feature_quality(self, features: Dict[str, float]) -> float:
        non_default = sum(
            1 for v in features.values() if abs(v - 50.0) > 0.01
        )
        total = len(features)
        if total == 0:
            return 0.0
        return non_default / total

    def _calculate_confidence(
        self, score: float, completeness: float, feature_quality: float
    ) -> Tuple[str, float]:
        quality_score = (completeness / 100.0) * 0.4 + feature_quality * 0.6
        extreme = abs(score - 50.0) / 50.0
        confidence_val = quality_score * 0.7 + extreme * 0.3
        confidence_val = min(1.0, max(0.0, confidence_val))

        if confidence_val >= cfg.confidence_high_threshold:
            level = "High"
        elif confidence_val >= cfg.confidence_medium_threshold:
            level = "Medium"
        else:
            level = "Low"

        return level, round(confidence_val * 100, 2)

    def _generate_evidence(
        self,
        features: Dict[str, float],
        scores: Dict[str, float],
        overall: float,
    ) -> List[str]:
        evidence: List[str] = []

        if scores.get("trend", 50) >= 75:
            evidence.append("Strong trend alignment across EMAs and VWAP")
        elif scores.get("trend", 50) <= 35:
            evidence.append("Weak trend structure - price below key EMAs")

        if scores.get("momentum", 50) >= 75:
            evidence.append("Strong momentum with positive RSI slope and MACD")
        elif scores.get("momentum", 50) <= 35:
            evidence.append("Losing momentum - RSI declining or oversold")

        vol_score = scores.get("volume", 50)
        vol_ratio = features.get("volume_expansion_ratio", 50)
        if vol_score >= 70 and vol_ratio >= 70:
            evidence.append(
                f"High conviction volume at {features.get('volume_expansion_ratio', 0):.0f}% of average"
            )
        elif vol_score <= 35:
            evidence.append("Below-average volume participation")

        oi_build = features.get("oi_build_type", 50)
        if oi_build >= 80:
            evidence.append("Fresh long build-up detected in derivatives")
        elif oi_build <= 25:
            evidence.append("Short build-up or long unwinding in derivatives")

        pcr_val = features.get("pcr", 50)
        if pcr_val >= 70:
            evidence.append(f"Bullish PCR at {features.get('pcr', 0):.2f} - put writers active")
        elif pcr_val <= 30:
            evidence.append(f"Bearish PCR at {features.get('pcr', 0):.2f} - call writers active")

        sector_rank = features.get("sector_rank", 50)
        if sector_rank >= 80:
            evidence.append("Sector ranked in top quintile")
        if sector_rank <= 25:
            evidence.append("Sector underperforming - ranked in bottom quartile")

        vix_level = features.get("vix_level", 50)
        if vix_level >= 75:
            evidence.append("Elevated VIX - caution advised for new positions")
        elif vix_level <= 30:
            evidence.append("Low VIX environment favorable for trend trades")

        delivery = features.get("delivery_percent", 50)
        if delivery >= 70:
            evidence.append(f"High delivery percentage ({features.get('delivery_percent', 0):.0f}%)")
        elif delivery <= 30:
            evidence.append("Low delivery participation - speculative activity")

        supertrend = features.get("supertrend_direction", 50)
        if supertrend >= 70:
            evidence.append("Supertrend indicates bullish trend")
        elif supertrend <= 30:
            evidence.append("Supertrend indicates bearish trend")

        adx = features.get("adx", 50)
        if adx >= 65:
            evidence.append(f"Strong trend (ADX: {features.get('adx', 0):.0f})")
        elif adx <= 25:
            evidence.append("Low ADX - ranging market conditions")

        bb_pos = features.get("bb_position", 50)
        if bb_pos >= 85:
            evidence.append("Price near upper Bollinger Band - overextended")
        elif bb_pos <= 15:
            evidence.append("Price near lower Bollinger Band - potential reversal")

        return evidence[:10]  # max 10 evidence points

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_symbols(
        self, scored_results: List[AIScoreResult]
    ) -> List[AIScoreResult]:
        sorted_results = sorted(
            scored_results,
            key=lambda r: r.overall_score,
            reverse=True,
        )
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
            result.total_scored = len(sorted_results)
        return sorted_results

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        self.weights.update(new_weights)
        self._normalize_weights()
        logger.info(f"Weights updated: {self.weights}")

    def get_top_n(
        self, scored_results: List[AIScoreResult], n: int = 10
    ) -> List[AIScoreResult]:
        ranked = self.rank_symbols(scored_results)
        return ranked[:n]

    def get_buy_candidates(
        self, scored_results: List[AIScoreResult], threshold: float = 70.0
    ) -> List[AIScoreResult]:
        return [
            r for r in scored_results
            if r.overall_score >= threshold
            and r.confidence_level in ("High", "Medium")
        ]
