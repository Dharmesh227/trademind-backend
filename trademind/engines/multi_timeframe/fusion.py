"""Timeframe fusion — weighted blend of per-timeframe features and scores."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from trademind.database.models import AIScoreResult
from datetime import datetime

logger = logging.getLogger(__name__)


class TimeframeFusion:
    """Fuses features and scores from multiple timeframes into a single result."""

    DEFAULT_WEIGHTS = {
        "daily": 0.50,
        "hourly": 0.30,
        "15min": 0.20,
    }

    def fuse_features(
        self,
        features_by_tf: Dict[str, Dict[str, float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Weighted average of each feature across timeframes.

        Missing features in a timeframe use 50.0 (neutral).
        """
        w = weights or self.DEFAULT_WEIGHTS
        all_feature_names = set()
        for tf_features in features_by_tf.values():
            all_feature_names.update(tf_features.keys())

        fused: Dict[str, float] = {}
        for fname in all_feature_names:
            weighted_sum = 0.0
            weight_total = 0.0
            for tf_name, tf_features in features_by_tf.items():
                tf_weight = w.get(tf_name, 0.0)
                if tf_weight <= 0:
                    continue
                val = tf_features.get(fname, 50.0)
                weighted_sum += val * tf_weight
                weight_total += tf_weight

            if weight_total > 0:
                fused[fname] = round(weighted_sum / weight_total, 2)
            else:
                fused[fname] = 50.0

        alignment = self.compute_alignment_score(features_by_tf)
        fused["timeframe_alignment"] = alignment

        return fused

    def fuse_scores(
        self,
        score_results: Dict[str, AIScoreResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> AIScoreResult:
        """Weighted blend of per-timeframe AIScoreResults into one."""
        w = weights or self.DEFAULT_WEIGHTS

        if not score_results:
            return AIScoreResult(
                symbol="",
                timestamp=datetime.now(),
                overall_score=50.0,
            )

        weighted_score = 0.0
        weight_total = 0.0
        all_evidence = []
        symbol = ""

        for tf_name, result in score_results.items():
            tf_weight = w.get(tf_name, 0.0)
            if tf_weight <= 0:
                continue
            weighted_score += result.overall_score * tf_weight
            weight_total += tf_weight
            all_evidence.extend(result.evidence or [])
            if not symbol:
                symbol = result.symbol

        final_score = weighted_score / weight_total if weight_total > 0 else 50.0

        alignment = self.compute_alignment_score_from_scores(score_results)
        confidence = "Low"
        if alignment > 80:
            confidence = "High"
        elif alignment > 60:
            confidence = "Medium"

        return AIScoreResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=round(final_score, 1),
            confidence_level=confidence,
            evidence=all_evidence[:10],
        )

    def compute_alignment_score(
        self, features_by_tf: Dict[str, Dict[str, float]]
    ) -> float:
        """Returns 0-100: how aligned are the timeframes directionally.

        Checks key directional features across timeframes.
        """
        if len(features_by_tf) < 2:
            return 50.0

        directional_features = [
            "rsi_value", "macd_signal", "trend_score",
            "adx_value", "supertrend_signal",
        ]

        agreements = 0
        comparisons = 0

        for fname in directional_features:
            values = []
            for tf_name in ["daily", "hourly", "15min"]:
                if tf_name in features_by_tf:
                    val = features_by_tf[tf_name].get(fname)
                    if val is not None:
                        values.append(val)

            if len(values) < 2:
                continue

            above_50 = sum(1 for v in values if v > 50)
            below_50 = sum(1 for v in values if v <= 50)

            total = len(values)
            if above_50 == total or below_50 == total:
                agreements += 1
            comparisons += 1

        if comparisons == 0:
            return 50.0

        alignment = (agreements / comparisons) * 100
        return round(alignment, 1)

    def compute_alignment_score_from_scores(
        self, scores: Dict[str, AIScoreResult]
    ) -> float:
        """Alignment based on overall scores agreeing on direction."""
        if len(scores) < 2:
            return 50.0

        vals = [r.overall_score for r in scores.values()]
        above_50 = sum(1 for v in vals if v > 50)
        below_50 = sum(1 for v in vals if v <= 50)

        total = len(vals)
        if above_50 == total or below_50 == total:
            return 90.0
        elif abs(above_50 - below_50) <= 1:
            return 30.0
        else:
            return 60.0


timeframe_fusion = TimeframeFusion()
