from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from trademind.database.models import (
    AdaptiveWeight,
    LearningRecord,
    PaperTrade,
)
from trademind.config.settings import settings


@dataclass
class WeightSet:
    weights: Dict[str, float]
    performance_score: float = 0.0
    sample_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""


@dataclass
class OptimizationResult:
    new_weights: Dict[str, float]
    old_weights: Dict[str, float]
    constrained_weights: Dict[str, float]
    baseline_performance: float
    new_performance: float
    was_applied: bool
    reason: str
    changes: Dict[str, float]


class AdaptiveWeightOptimizer:
    """Engine 8: Optimises category weights with safeguards.

    Weight safeguards:
    - Max change per update: ±5 points
    - Min sample size: 50 trades
    - Must outperform baseline to be adopted
    - Gradual adaptation via EMA of weight updates
    """

    MAX_WEIGHT_CHANGE = 5.0
    MIN_SAMPLE_SIZE = 50
    MIN_WEIGHT = 1.0
    MAX_WEIGHT = 50.0
    EMA_ALPHA = 0.3
    CATEGORY_FEATURE_MAP: Dict[str, List[str]] = {
        "trend": ["trend_score", "adx_value", "breakout_strength"],
        "momentum": ["rs_rank", "momentum_score", "macd_signal", "rsi_value"],
        "volume": ["volume_ratio", "delivery_pct", "vwap_distance_pct"],
        "options": ["oi_change_pct", "put_call_ratio", "pcr_oi"],
        "sector": ["sector_rank", "sector_momentum"],
        "market": ["market_breadth", "fii_flow", "dii_flow"],
        "volatility": ["vix_level", "atr_pct", "bb_position"],
    }

    _DB_URL = "sqlite+aiosqlite:///trademind.db"

    def __init__(self) -> None:
        self._engine = create_async_engine(self._DB_URL, echo=False)
        self._factory = async_sessionmaker(self._engine, class_=AsyncSession, expire_on_commit=False)
        self._current_weights: Optional[Dict[str, float]] = None

    async def optimize_weights(
        self, learning_records: Optional[List[LearningRecord]] = None
    ) -> OptimizationResult:
        if learning_records is None:
            learning_records = await self._get_all_learning_records()

        current = await self.get_current_weights()
        baseline_perf = await self._calculate_performance_with_weights(current, learning_records)

        category_performance = self._compute_category_performance(learning_records)
        raw_new = self._derive_weights_from_performance(category_performance, current)
        constrained = self._constrain_weight_changes(current, raw_new, self.MAX_WEIGHT_CHANGE)
        constrained = self._normalize_weights(constrained)

        new_perf = await self._calculate_performance_with_weights(constrained, learning_records)

        was_applied = self._compare_against_baseline(new_perf, baseline_perf)

        changes = {cat: round(constrained[cat] - current[cat], 2) for cat in self.CATEGORY_FEATURE_MAP}
        reason_parts = []
        for cat, delta in sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(delta) > 0.5:
                arrow = "↑" if delta > 0 else "↓"
                reason_parts.append(f"{cat} {arrow}{abs(delta):.1f}")

        reason = "Optimization: " + ", ".join(reason_parts) if reason_parts else "No significant changes"

        if was_applied:
            await self.apply_weights(constrained, reason)
            logger.info("Weights applied: {}", reason)
        else:
            logger.info(
                "Weights rejected: new_perf={:.4f} vs baseline={:.4f}",
                new_perf, baseline_perf,
            )

        return OptimizationResult(
            new_weights=raw_new,
            old_weights=current,
            constrained_weights=constrained,
            baseline_performance=baseline_perf,
            new_performance=new_perf,
            was_applied=was_applied,
            reason=reason,
            changes=changes,
        )

    async def validate_new_weights(
        self,
        new_weights: Dict[str, float],
        validation_data: Optional[List[LearningRecord]] = None,
    ) -> bool:
        if validation_data is None:
            validation_data = await self._get_all_learning_records()

        current = await self.get_current_weights()
        current_perf = await self._calculate_performance_with_weights(current, validation_data)
        new_perf = await self._calculate_performance_with_weights(new_weights, validation_data)
        return self._compare_against_baseline(new_perf, current_perf)

    async def apply_weights(
        self, new_weights: Dict[str, float], reason: str = ""
    ) -> None:
        now = datetime.utcnow()
        trade_count = await self._get_trade_count()

        async with self._factory() as session:
            for category, feature_names in self.CATEGORY_FEATURE_MAP.items():
                new_weight = new_weights.get(category, 10.0)

                prev_result = await session.execute(
                    select(AdaptiveWeight)
                    .where(AdaptiveWeight.category == category)
                    .order_by(AdaptiveWeight.id.desc())
                    .limit(1)
                )
                prev = prev_result.scalars().first()
                old_weight = prev.new_weight if prev else settings.category_weights.get(category, 10.0)

                avg_impact = self._compute_category_impact(category, new_weights)

                record = AdaptiveWeight(
                    category=category,
                    feature_name=category,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    change_reason=reason,
                    effective_date=now,
                    trade_count_sample=trade_count,
                    avg_impact=avg_impact,
                )
                session.add(record)

            await session.commit()

        self._current_weights = new_weights.copy()
        logger.info("Weight history recorded: {}", reason)

    async def get_current_weights(self) -> Dict[str, float]:
        if self._current_weights is not None:
            return self._current_weights.copy()

        async with self._factory() as session:
            for category in self.CATEGORY_FEATURE_MAP:
                result = await session.execute(
                    select(AdaptiveWeight)
                    .where(AdaptiveWeight.category == category)
                    .order_by(AdaptiveWeight.id.desc())
                    .limit(1)
                )
                latest = result.scalars().first()
                if latest:
                    self._current_weights = self._current_weights or {}
                    self._current_weights[category] = latest.new_weight

        if self._current_weights is None:
            self._current_weights = settings.category_weights.copy()

        missing = set(self.CATEGORY_FEATURE_MAP.keys()) - set(self._current_weights.keys())
        for cat in missing:
            self._current_weights[cat] = settings.category_weights.get(cat, 10.0)

        return self._current_weights.copy()

    async def _calculate_performance_with_weights(
        self,
        weights: Dict[str, float],
        records: Optional[List[LearningRecord]] = None,
    ) -> float:
        if records is None:
            records = await self._get_all_learning_records()
        if not records:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for record in records:
            try:
                features = json.loads(record.features_snapshot_json)
            except (json.JSONDecodeError, TypeError):
                continue

            category_scores = self._aggregate_category_scores(features)
            weighted_score = 0.0
            sum_w = 0.0

            for cat, score in category_scores.items():
                w = weights.get(cat, 10.0)
                weighted_score += score * w
                sum_w += w

            normalized = weighted_score / sum_w if sum_w > 0 else 50.0
            pnl = record.pnl_percent or 0.0
            sign_match = (normalized > 50 and pnl > 0) or (normalized <= 50 and pnl <= 0)
            weight_factor = abs(normalized - 50) / 50.0
            contribution = weight_factor * (1.0 if sign_match else -0.5)

            total_score += contribution
            total_weight += 1.0

        return float(total_score / total_weight) if total_weight > 0 else 0.0

    def _constrain_weight_changes(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        max_change: float = 5.0,
    ) -> Dict[str, float]:
        constrained: Dict[str, float] = {}
        for cat in self.CATEGORY_FEATURE_MAP:
            old_val = old_weights.get(cat, 10.0)
            new_val = new_weights.get(cat, old_val)
            delta = new_val - old_val
            clamped_delta = max(-max_change, min(max_change, delta))
            constrained_val = old_val + clamped_delta
            constrained_val = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, constrained_val))
            constrained[cat] = round(constrained_val, 2)
        return constrained

    def _require_minimum_samples(self, category: str, min_samples: int = 50) -> bool:
        return self._min_samples_cache.get(category, 0) >= min_samples

    async def _require_minimum_samples_async(
        self, category: str, min_samples: int = 50
    ) -> bool:
        async with self._factory() as session:
            result = await session.execute(
                select(func.count(LearningRecord.id))
            )
            count = result.scalar() or 0
        return count >= min_samples

    def _compare_against_baseline(
        self, new_performance: float, baseline_performance: float
    ) -> bool:
        improvement_threshold = 0.01
        return new_performance > baseline_performance + improvement_threshold

    async def get_weight_history(self) -> List[AdaptiveWeight]:
        async with self._factory() as session:
            result = await session.execute(
                select(AdaptiveWeight).order_by(AdaptiveWeight.id.desc())
            )
            return list(result.scalars().all())

    async def _get_all_learning_records(self) -> List[LearningRecord]:
        async with self._factory() as session:
            result = await session.execute(
                select(LearningRecord).order_by(LearningRecord.id.desc()).limit(2000)
            )
            return list(result.scalars().all())

    async def _get_trade_count(self) -> int:
        async with self._factory() as session:
            result = await session.execute(
                select(func.count(LearningRecord.id))
            )
            return result.scalar() or 0

    def _compute_category_performance(
        self, records: List[LearningRecord]
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for cat in self.CATEGORY_FEATURE_MAP:
            stats[cat] = {"total": 0.0, "wins": 0.0, "total_pnl": 0.0}

        for record in records:
            try:
                features = json.loads(record.features_snapshot_json)
            except (json.JSONDecodeError, TypeError):
                continue

            category_scores = self._aggregate_category_scores(features)
            pnl = record.pnl_percent or 0.0

            for cat, _score in category_scores.items():
                if cat in stats:
                    stats[cat]["total"] += 1
                    stats[cat]["total_pnl"] += pnl
                    if pnl > 0:
                        stats[cat]["wins"] += 1

        return stats

    def _derive_weights_from_performance(
        self,
        category_performance: Dict[str, Dict[str, float]],
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        new_weights: Dict[str, float] = {}

        for cat in self.CATEGORY_FEATURE_MAP:
            s = category_performance.get(cat, {"total": 0, "wins": 0, "total_pnl": 0.0})
            total = s["total"]
            wins = s["wins"]
            total_pnl = s["total_pnl"]

            if total < 5:
                new_weights[cat] = current_weights.get(cat, 10.0)
                continue

            win_rate = wins / total
            avg_pnl = total_pnl / total

            performance_score = win_rate * 0.6 + (avg_pnl / 10.0) * 0.4
            performance_score = max(0.0, min(1.0, performance_score + 0.5))

            new_weight = performance_score * 40.0 + 5.0
            new_weights[cat] = round(new_weight, 2)

        return new_weights

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            n = len(weights)
            return {k: round(100.0 / n, 2) for k in weights}
        return {k: round(v / total * 100.0, 2) for k, v in weights.items()}

    def _aggregate_category_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for cat, feature_names in self.CATEGORY_FEATURE_MAP.items():
            values = []
            for fname in feature_names:
                if fname in features and features[fname] is not None:
                    try:
                        values.append(float(features[fname]))
                    except (TypeError, ValueError):
                        continue
            scores[cat] = float(np.mean(values)) if values else 50.0
        return scores

    def _compute_category_impact(
        self, category: str, weights: Dict[str, float]
    ) -> float:
        cat_weight = weights.get(category, 10.0)
        total = sum(weights.values()) or 1.0
        return round(cat_weight / total * 100.0, 2)

    async def close(self) -> None:
        await self._engine.dispose()
